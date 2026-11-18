"""Phase 5 coding tool behavior, with no live model or shell side effects outside temp files."""

import base64
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding.tools import (
    create_bash_tool,
    create_coding_tools,
    create_edit_tool,
    create_read_tool,
    create_write_tool,
)


class CodingToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.cwd = Path(self.temporary.name)

    async def test_default_tools_and_read_line_slicing(self) -> None:
        self.assertEqual([tool.name for tool in create_coding_tools(self.cwd)],
                         ["read", "write", "edit", "bash"])
        (self.cwd / "lines.txt").write_bytes(b"one\ntwo\nthree\n")
        read = create_read_tool(self.cwd)

        result = await read.execute({"path": "lines.txt", "offset": 2, "limit": 1})
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "2: two\n[Truncated; continue with offset=3]")
        self.assertEqual(result.data["next_offset"], 3)

        first = await read.execute({"path": "lines.txt", "offset": 0, "limit": 1})
        self.assertTrue(first.content.startswith("1: one"))

    async def test_read_image_metadata(self) -> None:
        image = b"\x89PNG\r\n\x1a\nexample"
        (self.cwd / "sample.png").write_bytes(image)

        result = await create_read_tool(self.cwd).execute({"path": "sample.png"})

        self.assertTrue(result.ok)
        self.assertEqual(result.data["mime_type"], "image/png")
        self.assertEqual(base64.b64decode(result.data["base64"]), image)

    async def test_write_creates_parents_and_preserves_content(self) -> None:
        write = create_write_tool(self.cwd)
        result = await write.execute({"path": "nested/file.txt", "content": "a\r\nb\r\n"})

        self.assertTrue(result.ok)
        self.assertEqual((self.cwd / "nested/file.txt").read_bytes(), b"a\r\nb\r\n")

    async def test_edit_multiple_replacements_preserves_bom_and_line_endings(self) -> None:
        path = self.cwd / "file.txt"
        path.write_bytes(b"\xef\xbb\xbfalpha\r\nbeta\r\ngamma\r\n")
        edit = create_edit_tool(self.cwd)

        result = await edit.execute({"path": "file.txt", "edits": [
            {"oldText": "alpha", "newText": "first\nline"},
            {"oldText": "gamma", "newText": "last"},
        ]})

        self.assertTrue(result.ok)
        self.assertEqual(path.read_bytes(), b"\xef\xbb\xbffirst\r\nline\r\nbeta\r\nlast\r\n")
        self.assertEqual(result.data["first_changed_line"], 1)
        self.assertIn("-alpha", result.data["patch"])
        self.assertIn("+ first", result.data["diff"])

    async def test_edit_validation_rolls_back_and_rejects_overlap(self) -> None:
        path = self.cwd / "file.txt"
        original = b"one two two three"
        path.write_bytes(original)
        edit = create_edit_tool(self.cwd)

        missing = await edit.execute({"path": "file.txt", "edits": [
            {"oldText": "one", "newText": "ONE"},
            {"oldText": "missing", "newText": "MISSING"},
        ]})
        duplicate = await edit.execute({"path": "file.txt", "edits": [
            {"oldText": "two", "newText": "TWO"},
        ]})
        overlap_path = self.cwd / "overlap.txt"
        overlap_path.write_text("abcde", encoding="utf-8")
        overlap = await edit.execute({"path": "overlap.txt", "edits": [
            {"oldText": "abcd", "newText": "ABCD"},
            {"oldText": "bcde", "newText": "BCDE"},
        ]})

        self.assertFalse(missing.ok)
        self.assertFalse(duplicate.ok)
        self.assertFalse(overlap.ok)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(overlap_path.read_text(encoding="utf-8"), "abcde")

    async def test_edit_normalizes_legacy_and_json_string_edits(self) -> None:
        path = self.cwd / "file.txt"
        path.write_text("a b", encoding="utf-8")
        edit = create_edit_tool(self.cwd)

        legacy = await edit.execute({"path": "file.txt", "oldText": "a", "newText": "A"})
        encoded = await edit.execute({"path": "file.txt", "edits": json.dumps([
            {"oldText": "b", "newText": "B"}
        ])})

        self.assertTrue(legacy.ok)
        self.assertTrue(encoded.ok)
        self.assertEqual(path.read_text(encoding="utf-8"), "A B")

    async def test_concurrent_edits_to_one_path_keep_both_changes(self) -> None:
        path = self.cwd / "shared.txt"
        path.write_text("left right", encoding="utf-8")
        edit = create_edit_tool(self.cwd)

        results = await asyncio.gather(
            edit.execute({"path": "shared.txt", "oldText": "left", "newText": "LEFT"}),
            edit.execute({"path": "shared.txt", "oldText": "right", "newText": "RIGHT"}),
        )

        self.assertTrue(all(result.ok for result in results))
        self.assertEqual(path.read_text(encoding="utf-8"), "LEFT RIGHT")

    async def test_bash_output_exit_and_timeout(self) -> None:
        bash = create_bash_tool(self.cwd)
        command = f'"{sys.executable}" -c "print(123)"'
        result = await bash.execute({"command": command})

        self.assertTrue(result.ok)
        self.assertEqual(result.content.strip(), "123")
        self.assertEqual(result.data["exit_code"], 0)
        self.assertFalse(result.data["timed_out"])

        failed = await bash.execute({"command": f'"{sys.executable}" -c "import sys; sys.exit(7)"'})
        self.assertFalse(failed.ok)
        self.assertEqual(failed.data["exit_code"], 7)
        self.assertIn("Exit code: 7", failed.content)

        sleep_command = f'"{sys.executable}" -c "import time; time.sleep(2)"'
        timed_out = await bash.execute({"command": sleep_command, "timeout": 0.1})
        self.assertFalse(timed_out.ok)
        self.assertTrue(timed_out.data["timed_out"])

    async def test_bash_truncation_keeps_full_output(self) -> None:
        bash = create_bash_tool(self.cwd)
        command = f'"{sys.executable}" -c "print(\'x\' * 21000)"'
        result = await bash.execute({"command": command})

        self.assertTrue(result.ok)
        self.assertTrue(result.data["truncated"])
        full_output = Path(result.data["full_output_path"])
        self.assertTrue(full_output.is_file())
        self.assertEqual(len(full_output.read_text(encoding="utf-8").strip()), 21000)

    async def test_bash_standard_input_is_closed(self) -> None:
        command = f'"{sys.executable}" -c "import sys; print(len(sys.stdin.read()))"'
        result = await asyncio.wait_for(
            create_bash_tool(self.cwd).execute({"command": command, "timeout": 2}), 5,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.content.strip(), "0")
        self.assertFalse(result.data["timed_out"])

    @unittest.skipUnless(sys.platform == "win32", "Windows cmd.exe date command")
    async def test_windows_date_does_not_wait_for_terminal_input(self) -> None:
        # No date is entered: closed stdin makes date return without changing it.
        result = await asyncio.wait_for(
            create_bash_tool(self.cwd).execute({"command": "date", "timeout": 2}), 5,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.data["exit_code"], 1)
        self.assertFalse(result.data["timed_out"])

    async def test_bash_default_timeout_can_be_overridden(self) -> None:
        bash = create_bash_tool(self.cwd)
        command = f'"{sys.executable}" -c "import time; time.sleep(0.5); print(123)"'
        with patch("coding.tools.DEFAULT_BASH_TIMEOUT_SECONDS", 0.1):
            default = await asyncio.wait_for(bash.execute({"command": command}), 5)
            explicit = await asyncio.wait_for(
                bash.execute({"command": command, "timeout": 2}), 5,
            )
        self.assertFalse(default.ok)
        self.assertTrue(default.data["timed_out"])
        self.assertTrue(explicit.ok)
        self.assertEqual(explicit.content.strip(), "123")


if __name__ == "__main__":
    unittest.main()
