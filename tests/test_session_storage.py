"""本地文件追加、恢复和错误传播。"""

import tempfile
import unittest
from pathlib import Path

from agent.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage, Usage
from agent.session import (
    CompactionEntry, JsonlSessionStorage, MessageEntry, SessionInfoEntry,
    SessionJsonlError, SessionState,
)


class SessionStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "nested" / "session.jsonl"

    def test_missing_file_reads_as_empty_without_creating_it(self) -> None:
        storage = JsonlSessionStorage(str(self.path))
        self.assertEqual(storage.read_all(), [])
        self.assertFalse(self.path.parent.exists())

    def test_append_preserves_previous_bytes_and_restores_records(self) -> None:
        storage = JsonlSessionStorage(self.path)
        first = MessageEntry(message=UserMessage(content="你好\n第二行"))
        second = MessageEntry(message=UserMessage(content="接着聊"))
        storage.append(first)
        previous = self.path.read_bytes()
        storage.append(second)

        saved = self.path.read_bytes()
        self.assertTrue(saved.startswith(previous))
        self.assertEqual(len(saved.splitlines()), 2)
        self.assertIn("你好", saved.decode("utf-8"))
        self.assertEqual(JsonlSessionStorage(self.path).read_all(), [first, second])

    def test_bad_record_reports_physical_line_number(self) -> None:
        storage = JsonlSessionStorage(self.path)
        storage.append(MessageEntry(message=UserMessage(content="valid")))
        with self.path.open("a", encoding="utf-8") as file:
            file.write("\n{broken\n")
        with self.assertRaisesRegex(SessionJsonlError, "on line 3"):
            storage.read_all()

    def test_compaction_append_preserves_history_and_restores_source_ids(self) -> None:
        storage = JsonlSessionStorage(self.path)
        info = SessionInfoEntry(cwd=str(self.path.parent))
        old = MessageEntry(id="old", message=UserMessage(content="old task"))
        message = MessageEntry(id="kept", message=UserMessage(content="recent task"))
        storage.append(info)
        storage.append(old)
        storage.append(message)
        previous = self.path.read_bytes()
        compaction = CompactionEntry(
            id="compact", summary="摘要\n下一步", first_kept_entry_id="kept",
            tokens_before=20000,
            usage=Usage(input_tokens=100, output_tokens=20, total_tokens=120),
        )

        storage.append(compaction)
        entries = JsonlSessionStorage(self.path).read_all()
        state = SessionState.from_entries(entries)

        saved = self.path.read_bytes()
        self.assertTrue(saved.startswith(previous))
        self.assertEqual(entries, [info, old, message, compaction])
        self.assertEqual(state.context_entry_ids, ["compact", "kept"])
        self.assertEqual(state.messages, [
            UserMessage(content="Previous conversation summary:\n摘要\n下一步"), message.message,
        ])
        self.assertEqual(self.path.read_bytes(), saved)

    def test_other_file_errors_are_not_reported_as_empty_history(self) -> None:
        directory = Path(self.temp_dir.name)
        with self.assertRaises(OSError):
            JsonlSessionStorage(directory).read_all()

    def test_restore_tool_conversation_from_disk(self) -> None:
        messages = [
            UserMessage(content="读取文件"),
            AssistantMessage(content="", tool_calls=[
                ToolCall(id="call-1", name="read", arguments={"path": "README.md"}),
            ]),
            ToolResultMessage(tool_call_id="call-1", name="read", content="文件内容"),
            AssistantMessage(content="已经读取"),
        ]
        storage = JsonlSessionStorage(self.path)
        self.assertEqual(SessionState.from_entries(storage.read_all()).messages, [])
        info = SessionInfoEntry(cwd=str(self.path.parent))
        storage.append(info)
        for message in messages:
            storage.append(MessageEntry(message=message))

        entries = JsonlSessionStorage(self.path).read_all()
        self.assertIsInstance(entries[0], SessionInfoEntry)
        self.assertEqual(entries[0], info)
        state = SessionState.from_entries(entries)
        self.assertEqual(state.session_info, info)
        self.assertEqual(state.context_entry_ids, [entry.id for entry in entries[1:]])
        restored = state.messages
        self.assertEqual(restored, messages)
        self.assertEqual([type(message) for message in restored],
                         [type(message) for message in messages])
