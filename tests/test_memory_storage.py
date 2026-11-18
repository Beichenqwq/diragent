"""记忆文件、索引的写入和失败恢复。"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from coding.memory import load_existing_memories, load_memory_index, memory_filename, store_memory


class MemoryStorageTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(directory.cleanup)
        self.cwd = Path(directory.name)
        self.root = self.cwd / ".tau" / "memory"
        self.candidate = {"name": "Code Explanations", "type": "user", "scope": "persistent",
                          "description": "结合调用流程解释代码", "body": "第一段\n\n第二段"}

    def test_writes_utf8_body_and_index_and_can_reload_for_deduplication(self):
        path = store_memory(self.cwd, self.candidate)
        self.assertEqual(path.name, "code-explanations.md")
        self.assertEqual(path.read_bytes().decode("utf-8"),
                         "---\ntype: user\n---\n\n" + self.candidate["body"] + "\n")
        index = load_memory_index(self.cwd)
        self.assertEqual(index, "- [Code Explanations](code-explanations.md) - 结合调用流程解释代码\n")
        self.assertEqual(load_existing_memories(self.cwd, index), [
            {field: self.candidate[field] for field in ("name", "type", "description", "body")},
        ])
        self.assertEqual(len(list(self.root.iterdir())), 2)

    def test_filename_conversion_handles_spaces_fallback_and_reserved_names(self):
        self.assertEqual(memory_filename(" __Code   Explanations__ "), "code-explanations.md")
        self.assertEqual(memory_filename("---___"), "memory.md")
        self.assertEqual(memory_filename("CON"), "memory-con.md")
        self.assertEqual(memory_filename("LPT1"), "memory-lpt1.md")

    def test_filename_collision_does_not_overwrite_existing_body(self):
        self.root.mkdir(parents=True)
        existing = self.root / "code-explanations.md"
        existing.write_text("Old body", encoding="utf-8")
        path = store_memory(self.cwd, self.candidate)
        self.assertEqual(path.name, "code-explanations-1.md")
        self.assertEqual(existing.read_text(encoding="utf-8"), "Old body")

    def test_memory_named_memory_does_not_overwrite_the_index(self):
        self.root.mkdir(parents=True)
        (self.root / "MEMORY.md").write_text("# Existing index", encoding="utf-8")
        path = store_memory(self.cwd, {**self.candidate, "name": "MEMORY"})
        self.assertEqual(path.name, "memory-1.md")
        self.assertEqual(load_memory_index(self.cwd),
                         "# Existing index\n- [MEMORY](memory-1.md) - 结合调用流程解释代码\n")

    def test_index_labels_and_descriptions_round_trip_special_characters(self):
        candidate = {**self.candidate, "name": '字段 ["items"] & 缓冲\n约定',
                     "description": "第一行\n第二行 & 参考"}
        store_memory(self.cwd, candidate)
        index = load_memory_index(self.cwd)
        self.assertEqual(len(index.splitlines()), 1)
        self.assertEqual(load_existing_memories(self.cwd, index), [
            {field: candidate[field] for field in ("name", "type", "description", "body")},
        ])

    def test_failed_index_commit_preserves_old_index_and_removes_new_files(self):
        self.root.mkdir(parents=True)
        path = self.root / "MEMORY.md"
        original = "# 已有索引\n"
        path.write_text(original, encoding="utf-8")
        with patch.object(Path, "replace", side_effect=OSError("Index commit failed")):
            with self.assertRaises(OSError):
                store_memory(self.cwd, self.candidate)
        self.assertEqual(path.read_text(encoding="utf-8"), original)
        self.assertEqual(list(self.root.iterdir()), [path])
