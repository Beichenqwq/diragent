"""持久化资格检查及索引和正文的精确去重。"""

from pathlib import Path
import tempfile
import unittest

from coding.memory import load_existing_memories, should_store_memory


class MemoryEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {"name": "New preference", "type": "user", "scope": "persistent",
                          "description": "New description", "body": "New body"}
        self.existing = {"name": "Existing preference", "description": "Existing description",
                         "body": "Existing body"}

    def test_only_persistent_candidates_are_eligible(self):
        self.assertTrue(should_store_memory(self.candidate, []))
        self.assertFalse(should_store_memory({**self.candidate, "scope": "current_task"}, []))

    def test_any_matching_field_blocks_storage_independently(self):
        for field in ("name", "description", "body"):
            with self.subTest(field=field):
                candidate = {**self.candidate, field: self.existing[field]}
                self.assertFalse(should_store_memory(candidate, [self.existing]))

    def test_trims_edges_but_does_not_fold_case_or_internal_whitespace(self):
        self.assertFalse(should_store_memory({**self.candidate, "body": " Existing body\n"}, [self.existing]))
        for body in ("existing body", "Existing  body"):
            self.assertTrue(should_store_memory({**self.candidate, "body": body}, [self.existing]))

    def test_does_not_mutate_existing_records_or_deduplicate_current_batch(self):
        existing = [self.existing.copy()]
        candidates = [self.candidate.copy(), self.candidate.copy()]
        accepted = [candidate for candidate in candidates if should_store_memory(candidate, existing)]
        self.assertEqual(accepted, candidates)
        self.assertEqual(existing, [self.existing])


class ExistingMemoryLoadingTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(directory.cleanup)
        self.cwd = Path(directory.name)
        self.root = self.cwd / ".tau" / "memory"

    def test_empty_index_does_not_create_files(self):
        self.assertEqual(load_existing_memories(self.cwd, ""), [])
        self.assertFalse(self.root.exists())

    def test_reads_index_entries_and_only_linked_utf8_bodies(self):
        self.root.mkdir(parents=True)
        (self.root / "preference.md").write_text("\n正文第一段\n\n正文第二段\n", encoding="utf-8")
        # 未被索引引用的文件不会读取。
        (self.root / "unlinked.md").write_bytes(b"\xff")
        index = "# 记忆索引\n\n- [ 用户偏好 ](preference.md) -  结合调用流程解释代码  \n"
        self.assertEqual(load_existing_memories(self.cwd, index), [
            {"name": "用户偏好", "description": "结合调用流程解释代码", "body": "正文第一段\n\n正文第二段"},
        ])

    def test_missing_body_keeps_index_metadata_for_duplicate_checks(self):
        records = load_existing_memories(self.cwd, "- [Existing](missing.md) - Existing description\n")
        self.assertEqual(records, [{"name": "Existing", "description": "Existing description", "body": ""}])

    def test_rejects_links_outside_memory_directory_and_reserved_index(self):
        for filename in ("../outside.md", "subdir/entry.md", "MEMORY.md", "https://example.test/entry.md"):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                load_existing_memories(self.cwd, f"- [Memory]({filename}) - Description\n")
