"""阈值触发、独立模型请求及记忆重建的失败保护。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ai.events import ProviderErrorEvent, ProviderResponseEndEvent
from ai.provider import ModelProvider
from agent.messages import AssistantMessage, UserMessage
from agent.session import JsonlSessionStorage
from coding.memory import load_existing_memories, load_memory_index, store_memory
from coding.memory_consolidation import (
    MEMORY_CONSOLIDATION_SYSTEM_PROMPT, load_memory_snapshot,
    normalize_consolidated_memories, rebuild_memories,
)
from coding.session import CodingSession, CodingSessionConfig


def record(name, body=None):
    return {"name": name, "type": "project", "description": name + " description",
            "body": body or name + " body"}


class MemoryConsolidationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        retrieval = patch("coding.session.retrieve_memories", return_value=[])
        retrieval.start()
        self.addCleanup(retrieval.stop)
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(directory.cleanup)
        self.cwd = Path(directory.name)
        self.root = self.cwd / ".tau" / "memory"
        self.provider = Mock(spec=ModelProvider)
        self.seen = []

    def session(self, threshold=2):
        return CodingSession.load(CodingSessionConfig(
            provider=self.provider, model="test-model", system="coding instructions", tools=[],
            cwd=self.cwd, storage=JsonlSessionStorage(self.cwd / "session.jsonl"),
            auto_compact_enabled=False, memory_consolidation_threshold=threshold,
        ))

    def seed(self, count=3):
        for number in range(count):
            store_memory(self.cwd, {**record(f"Old {number}"), "scope": "persistent"})

    def set_responses(self, responses):
        def stream_response(**kwargs):
            self.seen.append(kwargs)
            async def iterator():
                for event in responses.pop(0):
                    yield event
            return iterator()
        self.provider.stream_response.side_effect = stream_response

    def response(self, records, finish_reason="stop"):
        content = records if isinstance(records, str) else json.dumps(records, ensure_ascii=False)
        return ProviderResponseEndEvent(
            message=AssistantMessage(content=content), finish_reason=finish_reason,
        )

    def files(self):
        return {path.name: path.read_bytes() for path in self.root.iterdir()}

    async def test_only_exceeding_threshold_calls_model(self):
        self.seed(2)
        await self.session()._try_consolidate_memories()
        await self.session(0)._try_consolidate_memories()
        self.provider.stream_response.assert_not_called()
        self.seed(1)
        self.set_responses([[self.response([record("Merged")])]])
        await self.session()._try_consolidate_memories()
        self.assertEqual(len(self.seen), 1)

    async def test_rebuilds_complete_memory_set_and_removes_original_backup(self):
        self.seed()
        session = self.session()
        messages_before = session.messages
        transcript_before = session.config.storage.read_all()
        expected = record("Project Rules", "第一段\n\n第二段")
        self.set_responses([[self.response([expected])]])
        await session._try_consolidate_memories()
        self.assertEqual(set(self.files()), {"MEMORY.md", "project-rules.md"})
        self.assertEqual(load_existing_memories(self.cwd, load_memory_index(self.cwd)), [expected])
        self.assertEqual(session.messages, messages_before)
        self.assertEqual(session.config.storage.read_all(), transcript_before)
        request = self.seen[0]
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["system"], MEMORY_CONSOLIDATION_SYSTEM_PROMPT)
        self.assertEqual(request["tools"], [])
        self.assertEqual(len(request["messages"]), 1)
        prompt = request["messages"][0].content
        for number in range(3):
            self.assertIn(f"Old {number} body", prompt)
        self.assertIn('"type": "project"', prompt)
        backups = list((self.cwd / ".codex_tmp").glob("memory-consolidation-*/original-memory"))
        self.assertEqual(backups, [])

    async def test_model_error_empty_invalid_or_truncated_response_keeps_originals(self):
        self.seed()
        original = self.files()
        cases = [
            [ProviderErrorEvent(message="request failed")],
            [self.response("no JSON")], [self.response([])],
            [self.response([record("Valid"), {"name": "Missing fields"}])],
            [self.response([{**record("Wrong type"), "type": "user preference"}])],
            [self.response([{**record("Empty body"), "body": "  "}])],
            [self.response([record("Truncated")], finish_reason="length")],
            [],
        ]
        for events in cases:
            with self.subTest(events=events):
                self.set_responses([events])
                with self.assertLogs("coding.session", level="WARNING"):
                    await self.session()._try_consolidate_memories()
                self.assertEqual(self.files(), original)

    def test_normalizes_four_fields_and_removes_identical_duplicates(self):
        item = {key: " " + value + " " for key, value in record("Rules").items()}
        text = "Here is the result:\n```json\n" + json.dumps([item, item]) + "\n```"
        self.assertEqual(normalize_consolidated_memories(text),
                         [{**record("Rules"), "scope": "persistent"}])
        with self.assertRaises(ValueError):
            normalize_consolidated_memories(json.dumps([record("Rules"), record("Rules", "Other fact")]))

    def test_legacy_plain_text_files_remain_supported(self):
        self.root.mkdir(parents=True)
        (self.root / "MEMORY.md").write_text("- [Old](old.md) - Old description\n", encoding="utf-8")
        (self.root / "old.md").write_text("Legacy body", encoding="utf-8")
        snapshot = load_memory_snapshot(self.cwd)
        self.assertEqual(snapshot.memories, [{"name": "Old", "description": "Old description",
                                             "body": "Legacy body"}])
        rebuild_memories(self.cwd, [record("New")], snapshot)
        self.assertEqual(load_existing_memories(self.cwd, load_memory_index(self.cwd)), [record("New")])

    def test_staging_write_failure_preserves_original_files(self):
        self.seed()
        snapshot = load_memory_snapshot(self.cwd)
        original = self.files()
        with patch("coding.memory_consolidation.store_memory", side_effect=OSError("Disk full")):
            with self.assertRaises(OSError):
                rebuild_memories(self.cwd, [record("Merged")], snapshot)
        self.assertEqual(self.files(), original)

    def test_failed_directory_swap_restores_original_directory(self):
        self.seed()
        snapshot = load_memory_snapshot(self.cwd)
        original = self.files()
        rename = Path.rename
        def fail_staged_move(path, destination):
            if path.parent.name == ".tau" and "staged" in path.parts:
                raise OSError("Swap failed")
            return rename(path, destination)
        with patch.object(Path, "rename", new=fail_staged_move), \
                patch("coding.memory_consolidation.shutil.rmtree") as cleanup:
            with self.assertRaises(OSError):
                rebuild_memories(self.cwd, [record("Merged")], snapshot)
            cleanup.assert_not_called()
        self.assertEqual(self.files(), original)

    def test_failed_backup_cleanup_keeps_new_memories_without_rollback(self):
        self.seed()
        snapshot = load_memory_snapshot(self.cwd)
        with patch("coding.memory_consolidation.shutil.rmtree", side_effect=PermissionError("File in use")), \
                self.assertLogs("coding.memory_consolidation", level="WARNING"):
            rebuild_memories(self.cwd, [record("Merged")], snapshot)
        self.assertEqual(load_existing_memories(self.cwd, load_memory_index(self.cwd)), [record("Merged")])
        backups = list((self.cwd / ".codex_tmp").glob("memory-consolidation-*/original-memory"))
        self.assertEqual(len(backups), 1)

    def test_concurrent_body_edit_or_new_record_prevents_overwrite(self):
        self.seed()
        for change in ("body", "new record"):
            with self.subTest(change=change):
                snapshot = load_memory_snapshot(self.cwd)
                if change == "body":
                    (self.root / "old-0.md").write_text("Edited body", encoding="utf-8")
                else:
                    store_memory(self.cwd, {**record("Concurrent"), "scope": "persistent"})
                current = self.files()
                with self.assertRaisesRegex(RuntimeError, "changed during consolidation"):
                    rebuild_memories(self.cwd, [record("Merged")], snapshot)
                self.assertEqual(self.files(), current)

    async def test_chat_extracts_saves_then_consolidates_without_polluting_history(self):
        self.seed(2)
        session = self.session()
        self.set_responses([
            [self.response("Chat answer")],
            [self.response([{**record("New fact"), "scope": "persistent"}])],
            [self.response([record("Merged")])],
        ])
        _ = [event async for event in session.prompt("question")]
        self.assertEqual(len(self.seen), 3)
        self.assertEqual([message.content for message in session.messages], ["question", "Chat answer"])
        self.assertEqual(load_existing_memories(self.cwd, load_memory_index(self.cwd)), [record("Merged")])

    async def test_no_newly_saved_memory_does_not_trigger_consolidation(self):
        self.seed()
        for candidates in ([], [{**record("Old 0"), "scope": "persistent"}],
                           [{**record("Temporary"), "scope": "current_task"}]):
            with self.subTest(candidates=candidates):
                self.set_responses([[self.response(candidates)]])
                with patch.object(CodingSession, "_try_consolidate_memories") as consolidate:
                    await self.session()._try_extract_memories([UserMessage(content="facts")])
                    consolidate.assert_not_called()
