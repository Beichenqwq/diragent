"""编号检索、按需读取和每轮系统提示词更新。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ai.events import ProviderErrorEvent, ProviderResponseEndEvent
from ai.provider import ModelProvider
from agent.messages import AssistantMessage, UserMessage
from agent.session import JsonlSessionStorage
from coding.memory import load_existing_memories, store_memory
from coding.memory_retrieval import retrieve_memories
from coding.session import CodingSession, CodingSessionConfig


class MemoryRetrievalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        root.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(directory.cleanup)
        self.cwd = Path(directory.name)
        self.provider = Mock(spec=ModelProvider)
        self.calls = []
        self.responses = []
        def stream(**kwargs):
            self.calls.append(kwargs)
            async def events():
                value = self.responses.pop(0)
                if isinstance(value, Exception):
                    yield ProviderErrorEvent(message=str(value))
                else:
                    yield ProviderResponseEndEvent(
                        message=AssistantMessage(content=value), finish_reason="stop",
                    )
            return events()
        self.provider.stream_response.side_effect = stream

    def seed(self):
        for number in range(3):
            store_memory(self.cwd, {
                "name": f"Rule {number}", "description": f"Description {number}",
                "body": f"BODY_SENTINEL_{number}", "type": "project", "scope": "persistent",
            })

    async def retrieve(self, messages=()):
        return await retrieve_memories(
            self.provider, model="test", cwd=self.cwd, messages=messages, content="CURRENT",
        )

    async def test_numbered_catalog_last_two_users_and_only_selected_bodies(self):
        self.seed()
        self.responses = [json.dumps([2, 0, 2, True, -1, 3, "1", 1.0])]
        messages = [UserMessage(content="TOO_OLD"), UserMessage(content="PREVIOUS_1"),
                    AssistantMessage(content="ASSISTANT_SECRET"), UserMessage(content="PREVIOUS_2")]
        with patch("coding.memory_retrieval.load_existing_memories", wraps=load_existing_memories) as loader:
            memories = await self.retrieve(messages)
        self.assertEqual([m["name"] for m in memories], ["Rule 2", "Rule 0"])
        self.assertNotIn("rule-1.md", loader.call_args.args[1])
        prompt = self.calls[0]["messages"][0].content
        self.assertIn("0: Rule 0 - Description 0\n1: Rule 1 - Description 1\n2: Rule 2 - Description 2", prompt)
        for value in ("PREVIOUS_1", "PREVIOUS_2", "CURRENT"):
            self.assertIn(value, prompt)
        for value in ("TOO_OLD", "ASSISTANT_SECRET", "BODY_SENTINEL", ".md"):
            self.assertNotIn(value, prompt)
        self.assertEqual(self.calls[0]["tools"], [])

    async def test_empty_catalog_skips_provider(self):
        self.assertEqual(await self.retrieve(), [])
        self.assertEqual(self.calls, [])

    async def test_empty_or_invalid_selection_never_reads_bodies(self):
        self.seed()
        for output in ("[]", "no JSON", '[true, -1, 99, "0", 0.5]'):
            self.responses = [output]
            with patch("coding.memory_retrieval.load_existing_memories") as loader:
                self.assertEqual(await self.retrieve(), [])
                loader.assert_not_called()

    async def test_missing_selected_file_is_skipped(self):
        self.seed()
        (self.cwd / ".tau" / "memory" / "rule-0.md").unlink()
        self.responses = ["[0, 1]"]
        self.assertEqual([m["name"] for m in await self.retrieve()], ["Rule 1"])

    async def test_selected_path_cannot_escape_memory_directory(self):
        root = self.cwd / ".tau" / "memory"
        root.mkdir(parents=True)
        (root / "MEMORY.md").write_text("- [Outside](../outside.md) - Invalid path\n", encoding="utf-8")
        self.responses = ["[0]"]
        with self.assertRaises(ValueError):
            await self.retrieve()

    async def test_each_prompt_replaces_memory_context_and_failure_clears_previous(self):
        self.seed()
        session = CodingSession.load(CodingSessionConfig(
            provider=self.provider, model="test", system="BASE", tools=[], cwd=self.cwd,
            storage=JsonlSessionStorage(self.cwd / "session.jsonl"), auto_compact_enabled=False,
        ))
        self.responses = ["[0]", "answer one", "[1]", "answer two", RuntimeError("offline"), "answer three"]
        with patch.object(session, "_try_extract_memories"):
            _ = [event async for event in session.prompt("first")]
            _ = [event async for event in session.prompt("second")]
            with self.assertLogs("coding.session", level="WARNING"):
                _ = [event async for event in session.prompt("third")]
        self.assertIn("BODY_SENTINEL_0", self.calls[1]["system"])
        self.assertIn("BODY_SENTINEL_1", self.calls[3]["system"])
        self.assertNotIn("BODY_SENTINEL_0", self.calls[3]["system"])
        self.assertTrue(self.calls[5]["system"].startswith("BASE"))
        self.assertNotIn("BODY_SENTINEL", self.calls[5]["system"])
        for call in (self.calls[1], self.calls[3], self.calls[5]):
            self.assertEqual(call["system"].count("<memory_index>"), 1)
            for number in range(3):
                self.assertIn(f"[Rule {number}](rule-{number}.md) - Description {number}", call["system"])
        self.assertEqual([m.content for m in session.messages],
                         ["first", "answer one", "second", "answer two", "third", "answer three"])
        self.assertEqual(len(session.config.storage.read_all()), 7)
