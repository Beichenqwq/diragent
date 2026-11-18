"""手动压缩的保存、回放、继续运行和失败边界。"""

import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from ai.events import ProviderErrorEvent, ProviderResponseEndEvent
from ai.provider import ModelProvider
from agent.context import request_context_fingerprint, response_context_fingerprint
from agent.messages import AssistantMessage, Usage, UserMessage
from agent.session import CompactionEntry, JsonlSessionStorage, MessageEntry, SessionInfoEntry
from coding.session import COMPACTION_SUMMARY_PREFIX, SUMMARIZATION_SYSTEM_PROMPT
from coding.session import CodingSession, CodingSessionConfig, CompactionPlan


class ManualCompactionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.enterContext(patch("coding.session.extract_memory_candidates", return_value=[]))
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(directory.cleanup)
        self.cwd = Path(directory.name)
        self.storage = JsonlSessionStorage(self.cwd / "session.jsonl")
        self.provider = Mock(spec=ModelProvider)
        self.config = CodingSessionConfig(provider=self.provider, model="fake", system="coding",
                                          tools=[], storage=self.storage, cwd=self.cwd)
        messages = [UserMessage(content="a" * 39984), UserMessage(content="b" * 39984),
                    AssistantMessage(content="c" * 39984, usage=Usage(
                        input_tokens=50000, output_tokens=10000, total_tokens=60000))]
        fingerprint = request_context_fingerprint(model="fake", system="coding", messages=messages[:2], tools=[])
        messages[2].usage_context = response_context_fingerprint(fingerprint, messages[2])
        self.entries = [SessionInfoEntry(cwd=str(self.cwd)),
                        *(MessageEntry(id=entry_id, message=message) for entry_id, message
                          in zip(("old", "kept", "answer"), messages, strict=True))]
        for entry in self.entries:
            self.storage.append(entry)
        self.session = CodingSession.load(self.config)
        self.saved = self.storage.path.read_bytes()
        self.seen = []

    def set_responses(self, responses) -> None:
        def stream_response(**kwargs):
            self.seen.append(kwargs)
            async def iterator():
                response = responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                yield response
            return iterator()
        self.provider.stream_response.side_effect = stream_response

    def summary_event(self, text="summary", usage=None):
        return ProviderResponseEndEvent(message=AssistantMessage(content=text, usage=usage))

    def assert_original_context(self) -> None:
        self.assertEqual(self.storage.path.read_bytes(), self.saved)
        self.assertEqual(self.session.state.context_entry_ids, ["old", "kept", "answer"])
        self.assertEqual(self.session.messages, tuple(entry.message for entry in self.entries[1:]))
        self.assertFalse(self.session._busy)

    async def test_compact_records_usage_boundary_and_tokens_then_reloads_same_context(self) -> None:
        usage = Usage(input_tokens=20, output_tokens=10, total_tokens=30)
        self.set_responses([self.summary_event(" summary ", usage)])
        harness = self.session._harness
        config = harness.config
        self.assertTrue(self.session.context_usage.uses_provider_usage)

        self.assertEqual(await self.session.compact("focus on tests"), "summary")

        entries = self.storage.read_all()
        self.assertEqual(entries[:-1], self.entries)
        self.assertTrue(self.storage.path.read_bytes().startswith(self.saved))
        compaction = entries[-1]
        self.assertIsInstance(compaction, CompactionEntry)
        self.assertEqual(compaction.first_kept_entry_id, "kept")
        self.assertEqual(compaction.tokens_before, 60000)
        self.assertEqual(compaction.usage, usage)
        self.assertEqual(self.session.state.context_entry_ids, [compaction.id, "kept", "answer"])
        self.assertEqual(self.session.messages, (UserMessage(content=COMPACTION_SUMMARY_PREFIX + "summary"),
                                               self.entries[2].message, self.entries[3].message))
        self.assertEqual(CodingSession.load(self.config).state, self.session.state)
        self.assertIs(self.session._harness, harness)
        self.assertIs(harness.config, config)
        self.assertIsNot(harness._messages, self.session.state.messages)
        self.assertFalse(self.session.context_usage.uses_provider_usage)
        self.assertEqual(entries[3].message.usage.total_tokens, 60000)
        self.assertIn("Additional focus: focus on tests", self.seen[0]["messages"][0].content)
        self.assertEqual(self.seen[0]["system"], SUMMARIZATION_SYSTEM_PROMPT)
        self.assertEqual(self.seen[0]["tools"], [])

    async def test_disabling_auto_compaction_does_not_disable_manual_compaction(self) -> None:
        self.session = CodingSession.load(replace(self.config, auto_compact_enabled=False))
        self.set_responses([self.summary_event()])
        self.assertIsNone(self.session.auto_compact_token_threshold)
        self.assertEqual(await self.session.compact(), "summary")
        self.assertIsInstance(self.storage.read_all()[-1], CompactionEntry)

    async def test_continue_and_prompt_after_compaction_save_only_new_messages(self) -> None:
        self.set_responses([self.summary_event(), self.summary_event("continued"), self.summary_event("answer")])
        await self.session.compact()
        compacted = self.session.messages
        async for _ in self.session.continue_():
            pass
        self.assertEqual(tuple(self.seen[1]["messages"][:3]), compacted)
        async for _ in self.session.prompt("new question"):
            pass
        entries = self.storage.read_all()
        self.assertEqual(len(entries), len(self.entries) + 4)
        self.assertEqual([entry.message.content for entry in entries[-3:]],
                         ["continued", "new question", "answer"])
        self.assertEqual(CodingSession.load(self.config).state, self.session.state)

    async def test_second_compaction_updates_old_summary_and_preserves_all_history(self) -> None:
        self.set_responses([self.summary_event("first summary"), self.summary_event("d" * 39984),
                            self.summary_event("second summary")])
        await self.session.compact()
        async for _ in self.session.prompt("e" * 39984):
            pass
        before_second = self.storage.read_all()
        first_summary = next(entry for entry in before_second if isinstance(entry, CompactionEntry))
        new_user = before_second[-2]
        await self.session.compact()
        prompt = self.seen[-1]["messages"][0].content
        self.assertIn("<previous-summary>\nfirst summary\n</previous-summary>", prompt)
        self.assertNotIn("a" * 39984, prompt)
        after = self.storage.read_all()
        self.assertEqual(after[:-1], before_second)
        self.assertEqual(after[-1].first_kept_entry_id, new_user.id)
        self.assertNotIn(first_summary.id, self.session.state.context_entry_ids)
        self.assertEqual(CodingSession.load(self.config).state, self.session.state)

    async def test_no_plan_raises_tau_value_error_without_request_or_write(self) -> None:
        storage = JsonlSessionStorage(self.cwd / "short.jsonl")
        storage.append(MessageEntry(message=UserMessage(content="short task")))
        session = CodingSession.load(CodingSessionConfig(
            provider=self.provider, model="fake", system="coding", tools=[], storage=storage, cwd=self.cwd))
        saved = storage.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Not enough context"):
            await session.compact()
        self.assertEqual(storage.path.read_bytes(), saved)
        self.assertFalse(session._busy)
        self.provider.stream_response.assert_not_called()
        self.assert_original_context()

    async def test_summary_failure_keeps_context_and_allows_retry(self) -> None:
        self.set_responses([ProviderErrorEvent(message="failed"), self.summary_event()])
        with self.assertRaisesRegex(RuntimeError, "failed"):
            await self.session.compact()
        self.assert_original_context()
        self.assertEqual(await self.session.compact(), "summary")

    async def test_append_failure_does_not_replace_context(self) -> None:
        self.set_responses([self.summary_event()])
        with patch.object(self.storage, "append", side_effect=OSError("write failed")):
            with self.assertRaisesRegex(OSError, "write failed"):
                await self.session.compact()
        self.assert_original_context()

    async def test_replay_failure_after_append_keeps_memory_and_reload_applies_record(self) -> None:
        self.set_responses([self.summary_event()])
        old_state, old_messages = self.session.state, self.session.messages
        with patch.object(self.storage, "read_all", side_effect=[self.entries, self.entries, OSError("read failed")]):
            with self.assertRaisesRegex(OSError, "read failed"):
                await self.session.compact()
        self.assertIs(self.session.state, old_state)
        self.assertEqual(self.session.messages, old_messages)
        self.assertIsInstance(self.storage.read_all()[-1], CompactionEntry)
        self.assertFalse(self.session._busy)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            await self.session.compact()
        reloaded = CodingSession.load(self.config)
        self.assertEqual(reloaded.messages[0].content, COMPACTION_SUMMARY_PREFIX + "summary")

    async def test_unsaved_harness_messages_or_state_changes_reject_before_request(self) -> None:
        self.session._harness.replace_messages([*self.session.messages, UserMessage(content="unsaved")])
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            await self.session.compact()
        self.session._harness.replace_messages(self.session.state.messages)
        self.session.state.context_entry_ids[0] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            await self.session.compact()
        self.provider.stream_response.assert_not_called()
        self.assertEqual(self.storage.path.read_bytes(), self.saved)
        self.assertFalse(self.session._busy)

    async def test_compaction_rejects_empty_or_missing_boundary(self) -> None:
        self.set_responses([self.summary_event(), self.summary_event()])
        for boundary in ("", "missing"):
            with self.subTest(boundary=boundary):
                plan = CompactionPlan(
                    first_kept_entry_id=boundary, replaced_entry_count=1,
                    messages_to_summarize=(self.session.messages[0],),
                )
                with patch.object(CodingSession, "_recent_preserving_compaction_plan", return_value=plan):
                    with self.assertRaisesRegex(ValueError, "boundary"):
                        await self.session.compact()
                self.assert_original_context()

    async def test_compaction_rejects_concurrent_runs_and_compaction(self) -> None:
        started, release = asyncio.Event(), asyncio.Event()
        async def iterator():
            started.set()
            await release.wait()
            yield self.summary_event()
        self.provider.stream_response.side_effect = lambda **kwargs: iterator()
        task = asyncio.create_task(self.session.compact())
        await started.wait()
        try:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                await self.session.compact()
            for run in (self.session.prompt("new"), self.session.continue_()):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    await anext(run)
        finally:
            release.set()
            await task
        self.assertFalse(self.session._busy)

    async def test_compaction_is_rejected_while_prompt_generator_is_active(self) -> None:
        self.set_responses([self.summary_event("answer")])
        run = self.session.prompt("new")
        await anext(run)
        with self.assertRaisesRegex(RuntimeError, "already running"):
            await self.session.compact()
        await run.aclose()
        self.assertFalse(self.session._busy)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            await self.session.compact()

    async def test_external_write_during_summary_rejects_compaction_append(self) -> None:
        async def iterator():
            self.storage.append(MessageEntry(id="external", message=UserMessage(content="external write")))
            yield self.summary_event()
        self.provider.stream_response.side_effect = lambda **kwargs: iterator()
        with self.assertRaisesRegex(RuntimeError, "changed during summarization"):
            await self.session.compact()
        self.assertFalse(any(isinstance(entry, CompactionEntry) for entry in self.storage.read_all()))
        self.assertEqual(self.session.state.context_entry_ids, ["old", "kept", "answer"])
        self.assertFalse(self.session._busy)

    async def test_cancellation_before_summary_completion_releases_busy_without_writing(self) -> None:
        started = asyncio.Event()
        async def iterator():
            started.set()
            await asyncio.Event().wait()
            yield self.summary_event()
        self.provider.stream_response.side_effect = lambda **kwargs: iterator()
        task = asyncio.create_task(self.session.compact())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_original_context()
