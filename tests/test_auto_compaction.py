"""自动压缩时机、provider 溢出识别和最多一次恢复重试。"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from ai.events import ProviderErrorEvent, ProviderResponseEndEvent
from ai.provider import ModelProvider
from agent.events import ErrorEvent
from agent.messages import AssistantMessage, ToolCall, UserMessage
from agent.session import CompactionEntry, JsonlSessionStorage, MessageEntry
from agent.tools import AgentTool, AgentToolResult
from coding.session import COMPACTION_SUMMARY_PREFIX, SUMMARIZATION_SYSTEM_PROMPT
from coding.session import CodingSession, CodingSessionConfig, is_context_overflow_error


class OverflowRecognitionTests(unittest.TestCase):
    def test_only_provider_errors_can_trigger_recovery(self) -> None:
        for source in ("loop", "tool", "provider"):
            with self.subTest(source=source):
                self.assertEqual(is_context_overflow_error(ErrorEvent(
                    message="maximum context length exceeded", source=source)), source == "provider")

    def test_http_body_and_context_error_code_are_recognized(self) -> None:
        event = ErrorEvent(source="provider", message="Provider request failed with status code 400",
                           data={"body": '{"error":{"code":"context_length_exceeded"}}'})
        self.assertTrue(is_context_overflow_error(event))

    def test_network_tool_and_rate_limit_messages_are_not_context_errors(self) -> None:
        for message in ("connection failed", "429 rate limit exceeded: too many tokens",
                        "token limit per minute exceeded", "authentication failed"):
            with self.subTest(message=message):
                self.assertFalse(is_context_overflow_error(ErrorEvent(source="provider", message=message)))


class AutoCompactionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # 压缩测试只预设聊天和摘要响应，记忆请求由专门的测试覆盖。
        self.enterContext(patch("coding.session.extract_memory_candidates", return_value=[]))
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(directory.cleanup)
        self.cwd = Path(directory.name)
        self.provider = Mock(spec=ModelProvider)
        self.storage = JsonlSessionStorage(self.cwd / "session.jsonl")
        self.config = CodingSessionConfig(provider=self.provider, model="fake", tools=[], system="coding",
                                          cwd=self.cwd, storage=self.storage, auto_compact_token_threshold=25000)
        self.seen = []

    def history(self, *, answer=True, config=None):
        messages = [UserMessage(content="a" * 39984), UserMessage(content="b" * 39984)]
        if answer:
            messages.append(AssistantMessage(content="c" * 39984))
        for index, message in enumerate(messages):
            self.storage.append(MessageEntry(id=f"history-{index}", message=message))
        return CodingSession.load(config or self.config)

    def response(self, text):
        return ProviderResponseEndEvent(message=AssistantMessage(content=text))

    def set_responses(self, responses):
        def stream_response(**kwargs):
            self.seen.append({**kwargs, "messages": list(kwargs["messages"])})
            async def iterator():
                response = responses.pop(0)
                yield response
            return iterator()
        self.provider.stream_response.side_effect = stream_response

    def compactions(self):
        return [entry for entry in self.storage.read_all() if isinstance(entry, CompactionEntry)]

    def assert_reloads(self, session):
        self.assertEqual(CodingSession.load(session.config).state, session.state)
        self.assertEqual(session.messages, tuple(session.state.messages))
        self.assertFalse(session._busy)

    async def test_prompt_compacts_before_input_and_records_correct_save_start(self):
        session = self.history()
        self.set_responses([self.response("summary"), self.response("answer")])
        events = [event async for event in session.prompt("new question")]
        self.assertTrue(events)
        self.assertEqual(self.seen[0]["system"], SUMMARIZATION_SYSTEM_PROMPT)
        self.assertNotIn("new question", self.seen[0]["messages"][0].content)
        self.assertEqual(self.seen[1]["messages"][0].content, COMPACTION_SUMMARY_PREFIX + "summary")
        self.assertEqual(self.seen[1]["messages"][-1].content, "new question")
        entries = self.storage.read_all()
        self.assertEqual(len(self.compactions()), 1)
        self.assertEqual([entry.message.content for entry in entries[-2:]], ["new question", "answer"])
        self.assertEqual(len(entries), 6)
        self.assert_reloads(session)

    async def test_prompt_defers_compaction_until_next_prompt(self):
        session = self.history(answer=False)
        self.set_responses([self.response("c" * 39984), self.response("summary"), self.response("answer")])
        async for _ in session.prompt("new question"):
            pass
        self.assertEqual([request["system"] for request in self.seen], ["coding"])
        entries = self.storage.read_all()
        self.assertFalse(self.compactions())
        self.assertEqual(entries[-2].message.content, "new question")
        self.assertEqual(entries[-1].message.content, "c" * 39984)
        async for _ in session.prompt("next question"):
            pass
        self.assertEqual([request["system"] for request in self.seen],
                         ["coding", SUMMARIZATION_SYSTEM_PROMPT, "coding"])
        self.assertEqual(len(self.compactions()), 1)
        self.assertNotIn("next question", self.seen[1]["messages"][0].content)
        self.assertEqual(self.seen[2]["messages"][-1].content, "next question")
        self.assert_reloads(session)

    async def test_continue_does_not_auto_compact(self):
        session = self.history()
        self.set_responses([self.response("answer")])
        async for _ in session.continue_():
            pass
        self.assertEqual([request["system"] for request in self.seen], ["coding"])
        self.assertEqual(len(self.seen[0]["messages"]), 3)
        self.assertFalse(self.compactions())
        self.assert_reloads(session)

    async def test_prompt_compacts_only_once_before_run_without_duplicate_records(self):
        session = self.history()
        self.set_responses([self.response("s" * 159984), self.response("answer")])
        async for _ in session.prompt("new question"):
            pass
        self.assertEqual(len(self.compactions()), 1)
        self.assertEqual(len(self.seen), 2)
        entries = self.storage.read_all()
        self.assertEqual(sum(isinstance(entry, MessageEntry) and entry.message.content == "new question"
                             for entry in entries), 1)
        self.assertEqual(len(entries), 6)
        self.assert_reloads(session)

    async def test_disabled_auto_compaction_and_zero_threshold_skip_checks(self):
        for index, options in enumerate(({"auto_compact_enabled": False}, {"auto_compact_token_threshold": 0})):
            with self.subTest(options=options):
                self.storage = JsonlSessionStorage(self.cwd / f"disabled-{index}.jsonl")
                config = replace(self.config, storage=self.storage, **options)
                session = self.history(config=config)
                self.set_responses([self.response("answer")])
                async for _ in session.prompt("new"):
                    pass
                self.assertFalse(self.compactions())
                self.assert_reloads(session)

    async def test_exact_threshold_does_not_compact_but_one_less_does(self):
        session = self.history()
        total = session.context_usage.total_tokens
        session = CodingSession.load(replace(self.config, auto_compact_token_threshold=total))
        self.provider.stream_response.assert_not_called()
        self.assertFalse(await session._try_auto_compact())
        session = CodingSession.load(replace(self.config, auto_compact_token_threshold=total - 1))
        self.set_responses([self.response("summary")])
        self.assertTrue(await session._try_auto_compact())

    async def test_no_plan_does_not_request_summary_or_write_compaction(self):
        session = CodingSession.load(replace(self.config, auto_compact_token_threshold=1))
        self.set_responses([self.response("answer")])
        async for _ in session.prompt("short"):
            pass
        self.assertEqual(len(self.seen), 1)
        self.assertFalse(self.compactions())
        self.assert_reloads(session)

    async def test_summary_failure_logs_and_continues_original_context(self):
        session = self.history()
        self.set_responses([ProviderErrorEvent(message="summary failed"), self.response("answer")])
        with self.assertLogs("coding.session", level="WARNING"):
            async for _ in session.prompt("new"):
                pass
        self.assertFalse(self.compactions())
        self.assertEqual(self.seen[1]["messages"][0].content, "a" * 39984)
        self.assertEqual(self.storage.read_all()[-1].message.content, "answer")
        self.assert_reloads(session)

    async def test_committed_auto_compaction_replay_failure_stops_before_new_input(self):
        session = self.history()
        self.set_responses([self.response("summary")])
        state_before, messages_before = session.state, session.messages
        with patch.object(session, "_refresh_persisted_state", side_effect=OSError("replay failed")):
            with self.assertLogs("coding.session", level="WARNING"):
                with self.assertRaisesRegex(OSError, "replay failed"):
                    async for _ in session.prompt("new"):
                        pass
        self.assertIs(session.state, state_before)
        self.assertEqual(session.messages, messages_before)
        self.assertFalse(session._busy)
        self.assertEqual(len(self.seen), 1)
        self.assertEqual(len(self.compactions()), 1)
        self.assertEqual(CodingSession.load(session.config).messages[0].content,
                         COMPACTION_SUMMARY_PREFIX + "summary")

    async def test_auto_compaction_append_failure_without_commit_continues_run(self):
        session = self.history()
        self.set_responses([self.response("summary"), self.response("answer"), self.response("summary")])
        original_append = self.storage.append
        def append(entry):
            if isinstance(entry, CompactionEntry):
                raise OSError("append failed")
            original_append(entry)
        with patch.object(self.storage, "append", side_effect=append):
            with self.assertLogs("coding.session", level="WARNING"):
                async for _ in session.prompt("new"):
                    pass
        self.assertFalse(self.compactions())
        self.assertEqual(self.storage.read_all()[-1].message.content, "answer")
        self.assert_reloads(session)

    async def test_overflow_recovery_works_when_auto_is_disabled_and_input_is_saved_once(self):
        session = self.history(config=replace(self.config, auto_compact_enabled=False))
        self.set_responses([ProviderErrorEvent(message="Provider request failed with status code 400",
                                              data={"body": '{"error":{"code":"context_length_exceeded"}}'}),
                            self.response("summary"), self.response("recovered")])
        with patch.object(session, "_try_extract_memories") as extract:
            events = [event async for event in session.prompt("new question")]
        extract.assert_awaited_once_with(session.messages)
        errors = [event for event in events if isinstance(event, ErrorEvent)]
        self.assertEqual(errors[0].source, "provider")
        self.assertEqual([request["system"] for request in self.seen],
                         ["coding", SUMMARIZATION_SYSTEM_PROMPT, "coding"])
        self.assertEqual(self.seen[-1]["messages"][-1].content, "new question")
        entries = self.storage.read_all()
        self.assertEqual(sum(isinstance(entry, MessageEntry) and entry.message.content == "new question"
                             for entry in entries), 1)
        self.assertEqual(entries[-1].message.content, "recovered")
        self.assertEqual(len(self.compactions()), 1)
        self.assert_reloads(session)

    async def test_retry_overflow_is_not_retried_or_compacted_again(self):
        session = self.history(config=replace(self.config, auto_compact_enabled=False))
        error = ProviderErrorEvent(message="context window exceeded")
        self.set_responses([error, self.response("summary"), error])
        with patch.object(session, "_try_extract_memories") as extract:
            events = [event async for event in session.prompt("new")]
        extract.assert_not_awaited()
        self.assertEqual(sum(isinstance(event, ErrorEvent) for event in events), 2)
        self.assertEqual(len(self.seen), 3)
        self.assertEqual(len(self.compactions()), 1)
        self.assert_reloads(session)

    async def test_overflow_without_plan_does_not_request_summary_or_retry(self):
        session = CodingSession.load(replace(self.config, auto_compact_enabled=False))
        self.set_responses([ProviderErrorEvent(message="context limit exceeded")])
        events = [event async for event in session.prompt("new")]
        self.assertTrue(any(isinstance(event, ErrorEvent) for event in events))
        self.assertEqual(len(self.seen), 1)
        self.assertFalse(self.compactions())
        self.assertEqual(session.messages, (UserMessage(content="new"),))
        self.assert_reloads(session)

    async def test_overflow_summary_failure_keeps_original_error_and_does_not_retry(self):
        session = self.history(config=replace(self.config, auto_compact_enabled=False))
        self.set_responses([ProviderErrorEvent(message="context length exceeded"),
                            ProviderErrorEvent(message="summary also too long")])
        with self.assertLogs("coding.session", level="WARNING"):
            events = [event async for event in session.prompt("new")]
        self.assertTrue(any(isinstance(event, ErrorEvent) and "context length" in event.message for event in events))
        self.assertEqual(len(self.seen), 2)
        self.assertFalse(self.compactions())
        self.assert_reloads(session)

    async def test_network_error_does_not_trigger_post_compaction_or_retry(self):
        session = self.history(config=replace(self.config, auto_compact_enabled=False))
        self.set_responses([ProviderErrorEvent(message="connection refused")])
        with patch.object(session, "_try_extract_memories") as extract:
            async for _ in session.prompt("new"):
                pass
        extract.assert_not_awaited()
        self.assertEqual(len(self.seen), 1)
        self.assertFalse(self.compactions())
        self.assert_reloads(session)

    async def test_continue_overflow_does_not_use_prompt_recovery(self):
        session = self.history(config=replace(self.config, auto_compact_enabled=False))
        self.set_responses([ProviderErrorEvent(message="context length exceeded")])
        async for _ in session.continue_():
            pass
        self.assertEqual(len(self.seen), 1)
        self.assertFalse(self.compactions())
        self.assert_reloads(session)

    async def test_tool_error_with_context_words_never_triggers_overflow_recovery(self):
        async def execute(arguments):
            return AgentToolResult(tool_call_id="call", name="fake", content="context length exceeded",
                                   error="context length exceeded", ok=False)
        tool = AgentTool("fake", "fake tool", {}, execute)
        session = self.history(config=replace(self.config, auto_compact_enabled=False, tools=[tool]))
        self.set_responses([ProviderResponseEndEvent(message=AssistantMessage(content="", tool_calls=[
            ToolCall(id="call", name="fake", arguments={})])), self.response("done")])
        events = [event async for event in session.prompt("new")]
        self.assertEqual([event.source for event in events if isinstance(event, ErrorEvent)], ["tool"])
        self.assertEqual(len(self.seen), 2)
        self.assertFalse(self.compactions())
        self.assert_reloads(session)
