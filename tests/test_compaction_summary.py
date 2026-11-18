"""摘要请求的消息序列化、提示词选择及独立 provider 调用。"""

import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import Mock

from ai.events import ProviderErrorEvent, ProviderResponseEndEvent, ProviderTextDeltaEvent
from ai.provider import ModelProvider
from agent.messages import AssistantMessage, ToolCall, ToolResultMessage, Usage, UserMessage
from agent.session import JsonlSessionStorage, MessageEntry
from coding.session import (
    COMPACTION_SUMMARY_PREFIX, SUMMARIZATION_PROMPT, SUMMARIZATION_SYSTEM_PROMPT,
    UPDATE_SUMMARIZATION_PROMPT,
)
from coding.session import CodingSession, CodingSessionConfig


class CompactionSummaryPromptTests(unittest.IsolatedAsyncioTestCase):
    async def conversation_text(self, messages):
        prompt = await self.request_summary_prompt(messages)
        return prompt.split("<conversation>\n", 1)[1].split("\n</conversation>", 1)[0]

    async def request_summary_prompt(self, messages, *, custom_instructions=None):
        async def stream_response(**kwargs):
            yield ProviderResponseEndEvent(message=AssistantMessage(content="summary"))

        provider = Mock(spec=ModelProvider)
        provider.stream_response.side_effect = stream_response
        session = SimpleNamespace(_harness=SimpleNamespace(
            config=SimpleNamespace(provider=provider, model="fake-model"),
        ))
        await CodingSession._generate_compaction_summary(
            session, messages, custom_instructions=custom_instructions,
        )
        return provider.stream_response.call_args.kwargs["messages"][0].content

    async def test_serialization_preserves_complete_content_and_tool_information(self) -> None:
        content = "文件内容\n" + "x" * 1000
        messages = (
            UserMessage(content="读取文件"),
            AssistantMessage(content="正在读取", tool_calls=[
                ToolCall(id="call", name="read", arguments={"path": "中文.md", "line": 3})]),
            ToolResultMessage(tool_call_id="call", name="read", content=content, ok=False),
        )
        serialized = await self.conversation_text(messages)
        self.assertIn("<message index=1 role=user>\n读取文件", serialized)
        self.assertIn("<message index=2 role=assistant>\n正在读取", serialized)
        self.assertIn('- read: {"path": "中文.md", "line": 3}', serialized)
        self.assertIn("<message index=3 role=tool name=read error=true>", serialized)
        self.assertIn(content, serialized)

    async def test_successful_tool_result_records_its_status(self) -> None:
        result = ToolResultMessage(tool_call_id="call", name="read", content="ok")
        self.assertIn("error=false", await self.conversation_text((result,)))

    async def test_first_summary_uses_all_messages_and_normal_prompt(self) -> None:
        messages = (UserMessage(content="task"), AssistantMessage(content="progress"))
        prompt = await self.request_summary_prompt(messages)
        self.assertIn(await self.conversation_text(messages), prompt)
        self.assertTrue(prompt.endswith(SUMMARIZATION_PROMPT))
        self.assertNotIn("<previous-summary>", prompt)
        for section in ("## Goal", "## Constraints & Preferences", "## Progress",
                        "## Key Decisions", "## Next Steps", "## Critical Context"):
            self.assertIn(section, prompt)

    async def test_update_separates_first_old_summary_from_new_messages(self) -> None:
        messages = (UserMessage(content=COMPACTION_SUMMARY_PREFIX + "old summary"),
                    AssistantMessage(content="new progress"))
        prompt = await self.request_summary_prompt(messages)
        self.assertIn("<previous-summary>\nold summary\n</previous-summary>", prompt)
        conversation = prompt.split("<conversation>\n", 1)[1].split("\n</conversation>", 1)[0]
        self.assertEqual(conversation, await self.conversation_text(messages[1:]))
        self.assertNotIn("old summary", conversation)
        self.assertTrue(prompt.endswith(UPDATE_SUMMARIZATION_PROMPT))

    async def test_later_summary_and_assistant_prefix_do_not_trigger_update(self) -> None:
        cases = ((UserMessage(content="ordinary"),
                  UserMessage(content=COMPACTION_SUMMARY_PREFIX + "old")),
                 (AssistantMessage(content=COMPACTION_SUMMARY_PREFIX + "old"),))
        for messages in cases:
            with self.subTest(messages=messages):
                prompt = await self.request_summary_prompt(messages)
                self.assertTrue(prompt.endswith(SUMMARIZATION_PROMPT))
                self.assertNotIn("<previous-summary>", prompt)
                self.assertIn(COMPACTION_SUMMARY_PREFIX + "old", prompt)

    async def test_no_new_messages_and_empty_old_summary_are_supported(self) -> None:
        self.assertIn("(no new messages)", await self.request_summary_prompt(()))
        prompt = await self.request_summary_prompt((UserMessage(content=COMPACTION_SUMMARY_PREFIX),))
        self.assertIn("<previous-summary>\n\n</previous-summary>", prompt)
        self.assertIn("(no new messages)", prompt)
        self.assertTrue(prompt.endswith(UPDATE_SUMMARIZATION_PROMPT))

    async def test_custom_instructions_are_appended_for_both_prompt_types(self) -> None:
        for messages in ((), (UserMessage(content=COMPACTION_SUMMARY_PREFIX + "old"),)):
            with self.subTest(messages=messages):
                prompt = await self.request_summary_prompt(messages, custom_instructions="  focus on tests  ")
                self.assertTrue(prompt.endswith("Additional focus: focus on tests"))
                self.assertEqual(await self.request_summary_prompt(messages, custom_instructions=" \n "),
                                 await self.request_summary_prompt(messages))


class CompactionSummaryRequestTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(temp_dir.cleanup)
        cwd = Path(temp_dir.name)
        self.provider = Mock(spec=ModelProvider)
        storage = JsonlSessionStorage(cwd / "session.jsonl")
        storage.append(MessageEntry(message=UserMessage(content="saved task")))
        self.session = CodingSession.load(CodingSessionConfig(
            provider=self.provider, model="fake-model", storage=storage, cwd=cwd, system="coding system",
        ))
        self.path = cwd / "session.jsonl"
        self.saved = self.path.read_bytes()
        self.state_before = self.session.state
        self.state_dump = [message.model_dump() for message in self.session.state.messages]
        self.ids_before = list(self.session.state.context_entry_ids)
        self.harness_before = self.session.messages
        self.config_before = self.session._harness.config
        self.addCleanup(self.assert_unchanged)

    def assert_unchanged(self) -> None:
        self.assertEqual(self.path.read_bytes(), self.saved)
        self.assertIs(self.session.state, self.state_before)
        self.assertEqual([message.model_dump() for message in self.session.state.messages], self.state_dump)
        self.assertEqual(self.session.state.context_entry_ids, self.ids_before)
        self.assertEqual(self.session.messages, self.harness_before)
        self.assertIs(self.session._harness.config, self.config_before)

    def set_events(self, events, *, error=None) -> None:
        async def iterator():
            for event in events:
                yield event
            if error is not None:
                raise error
        self.provider.stream_response.side_effect = lambda **kwargs: iterator()

    async def test_request_uses_current_provider_model_special_system_and_no_tools(self) -> None:
        usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        self.set_events([ProviderTextDeltaEvent(delta="partial"), ProviderResponseEndEvent(
            message=AssistantMessage(content="  complete summary\n", usage=usage))])
        messages = (UserMessage(content="summarize this"),)
        result = await self.session._generate_compaction_summary(messages, custom_instructions="tests")
        self.assertEqual(result.text, "complete summary")
        self.assertEqual(result.usage, usage)
        self.provider.stream_response.assert_called_once()
        request = self.provider.stream_response.call_args.kwargs
        self.assertEqual(request["model"], "fake-model")
        self.assertEqual(request["system"], SUMMARIZATION_SYSTEM_PROMPT)
        self.assertEqual(request["tools"], [])
        self.assertEqual(len(request["messages"]), 1)
        self.assertIsInstance(request["messages"][0], UserMessage)
        prompt = request["messages"][0].content
        self.assertIn("<message index=1 role=user>\nsummarize this\n</message>", prompt)
        self.assertTrue(prompt.endswith(SUMMARIZATION_PROMPT + "\n\nAdditional focus: tests"))

    async def test_final_text_without_usage_is_accepted(self) -> None:
        self.set_events([ProviderResponseEndEvent(message=AssistantMessage(content="summary"))])
        result = await self.session._generate_compaction_summary(())
        self.assertEqual(result.text, "summary")
        self.assertIsNone(result.usage)

    async def test_delta_only_stream_is_joined_when_final_event_is_absent(self) -> None:
        self.set_events([ProviderTextDeltaEvent(delta=" sum"), ProviderTextDeltaEvent(delta="mary ")])
        result = await self.session._generate_compaction_summary(())
        self.assertEqual(result.text, "summary")
        self.assertIsNone(result.usage)

    async def test_empty_final_text_is_not_replaced_with_partial_deltas(self) -> None:
        self.set_events([ProviderTextDeltaEvent(delta="partial"),
                         ProviderResponseEndEvent(message=AssistantMessage(content=" \n "))])
        with self.assertRaisesRegex(RuntimeError, "empty summary"):
            await self.session._generate_compaction_summary(())

    async def test_stream_without_text_fails(self) -> None:
        self.set_events([])
        with self.assertRaisesRegex(RuntimeError, "empty summary"):
            await self.session._generate_compaction_summary(())

    async def test_provider_error_event_rejects_partial_summary(self) -> None:
        self.set_events([ProviderTextDeltaEvent(delta="partial"),
                         ProviderErrorEvent(message="context limit exceeded")])
        with self.assertRaisesRegex(RuntimeError, "Compaction summarization failed: context limit exceeded"):
            await self.session._generate_compaction_summary(())

    async def test_provider_exception_propagates(self) -> None:
        self.set_events([ProviderTextDeltaEvent(delta="partial")], error=ConnectionError("offline"))
        with self.assertRaisesRegex(ConnectionError, "offline"):
            await self.session._generate_compaction_summary(())
