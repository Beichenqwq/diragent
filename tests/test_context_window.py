"""分项估算覆盖消息内容、工具参数和定义，不产生执行副作用。"""

import unittest
from copy import deepcopy
from unittest.mock import Mock

from agent.context import request_context_fingerprint, response_context_fingerprint
from agent.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage, Usage
from agent.tools import AgentTool
from coding.context_window import (
    ContextUsageEstimate,
    estimate_context_usage,
    estimate_message_tokens,
    estimate_text_tokens,
    estimate_tool_tokens,
)


class ContextComponentEstimateTests(unittest.TestCase):
    def test_text_rounding_including_empty_and_unicode(self) -> None:
        cases = [("", 0), ("a", 1), ("abcd", 1), ("abcde", 2),
                 ("abcdefgh", 2), ("abcdefghi", 3), ("你好，世界", 2)]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(estimate_text_tokens(text), expected)

    def test_user_and_tool_results_include_content_and_message_overhead(self) -> None:
        self.assertEqual(estimate_message_tokens(UserMessage(content="")), 4)
        self.assertEqual(estimate_message_tokens(UserMessage(content="abcd")), 5)
        result = ToolResultMessage(tool_call_id="call-1", name="read", content="abcdefgh")
        self.assertEqual(estimate_message_tokens(result), 7)
        alternate = result.model_copy(update={"tool_call_id": "different-call-id", "ok": False})
        self.assertEqual(estimate_message_tokens(alternate), 7)

    def test_assistant_counts_every_call_and_json_arguments(self) -> None:
        read = ToolCall(id="call-1", name="read", arguments={"path": "中文"})
        write = ToolCall(id="call-2", name="write", arguments={"value": True})
        message = AssistantMessage(content="", tool_calls=[read])
        # read 名称占 1；JSON 文本 {"path": "中文"} 的 14 个字符占 4。
        self.assertEqual(estimate_message_tokens(message), 9)
        message.tool_calls.append(write)
        self.assertEqual(estimate_message_tokens(message), 15)
        message.content = "abcd"
        self.assertEqual(estimate_message_tokens(message), 16)
        self.assertEqual(estimate_message_tokens(AssistantMessage(content="abcd")), 5)

    def test_tool_definition_counts_schema_and_excludes_prompt_metadata(self) -> None:
        executor = Mock(side_effect=AssertionError("Estimating must not execute tools"))
        tool = AgentTool("read", "abcdefgh", {}, executor)
        self.assertEqual(estimate_tool_tokens(tool), 20)
        rich = AgentTool("read", "abcdefgh", {}, executor,
                         prompt_snippet="Extra system text", prompt_guidelines=("A rule",))
        self.assertEqual(estimate_tool_tokens(rich), 20)
        schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        with_schema = AgentTool("read", "abcdefgh", schema, executor)
        self.assertGreater(estimate_tool_tokens(with_schema), 20)
        executor.assert_not_called()

    def test_repeated_estimates_preserve_nested_inputs(self) -> None:
        arguments = {"paths": ["中文", "README.md"], "options": {"enabled": True}}
        message = AssistantMessage(content="Check files", tool_calls=[
            ToolCall(id="call-1", name="read", arguments=arguments),
        ])
        schema = {"type": "object", "properties": {"paths": {"type": "array"}}}
        executor = Mock(side_effect=AssertionError("Unexpected tool execution"))
        tool = AgentTool("read", "Read files", schema, executor)
        before_message = message.model_dump()
        before_schema = deepcopy(schema)
        message_tokens = estimate_message_tokens(message)
        tool_tokens = estimate_tool_tokens(tool)
        self.assertEqual(estimate_message_tokens(message), message_tokens)
        self.assertEqual(estimate_tool_tokens(tool), tool_tokens)
        self.assertEqual(message.model_dump(), before_message)
        self.assertEqual(schema, before_schema)
        executor.assert_not_called()


class ContextUsageEstimateTests(unittest.TestCase):
    def test_empty_context(self) -> None:
        usage = estimate_context_usage(system="", messages=[], tools=[])
        self.assertEqual(usage, ContextUsageEstimate(
            total_tokens=0, system_tokens=0, message_tokens=0,
            tool_tokens=0, message_count=0,
        ))

    def test_mixed_context_accepts_sequences_without_side_effects(self) -> None:
        messages = (
            UserMessage(content="abcd"),
            AssistantMessage(content="", tool_calls=[
                ToolCall(id="call-1", name="read", arguments={"path": "中文"}),
            ]),
            ToolResultMessage(tool_call_id="call-1", name="read", content="abcdefgh"),
        )
        executor = Mock(side_effect=AssertionError("Unexpected tool execution"))
        tools = (
            AgentTool("read", "abcdefgh", {}, executor),
            AgentTool("write", "", {}, executor),
        )
        before_messages = [message.model_dump() for message in messages]
        before_schemas = deepcopy([tool.input_schema for tool in tools])
        usage = estimate_context_usage(system="abcde", messages=messages, tools=tools)
        # system=2；三条消息=5+9+7；两个工具=20+19。
        self.assertEqual(usage, ContextUsageEstimate(
            total_tokens=62, system_tokens=2, message_tokens=21,
            tool_tokens=39, message_count=3,
        ))
        self.assertEqual([message.model_dump() for message in messages], before_messages)
        self.assertEqual([tool.input_schema for tool in tools], before_schemas)
        executor.assert_not_called()

    def test_recalculation_reflects_new_messages_without_changing_old_snapshot(self) -> None:
        messages = [UserMessage(content="abcd")]
        before = estimate_context_usage(system="", messages=messages, tools=[])
        messages.append(UserMessage(content="abcde"))
        after = estimate_context_usage(system="", messages=messages, tools=[])
        self.assertEqual((before.message_count, before.total_tokens), (1, 5))
        self.assertEqual((after.message_count, after.total_tokens), (2, 11))


class ProviderContextPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = Mock(side_effect=AssertionError("Unexpected tool execution"))
        self.tool = AgentTool("read", "Read files", {"type": "object"}, self.executor)
        self.messages = [UserMessage(content="Read file")]
        self.reply = self.add_reply(self.messages, 1000)

    def add_reply(self, messages, total):
        reply = AssistantMessage(content="reply", usage=Usage(
            input_tokens=total - 20, output_tokens=20, total_tokens=total,
            cached_input_tokens=100,
        ))
        request = request_context_fingerprint(
            model="fake", system="system", messages=messages, tools=[self.tool],
        )
        reply.usage_context = response_context_fingerprint(request, reply)
        messages.append(reply)
        return reply

    def predict(self, **overrides):
        inputs = dict(model="fake", system="system", messages=self.messages, tools=[self.tool])
        inputs.update(overrides)
        return estimate_context_usage(**inputs)

    def assert_fallback(self, **overrides):
        usage = self.predict(**overrides)
        self.assertFalse(usage.uses_provider_usage)
        self.assertEqual(usage.provider_tokens, 0)
        self.assertEqual(usage.total_tokens,
                         usage.system_tokens + usage.message_tokens + usage.tool_tokens)
        self.assertEqual(usage.estimated_tokens, usage.total_tokens)

    def test_matching_usage_counts_only_trailing_messages(self):
        self.messages.extend([
            ToolResultMessage(tool_call_id="c1", name="read", content="file contents"),
            UserMessage(content="Check again"),
            AssistantMessage(content="reply without usage"),
        ])
        before = [message.model_dump() for message in self.messages]
        usage = self.predict()
        trailing = sum(estimate_message_tokens(message) for message in self.messages[2:])
        self.assertEqual(usage.total_tokens, 1000 + trailing)
        self.assertEqual(usage.provider_tokens, 1000)
        self.assertEqual(usage.estimated_tokens, trailing)
        self.assertTrue(usage.uses_provider_usage)
        self.assertEqual(usage.message_count, 5)
        self.assertEqual(usage.message_tokens,
                         sum(estimate_message_tokens(message) for message in self.messages))
        self.assertEqual([message.model_dump() for message in self.messages], before)
        self.executor.assert_not_called()
        # 不传 model，仍可显式调用原来的整体估算流程。
        self.assert_fallback(model=None)

    def test_latest_matching_usage_and_earlier_match_if_latest_is_invalid(self):
        self.messages.append(UserMessage(content="next question"))
        later = self.add_reply(self.messages, 1500)
        self.messages.append(UserMessage(content="last question"))
        self.assertEqual(self.predict().provider_tokens, 1500)
        later.usage_context = "invalid"
        usage = self.predict()
        self.assertEqual(usage.provider_tokens, 1000)
        self.assertEqual(usage.total_tokens, 1000 + sum(
            estimate_message_tokens(message) for message in self.messages[2:]
        ))

    def test_changed_model_system_or_tool_definition_falls_back(self):
        saved = self.reply.usage.model_dump()
        for overrides in (
            {"model": "different-model"},
            {"system": "different system"},
            {"tools": []},
            {"tools": [AgentTool("read", "Changed description", self.tool.input_schema, self.executor)]},
            {"tools": [AgentTool("read", "Read files", {"type": "string"}, self.executor)]},
        ):
            with self.subTest(overrides=overrides):
                self.assert_fallback(**overrides)
        self.assertEqual(self.reply.usage.model_dump(), saved)
        self.assertTrue(self.predict().uses_provider_usage)

    def test_history_and_saved_reply_edits_fall_back(self):
        history = deepcopy(self.messages)
        changes = [
            [UserMessage(content="modified question"), deepcopy(self.reply)],
            [deepcopy(self.reply)],
            [UserMessage(content="summary"), deepcopy(self.reply)],
            list(reversed(deepcopy(self.messages))),
        ]
        edited_reply = deepcopy(self.messages)
        edited_reply[-1].content = "edited answer"
        changes.append(edited_reply)
        edited_call = deepcopy(self.messages)
        edited_call[-1].tool_calls.append(ToolCall(id="new", name="read"))
        changes.append(edited_call)
        for messages in changes:
            with self.subTest(messages=messages):
                self.assert_fallback(messages=messages)
        self.assertEqual(self.messages, history)

    def test_unsigned_or_zero_usage_falls_back(self):
        for total, context in ((1000, None), (1000, "invalid"), (0, None)):
            messages = [UserMessage(content="question"), AssistantMessage(
                content="reply",
                usage=Usage(input_tokens=total, output_tokens=0, total_tokens=total),
                usage_context=context,
            )]
            self.assert_fallback(messages=messages)

    def test_local_metadata_and_executor_do_not_change_request_signature(self):
        earlier = AssistantMessage(content="earlier answer")
        messages = [earlier, UserMessage(content="question")]
        self.add_reply(messages, 1000)
        earlier.usage = Usage(input_tokens=1, output_tokens=1, total_tokens=2)
        earlier.usage_context = "old metadata"
        equivalent = AgentTool(
            "read", "Read files", {"type": "object"}, Mock(),
            prompt_snippet="Unused snippet", prompt_guidelines=("Unused rule",),
        )
        self.assertTrue(self.predict(messages=messages, tools=[equivalent]).uses_provider_usage)
