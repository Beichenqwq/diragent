"""按近期 token 预算和消息角色确定有效上下文的保留起点。"""

import unittest

from agent.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage, Usage
from coding.context_window import DEFAULT_COMPACTION_KEEP_RECENT_TOKENS, estimate_message_tokens
from types import SimpleNamespace
from unittest.mock import patch

from coding.session import CodingSession


class FirstRecentContextIndexTests(unittest.TestCase):
    def assert_plan_boundary(self, rows, expected_index, *,
                             keep_recent_tokens=DEFAULT_COMPACTION_KEEP_RECENT_TOKENS):
        session = SimpleNamespace(_state=SimpleNamespace(
            context_entry_ids=[entry_id for entry_id, _ in rows],
            messages=[message for _, message in rows],
        ))
        with patch("coding.session.DEFAULT_COMPACTION_KEEP_RECENT_TOKENS", keep_recent_tokens):
            plan = CodingSession._recent_preserving_compaction_plan(session)
        if expected_index <= 0 or expected_index >= len(rows):
            self.assertIsNone(plan)
        else:
            self.assertIsNotNone(plan)
            self.assertEqual(plan.first_kept_entry_id, rows[expected_index][0])
            self.assertEqual(plan.messages_to_summarize,
                             tuple(message for _, message in rows[:expected_index]))

    def test_budget_threshold_and_first_user_adjustment(self) -> None:
        # 每条消息为 4 token 开销加 1 token 内容，合计 5 tokens。
        rows = tuple((str(index), UserMessage(content="abcd")) for index in range(3))
        cases = [(1, 2), (5, 2), (6, 1), (10, 1), (11, 1), (15, 1), (16, 0)]
        before = [message.model_dump() for _, message in rows]
        for budget, expected in cases:
            with self.subTest(budget=budget):
                self.assert_plan_boundary(rows, expected, keep_recent_tokens=budget)
        self.assertEqual([message.model_dump() for _, message in rows], before)

    def test_default_budget_keeps_recent_twenty_thousand_tokens(self) -> None:
        self.assertEqual(DEFAULT_COMPACTION_KEEP_RECENT_TOKENS, 20_000)
        # 39,984 个字符占 9,996 tokens，加消息开销后每条为 10,000 tokens。
        rows = tuple((str(index), UserMessage(content="x" * 39984)) for index in range(3))
        self.assert_plan_boundary(rows, 1)

    def test_empty_or_short_history_keeps_all_messages(self) -> None:
        self.assert_plan_boundary((), 0)
        self.assert_plan_boundary((("only", UserMessage(content="short")),), 0)

    def test_nonpositive_budget_has_no_retained_messages(self) -> None:
        rows = (("only", UserMessage(content="content")),)
        for budget in (0, -1):
            with self.subTest(budget=budget):
                self.assert_plan_boundary(rows, 1, keep_recent_tokens=budget)

    def test_tool_information_is_counted_and_tool_only_suffix_is_skipped(self) -> None:
        assistant = AssistantMessage(content="", tool_calls=[
            ToolCall(id="call-1", name="read", arguments={"path": "README.md"}),
        ])
        result = ToolResultMessage(tool_call_id="call-1", name="read", content="file contents")
        rows = (("user", UserMessage(content="read file")), ("assistant", assistant), ("tool", result))

        self.assert_plan_boundary(rows, 3, keep_recent_tokens=estimate_message_tokens(result))
        self.assert_plan_boundary(rows, 1, keep_recent_tokens=estimate_message_tokens(result) + 1)

    def test_budget_uses_message_estimate_not_historical_request_usage(self) -> None:
        reply = AssistantMessage(content="abcd", usage=Usage(
            input_tokens=10000, output_tokens=10000, total_tokens=20000,
        ))
        self.assert_plan_boundary((("reply", reply),), 0)

    def test_assistant_candidate_moves_to_next_user(self) -> None:
        rows = (("old", UserMessage(content="abcd")),
                ("answer", AssistantMessage(content="abcd")),
                ("next", UserMessage(content="abcd")),
                ("last", AssistantMessage(content="abcd")))
        self.assert_plan_boundary(rows, 2, keep_recent_tokens=15)

    def test_tool_candidate_moves_to_next_user(self) -> None:
        result = ToolResultMessage(tool_call_id="call", name="read", content="abcd")
        rows = (("tool", result), ("next", UserMessage(content="abcd")),
                ("last", AssistantMessage(content="abcd")))
        budget = sum(estimate_message_tokens(message) for _, message in rows)
        self.assert_plan_boundary(rows, 1, keep_recent_tokens=budget)

    def test_without_later_user_assistant_candidate_itself_is_kept(self) -> None:
        rows = (("old", UserMessage(content="abcd")),
                ("answer", AssistantMessage(content="abcd")),
                ("last", AssistantMessage(content="abcd")))
        self.assert_plan_boundary(rows, 1, keep_recent_tokens=10)

    def test_without_later_user_tool_candidate_moves_to_assistant(self) -> None:
        result = ToolResultMessage(tool_call_id="call", name="read", content="abcd")
        rows = (("old", UserMessage(content="abcd")), ("tool", result),
                ("answer", AssistantMessage(content="abcd")))
        budget = estimate_message_tokens(result) + estimate_message_tokens(rows[2][1])
        self.assert_plan_boundary(rows, 2, keep_recent_tokens=budget)

    def test_first_user_without_later_user_keeps_everything(self) -> None:
        rows = (("user", UserMessage(content="abcd")),
                ("answer", AssistantMessage(content="abcd")))
        self.assert_plan_boundary(rows, 0, keep_recent_tokens=10)
