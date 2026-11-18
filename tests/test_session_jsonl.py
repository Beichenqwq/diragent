"""JSONL 转换的往返、顺序和错误处理。"""

import unittest

from pydantic import ValidationError

from agent.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage, Usage
from agent.session import (
    CompactionEntry,
    MessageEntry,
    SessionInfoEntry,
    SessionJsonlError,
    entries_from_json_lines,
    entry_from_json_line,
    entry_to_json_line,
)


class SessionJsonlTests(unittest.TestCase):
    def test_round_trip_keeps_messages_and_metadata_on_one_line(self) -> None:
        messages = [
            UserMessage(content="你好\n第二行"),
            AssistantMessage(content="", tool_calls=[
                ToolCall(id="call-1", name="read", arguments={"path": "中文.txt"}),
            ]),
            ToolResultMessage(tool_call_id="call-1", name="read", content="失败", ok=False),
        ]
        for message in messages:
            with self.subTest(role=message.role):
                entry = MessageEntry(id="saved-id", timestamp=1700000000.0, message=message)
                line = entry_to_json_line(entry)
                self.assertTrue(line.endswith("\n"))
                self.assertEqual(len(line.splitlines()), 1)
                restored = entry_from_json_line(line)
                self.assertEqual(restored, entry)
                self.assertIsInstance(restored.message, type(message))

    def test_multiple_lines_skip_blanks_and_keep_order(self) -> None:
        entries = [MessageEntry(message=UserMessage(content=text)) for text in ("first", "second")]
        lines = iter(["\n", entry_to_json_line(entries[0]), " \t\n", entry_to_json_line(entries[1])])
        self.assertEqual(entries_from_json_lines(lines), entries)
        self.assertEqual(entries_from_json_lines([]), [])

    def test_invalid_json_and_records_raise_session_error_with_line_number(self) -> None:
        for line in ("{broken", "{}", '{"message":{"role":"unknown","content":"text"}}'):
            with self.subTest(line=line):
                with self.assertRaisesRegex(SessionJsonlError, "on line 3") as caught:
                    entries_from_json_lines(["", "  ", line])
                self.assertIsInstance(caught.exception.__cause__, ValidationError)

    def test_single_invalid_line_works_without_line_number(self) -> None:
        with self.assertRaises(SessionJsonlError) as caught:
            entry_from_json_line("not JSON")
        self.assertNotIn("on line", str(caught.exception))

    def test_assistant_usage_round_trip_and_old_messages_without_usage(self) -> None:
        entry = MessageEntry(message=AssistantMessage(content="answer", usage=Usage(
            input_tokens=100, output_tokens=20, total_tokens=120, cached_input_tokens=40,
        )))
        line = entry_to_json_line(entry)
        self.assertEqual(entry_from_json_line(line), entry)
        old_line = '{"type":"message","message":{"role":"assistant","content":"old answer"}}'
        restored = entry_from_json_line(old_line)
        self.assertIsNone(restored.message.usage)

    def test_mixed_records_round_trip_with_compaction_and_usage(self) -> None:
        entries = [
            SessionInfoEntry(session_id="session-1", cwd="D:/project"),
            MessageEntry(id="kept", message=UserMessage(content="继续任务")),
            CompactionEntry(
                id="compact", timestamp=1700000000.0,
                summary="已完成第一步\n保留任务约束", first_kept_entry_id="kept",
                tokens_before=30000,
                usage=Usage(input_tokens=500, output_tokens=50, total_tokens=550),
            ),
        ]
        lines = [entry_to_json_line(entry) for entry in entries]
        self.assertTrue(all(len(line.splitlines()) == 1 for line in lines))
        restored = entries_from_json_lines(iter(lines))
        self.assertEqual(restored, entries)
        self.assertIsInstance(restored[-1], CompactionEntry)
        self.assertIsInstance(restored[-1].usage, Usage)

    def test_compaction_optional_fields_can_be_omitted(self) -> None:
        restored = entry_from_json_line('{"type":"compaction","summary":"summary"}')
        self.assertIsInstance(restored, CompactionEntry)
        self.assertIsNone(restored.first_kept_entry_id)
        self.assertIsNone(restored.tokens_before)
        self.assertIsNone(restored.usage)

    def test_invalid_compaction_reports_physical_line_number(self) -> None:
        line = '{"type":"compaction","summary":"summary","tokens_before":-1}'
        with self.assertRaisesRegex(SessionJsonlError, "on line 2"):
            entries_from_json_lines(["\n", line])
