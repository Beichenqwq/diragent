"""MessageEntry 的消息类型校验和序列化约定。"""

import unittest
from time import time
from uuid import UUID

from pydantic import ValidationError

from agent.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage, Usage
from agent.session import CompactionEntry, MessageEntry, SessionInfoEntry


class MessageEntryTests(unittest.TestCase):
    def test_round_trip_preserves_message_types_and_wire_aliases(self) -> None:
        messages = [
            UserMessage(content="读取文件"),
            AssistantMessage(
                content="",
                tool_calls=[ToolCall(id="call-1", name="read", arguments={"path": "README.md"})],
            ),
            ToolResultMessage(tool_call_id="call-1", name="read", content="文件内容", ok=True),
        ]
        for message in messages:
            with self.subTest(role=message.role):
                entry = MessageEntry(message=message)
                restored = MessageEntry.model_validate_json(entry.model_dump_json())
                self.assertIsInstance(restored.message, type(message))
                self.assertEqual(restored.message, message)
                self.assertEqual(restored.id, entry.id)
                self.assertEqual(restored.timestamp, entry.timestamp)
                self.assertEqual(entry.model_dump()["message"], message.model_dump())

    def test_metadata_is_created_for_each_record(self) -> None:
        before = time()
        first = MessageEntry(message=UserMessage(content="first"))
        second = MessageEntry(message=UserMessage(content="second"))
        after = time()

        self.assertEqual(UUID(hex=first.id).version, 4)
        self.assertNotEqual(first.id, second.id)
        self.assertLessEqual(before, first.timestamp)
        self.assertLessEqual(first.timestamp, after)

    def test_existing_metadata_is_preserved(self) -> None:
        entry = MessageEntry.model_validate({
            "id": "saved-entry",
            "timestamp": 1700000000.0,
            "message": {"role": "user", "content": "saved message"},
        })
        self.assertEqual(entry.id, "saved-entry")
        self.assertEqual(entry.timestamp, 1700000000.0)

    def test_rejects_invalid_record_data(self) -> None:
        invalid_records = [
            {},
            {"message": {"role": "unknown", "content": "text"}},
            {"message": {"role": "tool", "content": "missing call identity"}},
            {"message": {"role": "user", "content": "text"}, "model": "unused"},
        ]
        for record in invalid_records:
            with self.subTest(record=record):
                with self.assertRaises(ValidationError):
                    MessageEntry.model_validate(record)


class SessionInfoEntryTests(unittest.TestCase):
    def test_defaults_and_saved_metadata(self) -> None:
        before = time()
        first = SessionInfoEntry(cwd="D:/project")
        second = SessionInfoEntry(cwd="D:/project")
        self.assertEqual(UUID(hex=first.session_id).version, 4)
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertLessEqual(before, first.created_at)
        self.assertLessEqual(first.created_at, time())
        self.assertEqual(SessionInfoEntry.model_validate_json(first.model_dump_json()), first)

    def test_cwd_is_required(self) -> None:
        with self.assertRaises(ValidationError):
            SessionInfoEntry.model_validate({})


class CompactionEntryTests(unittest.TestCase):
    def test_defaults_create_record_metadata_and_allow_missing_boundary(self) -> None:
        before = time()
        first = CompactionEntry(summary="summary")
        second = CompactionEntry(summary="summary")

        self.assertEqual(first.type, "compaction")
        self.assertEqual(UUID(hex=first.id).version, 4)
        self.assertNotEqual(first.id, second.id)
        self.assertLessEqual(before, first.timestamp)
        self.assertLessEqual(first.timestamp, time())
        self.assertIsNone(first.first_kept_entry_id)
        self.assertIsNone(first.tokens_before)
        self.assertIsNone(first.usage)

    def test_round_trip_preserves_boundary_and_summary_request_usage(self) -> None:
        entry = CompactionEntry(
            id="compact-1", timestamp=1700000000.0,
            summary="目标：修复问题\n下一步：运行测试",
            first_kept_entry_id="kept-1", tokens_before=50000,
            usage=Usage(input_tokens=1000, output_tokens=100, total_tokens=1100,
                        cached_input_tokens=200),
        )
        restored = CompactionEntry.model_validate_json(entry.model_dump_json())
        self.assertEqual(restored, entry)
        self.assertIsInstance(restored.usage, Usage)

    def test_invalid_summary_usage_and_unknown_fields_are_rejected(self) -> None:
        records = [
            {},
            {"summary": "summary", "type": "message"},
            {"summary": "summary", "tokens_before": -1},
            {"summary": "summary", "usage": {"input_tokens": -1,
                                                "output_tokens": 1, "total_tokens": 0}},
            {"summary": "summary", "replaces_entry_ids": ["old"]},
        ]
        for record in records:
            with self.subTest(record=record):
                with self.assertRaises(ValidationError):
                    CompactionEntry.model_validate(record)
