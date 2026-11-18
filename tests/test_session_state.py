"""会话状态重建的空记录和元信息约束。"""

import unittest

from agent.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from agent.session import CompactionEntry, MessageEntry, SessionInfoEntry, SessionState


class SessionStateTests(unittest.TestCase):
    def test_empty_states_have_independent_message_lists(self) -> None:
        first = SessionState.from_entries([])
        second = SessionState.from_entries([])
        self.assertIsNone(first.session_info)
        self.assertEqual(first.messages, [])
        self.assertEqual(first.context_entry_ids, [])
        first.messages.append(UserMessage(content="new"))
        first.context_entry_ids.append("new-entry")
        self.assertEqual(second.messages, [])
        self.assertEqual(second.context_entry_ids, [])

    def test_message_only_history_can_be_restored(self) -> None:
        entries = [MessageEntry(message=UserMessage(content=text)) for text in ("first", "second")]
        state = SessionState.from_entries(iter(entries))
        self.assertIsNone(state.session_info)
        self.assertEqual(state.messages, [entry.message for entry in entries])
        self.assertEqual(state.context_entry_ids, [entry.id for entry in entries])

    def test_duplicate_metadata_is_rejected(self) -> None:
        info = SessionInfoEntry(cwd="D:/project")
        with self.assertRaisesRegex(ValueError, "multiple session_info"):
            SessionState.from_entries([info, info])

    def test_identical_messages_remain_associated_with_distinct_entry_ids(self) -> None:
        info = SessionInfoEntry(cwd="D:/project")
        first = MessageEntry(id="first", message=UserMessage(content="same text"))
        second = MessageEntry(id="second", message=UserMessage(content="same text"))

        state = SessionState.from_entries(iter([info, first, second]))

        self.assertEqual(state.session_info, info)
        self.assertEqual(state.messages, [first.message, second.message])
        self.assertEqual(state.context_entry_ids, ["first", "second"])
        self.assertEqual(list(zip(state.context_entry_ids, state.messages, strict=True)),
                         [("first", first.message), ("second", second.message)])

    def test_assistant_and_tool_result_keep_source_ids_and_call_identity(self) -> None:
        assistant = MessageEntry(id="assistant-entry", message=AssistantMessage(
            content="", tool_calls=[ToolCall(id="call-1", name="read", arguments={})],
        ))
        result = MessageEntry(id="result-entry", message=ToolResultMessage(
            tool_call_id="call-1", name="read", content="file content",
        ))

        state = SessionState.from_entries([assistant, result])

        self.assertEqual(state.context_entry_ids, ["assistant-entry", "result-entry"])
        self.assertEqual(state.messages, [assistant.message, result.message])
        self.assertEqual(state.messages[0].tool_calls[0].id, state.messages[1].tool_call_id)

    def test_compaction_summary_uses_compaction_id_as_message_source(self) -> None:
        old = MessageEntry(id="old", message=UserMessage(content="old context"))
        compaction = CompactionEntry(id="compact", summary="saved summary",
                                    first_kept_entry_id="old")
        after = MessageEntry(id="after", message=UserMessage(content="next task"))
        entries = [old, compaction, after]
        before = [entry.model_dump_json() for entry in entries]

        state = SessionState.from_entries(iter(entries))

        self.assertEqual(state.context_entry_ids, ["compact", "old", "after"])
        self.assertEqual(state.messages, [
            UserMessage(content="Previous conversation summary:\nsaved summary"),
            old.message,
            after.message,
        ])
        self.assertEqual([entry.model_dump_json() for entry in entries], before)

    def test_compaction_keeps_boundary_inclusively_at_each_message_position(self) -> None:
        messages = [
            MessageEntry(id=entry_id, message=UserMessage(content=entry_id))
            for entry_id in ("first", "middle", "last")
        ]
        after = MessageEntry(id="after", message=UserMessage(content="after"))
        for index, boundary in enumerate(messages):
            with self.subTest(boundary=boundary.id):
                compaction = CompactionEntry(
                    id="compact", summary="summary", first_kept_entry_id=boundary.id,
                )
                state = SessionState.from_entries(iter([*messages, compaction, after]))
                retained = messages[index:]
                self.assertEqual(state.context_entry_ids,
                                 ["compact", *[entry.id for entry in retained], "after"])
                self.assertEqual(state.messages, [
                    UserMessage(content="Previous conversation summary:\nsummary"),
                    *[entry.message for entry in retained], after.message,
                ])

    def test_missing_or_unresolvable_boundary_keeps_summary_and_successors(self) -> None:
        info = SessionInfoEntry(session_id="metadata", cwd="D:/project")
        old = MessageEntry(id="old", message=UserMessage(content="old"))
        after = MessageEntry(id="after", message=UserMessage(content="after"))
        for boundary in (None, "missing", "compact", "after", "metadata"):
            with self.subTest(boundary=boundary):
                compaction = CompactionEntry(id="compact", summary="summary",
                                            first_kept_entry_id=boundary)
                entries = [info, old, compaction, after]
                before = [entry.model_dump_json() for entry in entries]

                state = SessionState.from_entries(iter(entries))

                self.assertEqual(state.session_info, info)
                self.assertEqual(state.context_entry_ids, ["compact", "after"])
                self.assertEqual(state.messages, [
                    UserMessage(content="Previous conversation summary:\nsummary"), after.message,
                ])
                self.assertEqual([entry.model_dump_json() for entry in entries], before)

    def test_compaction_without_prior_messages_still_adds_summary(self) -> None:
        state = SessionState.from_entries([CompactionEntry(id="compact", summary="summary")])
        self.assertEqual(state.context_entry_ids, ["compact"])
        self.assertEqual(state.messages,
                         [UserMessage(content="Previous conversation summary:\nsummary")])

    def test_same_content_is_filtered_by_source_entry_id(self) -> None:
        first = MessageEntry(id="first", message=UserMessage(content="same"))
        kept = MessageEntry(id="kept", message=UserMessage(content="same"))
        compaction = CompactionEntry(id="compact", summary="summary", first_kept_entry_id="kept")

        state = SessionState.from_entries([first, kept, compaction])

        self.assertEqual(state.context_entry_ids, ["compact", "kept"])
        self.assertEqual(state.messages,
                         [UserMessage(content="Previous conversation summary:\nsummary"), kept.message])

    def test_repeated_compaction_retains_context_in_original_entry_order(self) -> None:
        old = MessageEntry(id="old", message=UserMessage(content="old"))
        kept = MessageEntry(id="kept", message=UserMessage(content="kept"))
        first = CompactionEntry(id="first-compact", summary="first summary",
                                first_kept_entry_id="kept")
        middle = MessageEntry(id="middle", message=UserMessage(content="middle"))
        second = CompactionEntry(id="second-compact", summary="second summary",
                                 first_kept_entry_id="kept")
        after = MessageEntry(id="after", message=UserMessage(content="after"))

        state = SessionState.from_entries([old, kept, first, middle, second, after])

        self.assertEqual(state.context_entry_ids,
                         ["second-compact", "kept", "first-compact", "middle", "after"])
        self.assertEqual(state.messages, [
            UserMessage(content="Previous conversation summary:\nsecond summary"),
            kept.message,
            UserMessage(content="Previous conversation summary:\nfirst summary"),
            middle.message, after.message,
        ])

    def test_repeated_compaction_does_not_restore_previously_removed_messages(self) -> None:
        old = MessageEntry(id="old", message=UserMessage(content="old"))
        kept = MessageEntry(id="kept", message=UserMessage(content="kept"))
        first = CompactionEntry(id="first-compact", summary="first summary",
                                first_kept_entry_id="kept")
        second = CompactionEntry(id="second-compact", summary="second summary",
                                 first_kept_entry_id="old")

        state = SessionState.from_entries([old, kept, first, second])

        self.assertEqual(state.context_entry_ids, ["second-compact", "kept", "first-compact"])
        self.assertEqual(state.messages, [
            UserMessage(content="Previous conversation summary:\nsecond summary"), kept.message,
            UserMessage(content="Previous conversation summary:\nfirst summary"),
        ])

    def test_boundary_can_reference_previous_compaction_record(self) -> None:
        old = MessageEntry(id="old", message=UserMessage(content="old"))
        kept = MessageEntry(id="kept", message=UserMessage(content="kept"))
        first = CompactionEntry(id="first-compact", summary="first summary",
                                first_kept_entry_id="kept")
        middle = MessageEntry(id="middle", message=UserMessage(content="middle"))
        second = CompactionEntry(id="second-compact", summary="second summary",
                                 first_kept_entry_id="first-compact")

        state = SessionState.from_entries([old, kept, first, middle, second])

        self.assertEqual(state.context_entry_ids, ["second-compact", "first-compact", "middle"])
        self.assertEqual(state.messages, [
            UserMessage(content="Previous conversation summary:\nsecond summary"),
            UserMessage(content="Previous conversation summary:\nfirst summary"), middle.message,
        ])
