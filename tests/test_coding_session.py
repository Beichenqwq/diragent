"""编程会话的新建、恢复和损坏记录处理。"""

import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from datetime import date
from unittest.mock import Mock, patch

from ai.provider import ModelProvider
from ai.events import ProviderResponseEndEvent
from agent.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from agent.session import (
    CompactionEntry, JsonlSessionStorage, MessageEntry, SessionInfoEntry, SessionJsonlError,
)
from coding import (
    CodingSession, CodingSessionConfig, ProjectContextFile, ResourceError,
    build_system_prompt, create_read_tool, load_skills, estimate_context_usage, estimate_text_tokens,
)


class CodingSessionLoadTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(temp_dir.cleanup)
        self.path = Path(temp_dir.name) / "session.jsonl"
        self.config = CodingSessionConfig(
            provider=Mock(spec=ModelProvider),
            model="fake", system="system",
            storage=JsonlSessionStorage(self.path), cwd=Path(temp_dir.name),
        )

    def test_new_session_metadata_is_written_once_and_reused(self) -> None:
        session = CodingSession.load(self.config)
        self.assertIs(session.storage, self.config.storage)
        entries = session.storage.read_all()
        self.assertEqual(len(entries), 1)
        self.assertIsInstance(entries[0], SessionInfoEntry)
        self.assertEqual(session.state.session_info, entries[0])
        self.assertEqual(entries[0].cwd, str(self.config.cwd.resolve()))
        self.assertEqual(session.state.messages, [])
        saved = self.path.read_bytes()
        reloaded = CodingSession.load(self.config)
        self.assertEqual(reloaded.state.session_info, session.state.session_info)
        self.assertEqual(self.path.read_bytes(), saved)
        self.config.provider.stream_response.assert_not_called()

    def test_existing_history_and_metadata_are_preserved(self) -> None:
        info = SessionInfoEntry(session_id="old-session", created_at=1700000000.0, cwd="D:/old-project")
        messages = [UserMessage(content="问题"), AssistantMessage(content="回答")]
        self.config.storage.append(info)
        for message in messages:
            self.config.storage.append(MessageEntry(message=message))
        saved = self.path.read_bytes()
        session = CodingSession.load(self.config)
        self.assertEqual(session.state.session_info, info)
        self.assertEqual(session.state.messages, messages)
        self.assertEqual(self.path.read_bytes(), saved)

    def test_compaction_plan_distinguishes_saved_messages_with_same_content(self) -> None:
        entries = [
            SessionInfoEntry(cwd=str(self.config.cwd)),
            MessageEntry(id="first", message=UserMessage(content="x" * 80000)),
            MessageEntry(id="second", message=UserMessage(content="x" * 80000)),
        ]
        for entry in entries:
            self.config.storage.append(entry)
        saved = self.path.read_bytes()
        session = CodingSession.load(self.config)

        plan = session._recent_preserving_compaction_plan()

        self.assertIsNotNone(plan)
        self.assertEqual(plan.first_kept_entry_id, "second")
        self.assertEqual(plan.messages_to_summarize, (entries[1].message,))
        self.assertEqual(self.path.read_bytes(), saved)
        self.config.provider.stream_response.assert_not_called()

    def test_compaction_plan_uses_state_and_rejects_mismatched_lengths(self) -> None:
        session = CodingSession.load(self.config)
        session._harness.replace_messages([UserMessage(content="not saved")])
        self.assertIsNone(session._recent_preserving_compaction_plan())

        session.state.context_entry_ids.append("missing-message")
        with self.assertRaises(ValueError):
            session._recent_preserving_compaction_plan()

    def test_compaction_plan_selects_saved_ids_and_messages_without_side_effects(self) -> None:
        messages = [UserMessage(content="x" * 39984) for _ in range(3)]
        self.config.storage.append(SessionInfoEntry(cwd=str(self.config.cwd)))
        for entry_id, message in zip(("first", "second", "third"), messages, strict=True):
            self.config.storage.append(MessageEntry(id=entry_id, message=message))
        session = CodingSession.load(self.config)
        saved = self.path.read_bytes()
        state_before = session.state
        messages_before = session.messages

        plan = session._recent_preserving_compaction_plan()

        self.assertIsNotNone(plan)
        self.assertEqual(plan.first_kept_entry_id, "second")
        self.assertEqual(plan.replaced_entry_count, 1)
        self.assertEqual(plan.messages_to_summarize, (messages[0],))
        self.assertIs(plan.messages_to_summarize[0], session.state.messages[0])
        self.assertIs(session.state, state_before)
        self.assertEqual(session.messages, messages_before)
        self.assertEqual(session.state.context_entry_ids, ["first", "second", "third"])
        self.assertEqual(self.path.read_bytes(), saved)
        self.config.provider.stream_response.assert_not_called()

    def test_compaction_plan_uses_replayed_summary_and_excludes_removed_messages(self) -> None:
        entries = [SessionInfoEntry(cwd=str(self.config.cwd)),
                   MessageEntry(id="removed", message=UserMessage(content="old")),
                   MessageEntry(id="kept", message=UserMessage(content="x" * 39984)),
                   CompactionEntry(id="summary", summary="old summary", first_kept_entry_id="kept"),
                   MessageEntry(id="answer", message=AssistantMessage(content="x" * 39984))]
        for entry in entries:
            self.config.storage.append(entry)
        session = CodingSession.load(self.config)

        plan = session._recent_preserving_compaction_plan()

        self.assertIsNotNone(plan)
        self.assertEqual(plan.first_kept_entry_id, "kept")
        self.assertEqual(plan.replaced_entry_count, 1)
        self.assertEqual(plan.messages_to_summarize,
                         (UserMessage(content="Previous conversation summary:\nold summary"),))

    def test_compaction_plan_is_absent_for_empty_single_or_short_context(self) -> None:
        histories = [[], [UserMessage(content="x" * 80000)],
                     [UserMessage(content="short"), AssistantMessage(content="answer")],
                     [UserMessage(content="x" * 39984), AssistantMessage(content="x" * 39984)]]
        for index, messages in enumerate(histories):
            with self.subTest(index=index):
                storage = JsonlSessionStorage(self.path.with_name(f"no-plan-{index}.jsonl"))
                for message in messages:
                    storage.append(MessageEntry(message=message))
                session = CodingSession.load(replace(self.config, storage=storage))
                self.assertIsNone(session._recent_preserving_compaction_plan())

    def test_compaction_plan_can_keep_an_assistant_when_no_later_user_exists(self) -> None:
        entries = [SessionInfoEntry(cwd=str(self.config.cwd)),
                   MessageEntry(id="user", message=UserMessage(content="read file")),
                   MessageEntry(id="call", message=AssistantMessage(content="", tool_calls=[
                       ToolCall(id="tool-call", name="read", arguments={"path": "file"})])),
                   MessageEntry(id="tool", message=ToolResultMessage(
                       tool_call_id="tool-call", name="read", content="contents")),
                   MessageEntry(id="answer", message=AssistantMessage(content="x" * 80000))]
        for entry in entries:
            self.config.storage.append(entry)
        session = CodingSession.load(self.config)

        plan = session._recent_preserving_compaction_plan()

        self.assertIsNotNone(plan)
        self.assertEqual(plan.first_kept_entry_id, "answer")
        self.assertEqual(plan.replaced_entry_count, 3)
        self.assertEqual(plan.messages_to_summarize, tuple(entry.message for entry in entries[1:4]))

    def test_compaction_plan_is_absent_when_retained_suffix_contains_only_tool_results(self) -> None:
        entries = [SessionInfoEntry(cwd=str(self.config.cwd)),
                   MessageEntry(message=UserMessage(content="read file")),
                   MessageEntry(message=AssistantMessage(content="", tool_calls=[
                       ToolCall(id="call", name="read", arguments={"path": "file"})])),
                   MessageEntry(message=ToolResultMessage(
                       tool_call_id="call", name="read", content="x" * 80000))]
        for entry in entries:
            self.config.storage.append(entry)
        session = CodingSession.load(self.config)
        self.assertIsNone(session._recent_preserving_compaction_plan())
        self.config.provider.stream_response.assert_not_called()

    def test_invalid_records_are_not_overwritten(self) -> None:
        self.path.write_text("{broken\n", encoding="utf-8")
        saved = self.path.read_bytes()
        with self.assertRaises(SessionJsonlError):
            CodingSession.load(self.config)
        self.assertEqual(self.path.read_bytes(), saved)

    def test_duplicate_session_metadata_is_rejected(self) -> None:
        self.config.storage.append(SessionInfoEntry(cwd="D:/one"))
        self.config.storage.append(SessionInfoEntry(cwd="D:/two"))
        with self.assertRaisesRegex(ValueError, "multiple session_info"):
            CodingSession.load(self.config)

    def test_load_applies_compaction_to_harness_without_rewriting_history(self) -> None:
        info = SessionInfoEntry(cwd=str(self.config.cwd))
        old = MessageEntry(id="old", message=UserMessage(content="old task"))
        kept = MessageEntry(id="kept", message=UserMessage(content="recent task"))
        compaction = CompactionEntry(id="compact", summary="saved summary",
                                    first_kept_entry_id="kept")
        after = MessageEntry(id="after", message=AssistantMessage(content="recent answer"))
        entries = [info, old, kept, compaction, after]
        for entry in entries:
            self.config.storage.append(entry)
        saved = self.path.read_bytes()

        session = CodingSession.load(self.config)
        reloaded = CodingSession.load(self.config)

        self.assertEqual(session.state.session_info, info)
        self.assertEqual(session.state.context_entry_ids, ["compact", "kept", "after"])
        self.assertEqual(list(session.messages), [
            UserMessage(content="Previous conversation summary:\nsaved summary"),
            kept.message, after.message,
        ])
        self.assertEqual(session.state.messages, list(session.messages))
        self.assertEqual(reloaded.state, session.state)
        self.assertEqual(session.storage.read_all(), entries)
        self.assertEqual(self.path.read_bytes(), saved)
        self.config.provider.stream_response.assert_not_called()


class CodingSessionAssemblyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.enterContext(patch("coding.session.extract_memory_candidates", return_value=[]))
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(temp_dir.cleanup)
        self.cwd = Path(temp_dir.name)
        self.config = CodingSessionConfig(
            provider=Mock(spec=ModelProvider), model="fake", system="test system",
            storage=JsonlSessionStorage(self.cwd / "session.jsonl"), cwd=self.cwd,
        )

    async def test_restored_history_and_default_tools_reach_model(self) -> None:
        (self.cwd / "README.md").write_text("项目说明", encoding="utf-8")
        history = [UserMessage(content="旧问题"), AssistantMessage(content="旧回答")]
        self.config.storage.append(SessionInfoEntry(cwd=str(self.cwd)))
        for message in history:
            self.config.storage.append(MessageEntry(message=message))
        responses = [
            AssistantMessage(content="", tool_calls=[
                ToolCall(id="read-1", name="read", arguments={"path": "README.md"}),
            ]),
            AssistantMessage(content="已经读取"),
        ]
        seen = []

        def stream_response(**kwargs):
            seen.append({**kwargs, "messages": list(kwargs["messages"])})

            async def iterator():
                yield ProviderResponseEndEvent(message=responses.pop(0))

            return iterator()

        self.config.provider.stream_response.side_effect = stream_response
        session = CodingSession.load(self.config)
        self.config.provider.stream_response.assert_not_called()
        self.assertIsNot(session._harness._messages, session.state.messages)
        events = [event async for event in session.prompt("读取 README.md")]
        self.assertTrue(events)
        self.assertEqual(seen[0]["messages"][:2], history)
        self.assertEqual(seen[0]["model"], "fake")
        self.assertEqual(seen[0]["system"], "test system")
        self.assertEqual([tool.name for tool in seen[0]["tools"]], ["read", "write", "edit", "bash"])
        self.assertIn("项目说明", seen[1]["messages"][-1].content)
        self.assertEqual(session.state.messages, list(session.messages))
        self.assertEqual(session.state.context_entry_ids, [
            entry.id for entry in session.storage.read_all() if isinstance(entry, MessageEntry)
        ])
        self.assertEqual(len(session.state.messages), len(history) + 4)
        self.assertEqual(CodingSession.load(self.config).state, session.state)

    def test_explicit_tools_and_empty_list_override_defaults(self) -> None:
        custom_tool = create_read_tool(self.cwd)
        for tools in ([custom_tool], []):
            with self.subTest(tools=tools):
                session = CodingSession.load(replace(self.config, tools=tools))
                self.assertEqual(session._harness._config.tools, tools)

    def set_responses(self, responses):
        seen = []

        def stream_response(**kwargs):
            seen.append(list(kwargs["messages"]))

            async def iterator():
                yield ProviderResponseEndEvent(message=responses.pop(0))

            return iterator()

        self.config.provider.stream_response.side_effect = stream_response
        return seen

    async def test_prompt_reload_continue_and_prompt_save_only_new_messages(self) -> None:
        seen = self.set_responses([
            AssistantMessage(content="first answer"),
            AssistantMessage(content="continued answer"),
            AssistantMessage(content="second answer"),
        ])
        config = replace(self.config, tools=[])
        session = CodingSession.load(config)
        initial_info = session.state.session_info
        async for _ in session.prompt("first question"):
            pass
        first_entries = config.storage.read_all()
        self.assertEqual(len(first_entries), 3)

        resumed = CodingSession.load(config)
        async for _ in resumed.continue_():
            pass
        self.assertEqual([message.content for message in seen[1]], ["first question", "first answer"])
        async for _ in resumed.prompt("second question"):
            pass
        entries = config.storage.read_all()
        self.assertEqual(entries[:3], first_entries)
        self.assertEqual(len(entries), 6)
        self.assertEqual(len({entry.id for entry in entries if isinstance(entry, MessageEntry)}), 5)
        self.assertEqual(resumed.state.context_entry_ids, [
            entry.id for entry in entries if isinstance(entry, MessageEntry)
        ])
        self.assertEqual(resumed.state.session_info, initial_info)
        self.assertEqual([message.content for message in resumed.state.messages], [
            "first question", "first answer", "continued answer", "second question", "second answer",
        ])
        self.assertEqual(CodingSession.load(config).state, resumed.state)

    async def test_incomplete_iteration_does_not_persist_run(self) -> None:
        self.set_responses([AssistantMessage(content="answer")])
        session = CodingSession.load(replace(self.config, tools=[]))
        snapshot = session.messages
        self.assertIsInstance(snapshot, tuple)
        saved = session.storage.path.read_bytes()
        events = session.prompt("question")
        await anext(events)
        self.assertEqual(snapshot, ())
        self.assertEqual([message.content for message in session.messages], ["question"] )
        self.assertEqual(session.state.messages, [])
        await events.aclose()
        self.assertEqual(self.config.storage.path.read_bytes(), saved)

    async def test_selected_skill_reaches_provider_and_survives_reload(self) -> None:
        skills_dir = self.cwd / ".tau" / "skills"
        for name, body in (("review", "检查逻辑错误。"), ("testing", "编写测试。")):
            path = skills_dir / name / "SKILL.md"
            path.parent.mkdir(parents=True)
            path.write_text(f"---\ndescription: {name}\n---\n{body}", encoding="utf-8")
        seen = self.set_responses([
            AssistantMessage(content="审查结果"),
            AssistantMessage(content="后续结果"),
            AssistantMessage(content="普通回复"),
        ])
        config = replace(self.config, tools=[])
        session = CodingSession.load(config)
        async for _ in session.prompt("检查 loop.py", skill_name="review"):
            pass
        expanded = seen[0][-1].content
        self.assertIn('<skill name="review"', expanded)
        self.assertIn("检查逻辑错误。", expanded)
        self.assertNotIn("编写测试。", expanded)
        self.assertTrue(expanded.endswith("</skill>\n\n检查 loop.py"))
        saved_entries = session.storage.read_all()
        self.assertEqual(saved_entries[1].message.content, expanded)
        resumed = CodingSession.load(config)
        async for _ in resumed.continue_():
            pass
        self.assertEqual(seen[1][0].content, expanded)
        async for _ in resumed.prompt("普通问题"):
            pass
        self.assertEqual(seen[2][-1].content, "普通问题")
        self.assertEqual(resumed.state.messages, list(resumed.messages))
        self.assertEqual(resumed.storage.read_all()[:3], saved_entries)

    async def test_unknown_skill_does_not_modify_messages_or_storage(self) -> None:
        session = CodingSession.load(replace(self.config, tools=[]))
        before_messages = session.messages
        before_bytes = session.storage.path.read_bytes()
        with self.assertRaisesRegex(ResourceError, "Unknown skill: missing"):
            async for _ in session.prompt("检查代码", skill_name="missing"):
                pass
        self.assertEqual(session.messages, before_messages)
        self.assertEqual(session.storage.path.read_bytes(), before_bytes)
        self.config.provider.stream_response.assert_not_called()

    async def test_automatic_system_reuses_skills_and_preserves_run_persistence(self) -> None:
        path = self.cwd / ".tau/skills/review/SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text("---\ndescription: Review code\n---\nREVIEW_BODY_SENTINEL", encoding="utf-8")
        config = CodingSessionConfig(
            provider=self.config.provider, model="fake", storage=self.config.storage, cwd=self.cwd,
            context_files=(ProjectContextFile("AGENTS.md", "Project rules."),),
            current_date=date(2026, 10, 2), extra_guidelines=("Run project tests.",),
        )
        calls = []
        responses = [AssistantMessage(content=text) for text in ("reviewed", "answered", "continued")]

        def stream_response(**kwargs):
            calls.append({**kwargs, "messages": list(kwargs["messages"])})

            async def iterator():
                yield ProviderResponseEndEvent(message=responses.pop(0))

            return iterator()

        config.provider.stream_response.side_effect = stream_response
        with patch("coding.session.load_skills", wraps=load_skills) as loader:
            with patch("coding.session.build_system_prompt", wraps=build_system_prompt) as builder:
                session = CodingSession.load(config)
                loader.assert_called_once_with(self.cwd)
                builder.assert_called_once()
                options = builder.call_args.args[0]
                self.assertIs(options.skills[0], session._skill_by_name["review"])
                self.assertIs(options.tools, session._harness._config.tools)
        config.provider.stream_response.assert_not_called()
        async for _ in session.prompt("Inspect code", skill_name="review"):
            pass
        generated_system = calls[0]["system"]
        self.assertIn("You are an expert coding assistant.", generated_system)
        self.assertIn("- read: read(path, offset?, limit?): inspect a local file.", generated_system)
        self.assertIn("<available_skills>", generated_system)
        self.assertIn("<name>review</name>", generated_system)
        self.assertIn("Project rules.", generated_system)
        self.assertIn("- Run project tests.", generated_system)
        self.assertIn("Current date: 2026-10-02", generated_system)
        self.assertIn(str(self.cwd).replace("\\", "/"), generated_system)
        self.assertNotIn("REVIEW_BODY_SENTINEL", generated_system)
        expanded = calls[0]["messages"][-1].content
        self.assertIn("REVIEW_BODY_SENTINEL", expanded)
        saved_entries = config.storage.read_all()
        self.assertEqual(saved_entries[1].message.content, expanded)

        async for _ in session.prompt("Ordinary question"):
            pass
        self.assertEqual(calls[1]["messages"][-1].content, "Ordinary question")
        resumed = CodingSession.load(config)
        async for _ in resumed.continue_():
            pass
        self.assertEqual([message.content for message in calls[2]["messages"]],
                         [expanded, "reviewed", "Ordinary question", "answered"])
        self.assertTrue(all(call["system"] == generated_system for call in calls))
        entries = config.storage.read_all()
        self.assertEqual(entries[:3], saved_entries)
        self.assertEqual(len(entries), 6)
        self.assertEqual(len({entry.id for entry in entries if isinstance(entry, MessageEntry)}), 5)
        self.assertEqual(resumed.state.messages, list(resumed.messages))

    async def test_explicit_system_including_empty_string_bypasses_builder(self) -> None:
        for system in ("Exact custom system", ""):
            with self.subTest(system=system):
                config = replace(self.config, system=system, tools=[],
                                 context_files=(ProjectContextFile("AGENTS.md", "Unused context"),),
                                 extra_guidelines=("Unused extra rule",))
                self.set_responses([AssistantMessage(content="answer")])
                with patch("coding.session.build_system_prompt", side_effect=AssertionError("Unexpected build")):
                    session = CodingSession.load(config)
                    async for _ in session.prompt("question"):
                        pass
                self.assertEqual(config.provider.stream_response.call_args.kwargs["system"], system)
                self.assertEqual(session._harness._config.tools, [])

    def test_automatic_system_respects_explicit_empty_tool_list(self) -> None:
        config = replace(self.config, system=None, tools=[], current_date=date(2026, 10, 2))
        with patch("coding.session.create_coding_tools", side_effect=AssertionError("Unexpected default tools")):
            session = CodingSession.load(config)
        self.assertEqual(session._harness._config.tools, [])
        self.assertIn("Available tools:\n(none)", session._harness._config.system)
        self.assertNotIn("<available_skills>", session._harness._config.system)

    def test_context_usage_reads_effective_config_without_rebuilding_or_io(self) -> None:
        for system in (None, "Explicit system", ""):
            for tools in (None, [create_read_tool(self.cwd)], []):
                with self.subTest(system=system, tools=tools):
                    config = replace(self.config, system=system, tools=tools)
                    session = CodingSession.load(config)
                    effective = session._harness.config
                    if system is None:
                        self.assertIn("You are an expert coding assistant.", effective.system)
                    else:
                        self.assertEqual(effective.system, system)
                    expected_names = (["read", "write", "edit", "bash"] if tools is None
                                      else [tool.name for tool in tools])
                    self.assertEqual([tool.name for tool in effective.tools], expected_names)
                    with patch.object(session.storage, "read_all", side_effect=AssertionError("Unexpected read")), \
                            patch("coding.session.load_skills", side_effect=AssertionError("Unexpected load")), \
                            patch("coding.session.build_system_prompt", side_effect=AssertionError("Unexpected build")):
                        usage = session.context_usage
                        self.assertEqual(usage, estimate_context_usage(
                            system=effective.system, messages=session.messages, tools=effective.tools,
                        ))
                        effective.system = "abcd"
                        effective.tools = []
                        updated = session.context_usage
                        self.assertEqual(updated.system_tokens, estimate_text_tokens("abcd"))
                        self.assertEqual(updated.tool_tokens, 0)
                        self.assertIsNot(updated, usage)
                    config.provider.stream_response.assert_not_called()

    async def test_context_usage_tracks_live_tool_run_continue_and_reload(self) -> None:
        (self.cwd / "README.md").write_text("Tool output", encoding="utf-8")
        self.set_responses([
            AssistantMessage(content="", tool_calls=[
                ToolCall(id="read-1", name="read", arguments={"path": "README.md"}),
            ]),
            AssistantMessage(content="Finished"),
            AssistantMessage(content="Continued"),
        ])
        config = replace(self.config, tools=[create_read_tool(self.cwd)])
        session = CodingSession.load(config)
        initial = session.context_usage
        self.assertEqual(initial.message_count, 0)
        saw_live_tool_result = False
        async for _ in session.prompt("Read file"):
            usage = session.context_usage
            self.assertEqual(usage.message_count, len(session.messages))
            if any(isinstance(message, ToolResultMessage) for message in session.messages):
                saw_live_tool_result = True
                self.assertGreaterEqual(usage.message_count, 3)
                # 保存前 state 尚未更新，统计依然包含 Harness 中的工具结果。
                self.assertEqual(session.state.messages, [])
        self.assertTrue(saw_live_tool_result)
        after_prompt = session.context_usage
        self.assertEqual(after_prompt.message_count, 4)
        self.assertGreater(after_prompt.message_tokens, initial.message_tokens)
        self.assertEqual(initial.message_count, 0)

        resumed = CodingSession.load(config)
        self.assertEqual(resumed.context_usage, after_prompt)
        async for _ in resumed.continue_():
            pass
        self.assertEqual(resumed.context_usage.message_count, 5)
        self.assertGreater(resumed.context_usage.message_tokens, after_prompt.message_tokens)
        self.assertEqual(CodingSession.load(config).context_usage, resumed.context_usage)
