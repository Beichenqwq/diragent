"""记忆模型请求独立于聊天上下文，失败不会破坏正常回复。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ai.events import ProviderErrorEvent, ProviderResponseEndEvent, ProviderTextDeltaEvent
from ai.provider import ModelProvider
from agent.messages import AssistantMessage, UserMessage
from agent.session import JsonlSessionStorage
from coding.memory import (
    MEMORY_EXTRACTION_SYSTEM_PROMPT, MEMORY_SCOPES, MEMORY_TYPES,
    extract_memory_candidates, load_existing_memories, load_memory_index,
    normalize_memory_candidate, parse_json_array, store_memory,
)
from coding.session import CodingSession, CodingSessionConfig


class JsonArrayParsingTests(unittest.TestCase):
    def test_accepts_json_with_markdown_and_explanation_before_and_after(self):
        text = '以下是提取结果：\n```json\n[{"name": "记忆"}]\n```\n以上是所有候选。'
        self.assertEqual(parse_json_array(text), [{"name": "记忆"}])

    def test_skips_unparseable_brackets_and_returns_first_valid_array(self):
        text = '[说明文字] [invalid JSON] [{"name": "first"}] [{"name": "second"}]'
        self.assertEqual(parse_json_array(text), [{"name": "first"}])

    def test_handles_nested_arrays_brackets_in_strings_and_escaped_quotes(self):
        expected = [{"body": '字符串包含 ] 和 [，以及 "引号"', "values": [1, 2]}]
        self.assertEqual(parse_json_array("结果：" + json.dumps(expected, ensure_ascii=False)), expected)

    def test_returns_empty_for_missing_invalid_and_incomplete_arrays(self):
        for text in ("", "no JSON", '{"name": "memory"}', '[{"name": "unfinished"}', '[invalid]'):
            with self.subTest(text=text):
                self.assertEqual(parse_json_array(text), [])

    def test_returns_first_empty_array_without_selecting_a_later_one(self):
        self.assertEqual(parse_json_array('[] then [{"name": "later"}]'), [])

    def test_parsing_arrays_does_not_require_object_elements(self):
        self.assertEqual(parse_json_array('Result: [1, "text", null]'), [1, "text", None])


class MemoryCandidateValidationTests(unittest.TestCase):
    def setUp(self):
        self.valid = {"name": "Code explanations", "type": "user", "scope": "persistent",
                      "description": "Explain call flow", "body": "结合实际调用流程解释代码。"}

    def test_accepts_all_allowed_types_and_scopes(self):
        for memory_type in MEMORY_TYPES:
            for scope in MEMORY_SCOPES:
                with self.subTest(memory_type=memory_type, scope=scope):
                    candidate = {**self.valid, "type": memory_type, "scope": scope}
                    self.assertEqual(normalize_memory_candidate(candidate), candidate)

    def test_rejects_missing_empty_and_non_string_fields(self):
        for field in self.valid:
            with self.subTest(field=field, missing=True):
                candidate = self.valid.copy()
                del candidate[field]
                self.assertIsNone(normalize_memory_candidate(candidate))
            for value in ("", " \n\t ", None, 42, False, [], {}):
                with self.subTest(field=field, value=value):
                    self.assertIsNone(normalize_memory_candidate({**self.valid, field: value}))

    def test_rejects_non_objects_and_invalid_enum_values(self):
        for candidate in (None, "memory", 1, []):
            self.assertIsNone(normalize_memory_candidate(candidate))
        for memory_type in ("user preference", "USER", "unknown"):
            self.assertIsNone(normalize_memory_candidate({**self.valid, "type": memory_type}))
        for scope in ("session", "temporary", "Persistent"):
            self.assertIsNone(normalize_memory_candidate({**self.valid, "scope": scope}))

    def test_strips_field_edges_without_changing_body_interior(self):
        expected = {**self.valid, "body": "First line\n\n    indented line"}
        padded = {field: " \t" + value + "\n " for field, value in expected.items()}
        self.assertEqual(normalize_memory_candidate(padded), expected)

    def test_rebuilds_required_fields_without_mutating_input(self):
        candidate = {**self.valid, "extra": {"unexpected": True}}
        result = normalize_memory_candidate(candidate)
        self.assertEqual(result, self.valid)
        self.assertIsNot(result, candidate)
        self.assertIn("extra", candidate)


class MemoryExtractionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        retrieval = patch("coding.session.retrieve_memories", return_value=[])
        retrieval.start()
        self.addCleanup(retrieval.stop)
        root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        root.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(directory.cleanup)
        self.cwd = Path(directory.name)
        self.provider = Mock(spec=ModelProvider)
        self.seen = []

    def set_responses(self, responses):
        def stream_response(**kwargs):
            self.seen.append(kwargs)
            async def iterator():
                for event in responses.pop(0):
                    yield event
            return iterator()
        self.provider.stream_response.side_effect = stream_response

    def response(self, content):
        return ProviderResponseEndEvent(message=AssistantMessage(content=content), finish_reason="stop")

    def session(self):
        return CodingSession.load(CodingSessionConfig(
            provider=self.provider, model="test-model", system="coding instructions", tools=[],
            cwd=self.cwd, storage=JsonlSessionStorage(self.cwd / "session.jsonl"),
            auto_compact_enabled=False,
        ))

    async def test_extracts_json_from_complete_response_without_duplicating_deltas(self):
        candidates = [{"name": "Code explanations", "type": "user", "scope": "persistent",
                       "description": "Explain call flow", "body": "结合实际调用流程解释代码。"}]
        text = json.dumps(candidates, ensure_ascii=False)
        self.set_responses([[ProviderTextDeltaEvent(delta=text[:20]),
                             ProviderTextDeltaEvent(delta=text[20:]), self.response(text)]])
        result = await extract_memory_candidates(self.provider, model="test-model", prompt="extract")
        self.assertEqual(result, candidates)
        self.assertEqual(self.seen[0]["system"], MEMORY_EXTRACTION_SYSTEM_PROMPT)
        self.assertEqual(self.seen[0]["tools"], [])
        self.assertEqual(self.seen[0]["messages"], [UserMessage(content="extract")])

    async def test_accepts_empty_candidate_list(self):
        self.set_responses([[self.response("[]")]])
        self.assertEqual(await extract_memory_candidates(self.provider, model="test", prompt="extract"), [])

    async def test_returns_empty_if_response_has_no_parseable_array(self):
        for text in ("not JSON", '{"name": "memory"}'):
            with self.subTest(text=text):
                self.set_responses([[self.response(text)]])
                self.assertEqual(await extract_memory_candidates(self.provider, model="test", prompt="extract"), [])

    async def test_extracts_candidates_from_markdown_wrapped_response(self):
        candidate = {"name": "memory", "type": "project", "scope": "persistent",
                     "description": "Project convention", "body": "记忆目录为 .tau/memory。"}
        self.set_responses([[self.response("结果：\n```json\n" + json.dumps([candidate]) + "\n```\n完成。")]])
        self.assertEqual(await extract_memory_candidates(self.provider, model="test", prompt="extract"),
                         [candidate])

    async def test_filters_non_objects_from_memory_candidates(self):
        self.set_responses([[self.response('["memory"]')]])
        self.assertEqual(await extract_memory_candidates(self.provider, model="test", prompt="extract"), [])

    async def test_filters_invalid_candidates_without_discarding_valid_items(self):
        valid = {"name": "Code explanations", "type": "user", "scope": "persistent",
                 "description": "Explain call flow", "body": "结合实际调用流程解释代码。"}
        current_task = {**valid, "scope": "current_task"}
        padded = {field: " " + value + " " for field, value in valid.items()}
        candidates = [{**padded, "extra": "ignored"}, {**valid, "body": " "}, {**valid, "type": "user preference"},
                      {**valid, "scope": "session"}, {"name": "missing fields"}, None, "memory", current_task]
        self.set_responses([[self.response(json.dumps(candidates))]])
        self.assertEqual(await extract_memory_candidates(self.provider, model="test", prompt="extract"),
                         [valid, current_task])

    async def test_requires_completed_response_and_propagates_provider_errors(self):
        for events in ([ProviderTextDeltaEvent(delta="[]")],
                       [ProviderErrorEvent(message="network failure")],
                       [ProviderResponseEndEvent(message=AssistantMessage(content="[]"),
                                                 finish_reason="length")]):
            with self.subTest(events=events):
                self.set_responses([events])
                with self.assertRaises(RuntimeError):
                    await extract_memory_candidates(self.provider, model="test", prompt="extract")

    async def test_session_includes_catalog_and_history_without_persisting_extraction_messages(self):
        session = self.session()
        path = self.cwd / ".tau" / "memory" / "MEMORY.md"
        path.parent.mkdir(parents=True)
        path.write_text("- [Existing](existing.md) - 已有项目约定\n", encoding="utf-8")
        self.set_responses([[self.response("answer one")], [self.response("[]")],
                            [self.response("answer two")], [self.response("[]")]])
        for question in ("question one", "question two"):
            events = [event async for event in session.prompt(question)]
            self.assertFalse(any(getattr(event, "delta", "") == "[]" for event in events))
        prompt = self.seen[-1]["messages"][0].content
        self.assertIn("已有项目约定", prompt)
        self.assertIn("user: question one", prompt)
        self.assertIn("assistant: answer one", prompt)
        self.assertIn("user: question two", prompt)
        self.assertIn("assistant: answer two", prompt)
        self.assertEqual([message.content for message in session.messages],
                         ["question one", "answer one", "question two", "answer two"])
        self.assertEqual(CodingSession.load(session.config).messages, session.messages)
        self.assertEqual(len(session.storage.read_all()), 5)
        self.assertEqual(list(path.parent.iterdir()), [path])

    async def test_memory_failure_logs_but_chat_completes_and_busy_is_released(self):
        session = self.session()
        self.set_responses([[self.response("answer")], [ProviderErrorEvent(message="extraction failed")]])
        with self.assertLogs("coding.session", level="WARNING"):
            events = [event async for event in session.prompt("question")]
        self.assertTrue(events)
        self.assertEqual(session.messages[-1].content, "answer")
        self.assertFalse(session._busy)

    async def test_session_saves_only_persistent_nonduplicate_candidates(self):
        session = self.session()
        path = self.cwd / ".tau" / "memory" / "MEMORY.md"
        path.parent.mkdir(parents=True)
        path.write_text("- [Existing](existing.md) - Existing description\n", encoding="utf-8")
        (path.parent / "existing.md").write_text("Existing body\n", encoding="utf-8")
        valid = {"name": "New", "type": "user", "scope": "persistent",
                 "description": "New description", "body": "New body"}
        candidates = [valid, {**valid, "scope": "current_task"}, {**valid, "name": "Existing"},
                      {**valid, "description": "Existing description"}, {**valid, "body": "Existing body"}, valid.copy()]
        self.set_responses([[self.response(json.dumps(candidates))]])
        await session._try_extract_memories([UserMessage(content="Long-term preferences")])
        self.assertEqual(len(list(path.parent.iterdir())), 3)
        self.assertEqual(load_existing_memories(self.cwd, load_memory_index(self.cwd))[-1],
                         {key: valid[key] for key in ("name", "type", "description", "body")})

    async def test_each_candidate_is_saved_before_checking_the_next(self):
        session = self.session()
        first = {"name": "First", "type": "project", "scope": "persistent",
                 "description": "First description", "body": "First body"}
        second = {**first, "name": "Second", "description": "Second description", "body": "Second body"}
        self.set_responses([[self.response(json.dumps([first, second]))]])
        def observe_store(cwd, candidate):
            if candidate == second:
                self.assertTrue((cwd / ".tau" / "memory" / "first.md").exists())
                self.assertIn("[First](first.md)", load_memory_index(cwd))
            return store_memory(cwd, candidate)
        with patch("coding.session.store_memory", side_effect=observe_store):
            await session._try_extract_memories([UserMessage(content="Stable project facts")])
        self.assertEqual(load_existing_memories(self.cwd, load_memory_index(self.cwd)), [
            {key: candidate[key] for key in ("name", "type", "description", "body")}
            for candidate in (first, second)
        ])

    async def test_failed_write_does_not_block_an_identical_later_candidate(self):
        session = self.session()
        candidate = {"name": "Preference", "type": "user", "scope": "persistent",
                     "description": "Description", "body": "Body"}
        self.set_responses([[self.response(json.dumps([candidate, candidate]))]])
        attempts = 0
        def fail_first(cwd, item):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("First write failed")
            return store_memory(cwd, item)
        with patch("coding.session.store_memory", side_effect=fail_first), self.assertLogs("coding.session", level="WARNING"):
            await session._try_extract_memories([UserMessage(content="Remember preference")])
        self.assertEqual(attempts, 2)
        self.assertEqual(load_existing_memories(self.cwd, load_memory_index(self.cwd)), [
            {key: candidate[key] for key in ("name", "type", "description", "body")},
        ])

    async def test_failed_chat_does_not_request_extraction(self):
        session = self.session()
        self.set_responses([[ProviderErrorEvent(message="network failure")]])
        _ = [event async for event in session.prompt("question")]
        self.assertEqual(len(self.seen), 1)

    async def test_empty_dialogue_does_not_request_model(self):
        session = self.session()
        await session._try_extract_memories([])
        self.provider.stream_response.assert_not_called()
        self.assertFalse((self.cwd / ".tau").exists())
