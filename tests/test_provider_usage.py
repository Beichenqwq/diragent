"""通过模拟 HTTP 流验证请求用量、历史序列化和会话持久化。"""

import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import httpx

from ai.env import OpenAICompatibleConfig
from ai.events import ProviderResponseEndEvent, ProviderTextDeltaEvent
from ai.openai_compatible import OpenAICompatibleProvider, _parse_chunk_usage
from agent.messages import AssistantMessage, UserMessage, Usage
from agent.session import JsonlSessionStorage
from coding import CodingSession, CodingSessionConfig, estimate_context_usage, estimate_message_tokens, create_read_tool


def text_chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}, "finish_reason": None}], "usage": None}


def usage_chunk(*, input_tokens=100, output_tokens=20, cached=40) -> dict:
    data = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cached is not None:
        data["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"choices": [], "usage": data}


def sse(chunks: list[dict]) -> bytes:
    return ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            + "data: [DONE]\n\n").encode("utf-8")


class ProviderUsageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch("coding.session.extract_memory_candidates", return_value=[]))

    async def make_provider(self, responses, *, include_usage=True, thinking=None):
        seen = []

        def handle(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, content=sse(responses.pop(0)),
                                  headers={"content-type": "text/event-stream"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        self.addAsyncCleanup(client.aclose)
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test-key", base_url="https://example.test/v1",
                                   include_usage=include_usage, thinking=thinking),
            client=client,
        )
        return provider, seen

    async def collect(self, provider):
        return [event async for event in provider.stream_response(
            model="fake", system="system", messages=[], tools=[],
        )]

    async def test_thinking_option_is_explicit_and_absent_by_default(self):
        for thinking in (None, "disabled"):
            provider, seen = await self.make_provider([[text_chunk("answer")]], thinking=thinking)
            await self.collect(provider)
            if thinking is None:
                self.assertNotIn("thinking", seen[0])
            else:
                self.assertEqual(seen[0]["thinking"], {"type": "disabled"})

    async def test_empty_choices_usage_is_saved_without_double_counting(self):
        provider, seen = await self.make_provider([[
            text_chunk("Hello"),
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            usage_chunk(input_tokens=80, output_tokens=10, cached=30),
            usage_chunk(),
            {"choices": [], "usage": None},
        ]])
        events = await self.collect(provider)
        end = events[-1]
        self.assertIsInstance(end, ProviderResponseEndEvent)
        self.assertEqual(end.message.content, "Hello")
        self.assertEqual(end.finish_reason, "stop")
        self.assertEqual(end.message.usage, Usage(
            input_tokens=100, output_tokens=20, total_tokens=120, cached_input_tokens=40,
        ))
        self.assertEqual([event.delta for event in events if isinstance(event, ProviderTextDeltaEvent)],
                         ["Hello"])
        self.assertEqual(seen[0]["stream_options"], {"include_usage": True})

    async def test_missing_usage_stays_none_and_request_option_can_be_disabled(self):
        for include_usage in (True, False):
            with self.subTest(include_usage=include_usage):
                provider, seen = await self.make_provider(
                    [[text_chunk("answer")]], include_usage=include_usage,
                )
                events = await self.collect(provider)
                self.assertIsNone(events[-1].message.usage)
                self.assertEqual("stream_options" in seen[0], include_usage)

    async def test_tool_call_message_keeps_usage(self):
        provider, _ = await self.make_provider([[
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call-1", "type": "function",
                "function": {"name": "read", "arguments": '{"path": "README.md"}'},
            }]}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            usage_chunk(cached=None),
        ]])
        events = await self.collect(provider)
        message = events[-1].message
        self.assertEqual(message.tool_calls[0].arguments, {"path": "README.md"})
        self.assertEqual(message.usage.input_tokens, 100)
        self.assertIsNone(message.usage.cached_input_tokens)

    async def test_empty_tool_identity_deltas_preserve_name_and_id(self):
        provider, _ = await self.make_provider([[
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call-1", "type": "function",
                "function": {"name": "read", "arguments": ""},
            }]}}]},
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "", "function": {"name": "", "arguments": '{"path":'},
            }]}}]},
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": None, "function": {"name": "", "arguments": '"README.md"}'},
            }]}, "finish_reason": "tool_calls"}]},
        ]])
        events = await self.collect(provider)
        tool_call = events[-1].message.tool_calls[0]
        self.assertEqual(tool_call.id, "call-1")
        self.assertEqual(tool_call.name, "read")
        self.assertEqual(tool_call.arguments, {"path": "README.md"})

    def test_usage_parser_rejects_invalid_counts_and_preserves_zero(self):
        for data in (None, [], {}, {"prompt_tokens": 3},
                     {"prompt_tokens": -1, "completion_tokens": 2, "total_tokens": 1},
                     {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3},
                     {"prompt_tokens": "3", "completion_tokens": 2, "total_tokens": 5}):
            with self.subTest(data=data):
                self.assertIsNone(_parse_chunk_usage({"usage": data}))
        parsed = _parse_chunk_usage(usage_chunk(input_tokens=0, output_tokens=0, cached=0))
        self.assertEqual(parsed, Usage(
            input_tokens=0, output_tokens=0, total_tokens=0, cached_input_tokens=0,
        ))
        data = usage_chunk()
        data["usage"]["prompt_tokens_details"] = {"cached_tokens": "unknown"}
        self.assertIsNone(_parse_chunk_usage(data).cached_input_tokens)

    async def test_real_adapter_usage_persists_reloads_and_is_not_sent_in_history(self):
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(temp_dir.cleanup)
        cwd = Path(temp_dir.name)
        provider, seen = await self.make_provider([
            [text_chunk("first answer"), usage_chunk()],
            [text_chunk("continued answer"), usage_chunk(input_tokens=150, cached=70)],
        ])
        config = CodingSessionConfig(
            provider=provider, model="fake", system="system", tools=[],
            storage=JsonlSessionStorage(cwd / "session.jsonl"), cwd=cwd,
        )
        session = CodingSession.load(config)
        async for _ in session.prompt("question"):
            pass
        first_usage = session.messages[-1].usage
        self.assertEqual(first_usage.total_tokens, 120)
        self.assertEqual(session.context_usage.provider_tokens, 120)
        self.assertEqual(session.context_usage.estimated_tokens, 0)
        resumed = CodingSession.load(config)
        self.assertEqual(resumed.context_usage, session.context_usage)
        self.assertEqual(resumed.messages[-1].usage_context, session.messages[-1].usage_context)
        self.assertFalse(CodingSession.load(replace(config, model="other-model")).context_usage.uses_provider_usage)
        self.assertEqual(resumed.messages[-1].usage, first_usage)
        async for _ in resumed.continue_():
            pass
        self.assertEqual(seen[1]["messages"][-1],
                         {"role": "assistant", "content": "first answer"})
        restored = CodingSession.load(config)
        self.assertEqual(restored.context_usage.provider_tokens, 170)
        self.assertEqual(restored.context_usage.estimated_tokens, 0)
        usages = [message.usage for message in restored.messages
                  if isinstance(message, AssistantMessage)]
        self.assertEqual([usage.input_tokens for usage in usages], [100, 150])
        self.assertEqual([usage.cached_input_tokens for usage in usages], [40, 70])

    async def test_signature_uses_request_snapshot_even_if_history_changes_during_stream(self):
        provider, _ = await self.make_provider([[text_chunk("answer"), usage_chunk()]])
        messages = [UserMessage(content="original")]
        async for event in provider.stream_response(
            model="fake", system="system", messages=messages, tools=[],
        ):
            if isinstance(event, ProviderTextDeltaEvent):
                messages[0].content = "edited during stream"
            if isinstance(event, ProviderResponseEndEvent):
                messages.append(event.message)
        self.assertFalse(estimate_context_usage(
            model="fake", system="system", messages=messages, tools=[],
        ).uses_provider_usage)
        messages[0].content = "original"
        self.assertEqual(estimate_context_usage(
            model="fake", system="system", messages=messages, tools=[],
        ).provider_tokens, 120)

    async def test_tool_result_and_unsigned_final_reply_use_previous_matching_usage(self):
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(temp_dir.cleanup)
        cwd = Path(temp_dir.name)
        (cwd / "README.md").write_text("file contents", encoding="utf-8")
        provider, _ = await self.make_provider([
            [
                {"choices": [{"delta": {"tool_calls": [{
                    "index": 0, "id": "read-1",
                    "function": {"name": "read", "arguments": '{"path": "README.md"}'},
                }]}, "finish_reason": "tool_calls"}]},
                usage_chunk(),
            ],
            [text_chunk("Finished")],
        ])
        config = CodingSessionConfig(
            provider=provider, model="fake", system="system",
            tools=[create_read_tool(cwd)],
            storage=JsonlSessionStorage(cwd / "session.jsonl"), cwd=cwd,
        )
        session = CodingSession.load(config)
        saw_tool_result = False
        async for _ in session.prompt("Read file"):
            if len(session.messages) >= 3:
                saw_tool_result = True
                usage = session.context_usage
                self.assertEqual(usage.provider_tokens, 120)
                self.assertEqual(usage.estimated_tokens, sum(
                    estimate_message_tokens(message) for message in session.messages[2:]
                ))
        self.assertTrue(saw_tool_result)
        self.assertEqual(session.context_usage.provider_tokens, 120)
        self.assertEqual(session.context_usage.message_count, 4)
        self.assertEqual(CodingSession.load(config).context_usage, session.context_usage)
