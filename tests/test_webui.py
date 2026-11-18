"""Web UI 消费事件、统计快照和后台保存行为。"""

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from agent.context import request_context_fingerprint, response_context_fingerprint
from agent.messages import AssistantMessage, ToolCall, Usage
from agent.session import JsonlSessionStorage, MessageEntry
from agent.tools import AgentTool, AgentToolResult
from ai.events import (
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
)
from coding.session import CodingSession, CodingSessionConfig
from coding.webui import create_app


class ScriptedProvider:
    """可阻塞的确定性 provider，不访问真实模型。"""

    def __init__(self, responses, *, blocked=False):
        self.responses = list(responses)
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()

    def stream_response(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        request_fingerprint = request_context_fingerprint(
            model=kwargs["model"], system=kwargs["system"],
            messages=kwargs["messages"], tools=kwargs["tools"],
        )

        async def iterator():
            yield ProviderResponseStartEvent(model=kwargs["model"])
            self.started.set()
            await self.release.wait()
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            if response.usage is not None:
                response.usage_context = response_context_fingerprint(request_fingerprint, response)
            if response.content:
                midpoint = max(1, len(response.content) // 2)
                yield ProviderTextDeltaEvent(delta=response.content[:midpoint])
                if response.content[midpoint:]:
                    yield ProviderTextDeltaEvent(delta=response.content[midpoint:])
            yield ProviderResponseEndEvent(message=response)

        return iterator()


def sse_frames(response):
    """解析接口的 data 帧，保留事件与快照顺序。"""
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ")]


class WebUiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch("coding.session.extract_memory_candidates", return_value=[]))
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)

    def make_session(self, provider, *, tools=()):
        config = CodingSessionConfig(
            provider=provider, model="fake", system="test system", tools=list(tools),
            cwd=self.cwd, storage=JsonlSessionStorage(self.cwd / "session.jsonl"),
        )
        return CodingSession.load(config), config

    async def make_client(self, app):
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                   base_url="http://testserver")
        self.addAsyncCleanup(client.aclose)
        return client

    def assert_snapshot(self, snapshot, session):
        self.assertEqual(snapshot["messages"], [
            message.model_dump(mode="json", by_alias=False) for message in session.messages
        ])
        expected = asdict(session.context_usage)
        expected.update(estimated_tokens=session.context_usage.estimated_tokens,
                        uses_provider_usage=session.context_usage.uses_provider_usage)
        self.assertEqual(snapshot["context_usage"], expected)
        self.assertEqual(snapshot["session_info"],
                         session.state.session_info.model_dump(mode="json"))

    async def test_page_and_snapshot_are_read_only_and_preserve_missing_vs_zero_usage(self):
        provider = ScriptedProvider([])
        session, config = self.make_session(provider)
        config.storage.append(MessageEntry(message=AssistantMessage(content="No usage")))
        config.storage.append(MessageEntry(message=AssistantMessage(
            content="Zero usage", usage=Usage(input_tokens=0, output_tokens=0,
                                              total_tokens=0, cached_input_tokens=0),
        )))
        session = CodingSession.load(config)
        saved = config.storage.path.read_bytes()
        client = await self.make_client(create_app(session))
        page = await client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertIn("/api/run", page.text)
        self.assertIn("/api/session", page.text)
        snapshot = (await client.get("/api/session")).json()
        self.assert_snapshot(snapshot, session)
        self.assertFalse(snapshot["running"])
        self.assertIsNone(snapshot["error"])
        self.assertIsNone(snapshot["messages"][0]["usage"])
        self.assertEqual(snapshot["messages"][1]["usage"]["cached_input_tokens"], 0)
        self.assertEqual(config.storage.path.read_bytes(), saved)
        self.assertEqual(provider.calls, [])

    async def test_prompt_tools_continue_and_reload_keep_events_usage_and_saved_messages(self):
        calls = []

        async def execute(arguments):
            calls.append(arguments)
            return AgentToolResult(tool_call_id="placeholder", name="echo",
                                   content="Tool result", ok=True)

        tool = AgentTool(name="echo", description="Return test text",
                         input_schema={"type": "object"}, executor=execute)
        provider = ScriptedProvider([
            AssistantMessage(content="", tool_calls=[
                ToolCall(id="echo-1", name="echo", arguments={"text": "中文"}),
            ], usage=Usage(input_tokens=100, output_tokens=20, total_tokens=120,
                           cached_input_tokens=40)),
            AssistantMessage(content="Finished", usage=Usage(
                input_tokens=130, output_tokens=10, total_tokens=140,
            )),
            AssistantMessage(content="Continued"),
        ])
        session, config = self.make_session(provider, tools=[tool])
        client = await self.make_client(create_app(session))
        response = await client.post("/api/run", json={"mode": "prompt", "content": "Run echo"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        frames = sse_frames(response)
        events = [frame["event"] for frame in frames if frame["type"] == "agent_event"]
        types = [event["type"] for event in events]
        self.assertLess(types.index("tool_execution_start"), types.index("tool_execution_end"))
        start = next(event for event in events if event["type"] == "tool_execution_start")
        end = next(event for event in events if event["type"] == "tool_execution_end")
        self.assertEqual(start["tool_call"]["id"], end["result"]["tool_call_id"])
        self.assertEqual(calls, [{"text": "中文"}])
        self.assertEqual("".join(event["delta"] for event in events
                                 if event["type"] == "message_delta"), "Finished")
        self.assertEqual(frames[-1]["type"], "run_finished")
        self.assert_snapshot(frames[-1]["session"], session)
        self.assertFalse(frames[-1]["session"]["running"])
        self.assertEqual(session.context_usage.provider_tokens, 140)
        self.assertTrue(session.context_usage.uses_provider_usage)
        saved = CodingSession.load(config)
        self.assertEqual(saved.messages, session.messages)
        self.assertEqual(saved.messages[1].usage.cached_input_tokens, 40)
        self.assertEqual(saved.context_usage, session.context_usage)
        self.assertTrue(any(frame["type"] == "snapshot" and
                            frame["session"]["context_usage"]["message_count"] == 3
                            for frame in frames))
        self.assertLess(next(i for i, frame in enumerate(frames)
                             if frame["type"] == "agent_event" and
                             frame["event"]["type"] == "agent_end"), len(frames) - 1)

        response = await client.post("/api/run", json={"mode": "continue"})
        self.assertEqual(sse_frames(response)[-1]["type"], "run_finished")
        self.assertEqual([message.role for message in session.messages],
                         ["user", "assistant", "tool", "assistant", "assistant"])
        self.assertEqual(len(provider.calls[2]["messages"]), 4)
        self.assertEqual(CodingSession.load(config).messages, session.messages)
        self.assert_snapshot((await client.get("/api/session")).json(), session)

    async def test_invalid_runs_do_not_start_model_or_modify_storage(self):
        provider = ScriptedProvider([])
        session, config = self.make_session(provider)
        saved = config.storage.path.read_bytes()
        client = await self.make_client(create_app(session))
        for payload in ({"mode": "prompt"}, {"mode": "prompt", "content": " \n\t"},
                        {"mode": "other", "content": "hello"},
                        {"mode": "prompt", "content": "hello", "extra": True}):
            with self.subTest(payload=payload):
                response = await client.post("/api/run", json=payload)
                self.assertEqual(response.status_code, 422)
        self.assertEqual(provider.calls, [])
        self.assertEqual(session.messages, ())
        self.assertEqual(config.storage.path.read_bytes(), saved)

    async def test_history_catalog_is_read_only_and_reports_unreadable_records(self):
        provider = ScriptedProvider([])
        session, config = self.make_session(provider)
        config.storage.append(MessageEntry(message=AssistantMessage(content="Saved answer")))
        session = CodingSession.load(config)
        saved = config.storage.path.read_bytes()
        broken = self.cwd / "broken.jsonl"
        broken.write_text("not json\n", encoding="utf-8")
        client = await self.make_client(create_app(session))
        response = await client.get("/api/sessions")
        self.assertEqual(response.status_code, 200)
        catalog = response.json()
        self.assertEqual(catalog["active_session_id"], session.state.session_info.session_id)
        self.assertEqual(len(catalog["sessions"]), 1)
        self.assertEqual(catalog["sessions"][0]["message_count"], 1)
        self.assertIn("broken.jsonl", catalog["errors"][0])
        self.assertEqual(provider.calls, [])
        self.assertEqual(config.storage.path.read_bytes(), saved)
        self.assertEqual(broken.read_text(encoding="utf-8"), "not json\n")

    async def test_new_session_is_empty_and_reuses_configuration_without_model_request(self):
        async def execute(arguments):
            return AgentToolResult(tool_call_id="", name="echo", content="unused", ok=True)

        tool = AgentTool(name="echo", description="Echo", input_schema={"type": "object"},
                         executor=execute)
        provider = ScriptedProvider([])
        session, config = self.make_session(provider, tools=[tool])
        original_id = session.state.session_info.session_id
        saved = config.storage.path.read_bytes()
        app = create_app(session)
        client = await self.make_client(app)
        response = await client.post("/api/sessions")
        self.assertEqual(response.status_code, 201)
        snapshot = response.json()
        new_id = snapshot["session_info"]["session_id"]
        self.assertNotEqual(new_id, original_id)
        self.assertEqual(snapshot["messages"], [])
        self.assertEqual(snapshot["context_usage"]["message_count"], 0)
        self.assertEqual(snapshot["context_usage"]["provider_tokens"], 0)
        self.assertFalse(snapshot["running"])
        current = app.state.runtime.session
        self.assertIs(current.config.provider, provider)
        self.assertIs(current.config.tools, config.tools)
        self.assertEqual(current.config.model, config.model)
        self.assertEqual(current.config.system, config.system)
        self.assertEqual(current.config.cwd, config.cwd)
        self.assertEqual(current.storage.path, self.cwd / f"{new_id}.jsonl")
        self.assertEqual(len(current.storage.read_all()), 1)
        self.assertEqual(config.storage.path.read_bytes(), saved)
        self.assertEqual(provider.calls, [])

    async def test_switch_and_restart_restore_separate_histories_and_context_usage(self):
        provider = ScriptedProvider([
            AssistantMessage(content="First answer", usage=Usage(
                input_tokens=100, output_tokens=20, total_tokens=120,
            )),
            AssistantMessage(content="Second answer", usage=Usage(
                input_tokens=200, output_tokens=30, total_tokens=230, cached_input_tokens=0,
            )),
            AssistantMessage(content="Continued first"),
        ])
        session, config = self.make_session(provider)
        first_id = session.state.session_info.session_id
        app = create_app(session)
        client = await self.make_client(app)
        await client.post("/api/run", json={"content": "First question", "session_id": first_id})
        first_snapshot = (await client.get("/api/session")).json()
        first_saved = config.storage.path.read_bytes()

        second_snapshot = (await client.post("/api/sessions")).json()
        second_id = second_snapshot["session_info"]["session_id"]
        await client.post("/api/run", json={"content": "Second question", "session_id": second_id})
        second_snapshot = (await client.get("/api/session")).json()
        second_path = app.state.runtime.session.storage.path
        second_saved = second_path.read_bytes()
        self.assertEqual(len(provider.calls[1]["messages"]), 1)
        self.assertEqual(provider.calls[1]["messages"][0].content, "Second question")
        self.assertEqual(config.storage.path.read_bytes(), first_saved)

        restored = (await client.post(f"/api/sessions/{first_id}/select")).json()
        self.assertEqual(restored, first_snapshot)
        self.assertEqual(len(provider.calls), 2)
        await client.post("/api/run", json={"mode": "continue", "session_id": first_id})
        self.assertEqual([message.content for message in provider.calls[2]["messages"]],
                         ["First question", "First answer"])
        self.assertEqual(second_path.read_bytes(), second_saved)

        restarted = create_app(CodingSession.load(config))
        fresh_client = await self.make_client(restarted)
        catalog = (await fresh_client.get("/api/sessions")).json()
        self.assertEqual({entry["session_id"] for entry in catalog["sessions"]},
                         {first_id, second_id})
        self.assertEqual({entry["title"] for entry in catalog["sessions"]},
                         {"First question", "Second question"})
        restored_second = (await fresh_client.post(f"/api/sessions/{second_id}/select")).json()
        self.assertEqual(restored_second, second_snapshot)
        self.assertTrue(restored_second["context_usage"]["uses_provider_usage"])
        self.assertEqual(restored_second["messages"][-1]["usage"]["cached_input_tokens"], 0)
        self.assertEqual(len(provider.calls), 3)

    async def test_stale_page_and_unknown_session_cannot_modify_active_history(self):
        provider = ScriptedProvider([])
        session, config = self.make_session(provider)
        original_id = session.state.session_info.session_id
        app = create_app(session)
        client = await self.make_client(app)
        snapshot = (await client.post("/api/sessions")).json()
        active_id = snapshot["session_info"]["session_id"]
        files = {path.name: path.read_bytes() for path in self.cwd.glob("*.jsonl")}
        rejected = await client.post("/api/run", json={
            "content": "Belongs to previous page", "session_id": original_id,
        })
        self.assertEqual(rejected.status_code, 409)
        missing = await client.post("/api/sessions/not-a-session/select")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(app.state.runtime.session_id, active_id)
        self.assertEqual((await client.get("/api/session")).json()["messages"], [])
        self.assertEqual({path.name: path.read_bytes() for path in self.cwd.glob("*.jsonl")}, files)
        self.assertEqual(provider.calls, [])

    async def test_new_session_loading_failure_leaves_current_session_and_files_intact(self):
        provider = ScriptedProvider([])
        session, config = self.make_session(provider)
        app = create_app(session)
        client = await self.make_client(app)
        saved = config.storage.path.read_bytes()
        with patch("coding.webui.CodingSession.load", side_effect=RuntimeError("Load failed")):
            with self.assertRaisesRegex(RuntimeError, "Load failed"):
                await client.post("/api/sessions")
        self.assertIs(app.state.runtime.session, session)
        self.assertEqual(list(self.cwd.glob("*.jsonl")), [config.storage.path])
        self.assertEqual(config.storage.path.read_bytes(), saved)
        self.assertEqual(provider.calls, [])

    async def test_active_run_rejects_second_request_and_snapshot_reads_live_messages(self):
        provider = ScriptedProvider([AssistantMessage(content="Finished")], blocked=True)
        session, _ = self.make_session(provider)
        app = create_app(session)
        client = await self.make_client(app)
        pending = asyncio.create_task(client.post("/api/run", json={
            "mode": "prompt", "content": "First question",
        }))
        try:
            await asyncio.wait_for(provider.started.wait(), timeout=2)
            self.assertTrue(app.state.runtime.running)
            snapshot = (await client.get("/api/session")).json()
            self.assertTrue(snapshot["running"])
            self.assertEqual(snapshot["messages"][0]["content"], "First question")
            self.assertEqual(snapshot["context_usage"]["message_count"], 1)
            self.assertEqual(session.state.messages, [])
            rejected = await client.post("/api/run", json={"mode": "continue"})
            self.assertEqual(rejected.status_code, 409)
            self.assertEqual(len(provider.calls), 1)
            rejected_new = await client.post("/api/sessions")
            self.assertEqual(rejected_new.status_code, 409)
            active_id = session.state.session_info.session_id
            rejected_switch = await client.post(f"/api/sessions/{active_id}/select")
            self.assertEqual(rejected_switch.status_code, 409)
            self.assertIs(app.state.runtime.session, session)
            self.assertEqual(len(list(self.cwd.glob("*.jsonl"))), 1)
        finally:
            provider.release.set()
            response = await asyncio.wait_for(pending, timeout=2)
        self.assertEqual(sse_frames(response)[-1]["type"], "run_finished")
        self.assertFalse(app.state.runtime.running)

    async def test_provider_exception_becomes_run_error_without_false_finished(self):
        provider = ScriptedProvider([RuntimeError("Provider failed")])
        session, config = self.make_session(provider)
        app = create_app(session)
        client = await self.make_client(app)
        response = await client.post("/api/run", json={"mode": "prompt", "content": "Question"})
        frames = sse_frames(response)
        self.assertEqual(frames[-1]["type"], "run_error")
        self.assertIn("Provider failed", frames[-1]["message"])
        self.assertNotIn("run_finished", [frame["type"] for frame in frames])
        self.assertFalse(app.state.runtime.running)
        self.assertIn("Provider failed", (await client.get("/api/session")).json()["error"])
        self.assertEqual(len(config.storage.read_all()), 1)

    async def test_save_failure_after_agent_end_is_not_reported_as_finished(self):
        provider = ScriptedProvider([AssistantMessage(content="Answer")])
        session, config = self.make_session(provider)
        client = await self.make_client(create_app(session))
        with patch.object(config.storage, "append", side_effect=OSError("Disk full")):
            response = await client.post("/api/run", json={"mode": "prompt", "content": "Question"})
        frames = sse_frames(response)
        self.assertTrue(any(frame["type"] == "agent_event" and
                            frame["event"]["type"] == "agent_end" for frame in frames))
        self.assertEqual(frames[-1]["type"], "run_error")
        self.assertIn("Disk full", frames[-1]["message"])
        self.assertNotIn("run_finished", [frame["type"] for frame in frames])
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(len(config.storage.read_all()), 1)

    async def test_client_disconnect_does_not_cancel_background_run_or_save(self):
        provider = ScriptedProvider([AssistantMessage(content="Saved answer")], blocked=True)
        session, config = self.make_session(provider)
        app = create_app(session)
        body = json.dumps({"mode": "prompt", "content": "Question"}).encode()
        disconnected = asyncio.Event()
        body_read = False

        async def receive():
            nonlocal body_read
            if not body_read:
                body_read = True
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                disconnected.set()

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/api/run", "raw_path": b"/api/run", "query_string": b"",
            "root_path": "", "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234), "server": ("testserver", 80),
        }
        request = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(disconnected.wait(), timeout=2)
            await asyncio.wait_for(request, timeout=2)
            await asyncio.wait_for(provider.started.wait(), timeout=2)
            task = app.state.runtime.task
            self.assertIsNotNone(task)
            self.assertFalse(task.done())
            self.assertTrue(app.state.runtime.running)
        finally:
            provider.release.set()
            if app.state.runtime.task is not None:
                await asyncio.wait_for(asyncio.shield(app.state.runtime.task), timeout=2)
            if not request.done():
                await asyncio.wait_for(request, timeout=2)
        self.assertFalse(app.state.runtime.running)
        self.assertEqual([message.content for message in CodingSession.load(config).messages],
                         ["Question", "Saved answer"])

    async def test_shutdown_waits_for_running_session_before_closing_provider(self):
        provider = ScriptedProvider([AssistantMessage(content="Saved")], blocked=True)
        session, config = self.make_session(provider)
        closed = AsyncMock()
        app = create_app(session, close_provider=closed)
        client = await self.make_client(app)
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        request = asyncio.create_task(client.post("/api/run", json={
            "mode": "prompt", "content": "Question",
        }))
        await asyncio.wait_for(provider.started.wait(), timeout=2)
        shutdown_started = asyncio.Event()

        async def shutdown():
            shutdown_started.set()
            await lifespan.__aexit__(None, None, None)

        exiting = asyncio.create_task(shutdown())
        try:
            await asyncio.wait_for(shutdown_started.wait(), timeout=2)
            # The closer must not run while the model still uses the provider.
            self.assertFalse(exiting.done())
            self.assertFalse(app.state.runtime.task.done())
            closed.assert_not_awaited()
        finally:
            provider.release.set()
            await asyncio.wait_for(request, timeout=2)
            await asyncio.wait_for(exiting, timeout=2)
        closed.assert_awaited_once_with()
        self.assertEqual(len(CodingSession.load(config).messages), 2)
