"""使用真实 Provider 和模拟 HTTP 响应验证 TUI 接线及资源释放。"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from textual.widgets import Input, Static

from ai.env import OpenAICompatibleConfig
from ai.openai_compatible import OpenAICompatibleProvider
from agent.events import MessageDeltaEvent
from agent.messages import UserMessage
from agent.session.entries import MessageEntry, SessionInfoEntry
from agent.session.storage import JsonlSessionStorage
from coding.tui_app import TUIApp, main, run_tui_app
from coding.session import CodingSession, CodingSessionConfig


def response_stream() -> httpx.Response:
    chunks = [
        {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
        for text in ("Hello ", "provider.")
    ]
    chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return httpx.Response(
        200, content=(body + "data: [DONE]\n\n").encode("utf-8"),
        headers={"content-type": "text/event-stream"},
    )


class TuiProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.enterContext(patch("coding.session.extract_memory_candidates", return_value=[]))
        root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        root.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)
        self.config = OpenAICompatibleConfig(
            api_key="test-key", base_url="https://example.test/v1",
        )

    async def test_default_startups_do_not_create_sessions(self) -> None:
        old_storage = JsonlSessionStorage(self.cwd / ".codex_tmp/tui/session.jsonl")
        old_storage.append(SessionInfoEntry(cwd=str(self.cwd)))
        old_storage.append(MessageEntry(message=UserMessage(content="old question")))
        old_bytes = old_storage.path.read_bytes()
        test = self

        async def inspect_startup(app):
            test.assertEqual(app.state.items, [])
            test.assertIsNone(app.session)
            async with app.run_test() as pilot:
                await pilot.press("enter")
                app.query_one("#prompt", Input).value = "/resume"
                await pilot.press("enter")
                test.assertIsNone(app.session)
                await pilot.press("ctrl+c")

        with patch.object(TUIApp, "run_async", inspect_startup):
            for _ in range(2):
                await run_tui_app(model="test-model", cwd=self.cwd, provider_config=self.config)
        self.assertEqual(old_storage.path.read_bytes(), old_bytes)
        self.assertEqual(len(list(old_storage.path.parent.glob("*.jsonl"))), 1)

    async def test_unused_explicit_session_path_is_not_created(self) -> None:
        path = self.cwd / "new-directory/new-session.jsonl"
        test = self

        async def inspect_startup(app):
            test.assertIsNone(app.session)
            test.assertEqual(app.state.items, [])

        with patch.object(TUIApp, "run_async", inspect_startup):
            await run_tui_app(
                model="test-model", cwd=self.cwd, provider_config=self.config, session_path=path,
            )
        self.assertFalse(path.parent.exists())

    async def test_explicit_session_path_still_restores_existing_history(self) -> None:
        storage = JsonlSessionStorage(self.cwd / "existing.jsonl")
        storage.append(SessionInfoEntry(cwd=str(self.cwd)))
        storage.append(MessageEntry(message=UserMessage(content="old question")))
        old_bytes = storage.path.read_bytes()
        test = self

        async def inspect_startup(app):
            test.assertEqual(app.session.storage.path, storage.path)
            test.assertEqual([(item.role, item.text) for item in app.state.items], [("user", "old question")])

        with patch.object(TUIApp, "run_async", inspect_startup):
            await run_tui_app(
                model="test-model", cwd=self.cwd, provider_config=self.config, session_path=storage.path,
            )
        self.assertEqual(storage.path.read_bytes(), old_bytes)

    async def test_startup_streams_real_provider_response_and_closes_it(self) -> None:
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            return response_stream()

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        self.addAsyncCleanup(client.aclose)
        provider = OpenAICompatibleProvider(self.config, client=client)
        observed_buffers = []
        test = self

        async def headless_run(app):
            original_apply = app.adapter.apply

            def observe(event):
                original_apply(event)
                if isinstance(event, MessageDeltaEvent):
                    observed_buffers.append(app.state.assistant_buffer)

            with patch.object(app.adapter, "apply", side_effect=observe):
                async with app.run_test() as pilot:
                    app.query_one("#prompt", Input).value = "hello"
                    await pilot.press("enter")
                    await asyncio.wait_for(app._prompt_worker.wait(), timeout=3)
                    test.assertEqual(
                        [(item.role, item.text) for item in app.state.items],
                        [("user", "hello"), ("assistant", "Hello provider.")],
                    )
                    rendered = "\n".join(
                        str(widget.content) for widget in app.query_one("#transcript").children
                    )
                    test.assertIn("assistant: Hello provider.", rendered)
                    test.assertEqual(str(app.query_one("#status", Static).content), "Ready")
                    test.assertFalse(app._busy)
                    test.assertEqual(len(app.session.storage.read_all()), 3)

        with patch("coding.tui_app.OpenAICompatibleProvider", return_value=provider), \
                patch.object(provider, "aclose", new_callable=AsyncMock, wraps=provider.aclose) as close, \
                patch.object(TUIApp, "run_async", headless_run):
            await run_tui_app(model="test-model", cwd=self.cwd, provider_config=self.config)
            close.assert_awaited_once()
        self.assertEqual(observed_buffers, ["Hello ", "Hello provider."])
        self.assertEqual(requests[0]["model"], "test-model")
        self.assertEqual(requests[0]["messages"][-1]["content"], "hello")
        self.assertTrue(requests[0]["stream"])

    async def test_provider_error_is_visible_and_next_prompt_can_succeed(self) -> None:
        responses = [httpx.Response(401, text="unauthorized"), response_stream()]
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: responses.pop(0)))
        self.addAsyncCleanup(client.aclose)
        provider = OpenAICompatibleProvider(self.config, client=client)
        session = CodingSession.load(CodingSessionConfig(
            provider=provider, model="test-model", cwd=self.cwd,
            storage=JsonlSessionStorage(self.cwd / "session.jsonl"),
            auto_compact_enabled=False,
        ))
        app = TUIApp(session)
        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "first"
            await pilot.press("enter")
            await asyncio.wait_for(app._prompt_worker.wait(), timeout=3)
            self.assertEqual(
                str(app.query_one("#status", Static).content),
                "Error: Provider request failed with status code 401",
            )
            self.assertFalse(app._busy)
            prompt.value = "retry"
            await pilot.press("enter")
            await asyncio.wait_for(app._prompt_worker.wait(), timeout=3)
            self.assertEqual(app.state.items[-1].text, "Hello provider.")
            self.assertEqual(str(app.query_one("#status", Static).content), "Ready")

    async def test_provider_is_closed_when_session_loading_fails(self) -> None:
        close = AsyncMock()
        storage = JsonlSessionStorage(self.cwd / "existing.jsonl")
        storage.append(SessionInfoEntry(cwd=str(self.cwd)))
        with patch("coding.tui_app.OpenAICompatibleProvider") as create_provider, \
                patch("coding.tui_app.CodingSession.load", side_effect=ValueError("invalid session")):
            create_provider.return_value.aclose = close
            with self.assertRaisesRegex(ValueError, "invalid session"):
                await run_tui_app(
                    model="test-model", cwd=self.cwd, provider_config=self.config, session_path=storage.path,
                )
        close.assert_awaited_once()


class TuiCommandLineTests(unittest.TestCase):
    def test_model_environment_and_usage_option_reach_startup(self) -> None:
        argv = ["coding.tui_app", "--model", "test-model", "--api-key-var", "TEST_API_KEY", "--no-usage"]
        with patch("sys.argv", argv), patch("coding.tui_app.load_dotenv"), \
                patch.dict("os.environ", {"TEST_API_KEY": "test-key"}, clear=True), \
                patch("coding.tui_app.run_tui_app", new_callable=AsyncMock) as run:
            main()
        run.assert_awaited_once()
        kwargs = run.await_args.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["provider_config"].api_key, "test-key")
        self.assertFalse(kwargs["provider_config"].include_usage)
