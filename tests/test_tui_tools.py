"""工具事件实时显示，并与持久化历史的恢复一致。"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Input

from ai.events import ProviderResponseEndEvent, ProviderResponseStartEvent, ProviderTextDeltaEvent
from agent.events import MessageEndEvent, ToolExecutionEndEvent, ToolExecutionStartEvent
from agent.messages import AssistantMessage, ToolCall, ToolResultMessage
from agent.session.storage import JsonlSessionStorage
from agent.tools import AgentTool, AgentToolResult
from coding.tui_app import TUIApp
from coding.session import CodingSession, CodingSessionConfig
from coding.tui_adapter import TuiEventAdapter
from coding.tui_state import TuiState


class ToolDisplayAdapterTests(unittest.TestCase):
    def test_assistant_tool_calls_are_only_displayed_when_execution_starts(self) -> None:
        state = TuiState()
        adapter = TuiEventAdapter(state)
        call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
        adapter.apply(MessageEndEvent(message=AssistantMessage(content="", tool_calls=[call])))
        self.assertEqual(state.items, [])
        adapter.apply(ToolExecutionStartEvent(tool_call=call))
        self.assertEqual([(item.role, item.text) for item in state.items], [
            ("tool", '→ read {"path": "README.md"}'),
        ])

    def test_failed_result_can_display_error_when_content_is_empty(self) -> None:
        state = TuiState()
        TuiEventAdapter(state).apply(ToolExecutionEndEvent(result=AgentToolResult(
            tool_call_id="call-1", name="read", content="", ok=False, error="Permission denied",
        )))
        self.assertEqual(state.items[0].text, "FAILED: read\nPermission denied")


class ToolDisplayAppTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch("coding.session.extract_memory_candidates", return_value=[]))

    async def test_running_tool_result_and_restored_history(self) -> None:
        root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        root.mkdir(exist_ok=True)
        for ok in (True, False):
            with self.subTest(ok=ok), tempfile.TemporaryDirectory(dir=root) as temporary:
                started, release = asyncio.Event(), asyncio.Event()
                call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
                result_text = "file contents" if ok else "Permission denied"

                class ToolProvider:
                    def __init__(self):
                        self.requests = []

                    async def stream_response(self, *, model, messages, **kwargs):
                        self.requests.append(list(messages))
                        message = (
                            AssistantMessage(content="Checking file.", tool_calls=[call])
                            if len(self.requests) == 1 else AssistantMessage(content="Done.")
                        )
                        yield ProviderResponseStartEvent(model=model)
                        yield ProviderTextDeltaEvent(delta=message.content)
                        yield ProviderResponseEndEvent(message=message)

                async def execute(arguments):
                    self.assertEqual(arguments, call.arguments)
                    started.set()
                    await release.wait()
                    return AgentToolResult(
                        tool_call_id="placeholder", name="placeholder", content=result_text, ok=ok,
                    )

                provider = ToolProvider()
                config = CodingSessionConfig(
                    provider=provider, model="test", cwd=Path(temporary), system="test",
                    storage=JsonlSessionStorage(Path(temporary) / "session.jsonl"),
                    tools=[AgentTool(name="read", description="test", input_schema={}, executor=execute)],
                    auto_compact_enabled=False,
                )
                app = TUIApp(CodingSession.load(config))
                async with app.run_test() as pilot:
                    app.query_one("#prompt", Input).value = "read file"
                    await pilot.press("enter")
                    await asyncio.wait_for(started.wait(), timeout=2)
                    await pilot.pause()
                    self.assertTrue(app._busy)
                    self.assertEqual(app.state.items[-1].text, '→ read {"path": "README.md"}')
                    self.assertEqual(len(app.state.items), 3)
                    release.set()
                    await asyncio.wait_for(app._prompt_worker.wait(), timeout=3)
                    expected = [
                        ("user", "read file"), ("assistant", "Checking file."),
                        ("tool", '→ read {"path": "README.md"}'),
                        ("tool", f"{'OK' if ok else 'FAILED'}: read\n{result_text}"),
                        ("assistant", "Done."),
                    ]
                    self.assertEqual([(item.role, item.text) for item in app.state.items], expected)
                    self.assertFalse(app._busy)
                    rendered = "\n".join(str(widget.content) for widget in app.query_one("#transcript").children)
                    self.assertIn(result_text, rendered)
                    self.assertEqual(len(provider.requests), 2)
                    self.assertIsInstance(provider.requests[1][-1], ToolResultMessage)
                    self.assertEqual(provider.requests[1][-1].tool_call_id, call.id)
                restored = TUIApp(CodingSession.load(config))
                self.assertEqual([(item.role, item.text) for item in restored.state.items], expected)
