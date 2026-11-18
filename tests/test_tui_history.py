"""历史显示使用原始记录，并能在重启后继续接收新消息。"""

import asyncio
import tempfile
import unittest
from pathlib import Path

from textual.widgets import Input

from ai.fake import FakeProvider
from agent.messages import AssistantMessage, ToolResultMessage, UserMessage
from agent.session.entries import CompactionEntry, MessageEntry, SessionInfoEntry
from agent.session.memory import SessionState
from agent.session.storage import JsonlSessionStorage
from coding.tui_app import TUIApp
from coding.session import CodingSession, CodingSessionConfig
from coding.tui_state import TuiState


class TuiHistoryStateTests(unittest.TestCase):
    def test_original_text_survives_compaction_and_repeated_loading(self) -> None:
        recent_user = MessageEntry(message=UserMessage(content="recent question"))
        entries = [
            SessionInfoEntry(cwd="."),
            MessageEntry(message=UserMessage(content="old question")),
            MessageEntry(message=AssistantMessage(content="old answer")),
            MessageEntry(message=AssistantMessage(content="")),
            MessageEntry(message=ToolResultMessage(
                tool_call_id="call-1", name="read", content="tool output",
            )),
            recent_user,
            MessageEntry(message=AssistantMessage(content="recent answer")),
            CompactionEntry(summary="summary", first_kept_entry_id=recent_user.id),
        ]
        context = SessionState.from_entries(entries)
        self.assertNotIn("old question", [message.content for message in context.messages])

        state = TuiState()
        state.add_item("user", "stale display")
        state.assistant_buffer = "unfinished reply"
        expected = [
            ("user", "old question"), ("assistant", "old answer"),
            ("tool", "OK: read\ntool output"),
            ("user", "recent question"), ("assistant", "recent answer"),
        ]
        for _ in range(2):
            state.load_entries(iter(entries))
            self.assertEqual([(item.role, item.text) for item in state.items], expected)
            self.assertEqual(state.assistant_buffer, "")

    def test_loading_empty_history_clears_previous_display(self) -> None:
        state = TuiState()
        state.add_item("user", "old question")
        state.assistant_buffer = "unfinished reply"
        state.load_entries([])
        self.assertEqual(state.items, [])
        self.assertEqual(state.assistant_buffer, "")


class TuiHistoryAppTests(unittest.IsolatedAsyncioTestCase):
    async def test_restart_restores_display_and_appends_without_duplicates(self) -> None:
        temporary_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            config = CodingSessionConfig(
                provider=FakeProvider(), model="fake", cwd=Path(temporary),
                storage=JsonlSessionStorage(Path(temporary) / "session.jsonl"),
                auto_compact_enabled=False,
            )
            expected: list[tuple[str, str]] = []
            for text in ("first", "second"):
                app = TUIApp(CodingSession.load(config))
                async with app.run_test() as pilot:
                    self.assertEqual([(item.role, item.text) for item in app.state.items], expected)
                    rendered = "\n".join(
                        str(widget.content) for widget in app.query_one("#transcript").children
                    )
                    for role, content in expected:
                        self.assertIn(f"{role}: {content}", rendered)
                    app.query_one("#prompt", Input).value = text
                    await pilot.press("enter")
                    self.assertIsNotNone(app._prompt_worker)
                    await asyncio.wait_for(app._prompt_worker.wait(), timeout=3)
                    expected.extend([("user", text), ("assistant", "Hello from FakeProvider.")])
                    self.assertEqual([(item.role, item.text) for item in app.state.items], expected)
                    self.assertEqual(app.state.assistant_buffer, "")
                    self.assertFalse(app._busy)

            restarted = TUIApp(CodingSession.load(config))
            self.assertEqual([(item.role, item.text) for item in restarted.state.items], expected)
            self.assertEqual(len(config.storage.read_all()), 5)
