"""增量显示保留历史控件，流式回答和历史在同一滚动区域。"""

from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch

from ai.fake import FakeProvider
from agent.session.storage import JsonlSessionStorage
from coding.session import CodingSessionConfig
from coding.tui_app import TUIApp
from coding.tui_state import TuiState
from coding.tui_widgets import TranscriptView


class TranscriptViewTests(unittest.IsolatedAsyncioTestCase):
    def make_app(self) -> TUIApp:
        return TUIApp(config=CodingSessionConfig(
            provider=FakeProvider(), model="test", cwd=Path.cwd(), tools=[],
            storage=JsonlSessionStorage(Path.cwd() / ".codex_tmp/unused-widget-test.jsonl"),
        ))

    async def test_streaming_keeps_history_and_finalizes_the_same_widget(self) -> None:
        app = self.make_app()
        app.state.add_item("user", "inspect project")
        app.state.add_item("tool", "source code line\n" * 2000)
        async with app.run_test() as pilot:
            view = app.query_one("#transcript", TranscriptView)
            history = tuple(view.children)
            with ExitStack() as patches:
                for widget in history:
                    patches.enter_context(patch.object(
                        widget, "update", side_effect=AssertionError("Old message was updated"),
                    ))
                app.state.assistant_buffer = "Hello"
                await app.update_from_state(app.state)
                streaming = view.children[-1]
                for _ in range(20):
                    app.state.assistant_buffer += " world"
                    await app.update_from_state(app.state)
                self.assertEqual(tuple(view.children[:2]), history)
                self.assertIs(view.children[-1], streaming)
                self.assertEqual(len(view.children), 3)

                final_text = app.state.assistant_buffer
                app.state.add_item("assistant", final_text)
                app.state.assistant_buffer = ""
                await app.update_from_state(app.state)
                self.assertIs(view.children[-1], streaming)
                self.assertEqual(str(streaming.content), f"assistant: {final_text}")
                self.assertEqual(len(view.children), 3)
                await app.update_from_state(app.state)
                self.assertEqual(len(view.children), 3)

            await pilot.pause()
            self.assertGreater(view.max_scroll_y, 0)
            self.assertTrue(all(widget.parent is view for widget in view.children))

    async def test_equal_length_session_switch_replaces_history_and_notices(self) -> None:
        app = self.make_app()
        app.state.add_item("user", "old conversation")
        async with app.run_test():
            view = app.query_one("#transcript", TranscriptView)
            old = view.children[0]
            await view.show_notice("temporary command output")
            replacement = TuiState()
            replacement.add_item("user", "new conversation")
            app.state = replacement
            await app.update_from_state(replacement)
            self.assertEqual(len(view.children), 1)
            self.assertIsNot(view.children[0], old)
            self.assertEqual(str(view.children[0].content), "user: new conversation")

    async def test_streaming_does_not_pull_reader_away_from_history(self) -> None:
        app = self.make_app()
        app.state.add_item("tool", "line\n" * 100)
        async with app.run_test() as pilot:
            view = app.query_one("#transcript", TranscriptView)
            app.state.assistant_buffer = "initial answer"
            await app.update_from_state(app.state)
            await pilot.pause()
            view.scroll_home(animate=False, immediate=True)
            await pilot.pause()
            self.assertGreater(view.max_scroll_y, 0)
            self.assertEqual(view.scroll_y, 0)
            app.state.assistant_buffer += " continued"
            await app.update_from_state(app.state)
            await pilot.pause()
            self.assertEqual(view.scroll_y, 0)
