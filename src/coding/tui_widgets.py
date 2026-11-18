"""将聊天条目保留为控件，只更新正在生成的回答。"""

from rich.console import RenderableType
from textual.containers import VerticalScroll
from textual.widgets import Static

from coding.tui_state import TuiState


class TranscriptView(VerticalScroll):
    DEFAULT_CSS = """
    TranscriptView > Static {
        height: auto;
        width: 1fr;
        margin-bottom: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state: TuiState | None = None
        self._rendered_count = 0
        self._streaming: Static | None = None
        self._streaming_text = ""
        self._notices: list[Static] = []

    async def update_from_state(self, state: TuiState) -> None:
        """追加新条目；回答结束时保留并完成已有的流式控件。"""
        follow = self.is_vertical_scroll_end
        changed = False
        if self._state is not state or len(state.items) < self._rendered_count:
            await self.remove_children()
            self._state = state
            self._rendered_count = 0
            self._streaming = None
            self._streaming_text = ""
            self._notices.clear()
            follow = True
            changed = True
        elif self._notices:
            await self.remove_children(self._notices)
            self._notices.clear()
            changed = True

        widgets: list[Static] = []
        for item in state.items[self._rendered_count:]:
            text = f"{item.role}: {item.text}"
            if self._streaming is not None:
                if item.role == "assistant":
                    if text != self._streaming_text:
                        self._streaming.update(text)
                    self._streaming = None
                    self._streaming_text = ""
                    changed = True
                    continue
                await self._streaming.remove()
                self._streaming = None
                self._streaming_text = ""
            widgets.append(Static(text, markup=False, classes="chat-message"))
        if widgets:
            await self.mount(*widgets)
            changed = True
        self._rendered_count = len(state.items)

        if state.assistant_buffer:
            text = f"assistant: {state.assistant_buffer}"
            if self._streaming is None:
                self._streaming = Static(text, markup=False, classes="chat-message")
                await self.mount(self._streaming)
                changed = True
            elif text != self._streaming_text:
                self._streaming.update(text)
                changed = True
            self._streaming_text = text
        elif self._streaming is not None:
            await self._streaming.remove()
            self._streaming = None
            self._streaming_text = ""
            changed = True

        if changed and follow:
            self.call_after_refresh(self.scroll_end, animate=False)

    async def show_notice(self, content: RenderableType) -> None:
        """临时命令输出也参与聊天区的整体滚动。"""
        notice = Static(content, markup=False)
        self._notices.append(notice)
        await self.mount(notice)
        self.call_after_refresh(self.scroll_end, animate=False)
