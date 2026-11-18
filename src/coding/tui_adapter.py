"""将消息事件转换为 TUI 显示状态的变化。"""

from agent.events import (
    AgentEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from coding.tui_state import TuiState


class TuiEventAdapter:
    """更新传入的 State，由 App 负责刷新控件。"""

    def __init__(self, state: TuiState) -> None:
        self.state = state

    def apply(self, event: AgentEvent) -> None:
        if isinstance(event, MessageStartEvent):
            self.state.assistant_buffer = ""
        elif isinstance(event, ToolExecutionStartEvent):
            self.state.add_tool_call(event.tool_call)
        elif isinstance(event, ToolExecutionEndEvent):
            result = event.result
            self.state.add_tool_result(
                name=result.name, content=result.content or result.error or "", ok=result.ok,
            )
        elif isinstance(event, MessageDeltaEvent):
            self.state.assistant_buffer += event.delta
        elif isinstance(event, MessageEndEvent):
            text = event.message.content or self.state.assistant_buffer
            if text:
                self.state.add_item("assistant", text)
            self.state.assistant_buffer = ""
