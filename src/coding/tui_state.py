"""TUI 显示数据，与模型上下文和会话存储独立。"""

from collections.abc import Iterable
from dataclasses import dataclass, field
import json
from typing import Literal

from agent.messages import ToolCall
from agent.session.entries import MessageEntry, SessionEntry


ChatItemRole = Literal["user", "assistant", "tool"]


@dataclass(slots=True)
class ChatItem:
    """聊天区域中的一条显示记录。"""

    role: ChatItemRole
    text: str


@dataclass(slots=True)
class TuiState:
    """保存已加入的显示条目和正在生成的助手文本。"""

    items: list[ChatItem] = field(default_factory=list)
    assistant_buffer: str = ""

    def add_item(self, role: ChatItemRole, text: str) -> None:
        self.items.append(ChatItem(role=role, text=text))

    def add_tool_call(self, call: ToolCall) -> None:
        """显示调用名称和 JSON 参数，供实时事件和历史恢复复用。"""
        arguments = json.dumps(call.arguments, ensure_ascii=False)
        self.add_item("tool", f"→ {call.name} {arguments}")

    def add_tool_result(self, *, name: str, content: str, ok: bool) -> None:
        """显示执行状态和结果文本。"""
        status = "OK" if ok else "FAILED"
        text = f"{status}: {name}"
        if content:
            text += f"\n{content}"
        self.add_item("tool", text)

    def load_entries(self, entries: Iterable[SessionEntry]) -> None:
        """从原始记录恢复文本历史，不应用模型上下文的压缩规则。"""
        self.items.clear()
        self.assistant_buffer = ""
        for entry in entries:
            if not isinstance(entry, MessageEntry):
                continue
            message = entry.message
            if message.role == "user":
                self.add_item("user", message.content)
            elif message.role == "assistant":
                if message.content:
                    self.add_item("assistant", message.content)
                for call in message.tool_calls:
                    self.add_tool_call(call)
            elif message.role == "tool":
                self.add_tool_result(name=message.name, content=message.content, ok=message.ok)
