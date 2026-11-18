"""定义agent loop执行过程中需要发出的一些关键事件，用来通知系统的其他模块(例如CLI/TUI)当前运行的进度"""

from typing import Literal

from agent.messages import WireModel, AssistantMessage, ToolCall
from agent.tools import AgentToolResult
from agent.types import JSONValue


class AgentStartEvent(WireModel):
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(WireModel):
    type: Literal["agent_end"] = "agent_end"


class TurnStartEvent(WireModel):
    type: Literal["turn_start"] = "turn_start"
    turn: int


class TurnEndEvent(WireModel):
    type: Literal["turn_end"] = "turn_end"
    turn: int


class MessageStartEvent(WireModel):
    type: Literal["message_start"] = "message_start"
    message_role: Literal["assistant"] = "assistant"


class MessageDeltaEvent(WireModel):
    type: Literal["message_delta"] = "message_delta"
    delta: str


class MessageEndEvent(WireModel):
    type: Literal["message_end"] = "message_end"
    message: AssistantMessage


class ToolExecutionStartEvent(WireModel):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call: ToolCall


class ToolExecutionUpdateEvent(WireModel):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    message: str
    data: dict[str, JSONValue] | None = None


class ToolExecutionEndEvent(WireModel):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    result: AgentToolResult


class ErrorEvent(WireModel):
    type: Literal["error"] = "error"
    message: str
    source: Literal["provider", "tool", "loop"] = "loop"
    recoverable: bool = False
    data: dict[str, JSONValue] | None = None


type AgentEvent = (AgentStartEvent | AgentEndEvent | TurnStartEvent | TurnEndEvent
                   | MessageStartEvent | MessageDeltaEvent | MessageEndEvent
                   | ToolExecutionStartEvent | ToolExecutionUpdateEvent | ToolExecutionEndEvent
                   | ErrorEvent)
