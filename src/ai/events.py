"""定义provider模型发出的事件，描述模型的行为，和agent无关"""

from typing import Literal

from agent.messages import WireModel, AssistantMessage, ToolCall
from agent.types import JSONValue


class ProviderResponseStartEvent(WireModel):
    type: Literal["response_start"] = "response_start"
    model: str


class ProviderTextDeltaEvent(WireModel):
    type: Literal["text_delta"] = "text_delta"
    delta: str


class ProviderToolCallEvent(WireModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call: ToolCall


class ProviderResponseEndEvent(WireModel):
    type: Literal["response_end"] = "response_end"
    message: AssistantMessage
    finish_reason: str | None = None


class ProviderErrorEvent(WireModel):
    type: Literal["error"] = "error"
    message: str
    data: dict[str, JSONValue] | None = None


type ProviderEvent = (ProviderResponseStartEvent | ProviderResponseEndEvent
                      | ProviderTextDeltaEvent | ProviderToolCallEvent
                      | ProviderErrorEvent )

