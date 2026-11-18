"""定义具体provider实现需要满足的接口"""

from collections.abc import AsyncIterator
from typing import Protocol

from ai.events import ProviderEvent
from agent.messages import AgentMessage
from agent.tools import AgentTool


class CancellationToken(Protocol):
    def is_cancelled(self) -> bool:
        ...


class ModelProvider(Protocol):
    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None
    ) -> AsyncIterator[ProviderEvent]:
        ...