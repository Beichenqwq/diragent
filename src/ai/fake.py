"""用固定的流式回答验证 Agent 和界面的接入。"""

import asyncio
from collections.abc import AsyncIterator

from ai.events import (
    ProviderEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
)
from ai.provider import CancellationToken
from agent.messages import AgentMessage, AssistantMessage
from agent.tools import AgentTool


class FakeProvider:
    async def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        response = "Hello from FakeProvider."

        yield ProviderResponseStartEvent(model=model)

        for index in range(0, len(response), 8):
            await asyncio.sleep(0.15)
            yield ProviderTextDeltaEvent(delta=response[index:index + 8])

        yield ProviderResponseEndEvent(
            message=AssistantMessage(content=response),
            finish_reason="stop",
        )
