"""Stateful, frontend-independent controller for the agent loop."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from ai.provider import ModelProvider
from agent.events import AgentEvent
from agent.loop import run_agent_loop
from agent.messages import AgentMessage, UserMessage
from agent.tools import AgentTool


@dataclass(slots=True)
class AgentHarnessConfig:
    provider: ModelProvider
    model: str
    system: str
    tools: list[AgentTool] = field(default_factory=list)
    max_turns: int = 8


class AgentHarness:
    """Own the transcript and delegate each run to the stateless loop."""

    def __init__(
        self,
        config: AgentHarnessConfig,
        *,
        messages: Sequence[AgentMessage] = (),
    ) -> None:
        self._config = config
        self._messages: list[AgentMessage] = list(messages)

    @property
    def config(self) -> AgentHarnessConfig:
        """Return the active configuration object; its fields remain mutable."""
        return self._config

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        """Return a snapshot; callers cannot change the transcript list."""
        return tuple(self._messages)

    def replace_messages(self, messages: Sequence[AgentMessage]) -> None:
        """Replace the context; keep an independent list of the supplied messages."""
        self._messages = list(messages)

    def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
        """Append a user message, then run the loop with current state."""
        self._messages.append(UserMessage(content=content))
        return self._run()

    def continue_(self) -> AsyncIterator[AgentEvent]:
        """Run from the existing transcript without adding a user message."""
        return self._run()

    async def _run(self) -> AsyncIterator[AgentEvent]:
        async for event in run_agent_loop(
            provider=self._config.provider,
            model=self._config.model,
            system=self._config.system,
            messages=self._messages,
            tools=self._config.tools,
            max_turns=self._config.max_turns,
        ):
            yield event
