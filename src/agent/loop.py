"""Run provider turns and feed tool results back into the transcript."""

from collections.abc import AsyncIterator, Mapping

from ai.events import (
    ProviderErrorEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
)
from ai.provider import CancellationToken, ModelProvider
from agent.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from agent.messages import AgentMessage, AssistantMessage, ToolCall, ToolResultMessage
from agent.tools import AgentTool, AgentToolResult

async def run_agent_loop(
    *,
    provider: ModelProvider,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    max_turns: int = 8,
    signal: CancellationToken | None = None,
) -> AsyncIterator[AgentEvent]:
    yield AgentStartEvent()

    if max_turns < 1:
        yield ErrorEvent(message="max_turns必须至少为1")
        yield AgentEndEvent()
        return

    tool_by_name = {tool.name: tool for tool in tools}

    for turn in range(1, max_turns + 1):
        if signal is not None and signal.is_cancelled():
            yield ErrorEvent(message="收到取消信号，终止turn循环", recoverable=True)
            break

        yield TurnStartEvent(turn=turn)
        assistant_message: AssistantMessage | None = None
        saw_provider_error = False

        async for provider_event in provider.stream_response(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            signal=signal,
        ):
            if isinstance(provider_event, ProviderResponseStartEvent):
                yield MessageStartEvent()
            elif isinstance(provider_event, ProviderTextDeltaEvent):
                yield MessageDeltaEvent(delta=provider_event.delta)
            elif isinstance(provider_event, ProviderResponseEndEvent):
                assistant_message = provider_event.message
                messages.append(assistant_message)
                yield MessageEndEvent(message=assistant_message)
            elif isinstance(provider_event, ProviderErrorEvent):
                saw_provider_error = True
                yield ErrorEvent(
                    message=provider_event.message,
                    source="provider",
                    data=provider_event.data,
                )

        if saw_provider_error or assistant_message is None:
            if not saw_provider_error:
                yield ErrorEvent(message="Provider流式传输完毕后没有发出assistant message")
            yield TurnEndEvent(turn=turn)
            break

        if not assistant_message.tool_calls:
            yield TurnEndEvent(turn=turn)
            break

        async for event in _execute_tool_calls(
            assistant_message.tool_calls, tool_by_name, messages
        ):
            yield event

        yield TurnEndEvent(turn=turn)
        if turn == max_turns:
            yield ErrorEvent(message="达到max_turns，尚未取得最终回复")

    yield AgentEndEvent()


async def _execute_tool_calls(
    tool_calls: list[ToolCall],
    tool_by_name: Mapping[str, AgentTool],
    messages: list[AgentMessage],
) -> AsyncIterator[AgentEvent]:
    """Execute calls in order, append model-facing results, and emit progress."""
    for tool_call in tool_calls:
        yield ToolExecutionStartEvent(tool_call=tool_call)
        tool = tool_by_name.get(tool_call.name)

        if tool is None:
            error = f"Unknown tool: {tool_call.name}"
            result = AgentToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=error,
                ok=False,
                error=error,
            )
        else:
            try:
                raw_result = await tool.execute(tool_call.arguments)
                # The call identity belongs to the provider request, even if
                # a tool returns its own placeholder ID or name.
                result = AgentToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=(raw_result.content or raw_result.error
                             or ("Tool failed without details" if not raw_result.ok else "")),
                    ok=raw_result.ok,
                    data=raw_result.data,
                    error=raw_result.error,
                )
            except Exception as exc:
                error = f"Tool execution failed: {exc or type(exc).__name__}"
                result = AgentToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=error,
                    ok=False,
                    error=error,
                )

        messages.append(ToolResultMessage(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=result.content,
            ok=result.ok,
        ))
        if not result.ok:
            yield ErrorEvent(
                message=result.error or result.content,
                source="tool",
                recoverable=True,
            )
        yield ToolExecutionEndEvent(result=result)
