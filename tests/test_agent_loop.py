"""Behavioral tests for the provider -> tool -> provider loop."""

import unittest
from collections.abc import AsyncIterator

from ai.events import (
    ProviderEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
)
from agent.events import (
    AgentEndEvent,
    ErrorEvent,
    MessageDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
)
from agent.loop import run_agent_loop
from agent.messages import AgentMessage, AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from agent.tools import AgentTool, AgentToolResult


class FakeProvider:
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = responses
        self.transcripts: list[list[AgentMessage]] = []

    def stream_response(self, *, model, system, messages, tools, signal=None) -> AsyncIterator[ProviderEvent]:
        async def iterator() -> AsyncIterator[ProviderEvent]:
            self.transcripts.append(list(messages))
            response = self.responses.pop(0)
            yield ProviderResponseStartEvent(model=model)
            if response.content:
                yield ProviderTextDeltaEvent(delta=response.content)
            yield ProviderResponseEndEvent(message=response)

        return iterator()


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_only_flow(self) -> None:
        messages: list[AgentMessage] = [UserMessage(content="Hi")]
        provider = FakeProvider([AssistantMessage(content="Hello")])

        events = [event async for event in run_agent_loop(
            provider=provider, model="fake", system="", messages=messages, tools=[]
        )]

        self.assertEqual([event.delta for event in events if isinstance(event, MessageDeltaEvent)], ["Hello"])
        self.assertEqual([event.turn for event in events if isinstance(event, TurnEndEvent)], [1])
        self.assertIsInstance(events[-1], AgentEndEvent)
        self.assertEqual(messages[-1].content, "Hello")

    async def test_multiple_tools_feed_results_into_next_provider_turn(self) -> None:
        calls = [ToolCall(id="c1", name="echo", arguments={"value": "a"}),
                 ToolCall(id="c2", name="echo", arguments={"value": "b"})]
        provider = FakeProvider([
            AssistantMessage(content="", tool_calls=calls),
            AssistantMessage(content="done"),
        ])
        executed: list[str] = []

        async def echo(arguments):
            executed.append(arguments["value"])
            return AgentToolResult(tool_call_id="placeholder", name="placeholder",
                                   content=arguments["value"], ok=True)

        tool = AgentTool(name="echo", description="Echo", input_schema={}, executor=echo)
        messages: list[AgentMessage] = [UserMessage(content="echo twice")]
        events = [event async for event in run_agent_loop(
            provider=provider, model="fake", system="", messages=messages, tools=[tool]
        )]

        self.assertEqual(executed, ["a", "b"])
        self.assertEqual([message.tool_call_id for message in messages if isinstance(message, ToolResultMessage)], ["c1", "c2"])
        self.assertEqual([message.content for message in provider.transcripts[1][-2:]], ["a", "b"])
        self.assertEqual([event.result.tool_call_id for event in events if isinstance(event, ToolExecutionEndEvent)], ["c1", "c2"])
        self.assertEqual(sum(isinstance(event, ToolExecutionStartEvent) for event in events), 2)
        self.assertEqual([event.turn for event in events if isinstance(event, TurnEndEvent)], [1, 2])

    async def test_unknown_and_raised_tool_errors_are_returned_to_model(self) -> None:
        calls = [ToolCall(id="missing", name="absent"), ToolCall(id="broken", name="explode")]
        provider = FakeProvider([
            AssistantMessage(content="", tool_calls=calls),
            AssistantMessage(content="recovered"),
        ])

        async def explode(arguments):
            raise ValueError("bad input")

        tool = AgentTool(name="explode", description="Fail", input_schema={}, executor=explode)
        messages: list[AgentMessage] = [UserMessage(content="try tools")]
        events = [event async for event in run_agent_loop(
            provider=provider, model="fake", system="", messages=messages, tools=[tool]
        )]

        results = [message for message in provider.transcripts[1] if isinstance(message, ToolResultMessage)]
        self.assertEqual([message.tool_call_id for message in results], ["missing", "broken"])
        self.assertTrue(all(not message.ok for message in results))
        self.assertIn("Unknown tool: absent", results[0].content)
        self.assertIn("bad input", results[1].content)
        self.assertEqual(sum(isinstance(event, ErrorEvent) and event.recoverable for event in events), 2)

    async def test_turn_limit_ends_after_tool_result(self) -> None:
        call = ToolCall(id="c1", name="echo", arguments={})
        provider = FakeProvider([AssistantMessage(content="", tool_calls=[call])])

        async def echo(arguments):
            return AgentToolResult(tool_call_id="c1", name="echo", content="ok", ok=True)

        tool = AgentTool(name="echo", description="Echo", input_schema={}, executor=echo)
        messages: list[AgentMessage] = [UserMessage(content="echo")]
        events = [event async for event in run_agent_loop(
            provider=provider, model="fake", system="", messages=messages,
            tools=[tool], max_turns=1
        )]

        self.assertEqual(len(provider.transcripts), 1)
        self.assertIsInstance(messages[-1], ToolResultMessage)
        self.assertTrue(any(isinstance(event, ErrorEvent) and "max_turns" in event.message for event in events))
        self.assertIsInstance(events[-1], AgentEndEvent)

    async def test_tool_reported_failure_uses_error_as_model_content(self) -> None:
        call = ToolCall(id="c1", name="lookup")
        provider = FakeProvider([
            AssistantMessage(content="", tool_calls=[call]),
            AssistantMessage(content="handled"),
        ])

        async def lookup(arguments):
            return AgentToolResult(tool_call_id="", name="lookup", content="",
                                   ok=False, error="item unavailable")

        tool = AgentTool(name="lookup", description="Lookup", input_schema={}, executor=lookup)
        messages: list[AgentMessage] = [UserMessage(content="lookup")]
        events = [event async for event in run_agent_loop(
            provider=provider, model="fake", system="", messages=messages, tools=[tool]
        )]

        result = provider.transcripts[1][-1]
        self.assertIsInstance(result, ToolResultMessage)
        self.assertEqual(result.content, "item unavailable")
        self.assertFalse(result.ok)
        self.assertTrue(any(isinstance(event, ErrorEvent) and event.recoverable for event in events))


if __name__ == "__main__":
    unittest.main()
