"""State ownership and loop delegation for AgentHarness."""

import unittest
from collections.abc import AsyncIterator

from ai.events import ProviderEvent, ProviderResponseEndEvent
from agent.events import AgentEndEvent
from agent.harness import AgentHarness, AgentHarnessConfig
from agent.messages import AgentMessage, AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from agent.session import CompactionEntry, MessageEntry, SessionState
from agent.tools import AgentTool, AgentToolResult


class FakeProvider:
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = responses
        self.seen: list[tuple[str, str, list[AgentMessage], list[AgentTool]]] = []

    def stream_response(self, *, model, system, messages, tools, signal=None) -> AsyncIterator[ProviderEvent]:
        async def iterator() -> AsyncIterator[ProviderEvent]:
            self.seen.append((model, system, list(messages), list(tools)))
            yield ProviderResponseEndEvent(message=self.responses.pop(0))

        return iterator()


class AgentHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_keeps_history_and_passes_config_to_loop(self) -> None:
        provider = FakeProvider([
            AssistantMessage(content="", tool_calls=[ToolCall(id="c1", name="echo")]),
            AssistantMessage(content="first answer"),
            AssistantMessage(content="second answer"),
        ])

        async def echo(arguments):
            return AgentToolResult(tool_call_id="", name="echo", content="tool output", ok=True)

        tool = AgentTool(name="echo", description="Echo", input_schema={}, executor=echo)
        config = AgentHarnessConfig(provider=provider, model="fake", system="system", tools=[tool])
        harness = AgentHarness(config)

        first_events = [event async for event in harness.prompt("first question")]
        self.assertIsInstance(first_events[-1], AgentEndEvent)
        self.assertEqual([type(message) for message in harness.messages],
                         [UserMessage, AssistantMessage, ToolResultMessage, AssistantMessage])
        self.assertEqual(provider.seen[1][2][-1].content, "tool output")

        second_events = [event async for event in harness.prompt("second question")]
        self.assertIsInstance(second_events[-1], AgentEndEvent)
        self.assertEqual([message.content for message in provider.seen[2][2][-2:]],
                         ["first answer", "second question"])
        self.assertTrue(all(model == "fake" and system == "system" and tools == [tool]
                            for model, system, _, tools in provider.seen))
        self.assertEqual(harness.messages[-1].content, "second answer")

    async def test_messages_property_is_a_snapshot(self) -> None:
        provider = FakeProvider([AssistantMessage(content="continued")])
        harness = AgentHarness(AgentHarnessConfig(provider=provider, model="fake", system=""))

        snapshot = harness.messages
        events = [event async for event in harness.prompt("question")]

        self.assertIsInstance(events[-1], AgentEndEvent)
        self.assertEqual(len(provider.seen[0][2]), 1)
        self.assertEqual([message.content for message in harness.messages], ["question", "continued"])
        self.assertEqual(len(snapshot), 0)
        self.assertIsInstance(snapshot, tuple)


    async def test_config_returns_active_object_and_changes_reach_loop(self) -> None:
        provider = FakeProvider([AssistantMessage(content="answer")])
        config = AgentHarnessConfig(provider=provider, model="fake", system="old system")
        harness = AgentHarness(config)
        self.assertIs(harness.config, config)
        harness.config.system = "updated system"
        async for _ in harness.prompt("question"):
            pass
        self.assertEqual(provider.seen[0][1], "updated system")
        with self.assertRaises(AttributeError):
            harness.config = config

    def test_replace_messages_replaces_context_and_does_not_share_input_list(self) -> None:
        provider = FakeProvider([])
        old = UserMessage(content="old context")
        harness = AgentHarness(AgentHarnessConfig(provider=provider, model="fake", system=""),
                               messages=[old])
        snapshot = harness.messages
        summary = UserMessage(content="Previous conversation summary:\nsummary")
        kept = UserMessage(content="recent task")
        replacement = [summary, kept]

        harness.replace_messages(replacement)
        replacement.pop()
        replacement.append(UserMessage(content="external addition"))

        self.assertEqual(harness.messages, (summary, kept))
        self.assertIs(harness.messages[0], summary)
        self.assertIs(harness.messages[1], kept)
        self.assertEqual(snapshot, (old,))
        self.assertEqual(provider.seen, [])

    async def test_replaced_session_context_reaches_continue_and_prompt(self) -> None:
        continued = AssistantMessage(content="continued answer")
        answer = AssistantMessage(content="new answer")
        provider = FakeProvider([continued, answer])

        async def echo(arguments):
            return AgentToolResult(tool_call_id="", name="echo", content="output", ok=True)

        tool = AgentTool(name="echo", description="Echo", input_schema={}, executor=echo)
        config = AgentHarnessConfig(provider=provider, model="fake", system="system",
                                    tools=[tool], max_turns=3)
        old = MessageEntry(id="old", message=UserMessage(content="old task"))
        kept = MessageEntry(id="kept", message=UserMessage(content="recent task"))
        compaction = CompactionEntry(id="compact", summary="summary", first_kept_entry_id="kept")
        state = SessionState.from_entries([old, kept, compaction])
        compacted_messages = list(state.messages)
        harness = AgentHarness(config, messages=[old.message, kept.message])

        harness.replace_messages(state.messages)
        async for _ in harness.continue_():
            pass
        async for _ in harness.prompt("new question"):
            pass

        self.assertIs(harness.config, config)
        self.assertEqual(config.max_turns, 3)
        self.assertEqual(provider.seen[0], ("fake", "system", compacted_messages, [tool]))
        self.assertEqual(provider.seen[1], (
            "fake", "system",
            [*compacted_messages, continued, UserMessage(content="new question")], [tool],
        ))
        self.assertEqual(harness.messages,
                         (*compacted_messages, continued, UserMessage(content="new question"), answer))
        self.assertEqual(state.messages, compacted_messages)
        self.assertEqual(state.context_entry_ids, ["compact", "kept"])

    async def test_replace_accepts_tuple_and_can_clear_context(self) -> None:
        provider = FakeProvider([AssistantMessage(content="continued"), AssistantMessage(content="answer")])
        harness = AgentHarness(AgentHarnessConfig(provider=provider, model="fake", system=""),
                               messages=[UserMessage(content="old context")])
        summary = UserMessage(content="summary")

        harness.replace_messages((summary,))
        async for _ in harness.continue_():
            pass
        self.assertEqual(provider.seen[0][2], [summary])

        harness.replace_messages(())
        self.assertEqual(harness.messages, ())
        async for _ in harness.prompt("fresh question"):
            pass
        self.assertEqual(provider.seen[1][2], [UserMessage(content="fresh question")])

if __name__ == "__main__":
    unittest.main()
