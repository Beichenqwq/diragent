"""Tool prompt metadata survives conversion without changing execution."""

import tempfile
import unittest
from pathlib import Path

from agent.tools import AgentTool, AgentToolResult
from agent.types import JSONValue
from coding.tools import (
    ToolDefinition,
    create_bash_tool_definition,
    create_edit_tool_definition,
    create_read_tool_definition,
    create_write_tool_definition,
)


class ToolPromptMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_tool_constructor_and_executor_still_work(self) -> None:
        async def executor(arguments: dict[str, JSONValue]) -> AgentToolResult:
            return AgentToolResult(
                tool_call_id="call-1", name="echo", content=str(arguments["text"]), ok=True,
            )

        tool = AgentTool("echo", "Echo text", {}, executor)

        self.assertIsNone(tool.prompt_snippet)
        self.assertEqual(tool.prompt_guidelines, ())
        result = await tool.execute({"text": "hello"})
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "hello")

    async def test_custom_definition_preserves_metadata_and_execution(self) -> None:
        arguments: dict[str, JSONValue] = {"text": "hello"}
        result = AgentToolResult(tool_call_id="call-1", name="echo", content="hello", ok=True)

        async def executor(received: dict[str, JSONValue]) -> AgentToolResult:
            self.assertIs(received, arguments)
            return result

        definition = ToolDefinition(
            name="echo", description="Echo text", input_schema={"type": "object"},
            executor=executor, prompt_snippet="Echo a message.",
            prompt_guidelines=("Preserve the original text.",),
        )
        tool = definition.to_agent_tool()

        self.assertEqual(tool.name, definition.name)
        self.assertEqual(tool.description, definition.description)
        self.assertIs(tool.input_schema, definition.input_schema)
        self.assertIs(tool.executor, executor)
        self.assertEqual(tool.prompt_snippet, definition.prompt_snippet)
        self.assertEqual(tool.prompt_guidelines, definition.prompt_guidelines)
        self.assertIs(await tool.execute(arguments), result)

    async def test_default_tool_factories_preserve_prompt_metadata(self) -> None:
        temporary_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            for factory in (
                create_read_tool_definition, create_write_tool_definition,
                create_edit_tool_definition, create_bash_tool_definition,
            ):
                definition = factory(Path(directory))
                with self.subTest(tool=definition.name):
                    tool = definition.to_agent_tool()
                    self.assertTrue(definition.prompt_snippet)
                    self.assertTrue(definition.prompt_guidelines)
                    self.assertEqual(tool.prompt_snippet, definition.prompt_snippet)
                    self.assertEqual(tool.prompt_guidelines, definition.prompt_guidelines)
