"""System prompt inputs preserve caller data without loading resources."""

import unittest
import xml.etree.ElementTree as ET
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path
from unittest.mock import patch

from agent.tools import AgentTool, AgentToolResult
from coding import (
    BuildSystemPromptOptions,
    ProjectContextFile,
    build_system_prompt,
    Skill,
    collect_prompt_guidelines,
    format_available_tools,
    format_guidelines,
    format_project_context,
)


class SystemPromptInputTests(unittest.TestCase):
    def test_minimal_options_use_empty_sequences_and_unspecified_date(self) -> None:
        cwd = Path("project")
        options = BuildSystemPromptOptions(cwd=cwd)

        self.assertIs(options.cwd, cwd)
        self.assertEqual(options.tools, ())
        self.assertEqual(options.skills, ())
        self.assertEqual(options.context_files, ())
        self.assertEqual(options.extra_guidelines, ())
        self.assertIsNone(options.current_date)
        with self.assertRaises(FrozenInstanceError):
            options.current_date = date(2026, 10, 2)

    def test_explicit_inputs_are_preserved_without_file_access_or_execution(self) -> None:
        async def executor(arguments) -> AgentToolResult:
            raise AssertionError("Constructing options must not execute tools")

        cwd = Path("project")
        tools = [AgentTool("read", "Read a file", {}, executor)]
        skills = [Skill("review", cwd / ".tau/skills/review/SKILL.md", "Review code", "Steps")]
        fixed_date = date(2026, 10, 2)
        extra_guidelines = ["Run relevant tests."]

        with patch.object(Path, "open", side_effect=AssertionError("Unexpected file read")):
            context = ProjectContextFile(path="project/AGENTS.md", content="Keep changes focused.\n")
            context_files = [context]
            options = BuildSystemPromptOptions(
                cwd=cwd, tools=tools, skills=skills, context_files=context_files,
                current_date=fixed_date, extra_guidelines=extra_guidelines,
            )

        self.assertIs(options.tools, tools)
        self.assertIs(options.skills, skills)
        self.assertIs(options.context_files, context_files)
        self.assertIs(options.extra_guidelines, extra_guidelines)
        self.assertEqual(options.current_date, fixed_date)
        self.assertEqual(context.path, "project/AGENTS.md")
        self.assertEqual(context.content, "Keep changes focused.\n")


class ToolGuidelineTests(unittest.TestCase):
    @staticmethod
    def make_tool(name, snippet=None, guidelines=()) -> AgentTool:
        async def executor(arguments) -> AgentToolResult:
            raise AssertionError("Formatting must not execute tools")

        return AgentTool(
            name=name, description="This description must not become a guideline.",
            input_schema={}, executor=executor,
            prompt_snippet=snippet, prompt_guidelines=guidelines,
        )

    def test_available_tools_preserve_order_and_skip_missing_snippets(self) -> None:
        tools = [
            self.make_tool("write", "Write files."),
            self.make_tool("hidden"),
            self.make_tool("empty", ""),
            self.make_tool("read", "Read files."),
        ]
        self.assertEqual(format_available_tools(tools), "- write: Write files.\n- read: Read files.")
        self.assertEqual([tool.name for tool in tools], ["write", "hidden", "empty", "read"])
        self.assertEqual(format_available_tools(()), "(none)")
        self.assertEqual(format_available_tools([self.make_tool("hidden")]), "(none)")

    def test_guidelines_trim_deduplicate_and_preserve_source_order(self) -> None:
        tools = [
            self.make_tool("read", guidelines=("  Inspect files. ", "", "Run tests.",
                                                "Be concise in your responses")),
            self.make_tool("bash", guidelines=("Inspect files.", "\n")),
        ]
        extra = [" Run tests. ", " Extra rule. ", " ", "Be concise in your responses"]
        self.assertEqual(collect_prompt_guidelines(tools, extra), [
            "Use bash for file operations like ls, rg, find",
            "Inspect files.", "Run tests.", "Be concise in your responses", "Extra rule.",
            "Show file paths clearly when working with files",
        ])
        self.assertEqual(tools[0].prompt_guidelines[0], "  Inspect files. ")
        self.assertEqual(extra, [" Run tests. ", " Extra rule. ", " ", "Be concise in your responses"])

    def test_exploration_advice_depends_on_enabled_tools(self) -> None:
        bash = self.make_tool("bash")
        preferred = "Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)"
        for name in ("grep", "find", "ls"):
            with self.subTest(tool=name):
                self.assertEqual(collect_prompt_guidelines([bash, self.make_tool(name)])[0], preferred)
                self.assertEqual(collect_prompt_guidelines([self.make_tool(name)]), [
                    "Be concise in your responses", "Show file paths clearly when working with files",
                ])
        self.assertEqual(collect_prompt_guidelines([bash])[0],
                         "Use bash for file operations like ls, rg, find")

    def test_format_guidelines_returns_bullets_in_collection_order(self) -> None:
        self.assertEqual(format_guidelines((), (" Extra rule. ", "Extra rule.")),
                         "- Extra rule.\n- Be concise in your responses\n"
                         "- Show file paths clearly when working with files")


class ProjectContextFormattingTests(unittest.TestCase):
    def test_no_files_returns_empty_string(self) -> None:
        self.assertEqual(format_project_context([]), "")
        self.assertEqual(format_project_context(()), "")

    def test_path_special_characters_roundtrip_as_one_attribute(self) -> None:
        path = 'project/a&b<review>"draft\'/AGENTS.md'
        formatted = format_project_context([ProjectContextFile(path=path, content="Instructions")])

        root = ET.fromstring(formatted)
        self.assertEqual(root.tag, "project_context")
        instructions = root.findall("project_instructions")
        self.assertEqual(len(instructions), 1)
        self.assertEqual(instructions[0].attrib, {"path": path})
        self.assertEqual(instructions[0].text, "\nInstructions\n")

    def test_preserves_body_and_file_order_without_file_access(self) -> None:
        body = '\r\n# Rules\r\n```html\r\n<div title="a&b">example</div>\r\n```\r\n'
        context_files = [
            ProjectContextFile(path="project/AGENTS.md", content=body),
            ProjectContextFile(path="project/OTHER.md", content=""),
        ]
        original_files = list(context_files)
        with patch.object(Path, "open", side_effect=AssertionError("Unexpected file read")):
            formatted = format_project_context(context_files)

        self.assertTrue(formatted.startswith("\n\n<project_context>\n"))
        self.assertIn("Project-specific instructions and guidelines:", formatted)
        self.assertIn(body, formatted)
        self.assertLess(formatted.index('path="project/AGENTS.md"'),
                        formatted.index('path="project/OTHER.md"'))
        self.assertIn('<project_instructions path="project/OTHER.md">\n\n</project_instructions>',
                      formatted)
        self.assertTrue(formatted.endswith("</project_context>"))
        self.assertEqual(context_files, original_files)
        self.assertEqual(context_files[0].content, body)


class SystemPromptBuilderTests(unittest.TestCase):
    def test_builds_sections_in_order_with_fixed_date_and_preserves_inputs(self) -> None:
        tools = [
            ToolGuidelineTests.make_tool("read", "Read files.", ("Inspect first.",)),
            ToolGuidelineTests.make_tool("bash", "Execute commands."),
        ]
        skills = [
            Skill("testing", Path("skills/testing/SKILL.md"), "Write tests", "EXCLUDED_BODY"),
            Skill("review", Path("skills/review/SKILL.md"), "Review code", "EXCLUDED_BODY"),
        ]
        options = BuildSystemPromptOptions(
            cwd=Path(r"C:\project\repo"), tools=tools, skills=skills,
            context_files=[ProjectContextFile("AGENTS.md", "Project instructions.")],
            current_date=date(2026, 10, 2), extra_guidelines=["Run tests."],
        )
        with patch.object(Path, "open", side_effect=AssertionError("Unexpected file read")):
            text = build_system_prompt(options)
            self.assertEqual(build_system_prompt(options), text)

        self.assertTrue(text.startswith("You are an expert coding assistant."))
        sections = ["Available tools:", "Guidelines:", "<project_context>",
                    "<available_skills>", "Current date:", "Current working directory:"]
        positions = [text.index(section) for section in sections]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("- read: Read files.", text)
        self.assertIn("- bash: Execute commands.", text)
        self.assertIn("- Inspect first.", text)
        self.assertIn("- Run tests.", text)
        self.assertIn("Project instructions.", text)
        self.assertNotIn("EXCLUDED_BODY", text)
        self.assertTrue(text.endswith("Current date: 2026-10-02\n"
                                      "Current working directory: C:/project/repo"))
        self.assertEqual([tool.name for tool in tools], ["read", "bash"])
        self.assertEqual([skill.name for skill in skills], ["testing", "review"])

    def test_skill_index_requires_read_and_nonempty_skills(self) -> None:
        skill = Skill("review", Path("skills/review/SKILL.md"), "Review code", "Body")
        cases = [
            ([], [skill], False),
            ([ToolGuidelineTests.make_tool("bash")], [skill], False),
            ([ToolGuidelineTests.make_tool("read")], [], False),
            ([ToolGuidelineTests.make_tool("read")], [skill], True),
        ]
        for tools, skills, expected in cases:
            with self.subTest(tools=[tool.name for tool in tools], skills=len(skills)):
                text = build_system_prompt(BuildSystemPromptOptions(
                    cwd=Path("project"), tools=tools, skills=skills, current_date=date(2026, 10, 2),
                ))
                self.assertEqual("<available_skills>" in text, expected)
                self.assertEqual("Use the read tool to load a skill's file" in text, expected)
                self.assertNotIn("<project_context>", text)
        empty = build_system_prompt(BuildSystemPromptOptions(
            cwd=Path("project"), current_date=date(2026, 10, 2),
        ))
        self.assertIn("Available tools:\n(none)", empty)
        self.assertIn("- Be concise in your responses", empty)

    def test_default_date_uses_today_and_explicit_date_takes_precedence(self) -> None:
        with patch("coding.system_prompt.date") as clock:
            clock.today.return_value = date(2026, 9, 30)
            text = build_system_prompt(BuildSystemPromptOptions(cwd=Path("project")))
            self.assertIn("Current date: 2026-09-30", text)
            clock.today.assert_called_once_with()
            clock.today.reset_mock()
            explicit = build_system_prompt(BuildSystemPromptOptions(
                cwd=Path("project"), current_date=date(2026, 10, 2),
            ))
            self.assertIn("Current date: 2026-10-02", explicit)
            clock.today.assert_not_called()
