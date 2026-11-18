"""Concrete tools for local coding work."""

from coding.context_window import (
    ContextUsageEstimate,
    auto_compaction_threshold_for_context_window,
    estimate_context_usage,
    estimate_message_tokens,
    estimate_text_tokens,
    estimate_tool_tokens,
)
from coding.resources import ResourceError, parse_markdown_resource
from coding.session import CodingSession, CodingSessionConfig
from coding.skills import Skill, expand_skill, load_skills
from coding.system_prompt import (
    BuildSystemPromptOptions,
    ProjectContextFile,
    build_system_prompt,
    collect_prompt_guidelines,
    format_available_tools,
    format_guidelines,
    format_project_context,
    format_skills_for_prompt,
)

from coding.tools import (
    ToolDefinition,
    create_bash_tool,
    create_bash_tool_definition,
    create_coding_tools,
    create_edit_tool,
    create_edit_tool_definition,
    create_read_tool,
    create_read_tool_definition,
    create_write_tool,
    create_write_tool_definition,
)

__all__ = [
    "ContextUsageEstimate",
    "auto_compaction_threshold_for_context_window",
    "estimate_context_usage",
    "estimate_message_tokens",
    "estimate_text_tokens",
    "estimate_tool_tokens",
    "BuildSystemPromptOptions",
    "ProjectContextFile",
    "build_system_prompt",
    "collect_prompt_guidelines",
    "format_available_tools",
    "format_guidelines",
    "format_project_context",
    "Skill",
    "load_skills",
    "expand_skill",
    "format_skills_for_prompt",
    "ResourceError",
    "parse_markdown_resource",
    "CodingSession",
    "CodingSessionConfig",
    "ToolDefinition",
    "create_bash_tool",
    "create_bash_tool_definition",
    "create_coding_tools",
    "create_edit_tool",
    "create_edit_tool_definition",
    "create_read_tool",
    "create_read_tool_definition",
    "create_write_tool",
    "create_write_tool_definition",
]
