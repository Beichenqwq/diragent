"""系统提示词构建所需的数据。"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path

from agent.tools import AgentTool
from coding.skills import Skill


@dataclass(frozen=True, slots=True)
class ProjectContextFile:
    """调用方提供的项目指令文件路径和正文；不读取文件。"""

    path: str
    content: str


@dataclass(frozen=True, slots=True)
class BuildSystemPromptOptions:
    """系统提示词的构建输入；只保存数据，不加载资源。"""

    cwd: Path
    tools: Sequence[AgentTool] = ()
    skills: Sequence[Skill] = ()
    context_files: Sequence[ProjectContextFile] = ()
    current_date: date | None = None
    extra_guidelines: Sequence[str] = ()


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    """从已准备的数据构建默认系统提示词，不读取文件或调用模型。"""
    current_date = options.current_date or date.today()
    cwd = str(options.cwd).replace("\\", "/")
    prompt = (
        "You are an expert coding assistant. "
        "You help users by reading files, executing commands, editing code, and writing new files."
        f"\n\nAvailable tools:\n{format_available_tools(options.tools)}"
        f"\n\nGuidelines:\n{format_guidelines(options.tools, options.extra_guidelines)}"
    )
    prompt += format_project_context(options.context_files)
    if any(tool.name == "read" for tool in options.tools):
        prompt += format_skills_for_prompt(options.skills)
    prompt += f"\nCurrent date: {current_date.isoformat()}"
    prompt += f"\nCurrent working directory: {cwd}"
    return prompt


def format_available_tools(tools: Sequence[AgentTool]) -> str:
    """将有 snippet 的工具按输入顺序格式化为列表。"""
    lines = [f"- {tool.name}: {tool.prompt_snippet}" for tool in tools if tool.prompt_snippet]
    return "\n".join(lines) if lines else "(none)"


def format_memory_index_for_prompt(index: str) -> str:
    """注入完整索引，并说明索引内文件链接的解析位置。"""
    if not index.strip():
        return ""
    return (
        "\n\n<memory_index>\n"
        "Catalog of stored memories. File links are relative to .tau/memory/ "
        "under the current working directory. Read a memory file when its details "
        "are relevant and not already included below.\n"
        + escape(index, quote=False) + "\n</memory_index>"
    )


def format_memories_for_prompt(memories: Sequence[dict[str, str]]) -> str:
    """格式化本轮选中的记忆正文，不读取文件。"""
    if not memories:
        return ""
    lines = [
        "\n\n<relevant_memories>",
        "Previously stored context relevant to this request. "
        "Use it as background; follow the user's current instructions when they differ.",
    ]
    for memory in memories:
        lines.extend([
            f"<memory name=\"{escape(memory['name'], quote=True)}\">",
            escape(memory["body"], quote=False),
            "</memory>",
        ])
    lines.append("</relevant_memories>")
    return "\n".join(lines)


def collect_prompt_guidelines(
    tools: Sequence[AgentTool], extra_guidelines: Sequence[str] = (),
) -> list[str]:
    """收集工具组合建议、工具规则、额外规则和默认规则，按首次出现顺序去重。"""
    names = {tool.name for tool in tools}
    guidelines: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        normalized = value.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        guidelines.append(normalized)

    has_bash = "bash" in names
    has_exploration_tools = bool({"grep", "find", "ls"} & names)
    if has_bash and not has_exploration_tools:
        add("Use bash for file operations like ls, rg, find")
    elif has_bash and has_exploration_tools:
        add("Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)")

    for tool in tools:
        for guideline in tool.prompt_guidelines:
            add(guideline)
    for guideline in extra_guidelines:
        add(guideline)

    add("Be concise in your responses")
    add("Show file paths clearly when working with files")
    return guidelines


def format_guidelines(tools: Sequence[AgentTool], extra_guidelines: Sequence[str] = ()) -> str:
    """收集规则并格式化为 Markdown 列表文本。"""
    return "\n".join(
        f"- {guideline}" for guideline in collect_prompt_guidelines(tools, extra_guidelines)
    )


def format_project_context(context_files: Sequence[ProjectContextFile]) -> str:
    """按输入顺序包装项目指令；保留正文，转义路径属性，不读取文件。"""
    if not context_files:
        return ""

    lines = [
        "\n\n<project_context>",
        "",
        "Project-specific instructions and guidelines:",
        "",
    ]
    for context_file in context_files:
        lines.append(f'<project_instructions path="{escape(context_file.path, quote=True)}">')
        lines.append(context_file.content)
        lines.append("</project_instructions>")
        lines.append("")
    lines.append("</project_context>")
    return "\n".join(lines)


def format_skills_for_prompt(skills: Sequence[Skill]) -> str:
    """生成按需读取指引和 XML 技能索引，不包含技能正文。"""
    if not skills:
        return ""

    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory "
        "(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in sorted(skills, key=lambda item: item.name):
        lines.extend([
            "  <skill>",
            f"    <name>{escape(skill.name, quote=False)}</name>",
            f"    <description>{escape(skill.description, quote=False)}</description>",
            f"    <location>{escape(str(skill.path), quote=False)}</location>",
            "  </skill>",
        ])
    lines.append("</available_skills>")
    return "\n".join(lines)
