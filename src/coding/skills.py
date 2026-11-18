"""加载当前项目的 Markdown 技能。"""

from dataclasses import dataclass
from html import escape
from pathlib import Path

from coding.resources import ResourceError, parse_markdown_resource


@dataclass(frozen=True, slots=True)
class Skill:
    """已加载的技能；名称来自子目录，描述来自 frontmatter。"""

    name: str
    path: Path
    description: str
    content: str # skill正文，其余三项是skill元信息


def load_skills(cwd: Path) -> list[Skill]:
    """按目录名排序加载 cwd/.tau/skills/*/SKILL.md。

    资源目录不存在时返回空列表，没有 SKILL.md 的子目录跳过。
    文件读取、元信息解析或必填描述校验失败时抛出包含路径的
    ResourceError；不从正文推导描述。
    """
    skills_dir = cwd.resolve() / ".tau" / "skills"
    if not skills_dir.exists():
        return []

    skills: list[Skill] = []
    for directory in sorted(skills_dir.iterdir(), key=lambda item: item.name):
        if not directory.is_dir():
            continue
        path = directory / "SKILL.md"
        if not path.is_file():
            continue

        try:
            with path.open("r", encoding="utf-8", newline="") as file:
                metadata, content = parse_markdown_resource(file.read())
        except (OSError, UnicodeError, ResourceError) as exc:
            raise ResourceError(f"Failed to load skill {path}: {exc}") from exc

        description = metadata.get("description", "").strip()
        if not description:
            raise ResourceError(f"Skill {path} must include a non-empty description in frontmatter")

        skills.append(Skill(
            name=directory.name,
            path=path,
            description=description,
            content=content,
        ))
    return skills


def expand_skill(skill: Skill, request: str) -> str:
    """用技能标签包围完整正文，再附加本次任务；不替换正文占位符。"""
    return (
        "Use the following skill instructions:\n\n"
        f'<skill name="{escape(skill.name, quote=True)}" '
        f'location="{escape(str(skill.path), quote=True)}">\n'
        f"References are relative to {skill.path.parent}.\n\n"
        f"{skill.content}\n</skill>\n\n"
        f"{request}"
    )
