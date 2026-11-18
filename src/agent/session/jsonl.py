"""会话记录与 JSONL 文本之间的转换。"""

from collections.abc import Iterable

from pydantic import TypeAdapter, ValidationError

from agent.session.entries import SessionEntry

_ENTRY_ADAPTER = TypeAdapter(SessionEntry)


class SessionJsonlError(ValueError):
    """JSONL 记录无法解析或不符合消息记录结构。"""


def entry_to_json_line(entry: SessionEntry) -> str:
    """将一条记录序列化为带换行符的 JSONL 行。"""
    return entry.model_dump_json() + "\n"


def entry_from_json_line(line: str, *, line_number: int | None = None) -> SessionEntry:
    """解析并校验一条 JSONL 记录，错误时附带可选行号。"""
    try:
        return _ENTRY_ADAPTER.validate_json(line)
    except ValidationError as exc:
        location = f" on line {line_number}" if line_number is not None else ""
        raise SessionJsonlError(f"Invalid session entry{location}: {exc}") from exc


def entries_from_json_lines(lines: Iterable[str]) -> list[SessionEntry]:
    """按顺序解析多行记录，跳过空行并保留原始行号。"""
    entries: list[SessionEntry] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip(): # 即：如果这一行为空
            continue
        entries.append(entry_from_json_line(line, line_number=line_number))
    return entries
