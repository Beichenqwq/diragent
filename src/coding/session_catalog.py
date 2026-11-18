"""查找和恢复存储目录中的独立会话，不依赖具体 UI。"""

from dataclasses import dataclass, replace
from pathlib import Path

from agent.session.entries import SessionInfoEntry
from agent.session.jsonl import entry_from_json_line, entry_to_json_line
from agent.session.storage import JsonlSessionStorage
from coding.session import CodingSession, CodingSessionConfig


@dataclass(frozen=True, slots=True)
class SavedSession:
    path: Path
    session_id: str
    updated_at: float


def create_new_session(config: CodingSessionConfig) -> CodingSession:
    """第一次提交输入时，创建以新 ID 命名的文件并加载会话。"""
    info = SessionInfoEntry(cwd=str(config.cwd.resolve()))
    path = config.storage.path.resolve().parent / f"{info.session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        file.write(entry_to_json_line(info))
    return CodingSession.load(replace(config, storage=JsonlSessionStorage(path)))


def list_saved_sessions(directory: Path) -> tuple[list[SavedSession], list[str]]:
    """只读取首条元信息，用文件修改时间排序；正文在恢复时校验。"""
    sessions: list[SavedSession] = []
    errors: list[str] = []
    for path in directory.glob("*.jsonl"):
        try:
            with path.open("r", encoding="utf-8") as file:
                first_line = next((line for line in file if line.strip()), "")
            info = entry_from_json_line(first_line)
            if not isinstance(info, SessionInfoEntry):
                raise ValueError("Expected session_info as the first entry")
            sessions.append(SavedSession(
                path=path,
                session_id=info.session_id,
                updated_at=path.stat().st_mtime,
            ))
        except (OSError, UnicodeError, ValueError):
            errors.append(f"Invalid or unreadable session: {path.name}")
    sessions.sort(key=lambda session: (-session.updated_at, session.path.name))
    return sessions, errors


def resume_session(config: CodingSessionConfig, target: str) -> CodingSession:
    """按文件名、完整 ID 或唯一 ID 前缀恢复，沿用当前运行配置。"""
    directory = config.storage.path.resolve().parent
    sessions, _ = list_saved_sessions(directory)
    matches = [
        session for session in sessions
        if session.path.name.casefold() == target.casefold() or session.session_id == target
    ]
    if not matches:
        matches = [session for session in sessions if session.session_id.startswith(target)]
    if not matches:
        raise ValueError(f"Session not found: {target}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous session: {target}. Use a filename or a longer ID.")
    path = matches[0].path
    if not path.is_file():
        raise ValueError(f"Session file no longer exists: {path.name}")
    return CodingSession.load(replace(config, storage=JsonlSessionStorage(path)))
