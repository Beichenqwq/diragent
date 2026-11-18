"""从会话记录中恢复元信息和历史消息。"""

from collections.abc import Iterable
from dataclasses import dataclass

from agent.messages import AgentMessage, UserMessage
from agent.session.entries import CompactionEntry, MessageEntry, SessionEntry, SessionInfoEntry


@dataclass
class SessionState:
    """由已读取的记录重建出的会话状态。"""

    session_info: SessionInfoEntry | None
    messages: list[AgentMessage]
    context_entry_ids: list[str]

    @classmethod
    def from_entries(cls, entries: Iterable[SessionEntry]) -> "SessionState":
        """按记录顺序恢复状态，并应用压缩记录的保留边界。"""
        replay_entries = list(entries)
        session_info: SessionInfoEntry | None = None
        message_rows: list[tuple[str, AgentMessage]] = []
        for entry_index, entry in enumerate(replay_entries):
            if isinstance(entry, SessionInfoEntry):
                if session_info is not None:
                    raise ValueError("Session contains multiple session_info entries")
                session_info = entry
            elif isinstance(entry, MessageEntry):
                message_rows.append((entry.id, entry.message))
            elif isinstance(entry, CompactionEntry):
                message_rows = _apply_compaction(
                    message_rows,
                    entry,
                    entries_before=replay_entries[:entry_index],
                )
        return cls(
            session_info=session_info,
            messages=[message for _, message in message_rows],
            context_entry_ids=[entry_id for entry_id, _ in message_rows],
        )


def _apply_compaction( # 遇到压缩entry时的session回放
    message_rows: list[tuple[str, AgentMessage]], # 回放到当前entry时，已经积累的上下文消息
    entry: CompactionEntry,
    *,
    entries_before: list[SessionEntry],
) -> list[tuple[str, AgentMessage]]:
    """按历史记录中的边界保留当前有效消息，不恢复已经被替代的消息。"""
    summary_row = (
        entry.id,
        UserMessage(content=f"Previous conversation summary:\n{entry.summary}"),
    )
    first_kept_index = next(
        (
            index
            for index, prior_entry in enumerate(entries_before)
            if isinstance(prior_entry, (MessageEntry, CompactionEntry))
            and prior_entry.id == entry.first_kept_entry_id
        ),
        None,
    )
    if first_kept_index is None: # 不存在first_kept_index时，只保留压缩消息，其以前的都丢弃
        return [summary_row]

    retained_by_id = dict(message_rows)
    retained = [
        (prior_entry.id, retained_by_id[prior_entry.id])
        for prior_entry in entries_before[first_kept_index:]
        if isinstance(prior_entry, (MessageEntry, CompactionEntry))
        and prior_entry.id in retained_by_id # 如果前面已经压缩过一次了，kept index之后的有些消息已经从当前上下文中被移除，这里需要跳过这些消息。也就是说需要确保当前压缩后返回的消息上下文是继承了上次压缩结果的，不会把多余的消息恢复进上下文。
    ]
    return [summary_row, *retained]
