"""本地 JSONL 会话记录的追加写入和读取。"""

from pathlib import Path

from agent.session.entries import SessionEntry
from agent.session.jsonl import entries_from_json_lines, entry_to_json_line


class JsonlSessionStorage:
    """保存目标文件路径，提供同步的磁盘读写操作。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path) # 待存入信息的文件路径

    def append(self, entry: SessionEntry) -> None:
        """向文件末尾追加一条记录，必要时创建父目录。"""
        line = entry_to_json_line(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(line)

    def read_all(self) -> list[SessionEntry]:
        """按文件顺序读取全部记录；文件不存在时返回空列表。"""
        try:
            with self.path.open("r", encoding="utf-8") as file:
                return entries_from_json_lines(file)
        except FileNotFoundError:
            return []
