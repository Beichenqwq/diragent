"""记忆提取对话的过滤范围和时间顺序。"""

import tempfile
import unittest
from pathlib import Path

from agent.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from coding.memory import build_memory_dialogue, load_memory_index


class MemoryDialogueTests(unittest.TestCase):
    def test_filters_tool_messages_and_empty_text_before_selecting_recent_twelve(self):
        messages = []
        call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
        for index in range(7):
            messages.extend([
                UserMessage(content=f"question {index}"),
                AssistantMessage(content="Reading the project.", tool_calls=[call]),
                ToolResultMessage(tool_call_id=call.id, name=call.name, content="file contents"),
                AssistantMessage(content=f"answer {index}"),
                UserMessage(content=" \n "),
            ])

        expected = []
        for index in range(1, 7):
            expected.extend([f"user: question {index}", f"assistant: answer {index}"])
        self.assertEqual(build_memory_dialogue(messages), "\n".join(expected))

    def test_preserves_multiline_text_and_custom_message_limit(self):
        messages = [
            UserMessage(content="old question"),
            AssistantMessage(content="  First line\nSecond line  "),
            UserMessage(content=" new question "),
        ]
        self.assertEqual(build_memory_dialogue(messages, max_messages=2),
                         "assistant: First line\nSecond line\nuser: new question")

    def test_empty_input_and_nonpositive_limits_produce_empty_dialogue(self):
        self.assertEqual(build_memory_dialogue([]), "")
        for limit in (0, -1):
            self.assertEqual(build_memory_dialogue([UserMessage(content="question")],
                                                  max_messages=limit), "")


class MemoryIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(directory.cleanup)
        self.cwd = Path(directory.name)

    def test_missing_index_returns_empty_without_creating_files(self):
        self.assertEqual(load_memory_index(self.cwd), "")
        self.assertFalse((self.cwd / ".tau").exists())
        (self.cwd / ".tau" / "memory").mkdir(parents=True)
        self.assertEqual(load_memory_index(self.cwd), "")
        self.assertFalse((self.cwd / ".tau" / "memory" / "MEMORY.md").exists())

    def test_reads_utf8_index_without_reading_memory_bodies(self):
        path = self.cwd / ".tau" / "memory" / "MEMORY.md"
        path.parent.mkdir(parents=True)
        content = "- [用户偏好](user-preferences.md) - 结合实际调用流程解释代码\n"
        path.write_text(content, encoding="utf-8")
        # 正文文件并不存在；读取索引不解析链接或加载正文。
        self.assertEqual(load_memory_index(self.cwd), content)
