"""Markdown 元信息分离及错误处理。"""

import unittest

from coding.resources import ResourceError, parse_markdown_resource


class MarkdownResourceTests(unittest.TestCase):
    def test_metadata_and_body_preserve_markdown_and_newlines(self) -> None:
        body = '\r\n# 步骤\r\n\r\n```python\r\nprint("你好")\r\n```\r\n---\r\n'
        text = '---\r\nname: review\r\ndescription: "重点包括：异常处理"\r\n---\r\n' + body
        metadata, content = parse_markdown_resource(text)
        self.assertEqual(metadata, {"name": "review", "description": "重点包括：异常处理"})
        self.assertEqual(content, body)

    def test_comments_blanks_and_colons_in_values(self) -> None:
        metadata, content = parse_markdown_resource(
            "---\n\n# 注释\nurl: https://example.com:8080\ndescription: '审查代码'\n---\n正文"
        )
        self.assertEqual(metadata, {"url": "https://example.com:8080", "description": "审查代码"})
        self.assertEqual(content, "正文")

    def test_missing_delimiters_raise(self) -> None:
        for text in ("", "# 正文\n", "\n---\n---\n", "---\ndescription: review\n", "---\n---suffix\n"):
            with self.subTest(text=text):
                with self.assertRaises(ResourceError):
                    parse_markdown_resource(text)

    def test_invalid_and_duplicate_metadata_raise_with_line_number(self) -> None:
        for text, line in (
            ("---\ninvalid\n---\n", 2),
            ("---\n: value\n---\n", 2),
            ("---\nname: first\nname: second\n---\n", 3),
        ):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ResourceError, f"line {line}"):
                    parse_markdown_resource(text)

    def test_required_skill_fields_are_left_to_loader(self) -> None:
        self.assertEqual(parse_markdown_resource("---\n---"), ({}, ""))
        self.assertEqual(parse_markdown_resource("---\ndescription: \"\"\n---\n"), ({"description": ""}, ""))

    def test_only_matching_outer_quotes_are_removed(self) -> None:
        metadata, _ = parse_markdown_resource("---\ntext: don't strip\n---\n")
        self.assertEqual(metadata["text"], "don't strip")
