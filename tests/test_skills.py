"""项目技能的查找范围、内容读取和元信息校验。"""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from coding import ResourceError, Skill, expand_skill, format_skills_for_prompt, load_skills


class SkillLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(temp_dir.cleanup)
        self.cwd = Path(temp_dir.name)
        self.skills_dir = self.cwd / ".tau" / "skills"

    def write_skill(self, name: str, text: str) -> Path:
        path = self.skills_dir / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        return path

    def test_absent_directory_returns_empty_without_creating_it(self) -> None:
        self.assertEqual(load_skills(self.cwd), [])
        self.assertFalse(self.skills_dir.exists())

    def test_multiple_skills_keep_order_body_and_directory_names(self) -> None:
        body = '\r\n# 操作步骤\r\n\r\n```python\r\nprint("你好")\r\n```\r\n'
        testing_path = self.write_skill("testing", "---\ndescription: 编写测试\n---\n测试正文")
        review_path = self.write_skill(
            "review",
            '---\r\nname: ignored-name\r\ndescription: "  审查代码  "\r\n---\r\n' + body,
        )
        skills = load_skills(self.cwd)
        self.assertEqual(skills, [
            Skill(name="review", path=review_path, description="审查代码", content=body),
            Skill(name="testing", path=testing_path, description="编写测试", content="测试正文"),
        ])

    def test_only_direct_skill_subdirectories_are_loaded(self) -> None:
        self.write_skill("valid", "---\ndescription: 有效技能\n---\n正文")
        (self.skills_dir / "flat.md").write_text("not a skill", encoding="utf-8")
        (self.skills_dir / "missing").mkdir()
        nested = self.skills_dir / "nested" / "child"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("not a direct skill", encoding="utf-8")
        self.assertEqual([skill.name for skill in load_skills(self.cwd)], ["valid"])

    def test_missing_or_blank_description_raises_with_path(self) -> None:
        for frontmatter in ("", "name: review\n", "description:\n", 'description: "  "\n'):
            with self.subTest(frontmatter=frontmatter):
                path = self.write_skill("review", "---\n" + frontmatter + "---\n# 有意义的标题\n")
                with self.assertRaises(ResourceError) as caught:
                    load_skills(self.cwd)
                self.assertIn(str(path), str(caught.exception))
                self.assertIn("non-empty description", str(caught.exception))

    def test_invalid_frontmatter_raises_with_path_and_preserves_cause(self) -> None:
        for text in ("# 没有 frontmatter", "---\ndescription: 审查\n", "---\ninvalid\n---\n"):
            with self.subTest(text=text):
                path = self.write_skill("review", text)
                with self.assertRaises(ResourceError) as caught:
                    load_skills(self.cwd)
                self.assertIn(str(path), str(caught.exception))
                self.assertIsInstance(caught.exception.__cause__, ResourceError)

    def test_invalid_utf8_raises_with_path(self) -> None:
        path = self.write_skill("review", "")
        path.write_bytes(b"\xff\xfe")
        with self.assertRaises(ResourceError) as caught:
            load_skills(self.cwd)
        self.assertIn(str(path), str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, UnicodeDecodeError)


class SkillExpansionTests(unittest.TestCase):
    def test_expansion_keeps_body_and_request_without_rendering_placeholders(self) -> None:
        body = "# 审查步骤\n\n检查 {{ target }} 和 $1。\n"
        request = "检查 loop.py，重点关注取消处理。"
        skill = Skill(name="review", path=Path("skills/review/SKILL.md"), description="审查", content=body)
        text = expand_skill(skill, request)
        self.assertIn('<skill name="review" location="skills', text)
        self.assertIn(f"References are relative to {skill.path.parent}.", text)
        self.assertIn(body, text)
        self.assertTrue(text.endswith("</skill>\n\n" + request))

    def test_attribute_values_are_escaped(self) -> None:
        skill = Skill(name='review"&', path=Path("skills/a&b/SKILL.md"), description="审查", content="正文")
        text = expand_skill(skill, "任务")
        self.assertIn('name="review&quot;&amp;"', text)
        self.assertIn("a&amp;b", text)


class SkillIndexTests(unittest.TestCase):
    def test_empty_index_has_no_instructions_or_xml(self) -> None:
        self.assertEqual(format_skills_for_prompt([]), "")
        self.assertEqual(format_skills_for_prompt(()), "")

    def test_index_sorted_without_changing_inputs_or_reading_body(self) -> None:
        skills = [
            Skill("testing", Path("skills/testing/SKILL.md"), "Write tests", "TESTING_BODY_SENTINEL"),
            Skill("review", Path("skills/review/SKILL.md"), "Review code", "REVIEW_BODY_SENTINEL"),
        ]
        with patch.object(Path, "open", side_effect=AssertionError("Unexpected skill file read")):
            text = format_skills_for_prompt(skills)
        root = ET.fromstring(text[text.index("<available_skills>"):])

        self.assertEqual([item.findtext("name") for item in root], ["review", "testing"])
        self.assertEqual([item.findtext("description") for item in root], ["Review code", "Write tests"])
        self.assertEqual([item.findtext("location") for item in root],
                         [str(skills[1].path), str(skills[0].path)])
        self.assertEqual([skill.name for skill in skills], ["testing", "review"])
        self.assertNotIn("REVIEW_BODY_SENTINEL", text)
        self.assertNotIn("TESTING_BODY_SENTINEL", text)
        self.assertIn("Use the read tool to load a skill's file", text)
        self.assertIn("resolve it against the skill directory", text)
        self.assertTrue(text.startswith("\n\nThe following skills"))

    def test_xml_fields_roundtrip_special_characters(self) -> None:
        skill = Skill(
            name='review<&"', path=Path('skills/a&b<draft>"/SKILL.md'),
            description='Check <tags> & "quotes"', content="Excluded body",
        )
        text = format_skills_for_prompt((skill,))
        root = ET.fromstring(text[text.index("<available_skills>"):])
        item = root.find("skill")
        self.assertEqual(item.findtext("name"), skill.name)
        self.assertEqual(item.findtext("description"), skill.description)
        self.assertEqual(item.findtext("location"), str(skill.path))
        self.assertEqual(len(root), 1)
