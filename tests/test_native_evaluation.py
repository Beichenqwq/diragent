"""Native evaluation preserves test commands and declares its environment limits."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.native import adapt_eval_commands, grade_native_django, prepare_grader_log
from evaluation.runner import write_json


class NativeEvaluationTests(unittest.TestCase):
    def test_unittest_subtest_failure_keeps_full_failed_and_next_passed_names(self):
        root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            raw, decoded = Path(temporary) / "raw.log", Path(temporary) / "grader.log"
            content = ("test_first (suite.Case) ... test_second (suite.Case) ... ok\n"
                       "FAIL: test_first (suite.Case) (value=1)\nAssertionError\n")
            raw.write_text(content, encoding="utf-8")
            prepare_grader_log(raw, decoded, normalize_unittest=True)
            self.assertEqual(raw.read_text(encoding="utf-8"), content)
            normalized = decoded.read_text(encoding="utf-8")
            self.assertIn("test_first (suite.Case) ... \ntest_second (suite.Case) ... ok", normalized)
            self.assertIn("test_first (suite.Case) ... FAIL", normalized)

    def test_grader_log_preserves_test_output_and_keeps_raw_bytes(self):
        root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            raw, decoded = Path(temporary) / "raw.log", Path(temporary) / "grader.log"
            content = b"git show: \xd8\n>>>>> Start Test Output\ntest_case ... ok\n>>>>> End Test Output\n"
            raw.write_bytes(content)
            prepare_grader_log(raw, decoded)
            self.assertEqual(raw.read_bytes(), content)
            self.assertEqual(decoded.read_text(encoding="utf-8"), content.decode("utf-8", errors="replace"))

    def test_only_container_setup_is_removed_not_test_or_patch_content(self):
        apply_patch = "git apply - <<'EOF'\n+source /opt/miniconda3/bin/activate\nEOF"
        tests = "./tests/runtests.py --verbosity 2 --settings=test_sqlite auth_tests.test_forms"
        commands = ["source /opt/miniconda3/bin/activate", "conda activate testbed",
                    "cd /task", "git config --global --add safe.directory /task",
                    apply_patch, ": '>>>>> Start Test Output'", tests,
                    ": '>>>>> End Test Output'", "git checkout base tests/test.py"]
        self.assertEqual(adapt_eval_commands(commands), commands[2:3] + commands[4:])
        self.assertIn(apply_patch, adapt_eval_commands(commands))

    def test_unsupported_repository_is_recorded_without_claiming_docker_or_resolution(self):
        root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            output = Path(temporary)
            write_json(output / "result.json", {"instance_id": "other__repo-1", "repo": "other/repo"})
            with patch("evaluation.native.platform.system", return_value="Linux"):
                result = grade_native_django(output, cwd=output, task_python=output / "python",
                                             dataset="unused")
            self.assertEqual(result["status"], "grading_error")
            self.assertIsNone(result["resolved"])
            self.assertFalse(result["official_docker_environment"])
            self.assertTrue((output / "native_result.json").is_file())
