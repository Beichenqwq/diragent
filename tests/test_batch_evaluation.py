"""Batch resume, answer filtering, and environment failures remain distinguishable."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.batch import execute_one, fetch_judging_metadata, grade_one, summarize
from evaluation.runner import write_json


class BatchEvaluationTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.batch = Path(self.temp.name)

    def test_metadata_fetch_discards_reference_solution(self):
        row = dict(instance_id="task-1", repo="django/django", base_commit="abc", version="3.1",
                   problem_statement="issue", test_patch="tests", FAIL_TO_PASS=["f"], PASS_TO_PASS=["p"],
                   patch="reference solution")
        response = io.BytesIO(json.dumps({"rows": [{"row": row}], "num_rows_total": 1}).encode())
        with patch("evaluation.batch.urllib.request.urlopen", return_value=response):
            metadata = fetch_judging_metadata({"task-1"})
        self.assertNotIn("patch", metadata["task-1"])
        self.assertEqual(metadata["task-1"]["test_patch"], "tests")

    def test_summary_excludes_environment_errors_from_valid_results(self):
        rows = [{"instance_id": str(i), "repo": "django/django"} for i in range(3)]
        for row, status in zip(rows, ({"status": "completed", "resolved": True},
                                     {"status": "empty_patch", "resolved": False},
                                     {"status": "environment_error", "resolved": None}), strict=True):
            case = self.batch / "cases" / row["instance_id"]
            case.mkdir(parents=True)
            write_json(case / "grading-state.json", status)
        result = summarize(self.batch, rows)
        self.assertEqual((result["graded"], result["valid_results"], result["resolved"]), (3, 2, 1))
        self.assertFalse(result["official_docker_environment"])
        self.assertIsNone(result["resolved_rate"])
        self.assertEqual(result["resolved_fraction_lower_bound"], 1 / 3)

    def test_resume_does_not_repeat_a_completed_model_execution(self):
        row = {"instance_id": "task-1", "repo": "django/django"}
        case = self.batch / "cases/task-1"
        (case / "run").mkdir(parents=True)
        write_json(case / "run/result.json", {"status": "completed", "model_requests": 3,
                                              "usage": {}, "patch_bytes": 10})
        write_json(case / "execution-state.json", {"phase": "executing"})
        with patch("evaluation.batch.prepare_repository") as prepare:
            self.assertEqual(execute_one(self.batch, self.batch, row)["phase"], "executed")
        prepare.assert_not_called()

    def test_empty_patch_is_unsolved_without_claiming_tests_were_run(self):
        row = {"instance_id": "task-1", "repo": "django/django"}
        run = self.batch / "cases/task-1/run"
        run.mkdir(parents=True)
        write_json(run / "result.json", {"status": "completed"})
        (run / "patch.diff").write_text("")
        result = grade_one(self.batch, self.batch, row)
        self.assertEqual(result["status"], "empty_patch")
        self.assertFalse(result["resolved"])
        self.assertFalse(result["tests_executed"])
