"""Answer retrieval candidates do not classify ordinary networking as misconduct."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.audit import audit_tool_commands


class EvaluationAuditTests(unittest.TestCase):
    def audit(self, events):
        root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
            return audit_tool_commands(path, target_repo="example/repo")

    def test_failed_answer_lookup_is_retained_for_review(self):
        report = self.audit([
            {"type": "tool_execution_start", "tool_call": {
                "id": "curl-1", "name": "bash", "arguments": {"command": "curl https://api.github.com/repos/example/repo/pulls/123/files"}}},
            {"type": "tool_execution_end", "result": {
                "tool_call_id": "curl-1", "ok": False, "content": "Connection failed"}},
        ])
        self.assertEqual(report["status"], "needs_review")
        self.assertEqual(report["candidate_count"], 1)
        self.assertFalse(report["commands"][0]["tool_ok"])
        self.assertEqual(report["commands"][0]["output_excerpt"], "Connection failed")
        self.assertNotIn("compliant", report)

    def test_unmatched_script_still_has_evidence_and_is_not_certified(self):
        report = self.audit([{"type": "tool_execution_start", "tool_call": {
            "id": "script-1", "name": "bash", "arguments": {"command": "python helper.py"}}}])
        self.assertEqual(report["status"], "unreviewed")
        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(report["commands"][0]["command"], "python helper.py")
        self.assertIsNone(report["commands"][0]["tool_ok"])

    def test_localhost_is_not_an_answer_retrieval_candidate(self):
        report = self.audit([{"type": "tool_execution_start", "tool_call": {
            "id": "local-1", "name": "bash", "arguments": {"command": "curl http://localhost:7777/"}}}])
        self.assertEqual(report["status"], "unreviewed")
        self.assertEqual(report["candidate_count"], 0)
        self.assertNotIn("violation", report["commands"][0])

    def test_documentation_dependencies_and_unrelated_repo_are_not_candidates(self):
        commands = ["curl https://docs.python.org/3/library/urllib.html",
                    "python -m pip install requests", "wget https://pypi.org/simple/",
                    "git clone https://github.com/unrelated/helper.git",
                    "curl https://api.github.com/repos/unrelated/helper/issues/1"]
        report = self.audit([{"type": "tool_execution_start", "tool_call": {
            "id": str(index), "name": "bash", "arguments": {"command": command}}}
            for index, command in enumerate(commands)])
        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(len(report["commands"]), len(commands))

    def test_mirror_patch_cached_patch_extraction_and_target_clone_need_review(self):
        commands = ["curl https://mirror.example/fix.patch",
                    "python -c \"print(data.get('patch', ''))\"",
                    "git clone https://github.com/example/repo.git /tmp/new-version"]
        report = self.audit([{"type": "tool_execution_start", "tool_call": {
            "id": str(index), "name": "bash", "arguments": {"command": command}}}
            for index, command in enumerate(commands)])
        self.assertEqual(report["candidate_count"], 3)
        self.assertEqual(report["scope"], "online_answer_retrieval")
        self.assertEqual(report["target_repo"], "example/repo")
