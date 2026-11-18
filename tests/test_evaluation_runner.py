"""Task execution artifacts, budgets, patch completeness, and grading boundaries."""

import asyncio
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from ai.events import ProviderErrorEvent, ProviderResponseEndEvent
from agent.messages import AssistantMessage, ToolCall, Usage
from evaluation.runner import CodingTask, export_patch, git, run_task, write_json
from evaluation.swebench import grade


class EvaluationRunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch("coding.session.extract_memory_candidates", return_value=[]))
        root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        (self.repo / "file.txt").write_text("old\n", encoding="utf-8")
        git(self.repo, "add", ".")
        git(self.repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "initial")
        self.task = CodingTask("example__repo-1", "example/repo",
                               git(self.repo, "rev-parse", "HEAD").strip(), "Fix the file")
        self.output = self.root / "attempt"

    def test_task_drops_hidden_answers(self):
        task = CodingTask.from_dict(dict(self.task.to_dict(), patch="secret answer",
                                         test_patch="hidden tests", FAIL_TO_PASS=["hidden"]))
        self.assertEqual(task.to_dict(), self.task.to_dict())

    def test_patch_contains_new_and_staged_files_without_changing_real_index(self):
        self.output.mkdir()
        (self.repo / "file.txt").write_text("new\n", encoding="utf-8")
        git(self.repo, "add", "file.txt")
        (self.repo / "added.py").write_text("print('new')\n", encoding="utf-8")
        (self.repo / ".codex_tmp").mkdir()
        (self.repo / ".codex_tmp" / "log.txt").write_text("tool output", encoding="utf-8")
        before = git(self.repo, "diff", "--cached")
        result = export_patch(self.repo, self.task.base_commit, self.output)
        self.assertIn("+new", result)
        self.assertIn("added.py", result)
        self.assertNotIn("tool output", result)
        self.assertEqual(git(self.repo, "diff", "--cached"), before)

    def test_non_utf8_generated_file_round_trips_without_changing_index(self):
        self.output.mkdir()
        git(self.repo, "config", "core.autocrlf", "false")
        inventory = b"# Sphinx inventory version 2\ncompressed-data:\xd0\xff\n"
        (self.repo / "objects.inv").write_bytes(inventory)
        (self.repo / "file.txt").write_text("fixed\n", encoding="utf-8")
        before = (self.repo / ".git" / "index").read_bytes()
        model_patch = export_patch(self.repo, self.task.base_commit, self.output)
        self.assertIn("GIT binary patch", model_patch)
        self.assertEqual((self.repo / ".git" / "index").read_bytes(), before)
        clone = self.root / "clean-clone"
        subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "-q", str(self.repo), str(clone)], check=True)
        git(clone, "config", "core.autocrlf", "false")
        patch_path = self.output / "patch.diff"
        patch_path.write_text(model_patch, encoding="utf-8", newline="\n")
        git(clone, "apply", str(patch_path))
        self.assertEqual((clone / "objects.inv").read_bytes(), inventory)
        self.assertEqual((clone / "file.txt").read_text(encoding="utf-8"), "fixed\n")

    async def test_real_coding_session_executes_tool_saves_history_and_prediction(self):
        class Provider:
            calls = 0

            async def stream_response(inner, **kwargs):
                inner.calls += 1
                self.assertIn("Network access is allowed", kwargs["system"])
                self.assertIn("do not search for or retrieve existing solutions to this task", kwargs["system"])
                if inner.calls == 1:
                    message = AssistantMessage(content="", tool_calls=[ToolCall(
                        id="write-1", name="write", arguments={"path": "file.txt", "content": "fixed\n"},
                    )])
                else:
                    self.assertEqual(kwargs["messages"][-1].role, "tool")
                    message = AssistantMessage(content="Finished", usage=Usage(
                        input_tokens=5, output_tokens=2, total_tokens=7,
                    ))
                yield ProviderResponseEndEvent(message=message)

        result = await run_task(self.task, provider=Provider(), model="fake",
                                cwd=self.repo, output_dir=self.output)
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["resolved"])
        self.assertEqual(result["model_requests"], 2)
        self.assertEqual(result["usage"]["total_tokens"], 7)
        self.assertFalse(result["usage_complete"])
        prediction = json.loads((self.output / "predictions.jsonl").read_text(encoding="utf-8"))
        self.assertIn("+fixed", prediction["model_patch"])
        entries = [json.loads(line) for line in
                   (self.output / "session.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(entries), 5)
        self.assertTrue((self.output / "events.jsonl").is_file())
        events = [json.loads(line) for line in
                  (self.output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(events[0]["type"], "agent_start")
        self.assertFalse(json.loads((self.output / "evaluation_rules.json").read_text(encoding="utf-8"))["network_enforced"])
        self.assertEqual(json.loads((self.output / "network-audit.json").read_text(encoding="utf-8"))["status"], "unreviewed")

    async def test_provider_error_is_recorded_and_credentials_redacted(self):
        class Provider:
            async def stream_response(inner, **kwargs):
                yield ProviderErrorEvent(message="failed key=private-key")

        result = await run_task(self.task, provider=Provider(), model="fake", cwd=self.repo,
                                output_dir=self.output, secrets=("private-key",))
        self.assertEqual(result["status"], "agent_error")
        self.assertIsNone(result["resolved"])
        for file in self.output.iterdir():
            self.assertNotIn("private-key", file.read_text(encoding="utf-8"))

    async def test_global_model_request_budget_terminates_tool_loop(self):
        class Provider:
            async def stream_response(inner, **kwargs):
                yield ProviderResponseEndEvent(message=AssistantMessage(content="", tool_calls=[
                    ToolCall(id="call", name="read", arguments={"path": "file.txt"}),
                ]))

        result = await run_task(self.task, provider=Provider(), model="fake", cwd=self.repo,
                                output_dir=self.output, max_model_requests=1)
        self.assertEqual(result["status"], "model_request_limit")
        self.assertEqual(result["model_requests"], 1)

    async def test_default_budgets_allow_more_than_twenty_four_requests(self):
        class Provider:
            calls = 0

            async def stream_response(inner, **kwargs):
                inner.calls += 1
                if inner.calls <= 25:
                    message = AssistantMessage(content="", tool_calls=[ToolCall(
                        id=f"call-{inner.calls}", name="read", arguments={"path": "file.txt"},
                    )])
                else:
                    message = AssistantMessage(content="Finished")
                yield ProviderResponseEndEvent(message=message)

        result = await run_task(self.task, provider=Provider(), model="fake",
                                cwd=self.repo, output_dir=self.output)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model_requests"], 26)
        self.assertEqual(result["budgets"], {
            "max_turns": 1_000_000, "max_model_requests": 1_000_000,
            "timeout_seconds": 600,
        })

    async def test_timeout_preserves_diagnostics_and_exported_prediction(self):
        class Provider:
            async def stream_response(inner, **kwargs):
                await asyncio.sleep(10)
                yield ProviderResponseEndEvent(message=AssistantMessage(content="never"))

        result = await run_task(self.task, provider=Provider(), model="fake", cwd=self.repo,
                                output_dir=self.output, timeout_seconds=.01)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue((self.output / "messages_snapshot.json").is_file())
        self.assertTrue((self.output / "predictions.jsonl").is_file())

    async def test_coding_session_respects_configured_turn_budget(self):
        class Provider:
            async def stream_response(inner, **kwargs):
                yield ProviderResponseEndEvent(message=AssistantMessage(content="", tool_calls=[
                    ToolCall(id="call", name="read", arguments={"path": "file.txt"}),
                ]))

        result = await run_task(self.task, provider=Provider(), model="fake", cwd=self.repo,
                                output_dir=self.output, max_turns=1)
        self.assertEqual(result["status"], "agent_error")
        self.assertEqual(result["model_requests"], 1)

    async def test_cancellation_records_snapshot_and_propagates_to_caller(self):
        started = asyncio.Event()

        class Provider:
            async def stream_response(inner, **kwargs):
                started.set()
                await asyncio.Event().wait()
                yield ProviderResponseEndEvent(message=AssistantMessage(content="never"))

        execution = asyncio.create_task(run_task(
            self.task, provider=Provider(), model="fake", cwd=self.repo, output_dir=self.output,
        ))
        await started.wait()
        execution.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await execution
        result = json.loads((self.output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue((self.output / "messages_snapshot.json").is_file())

    async def test_dirty_repo_is_rejected_before_provider_call(self):
        (self.repo / "file.txt").write_text("already modified", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "clean"):
            await run_task(self.task, provider=None, model="fake", cwd=self.repo,
                           output_dir=self.output)
        self.assertFalse(self.output.exists())

    def test_no_docker_is_environment_error_not_unresolved(self):
        self.output.mkdir()
        write_json(self.output / "result.json", {"instance_id": self.task.instance_id,
                                                 "grading_status": "not_run", "resolved": None})
        write_json(self.output / "predictions.jsonl", {"instance_id": self.task.instance_id,
                                                       "model_patch": "patch"})
        with patch("evaluation.swebench.shutil.which", return_value=None):
            result = grade(self.output)
        self.assertEqual(result["grading_status"], "environment_error")
        self.assertIsNone(result["resolved"])

    def test_official_report_is_used_even_when_harness_exit_code_is_zero(self):
        self.output.mkdir()
        write_json(self.output / "result.json", {"instance_id": self.task.instance_id,
                                                 "grading_status": "not_run", "resolved": None})
        prediction = {"instance_id": self.task.instance_id, "model_name_or_path": "my-agent/fake",
                      "model_patch": "patch"}
        write_json(self.output / "predictions.jsonl", prediction)
        report = self.output / "logs/evaluation/single-test-id/my-agent__fake" / self.task.instance_id
        report.mkdir(parents=True)
        write_json(report / "report.json", {self.task.instance_id: {"resolved": False}})
        with patch("evaluation.swebench.shutil.which", return_value="docker"), \
                patch("evaluation.swebench.uuid.uuid4", return_value=SimpleNamespace(hex="test-id")), \
                patch("evaluation.swebench.subprocess.check_output", return_value="4.1.0\n"), \
                patch("evaluation.swebench.subprocess.run", return_value=subprocess.CompletedProcess([], 0)):
            result = grade(self.output)
        self.assertEqual(result["grading_status"], "completed")
        self.assertFalse(result["resolved"])

    def test_empty_patch_is_not_an_environment_failure(self):
        self.output.mkdir()
        write_json(self.output / "result.json", {"instance_id": self.task.instance_id,
                                                 "grading_status": "not_run", "resolved": None})
        write_json(self.output / "predictions.jsonl", {"instance_id": self.task.instance_id,
                                                       "model_patch": ""})
        result = grade(self.output)
        self.assertEqual(result["grading_status"], "empty_patch")
        self.assertIsNone(result["resolved"])

    def test_no_official_report_is_grading_error_not_environment_failure(self):
        self.output.mkdir()
        write_json(self.output / "result.json", {"instance_id": self.task.instance_id,
                                                 "grading_status": "not_run", "resolved": None})
        write_json(self.output / "predictions.jsonl", {"instance_id": self.task.instance_id,
                                                       "model_name_or_path": "fake", "model_patch": "patch"})
        with patch("evaluation.swebench.shutil.which", return_value="docker"), \
                patch("evaluation.swebench.subprocess.check_output", return_value="4.1.0\n"), \
                patch("evaluation.swebench.subprocess.run", return_value=subprocess.CompletedProcess([], 0)):
            result = grade(self.output)
        self.assertEqual(result["grading_status"], "grading_error")
        self.assertIsNone(result["resolved"])
