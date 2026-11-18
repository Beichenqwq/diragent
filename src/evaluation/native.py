"""Django and Sphinx evaluation using official tests and grader in native Linux."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shlex
import signal
import subprocess
from pathlib import Path

from evaluation.runner import git, write_json
from evaluation.swebench import HARNESS_VERSION, _fetch_instance


def adapt_eval_commands(commands: list[str]) -> list[str]:
    """Replace only container environment setup; keep official test/reset/apply commands."""
    omitted = {"source /opt/miniconda3/bin/activate", "conda activate testbed"}
    return [command for command in commands if command not in omitted
            and not command.startswith("git config --global --add safe.directory ")]


def prepare_grader_log(raw_log: Path, grader_log: Path, *, normalize_unittest: bool = False) -> None:
    """Preserve subprocess bytes and provide UTF-8 text to the official parser."""
    text = raw_log.read_bytes().decode("utf-8", errors="replace")
    if normalize_unittest:
        # unittest subtest failures leave the verbose header unfinished. The next
        # header shares that line, while the full failed name appears in FAIL:/ERROR:.
        text = re.sub(r"(test\S+ \([^)]+\) \.\.\. )(?=test\S+ \([^)]+\) \.\.\. )", r"\1\n", text)
        text = re.sub(r"(?m)^(FAIL|ERROR): (\S+ \([^)]+\))([^\n]*)$",
                      lambda match: match.group(0) + "\n" + match.group(2) + " ... " + match.group(1), text)
    grader_log.write_text(text, encoding="utf-8")


def _run_script(script: Path, log_path: Path, *, env: dict, timeout: int) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(["bash", str(script)], env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def grade_native_python(
    output_dir: Path, *, cwd: Path, task_python: Path, dataset: str,
    timeout_seconds: int = 300,
    instance_metadata: dict | None = None,
) -> dict:
    """Run negative control and model patch in separate clean clones, after agent execution."""
    output_dir, cwd, task_python = output_dir.resolve(), cwd.resolve(), task_python.absolute()
    attempt = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    result = {
        "instance_id": attempt["instance_id"], "status": "running", "resolved": None,
        "backend": "swebench.harness.grading.get_eval_report",
        "environment": "native_linux", "official_docker_environment": False,
        "harness_version": HARNESS_VERSION,
    }
    native_dir = output_dir / "native-evaluation"
    native_dir.mkdir(exist_ok=False)
    try:
        if platform.system() != "Linux" or attempt["repo"] not in ("django/django", "sphinx-doc/sphinx"):
            raise ValueError("This native adapter supports Django and Sphinx on Linux only")
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        if importlib.metadata.version("swebench") != HARNESS_VERSION:
            raise RuntimeError(f"Install swebench=={HARNESS_VERSION}")
        prediction = json.loads((output_dir / "predictions.jsonl").read_text(encoding="utf-8"))
        if prediction["instance_id"] != attempt["instance_id"]:
            raise ValueError("Prediction ID does not match the attempt")
        if not prediction["model_patch"]:
            raise ValueError("Empty model patch; no positive evaluation can run")
        if git(cwd, "rev-parse", "HEAD").strip() != attempt["base_commit"]:
            raise ValueError("Source repository HEAD does not match base commit")
        # Fetch test metadata only after the run, and never retain the reference solution.
        raw = instance_metadata if instance_metadata is not None else _fetch_instance(dataset, attempt["instance_id"])
        keys = ("instance_id", "repo", "base_commit", "version", "problem_statement",
                "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS")
        instance = {key: raw[key] for key in keys}
        if "environment_setup_commit" in raw:
            instance["environment_setup_commit"] = raw["environment_setup_commit"]
        if any(instance[key] != attempt[key] for key in ("instance_id", "repo", "base_commit")):
            raise ValueError("Judging metadata does not match the executed task")
        write_json(native_dir / "judging_metadata.json", instance)
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
        from swebench.harness.grading import get_eval_report, get_logs_eval
        from swebench.harness.test_spec.test_spec import make_test_spec
        from swebench.harness.test_spec.python import make_eval_script_list_py

        repo_spec = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]
        actual_python = subprocess.check_output(
            [str(task_python), "-c", "import sys; print('%s.%s' % sys.version_info[:2])"],
            text=True,
        ).strip()
        result["expected_task_python"] = repo_spec["python"]
        result["actual_task_python"] = actual_python
        if actual_python != repo_spec["python"]:
            raise RuntimeError(f"Task Python must be {repo_spec['python']}; got {actual_python}")
        spec = make_test_spec(instance)
        report_paths = {}
        for label in ("baseline", "model"):
            repo = native_dir / label
            subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(cwd), str(repo)], check=True)
            git(repo, "checkout", "--detach", attempt["base_commit"])
            # The official image applies these public repository setup commands before evaluation.
            if attempt["repo"] == "sphinx-doc/sphinx":
                for command in repo_spec.get("pre_install", []):
                    if not command.startswith("apt-get "):
                        subprocess.run(["bash", "-ec", command], cwd=repo, check=True)
            current_prediction = dict(prediction)
            if label == "model":
                git(repo, "apply", "--check", str(output_dir / "patch.diff"))
                git(repo, "apply", str(output_dir / "patch.diff"))
            else:
                current_prediction["model_patch"] = ""
            commands = make_eval_script_list_py(
                instance, repo_spec,
                "testbed", shlex.quote(str(repo)), attempt["base_commit"], instance["test_patch"],
            )
            script = native_dir / f"{label}-eval.sh"
            script.write_text("#!/bin/bash\nset -uxo pipefail\n" +
                              "\n".join(adapt_eval_commands(commands)) + "\n", encoding="utf-8")
            env = dict(os.environ, PATH=str(task_python.parent) + os.pathsep + os.environ.get("PATH", ""),
                       PYTHONPATH=str(repo), PIP_INDEX_URL="https://pypi.org/simple")
            if attempt["repo"] == "sphinx-doc/sphinx":
                # tox-current-env otherwise creates interpreter symlinks outside
                # the venv, which lose its site-packages when commands use python -m.
                env["TOX_OVERRIDE"] = f"testenv.env_dir={task_python.parent.parent}"
                env["PYTEST_ADDOPTS"] = "--color=no"
                # Linkcheck tests start local HTTP servers; keep these requests local.
                env["NO_PROXY"] = ",".join(filter(None, [env.get("NO_PROXY", ""), "localhost", "127.0.0.1", "::1"]))
                env["no_proxy"] = env["NO_PROXY"]
            log = native_dir / f"{label}-test-output.txt"
            exit_code = _run_script(script, log, env=env, timeout=timeout_seconds)
            grader_log = native_dir / f"{label}-grader-log.txt"
            prepare_grader_log(log, grader_log, normalize_unittest=attempt["repo"] == "django/django")
            report = get_eval_report(spec, current_prediction, str(grader_log), include_tests_status=True)
            report_path = native_dir / f"{label}-report.json"
            write_json(report_path, report)
            report_paths[label] = report_path
            result[f"{label}_script_exit_code"] = exit_code
        baseline = json.loads(report_paths["baseline"].read_text(encoding="utf-8"))[attempt["instance_id"]]
        baseline_tests = baseline.get("tests_status", {})
        baseline_statuses, _ = get_logs_eval(spec, str(native_dir / "baseline-grader-log.txt"))
        expected_failures = instance["FAIL_TO_PASS"]
        if isinstance(expected_failures, str):
            expected_failures = json.loads(expected_failures)
        if not expected_failures or any(test not in baseline_statuses for test in expected_failures):
            raise RuntimeError("Negative control is missing expected target test results")
        if (baseline["resolved"] or not baseline_tests.get("FAIL_TO_PASS", {}).get("failure")
                or baseline_tests.get("PASS_TO_PASS", {}).get("failure")):
            raise RuntimeError("Negative control did not reproduce the original failure with passing regression tests")
        model_report = json.loads(report_paths["model"].read_text(encoding="utf-8"))[attempt["instance_id"]]
        if not model_report["patch_successfully_applied"] or "tests_status" not in model_report:
            raise RuntimeError("Official grader could not parse a complete model test log")
        result["status"] = "completed"
        result["resolved"] = model_report["resolved"]
        result["negative_control_valid"] = True
        result["tests_status"] = model_report["tests_status"]
        result["report_path"] = str(report_paths["model"].relative_to(output_dir))
        result["task_python_version"] = subprocess.check_output(
            [str(task_python), "--version"], text=True, stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        result["status"] = "grading_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    write_json(output_dir / "native_result.json", result)
    return result


# Keep the original single-task entry point compatible.
grade_native_django = grade_native_python
