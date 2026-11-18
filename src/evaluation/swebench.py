"""Single-instance SWE-bench preparation, real-agent execution, and official grading."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from evaluation.runner import (
    DEFAULT_MAX_MODEL_REQUESTS, DEFAULT_MAX_TURNS, CodingTask, run_task, write_json,
)

DEFAULT_DATASET = "MariusHobbhahn/swe-bench-verified-mini"
GRADING_DATASET = "SWE-bench/SWE-bench_Verified"
HARNESS_VERSION = "4.1.0"


def _fetch_instance(dataset: str, instance_id: str) -> dict:
    offset = 0
    while True:
        query = urllib.parse.urlencode({
            "dataset": dataset, "config": "default", "split": "test",
            "offset": offset, "length": 100,
        })
        with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + query,
                                    timeout=60) as response:
            page = json.load(response)
        for item in page["rows"]:
            if item["row"]["instance_id"] == instance_id:
                return item["row"]
        offset += len(page["rows"])
        if not page["rows"] or offset >= page["num_rows_total"]:
            raise ValueError(f"Instance not found: {instance_id}")


def fetch_task(dataset: str, instance_id: str) -> CodingTask:
    """Fetch public rows, retaining only the four fields needed by the agent."""
    return CodingTask.from_dict(_fetch_instance(dataset, instance_id))


def prepare_repository(task: CodingTask, cwd: Path) -> None:
    if cwd.exists():
        raise ValueError("Repository destination already exists; choose a fresh directory")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", task.repo):
        raise ValueError("Invalid GitHub repository name")
    if not re.fullmatch(r"[0-9a-f]{40}", task.base_commit):
        raise ValueError("Invalid base commit")
    cwd.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(cwd)], check=True)
    subprocess.run(["git", "-C", str(cwd), "remote", "add", "origin",
                    f"https://github.com/{task.repo}.git"], check=True)
    subprocess.run(["git", "-C", str(cwd), "-c", "core.autocrlf=false", "fetch",
                    "--depth", "1", "origin", task.base_commit], check=True)
    subprocess.run(["git", "-C", str(cwd), "-c", "core.autocrlf=false", "checkout",
                    "--detach", "FETCH_HEAD"], check=True)
    subprocess.run(["git", "-C", str(cwd), "remote", "remove", "origin"], check=True)


def grade(output_dir: Path, *, python: str = sys.executable,
          dataset: str = GRADING_DATASET, timeout_seconds: int = 900) -> dict:
    """Run the unmodified, pinned official Docker harness and inspect its report."""
    output_dir = output_dir.resolve()
    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("grading_status") not in ("not_run", "environment_error", "grading_error"):
        raise ValueError("This attempt already has a grading result; use a new attempt")
    result["grading_backend"] = "swebench.harness.run_evaluation"
    result["harness_version"] = HARNESS_VERSION
    failure_status = "grading_error"
    try:
        prediction = json.loads((output_dir / "predictions.jsonl").read_text(encoding="utf-8"))
        if prediction["instance_id"] != result["instance_id"]:
            raise ValueError("Prediction instance ID does not match the attempt")
        if not prediction["model_patch"]:
            result["grading_status"] = "empty_patch"
            result["grading_error"] = "Empty prediction; no tests were executed"
            write_json(result_path, result)
            return result
        failure_status = "environment_error"
        if shutil.which("docker") is None:
            raise RuntimeError("Docker CLI is unavailable; official grading requires Docker")
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=30)
        version = subprocess.check_output([
            python, "-c", "import importlib.metadata; print(importlib.metadata.version('swebench'))",
        ], text=True).strip()
        if version != HARNESS_VERSION:
            raise RuntimeError(f"Install swebench=={HARNESS_VERSION} in the grading environment")
        failure_status = "grading_error"
        run_id = "single-" + uuid.uuid4().hex[:12]
        result["grading_run_id"] = run_id
        command = [
            python, "-m", "swebench.harness.run_evaluation", "--dataset_name", dataset,
            "--predictions_path", str(output_dir / "predictions.jsonl"),
            "--instance_ids", result["instance_id"], "--max_workers", "1",
            "--run_id", run_id, "--timeout", str(timeout_seconds),
        ]
        with (output_dir / "grader.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=output_dir, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=timeout_seconds + 600)
        report_path = (output_dir / "logs" / "evaluation" / run_id
                       / prediction["model_name_or_path"].replace("/", "__")
                       / result["instance_id"] / "report.json")
        if not report_path.is_file():
            raise RuntimeError(f"Official harness produced no report (exit {completed.returncode}); see grader.log")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        resolved = report[result["instance_id"]]["resolved"]
        if not isinstance(resolved, bool):
            raise ValueError("Official report resolved field must be a boolean")
        result["resolved"] = resolved
        result["grading_status"] = "completed"
        result["report_path"] = str(report_path.relative_to(output_dir))
        result.pop("grading_error", None)
    except Exception as exc:
        result["grading_status"] = failure_status
        result["grading_error"] = f"{type(exc).__name__}: {exc}"
    write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Save safe task input and clone its base commit")
    prepare.add_argument("--dataset", default=DEFAULT_DATASET)
    prepare.add_argument("--instance-id", required=True)
    prepare.add_argument("--task", type=Path, required=True)
    prepare.add_argument("--cwd", type=Path, required=True)
    execute = commands.add_parser("execute", help="Run our CodingSession on a prepared repository")
    execute.add_argument("--task", type=Path, required=True)
    execute.add_argument("--cwd", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--env-file", type=Path, default=Path(".env"))
    execute.add_argument("--model")
    execute.add_argument("--model-var", default="OPENAI_MODEL")
    execute.add_argument("--api-key-var", default="OPENAI_API_KEY")
    execute.add_argument("--base-url-var", default="OPENAI_BASE_URL")
    execute.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    execute.add_argument("--max-model-requests", type=int, default=DEFAULT_MAX_MODEL_REQUESTS)
    execute.add_argument("--timeout", type=float, default=600)
    execute.add_argument("--no-usage", action="store_true")
    execute.add_argument("--disable-thinking", action="store_true")
    grading = commands.add_parser("grade", help="Grade exported predictions with the official harness")
    grading.add_argument("--output", type=Path, required=True)
    grading.add_argument("--python", default=sys.executable)
    grading.add_argument("--dataset", default=GRADING_DATASET)
    grading.add_argument("--timeout", type=int, default=900)
    native = commands.add_parser("native-grade", help="Django/Sphinx native environment using official grading rules")
    native.add_argument("--output", type=Path, required=True)
    native.add_argument("--cwd", type=Path, required=True)
    native.add_argument("--task-python", type=Path, required=True)
    native.add_argument("--dataset", default=GRADING_DATASET)
    native.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.task.exists():
            parser.error("Task file already exists")
        task = fetch_task(args.dataset, args.instance_id)
        args.task.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.task, task.to_dict())
        prepare_repository(task, args.cwd.resolve())
    elif args.command == "execute":
        from dataclasses import replace
        from dotenv import load_dotenv
        from ai.env import openai_compatible_config_from_env
        from ai.openai_compatible import OpenAICompatibleProvider

        load_dotenv(args.env_file)
        model = args.model or os.getenv(args.model_var)
        if not model:
            parser.error("Specify --model or configure --model-var")
        config = openai_compatible_config_from_env(
            api_key_var=args.api_key_var, base_url_var=args.base_url_var,
        )
        if args.no_usage:
            config = replace(config, include_usage=False)
        if args.disable_thinking:
            config = replace(config, thinking="disabled")
        task = CodingTask.from_dict(json.loads(args.task.read_text(encoding="utf-8")))

        async def execute_task():
            provider = OpenAICompatibleProvider(config)
            try:
                return await run_task(
                    task, provider=provider, model=model, cwd=args.cwd, output_dir=args.output,
                    max_turns=args.max_turns, max_model_requests=args.max_model_requests,
                    timeout_seconds=args.timeout, secrets=(config.api_key,),
                )
            finally:
                await provider.aclose()

        result = asyncio.run(execute_task())
        print(json.dumps(result, ensure_ascii=False))
        if result["status"] != "completed" or "patch_export_error" in result:
            raise SystemExit(1)
    elif args.command == "native-grade":
        from evaluation.native import grade_native_python

        result = grade_native_python(args.output, cwd=args.cwd, task_python=args.task_python,
                                     dataset=args.dataset, timeout_seconds=args.timeout)
        print(json.dumps(result, ensure_ascii=False))
        if result["status"] != "completed":
            raise SystemExit(1)
    else:
        result = grade(args.output, python=args.python, dataset=args.dataset,
                       timeout_seconds=args.timeout)
        print(json.dumps(result, ensure_ascii=False))
        if result["grading_status"] != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
