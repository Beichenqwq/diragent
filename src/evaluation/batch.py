"""Resume a Flash Mini batch; keep execution separate from hidden-test grading."""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from evaluation.runner import CodingTask, git, write_json
from evaluation.swebench import GRADING_DATASET, prepare_repository


def fetch_judging_metadata(ids: set[str]) -> dict[str, dict]:
    """Fetch after execution, discard reference patches before writing anything."""
    found = {}
    offset = 0
    keys = ("instance_id", "repo", "base_commit", "version", "problem_statement",
            "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit")
    while len(found) < len(ids):
        query = urllib.parse.urlencode({"dataset": GRADING_DATASET, "config": "default",
                                       "split": "test", "offset": offset, "length": 100})
        with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + query,
                                    timeout=60) as response:
            page = json.load(response)
        for item in page["rows"]:
            row = item["row"]
            if row["instance_id"] in ids:
                found[row["instance_id"]] = {key: row[key] for key in keys if key in row}
        offset += len(page["rows"])
        if not page["rows"] or offset >= page["num_rows_total"]:
            break
    if found.keys() != ids:
        raise ValueError(f"Official dataset missing IDs: {sorted(ids - found.keys())}")
    return found


def run_command(command: list[str], *, log: Path, env: dict | None = None,
                cwd: Path | None = None, timeout: int = 600) -> None:
    with log.open("ab") as stream:
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"Command exit {completed.returncode}; see {log.name}")


def create_environment(root: Path, destination: Path, version: str, log: Path) -> Path:
    if version == "3.6":
        base = root / "task-env36/bin/python"
        run_command([str(base), "-m", "venv", "--copies", "--without-pip", str(destination)], log=log)
        # This archived Miniconda lacks ensurepip. Seed its existing pip without sharing imports.
        site = destination / "lib/python3.6/site-packages"
        source = root / "task-env36/lib/python3.6/site-packages"
        shutil.copytree(source / "pip", site / "pip")
        for info in source.glob("pip-*.dist-info"):
            shutil.copytree(info, site / info.name)
    else:
        run_command([str(root / "uv"), "venv", "--seed", str(destination), "--python", version], log=log)
    return destination / "bin/python"


def install_repository(root: Path, repo: Path, python: Path, row: dict,
                       log: Path, *, full_requirements: bool) -> None:
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.test_spec.python import get_requirements

    specs = MAP_REPO_VERSION_TO_SPECS[row["repo"]][row["version"]]
    env = dict(os.environ, PATH=str(python.parent) + os.pathsep + os.environ.get("PATH", ""),
               PIP_INDEX_URL="https://pypi.org/simple", PIP_DISABLE_PIP_VERSION_CHECK="1")
    if row["repo"] == "django/django":
        packages = ["setuptools<60", "wheel<0.38"] if specs["python"] == "3.6" and full_requirements else ["setuptools", "wheel"]
        run_command([str(python), "-m", "pip", "install", *packages], log=log, env=env)
        if full_requirements:
            requirements = repo.parent / "requirements-official.txt"
            requirements.write_text(get_requirements(row), encoding="utf-8")
            run_command([str(python), "-m", "pip", "install", "-r", str(requirements)], log=log, env=env)
    else:
        for command in specs.get("pre_install", []):
            if not command.startswith("apt-get "):
                run_command(["bash", "-ec", command], cwd=repo, log=log, env=env)
        run_command([str(python), "-m", "pip", "install", "setuptools<81", "wheel",
                     *specs.get("pip_packages", [])], log=log, env=env)
    run_command(["bash", "-ec", specs["install"]], cwd=repo, log=log, env=env)
    if row["repo"] == "sphinx-doc/sphinx" and int(row["version"].split(".")[0]) < 5:
        # Old Sphinx imports pkg_resources and uses pre-docutils-0.18/pytest-8 APIs.
        docutils = "docutils<0.17" if row["version"].startswith("3.") else "docutils<0.18"
        run_command([str(python), "-m", "pip", "install", "setuptools<81", docutils,
                     "pytest<8", "pytest-cov<6"], log=log, env=env)


def execute_one(root: Path, batch: Path, row: dict) -> dict:
    case = batch / "cases" / row["instance_id"]
    state_path = case / "execution-state.json"
    if (case / "run/result.json").exists():
        state = json.loads(state_path.read_text())
        result = json.loads((case / "run/result.json").read_text())
        state.update(phase="executed", execution_status=result["status"],
                     model_requests=result["model_requests"], usage=result["usage"],
                     patch_bytes=result.get("patch_bytes", 0))
        write_json(state_path, state)
        return state
    if case.exists():
        raise ValueError(f"Unfinished setup exists: {case}; preserve it before retrying")
    case.mkdir(parents=True)
    started = time.time()
    state = {"instance_id": row["instance_id"], "repo": row["repo"], "phase": "preparing"}
    write_json(state_path, state)
    try:
        write_json(case / "task.json", CodingTask.from_dict(row).to_dict())
        prepare_repository(CodingTask.from_dict(row), case / "repo")
        subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(case / "repo"),
                        str(case / "setup-repo")], check=True)
        version = "3.8" if row["repo"] == "django/django" else "3.9"
        python = create_environment(root, case / "agent-env", version, case / "setup.log")
        install_repository(root, case / "setup-repo", python, row, case / "setup.log",
                           full_requirements=False)
        env = dict(os.environ, PATH=str(python.parent) + os.pathsep + os.environ.get("PATH", ""),
                   PYTHONPATH=str(case / "repo"))
        state["phase"] = "executing"
        write_json(state_path, state)
        command = [str(root / "project/.venv/bin/python"), "-m", "evaluation.swebench", "execute",
                   "--task", str(case / "task.json"), "--cwd", str(case / "repo"),
                   "--output", str(case / "run"), "--env-file", str(root / "deepseek.env"),
                   "--model", "deepseek-flash", "--disable-thinking", "--timeout", "600"]
        try:
            run_command(command, log=case / "agent.log", env=env, timeout=660)
        except RuntimeError:
            if not (case / "run/result.json").exists():
                raise
        result = json.loads((case / "run/result.json").read_text())
        state.update(phase="executed", execution_status=result["status"],
                     model_requests=result["model_requests"], usage=result["usage"],
                     patch_bytes=result.get("patch_bytes", 0))
    except Exception as exc:
        state.update(phase="preparation_error", error=f"{type(exc).__name__}: {exc}")
    state["wall_seconds"] = round(time.time() - started, 3)
    write_json(state_path, state)
    return state


def grade_one(root: Path, batch: Path, row: dict) -> dict:
    case = batch / "cases" / row["instance_id"]
    result_path = case / "grading-state.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    state = {"instance_id": row["instance_id"], "status": "running", "resolved": None}
    try:
        if not (case / "run/result.json").exists():
            state["status"] = "preparation_error"
        elif not (case / "run/patch.diff").stat().st_size:
            state["status"] = "empty_patch"
            state["resolved"] = False
            state["tests_executed"] = False
        else:
            from evaluation.native import grade_native_python
            from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

            grade_setup = case / "grading-setup"
            grade_setup.mkdir()
            subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(case / "repo"),
                            str(grade_setup / "repo")], check=True)
            specs = MAP_REPO_VERSION_TO_SPECS[row["repo"]][row["version"]]
            python = create_environment(root, grade_setup / "env", specs["python"], case / "grading-setup.log")
            install_repository(root, grade_setup / "repo", python, row,
                               case / "grading-setup.log", full_requirements=True)
            state = grade_native_python(case / "run", cwd=case / "repo", task_python=python,
                                        dataset=GRADING_DATASET, timeout_seconds=300,
                                        instance_metadata=row)
    except Exception as exc:
        state.update(status="environment_error", error=f"{type(exc).__name__}: {exc}")
    write_json(result_path, state)
    return state


def summarize(batch: Path, rows: list[dict]) -> dict:
    cases = []
    for row in rows:
        case = batch / "cases" / row["instance_id"]
        item = {"instance_id": row["instance_id"], "repo": row["repo"]}
        for key, name in (("execution", "execution-state.json"), ("grading", "grading-state.json")):
            if (case / name).exists():
                item[key] = json.loads((case / name).read_text())
        cases.append(item)
    judged = [r["grading"] for r in cases if "grading" in r]
    executed = [r["execution"] for r in cases if r.get("execution", {}).get("phase") == "executed"]
    valid = [r for r in judged if isinstance(r.get("resolved"), bool)]
    resolved = sum(r["resolved"] for r in valid)
    summary = {"total": len(rows), "executed": len(executed), "graded": len(judged),
               "valid_results": len(valid), "resolved": resolved,
               "resolved_rate": resolved / len(rows) if len(valid) == len(rows) else None,
               "resolved_fraction_lower_bound": resolved / len(rows),
               "execution_status_counts": dict(Counter(r.get("execution_status") for r in executed)),
               "grading_status_counts": dict(Counter(r["status"] for r in judged)),
               "model_requests": sum(r.get("model_requests", 0) for r in executed),
               "usage": {key: sum(r.get("usage", {}).get(key, 0) for r in executed)
                         for key in ("input_tokens", "output_tokens", "total_tokens")},
               "environment": "native_linux", "official_docker_environment": False,
               "model": "deepseek-flash", "thinking": "disabled", "cases": cases}
    temporary = batch / "summary.tmp"
    write_json(temporary, summary)
    temporary.replace(batch / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--phase", choices=("execute", "grade", "summary"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    rows = json.loads(args.inputs.read_text(encoding="utf-8"))
    if len({r["instance_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate task IDs")
    args.batch.mkdir(parents=True, exist_ok=True)
    if args.phase != "summary":
        if args.phase == "grade":
            if any((args.batch / "cases" / r["instance_id"] / "execution-state.json").exists()
                   and json.loads((args.batch / "cases" / r["instance_id"] / "execution-state.json").read_text()).get("phase")
                   in ("preparing", "executing") for r in rows):
                raise RuntimeError("Do not fetch hidden tests while agent execution is active")
            metadata_path = args.batch / "judging-metadata.json"
            metadata = (json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists()
                        else fetch_judging_metadata({r["instance_id"] for r in rows}))
            if set(metadata) != {r["instance_id"] for r in rows}:
                raise ValueError("Judging metadata IDs do not match batch inputs")
            for row in rows:
                if any(metadata[row["instance_id"]][key] != row[key]
                       for key in ("instance_id", "repo", "base_commit")):
                    raise ValueError("Judging metadata does not match executed task")
            write_json(metadata_path, metadata)
            rows = [metadata[r["instance_id"]] for r in rows]
        operation = execute_one if args.phase == "execute" else grade_one
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(operation, args.root.resolve(), args.batch.resolve(), row) for row in rows]
            for future in concurrent.futures.as_completed(futures):
                print(json.dumps(future.result(), ensure_ascii=False), flush=True)
                summarize(args.batch, rows)
    print(json.dumps({k:v for k,v in summarize(args.batch, rows).items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
