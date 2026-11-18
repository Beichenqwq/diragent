"""Run identical SWE-bench tasks through native Claude Code and Codex CLIs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from uuid import uuid4

from evaluation.audit import audit_tool_commands
from evaluation.cli_proxy import MeteredEndpoint
from evaluation.cli_cost import PRICE_SNAPSHOT, estimate_cost
from evaluation.runner import EVALUATION_GUIDELINES, export_patch, write_json
from evaluation.swebench import GRADING_DATASET

CLI_VERSIONS = {"claude-code": "2.1.288", "codex": "0.160.0"}
TOOLS = Path("/opt/swe-harness-tools")
HOMES = Path("/opt/swe-harness-runs")
PROOT = Path("/opt/swe-proot/proot")


def task_prompt(row: dict) -> str:
    return ("Fix the following issue in the current repository. Implement and verify your own "
            "solution, leaving the final changes in the working tree for patch export. "
            "Do not commit the changes.\n\n" + row["problem_statement"])


def native_terminal_status(harness: str, events: Path) -> str | None:
    """Interpret the CLI's own completion event independently of its outer wrapper."""
    terminal = None
    for line in events.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if harness == "codex" and event.get("type") in ("turn.completed", "turn.failed"):
            terminal = "completed" if event["type"] == "turn.completed" else "error"
        elif harness == "claude-code" and event.get("type") == "result":
            terminal = ("completed" if event.get("subtype") == "success" and
                        not event.get("is_error", False) else "error")
    return terminal


def native_calls(records: list[dict]) -> list[dict]:
    """Deduplicate native tool requests across retransmitted conversation history."""
    calls = {}
    for record in records:
        request = record.get("request", {})
        if record.get("protocol") == "anthropic":
            for message in request.get("messages", []):
                content = message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") in ("tool_use", "server_tool_use"):
                            calls[block["id"]] = {"id": block["id"], "name": block["name"],
                                                 "arguments": block.get("input", {})}
            # The last tool invocation may not yet occur in another request.
            blocks = {}
            for event in record.get("response_events", []):
                kind = event.get("type")
                if kind == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") in ("tool_use", "server_tool_use"):
                        blocks[event["index"]] = {"block": block, "json": ""}
                elif kind == "content_block_delta" and event.get("index") in blocks:
                    blocks[event["index"]]["json"] += event.get("delta", {}).get("partial_json", "")
            for item in blocks.values():
                block = item["block"]
                try:
                    args = json.loads(item["json"]) if item["json"] else block.get("input", {})
                except json.JSONDecodeError:
                    args = {"partial_input": item["json"]}
                calls[block["id"]] = {"id": block["id"], "name": block["name"], "arguments": args}
        else:
            items = list(request.get("input", [])) if isinstance(request.get("input"), list) else []
            items += [e.get("item", {}) for e in record.get("response_events", [])
                      if e.get("type") == "response.output_item.done"]
            for item in items:
                if item.get("type") == "function_call":
                    try:
                        args = json.loads(item.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {"partial_input": item.get("arguments")}
                    ident = item.get("call_id", item.get("id"))
                    calls[ident] = {"id": ident, "name": item.get("name"), "arguments": args}
                elif item.get("type") == "custom_tool_call":
                    ident = item.get("call_id", item.get("id"))
                    calls[ident] = {"id": ident, "name": item.get("name"),
                                    "arguments": {"input": item.get("input")}}
    return list(calls.values())


def write_audit(run: Path, records: list[dict], row: dict) -> dict:
    calls = native_calls(records)
    write_json(run / "native-tool-calls.json", calls)
    results = {}
    for record in records:
        request = record.get("request", {})
        if record.get("protocol") == "anthropic":
            for message in request.get("messages", []):
                content = message.get("content", [])
                if not isinstance(content, list):
                    continue
                for block in content:
                    if block.get("type") in ("tool_result", "web_search_tool_result", "web_fetch_tool_result"):
                        ident = block.get("tool_use_id")
                        raw = block.get("content", "")
                        results[ident] = {"tool_call_id": ident, "ok": not block.get("is_error", False),
                                          "content": raw if isinstance(raw, str) else json.dumps(raw)}
        else:
            inputs = request.get("input", [])
            if not isinstance(inputs, list):
                continue
            for item in inputs:
                if item.get("type") in ("function_call_output", "custom_tool_call_output"):
                    ident = item.get("call_id")
                    raw = item.get("output", "")
                    results[ident] = {"tool_call_id": ident, "ok": None,
                                      "content": raw if isinstance(raw, str) else json.dumps(raw)}
    write_json(run / "native-tool-results.json", list(results.values()))
    events = []
    web = []
    for call in calls:
        args = call["arguments"]
        if call["name"] in ("Bash", "bash", "shell_command", "exec_command", "shell"):
            command = args.get("command", args.get("cmd", ""))
            if isinstance(command, list):
                command = " ".join(command)
            events.append({"type": "tool_execution_start", "tool_call":
                           {"id": call["id"], "name": "bash", "arguments": {"command": command}}})
            if call["id"] in results:
                events.append({"type": "tool_execution_end", "result": results[call["id"]]})
        elif call["name"] in ("WebFetch", "WebSearch", "web_search", "web.run"):
            web.append(call)
    events_path = run / "audit-events.jsonl"
    events_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    report = audit_tool_commands(events_path, target_repo=row["repo"])
    report["native_web_calls"] = web
    report["native_tools_count"] = len(calls)
    report["limitations"] += " Also inspect native web calls, assistant text, read scripts and raw CLI logs. "
    write_json(run / "network-audit.json", report)
    return report


def isolated_home(harness: str, index: int, row: dict, source: Path) -> tuple[str, Path]:
    username = f"hs{'cc' if harness == 'claude-code' else 'cx'}{index:03d}"
    home = HOMES / username
    if home.exists():
        raise ValueError(f"Unfinished attempt already exists: {home}; preserve before retrying")
    subprocess.run(["useradd", "--no-create-home", "--home-dir", str(home),
                    "--shell", "/bin/bash", username], check=True)
    home.mkdir(mode=0o700, parents=True)
    source_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if source_head != row["base_commit"]:
        raise ValueError("Prepared source HEAD does not match the task base commit")
    # Transfer only the base snapshot, never objects/history fetched by previous attempts.
    subprocess.run(["git", "clone", "--quiet", "--no-local", "--single-branch", "--no-tags",
                    "--depth", "1", source.as_uri(), str(home / "repo")], check=True)
    subprocess.run(["git", "checkout", "--quiet", "--detach", row["base_commit"]], cwd=home / "repo", check=True)
    subprocess.run(["git", "remote", "remove", "origin"], cwd=home / "repo", check=True)
    return username, home


def cli_command(harness: str, home: Path, endpoint: MeteredEndpoint, root: Path) -> tuple[list[str], dict]:
    binary = str(TOOLS / "node_modules/.bin" / ("claude" if harness == "claude-code" else "codex"))
    rules = "\n".join(EVALUATION_GUIDELINES)
    if harness == "claude-code":
        settings = {"alwaysThinkingEnabled": False, "autoUpdatesChannel": "stable"}
        write_json(home / "settings.json", settings)
        # The CLI retains its own complete system prompt; only the shared evaluation rules are appended.
        command = [binary, "-p", "--verbose", "--output-format", "stream-json",
                   "--model", "deepseek-flash[1m]", "--dangerously-skip-permissions",
                   "--setting-sources", "", "--settings", str(home / "settings.json"),
                   "--append-system-prompt", rules]
        env = {"ANTHROPIC_BASE_URL": endpoint.base_url + "/anthropic",
               "ANTHROPIC_AUTH_TOKEN": endpoint.token,
               "ANTHROPIC_MODEL": "deepseek-flash[1m]",
               "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-flash[1m]",
               "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-flash[1m]",
               "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-flash",
               "ANTHROPIC_DEFAULT_FABLE_MODEL": "deepseek-flash[1m]",
               "CLAUDE_CODE_SUBAGENT_MODEL": "deepseek-flash",
               "MAX_THINKING_TOKENS": "0", "DISABLE_AUTOUPDATER": "1",
               "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
               "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "786432"}
        return command, env
    config = home / ".codex"
    config.mkdir()
    shutil.copy2(root / "flash-models.json", config / "models.json")
    (config / "config.toml").write_text(
        'model = "deepseek-flash"\nmodel_provider = "deepseek"\n'
        'model_reasoning_effort = "none"\nweb_search = "disabled"\n'
        'model_catalog_json = ' + json.dumps(str(config / "models.json")) + '\n'
        'developer_instructions = ' + json.dumps(rules) + '\n'
        '[model_providers.deepseek]\nname = "DeepSeek"\nwire_api = "responses"\n'
        'base_url = ' + json.dumps(endpoint.base_url) + '\n'
        'env_key = "DEEPSEEK_EVAL_TOKEN"\nsupports_websockets = false\n', encoding="utf-8")
    command = [binary, "exec", "--json", "--dangerously-bypass-approvals-and-sandbox",
               "--cd", str(home / "repo"), "--output-last-message", str(home / "answer.txt"), "-"]
    return command, {"DEEPSEEK_EVAL_TOKEN": endpoint.token}


def execute_one(root: Path, output: Path, row: dict, harness: str, index: int, key: str,
                timeout: int, *, prompt_override: str | None = None) -> dict:
    import pwd
    case = output / harness / "cases" / row["instance_id"]
    run = case / "run"
    if (run / "result.json").exists():
        return json.loads((run / "result.json").read_text())
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {k: row[k] for k in ("instance_id", "repo", "base_commit")}
    result.update(harness=harness, cli_version=CLI_VERSIONS[harness], model="deepseek-flash",
                  thinking="disabled", status="preparing", started_at=datetime.now(timezone.utc).isoformat())
    write_json(case / "state.json", result)
    endpoint = MeteredEndpoint(key, run)
    try:
        username, home = isolated_home(harness, index, row, root / "batch-flash-mini50/cases" / row["instance_id"] / "repo")
        command, additions = cli_command(harness, home, endpoint, root)
        subprocess.run(["chown", "-R", f"{username}:{username}", str(home)], check=True)
        repo = home / "repo"
        result["cwd"] = str(repo)
        case.joinpath("repo").symlink_to(repo, target_is_directory=True)
        env_dir = Path("/opt/swe-task-envs") / row["instance_id"]
        env = {"HOME": str(home), "USER": username, "LOGNAME": username,
               "PATH": str(env_dir / "bin") + ":" + str(TOOLS / "node/bin") + ":/usr/local/bin:/usr/bin:/bin",
               "PYTHONPATH": str(repo), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
               "TMPDIR": str(home / "tmp"), "NO_PROXY": "localhost,127.0.0.1,::1",
               "LD_LIBRARY_PATH": "/opt/swe-proot/usr/lib/x86_64-linux-gnu",
               "PROOT_NO_SECCOMP": "1",
               **additions}
        (home / "tmp").mkdir()
        shutil.chown(home / "tmp", user=username, group=username)
        prompt = task_prompt(row) if prompt_override is None else prompt_override
        (run / "prompt.txt").write_text(prompt, encoding="utf-8")
        write_json(run / "evaluation_rules.json", {"version": "no_online_answers_prompt_v2",
                                                    "system_guidelines": EVALUATION_GUIDELINES,
                                                    "network_enforced": False})
        result["status"] = "executing"
        write_json(case / "state.json", result)
        with (run / "cli-events.jsonl").open("wb") as stdout, (run / "cli-stderr.txt").open("wb") as stderr:
            process = subprocess.Popen(["runuser", "-u", username, "--", "env", "-i",
                                        *[k + "=" + v for k, v in env.items()], str(PROOT),
                                        "-b", str(home / "tmp") + ":/tmp", *command],
                                       cwd=repo, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            try:
                process.communicate(prompt.encode(), timeout=timeout)
                result["exit_code"] = process.returncode
                result["status"] = "completed" if process.returncode == 0 else "cli_error"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                result.update(status="timeout", exit_code=process.returncode)
        # Stop detached/background tools too, before reading the final working tree.
        terminal = native_terminal_status(harness, run / "cli-events.jsonl")
        result["native_terminal_status"] = terminal
        if result["status"] != "timeout" and terminal is not None:
            result["status"] = "completed" if terminal == "completed" else "cli_error"
        result["wrapper_exit_nonzero"] = result.get("exit_code", 0) != 0
        subprocess.run(["pkill", "-KILL", "-u", str(pwd.getpwnam(username).pw_uid)], check=False)
        # Old distro Git may not honor safe.directory supplied through environment config.
        # The attempt has ended; root owns the repository while exporting and judging it.
        subprocess.run(["chown", "-R", "root:root", str(repo)], check=True)
        patch = export_patch(repo, row["base_commit"], run)
        (run / "patch.diff").write_text(patch, encoding="utf-8")
        result["patch_bytes"] = len(patch.encode())
        (run / "predictions.jsonl").write_text(json.dumps({"instance_id": row["instance_id"],
                "model_name_or_path": harness + "/deepseek-flash", "model_patch": patch}) + "\n", encoding="utf-8")
        if (home / "answer.txt").exists():
            shutil.copy2(home / "answer.txt", run / "answer.txt")
    except Exception as exc:
        result.update(status="setup_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        endpoint.close()
        # Complete upstream in-flight requests after a CLI timeout so billed tokens remain visible.
        records = endpoint.records
        totals = {k: sum((r.get("usage") or {}).get(k, 0) for r in records)
                  for k in ("input_tokens", "cached_input_tokens", "cache_creation_input_tokens", "output_tokens", "total_tokens")}
        result.update(duration_seconds=round(time.monotonic() - started, 3),
                      model_requests=len(records), usage=totals,
                      usage_complete=bool(records) and all(r.get("usage") is not None and r.get("complete")
                                   for r in records if "/count_tokens" not in r["path"]),
                      timeout_seconds=timeout)
        audit = write_audit(run, records, row)
        result["answer_retrieval_candidates"] = audit["candidate_count"]
        write_json(run / "result.json", result)
        write_json(case / "state.json", result)
    return result


def grade_one(root: Path, output: Path, row: dict, harness: str) -> dict:
    from evaluation.native import grade_native_python
    case = output / harness / "cases" / row["instance_id"]
    run = case / "run"
    if (run / "native_result.json").exists():
        return json.loads((run / "native_result.json").read_text())
    if not (run / "patch.diff").exists() or not (run / "patch.diff").stat().st_size:
        result = {"instance_id": row["instance_id"], "status": "empty_patch", "resolved": False}
        write_json(run / "native_result.json", result)
        return result
    # Reuse the validated grading dependencies, with a new clean baseline/model checkout.
    python = root / "batch-flash-mini50/cases" / row["instance_id"] / "grading-setup/env/bin/python"
    return grade_native_python(run, cwd=case / "repo", task_python=python,
                               dataset=GRADING_DATASET, timeout_seconds=300, instance_metadata=row)


def summarize(output: Path, rows: list[dict]) -> dict:
    summary = {"model": "deepseek-flash", "thinking": "requested_disabled",
               "thinking_request_setting": {"anthropic": "disabled", "responses": "none"},
               "thinking_behavior_note": "Request settings do not verify response behavior; see integrity-review.json",
               "total_per_harness": len(rows),
               "official_docker_environment": False, "price_snapshot": PRICE_SNAPSHOT,
               "harnesses": {}}
    for harness in CLI_VERSIONS:
        results, grades = [], []
        for row in rows:
            run = output / harness / "cases" / row["instance_id"] / "run"
            if (run / "result.json").exists():
                results.append(json.loads((run / "result.json").read_text()))
            if (run / "native_result.json").exists():
                grades.append(json.loads((run / "native_result.json").read_text()))
        valid = [g for g in grades if isinstance(g.get("resolved"), bool)]
        summary["harnesses"][harness] = {"cli_version": CLI_VERSIONS[harness],
            "executed": len(results), "graded": len(grades), "valid_grades": len(valid),
            "resolved": sum(g["resolved"] for g in valid),
            "usage_complete": all(r.get("usage_complete") for r in results) if results else None,
            "status_counts": {status: sum(r["status"] == status for r in results)
                              for status in sorted({r["status"] for r in results})},
            "usage": {k: sum(r.get("usage", {}).get(k, 0) for r in results)
                      for k in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")},
            "cases": results, "grades": grades}
        data = summary["harnesses"][harness]
        data["resolved_rate"] = data["resolved"] / len(rows) if len(valid) == len(rows) else None
        data["resolved_fraction_lower_bound"] = data["resolved"] / len(rows)
        data["cost_usd_estimate"] = estimate_cost(data["usage"])
        data["cost_estimate_complete"] = len(results) == len(rows) and bool(data["usage_complete"])
        data["cost_usd_per_resolved_estimate"] = (
            {period: value / data["resolved"] for period, value in data["cost_usd_estimate"].items()}
            if data["resolved"] and len(valid) == len(rows) and data["cost_estimate_complete"] else None)
    temporary = output / f"summary.{uuid4().hex}.tmp"
    write_json(temporary, summary)
    temporary.replace(output / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "execute", "grade", "summary"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    # Applies only to this trusted evaluation driver's Git calls; CLI env is separately constructed.
    os.environ.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory", GIT_CONFIG_VALUE_0="*")
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    HOMES.mkdir(exist_ok=True, mode=0o711)
    rows = json.loads((args.root / "mini50-inputs.json").read_text())
    if args.phase in ("smoke", "execute"):
        from dotenv import dotenv_values
        key = dotenv_values(args.root / "deepseek.env")["OPENAI_API_KEY"]
        work = [(i, row, harness) for i, row in enumerate(rows) for harness in CLI_VERSIONS]
        if args.phase == "smoke":
            work = work[:2]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(execute_one, args.root, args.output, row, harness, i, key, args.timeout)
                       for i, row, harness in work]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
                summarize(args.output, rows)
    elif args.phase == "grade":
        if any(not (args.output / harness / "cases" / row["instance_id"] / "run/result.json").exists()
               for row in rows for harness in CLI_VERSIONS):
            raise RuntimeError("Finish both complete execution batches before grading")
        metadata = json.loads((args.root / "batch-flash-mini50/judging-metadata.json").read_text())
        # Grade both harnesses for one case sequentially: their shared grade env uses editable installs.
        def grade_pair(row):
            for harness in CLI_VERSIONS:
                result = grade_one(args.root, args.output, metadata[row["instance_id"]], harness)
                print(json.dumps({"harness": harness, **result}), flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for _ in pool.map(grade_pair, rows):
                summarize(args.output, rows)
    print(json.dumps({k: {f: v for f,v in val.items() if f not in ("cases","grades")}
                      for k,val in summarize(args.output, rows)["harnesses"].items()}), flush=True)


if __name__ == "__main__":
    main()
