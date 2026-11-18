"""Run the real CodingSession and collect artifacts without judging its answer."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ai.events import ProviderResponseEndEvent
from ai.provider import ModelProvider
from agent.events import ErrorEvent, MessageEndEvent
from agent.session import JsonlSessionStorage
from coding.session import CodingSession, CodingSessionConfig
from evaluation.audit import audit_tool_commands

DEFAULT_MAX_TURNS = 1_000_000
DEFAULT_MAX_MODEL_REQUESTS = 1_000_000
EVALUATION_GUIDELINES = (
    "Network access is allowed for general documentation, dependency installation, and "
    "other information needed to solve the issue independently.",
    "Evaluation rule: do not search for or retrieve existing solutions to this task. "
    "Do not obtain the resolving pull request, repair commits or patches, reference answers, "
    "or a newer version of the target repository to extract its fix. This applies to GitHub, "
    "mirrors, search engines, and any other online source. "
    "Implement your own solution from the supplied problem statement and local code.",
    "Do not inspect other attempts, grading artifacts, reference patches, "
    "or hidden evaluation tests, including files outside the task repository.",
)


@dataclass(frozen=True, slots=True)
class CodingTask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str

    @classmethod
    def from_dict(cls, row: dict) -> CodingTask:
        # Evaluation patches and hidden tests never enter the agent's input.
        fields = ("instance_id", "repo", "base_commit", "problem_statement")
        values = {key: row[key] for key in fields}
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError("Task fields must be non-empty strings")
        return cls(**values)

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git(cwd: Path, *arguments: str, env: dict | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=cwd, env=env, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8")


def export_patch(cwd: Path, base_commit: str, output_dir: Path) -> str:
    """Include new files and staged changes without modifying the real Git index."""
    with tempfile.TemporaryDirectory(dir=output_dir, prefix="patch-index-") as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        git(cwd, "read-tree", base_commit, env=env)
        git(cwd, "add", "-A", "--", ".", ":(exclude).codex_tmp", env=env)
        arguments = ["-c", "core.fileMode=false", "diff", "--cached", "--binary",
                     "--no-ext-diff", "--no-color", base_commit, "--"]
        try:
            return git(cwd, *arguments, env=env)
        except UnicodeDecodeError:
            # Git's NUL-byte heuristic can misidentify compressed/non-UTF-8 files as text.
            # Binary patches preserve their exact bytes in an ASCII representation.
            attributes = Path(temporary) / "attributes"
            attributes.write_text("* -diff\n", encoding="utf-8")
            patch = git(cwd, "-c", f"core.attributesFile={attributes}", *arguments, env=env)
            write_json(output_dir / "patch-export.json", {"encoding": "git_binary_patch",
                       "reason": "Default Git diff contains non-UTF-8 bytes"})
            return patch


class ModelRequestLimitExceeded(RuntimeError):
    pass


class _MeasuredProvider:
    """Count all calls, including compaction and overflow recovery requests."""

    def __init__(self, provider: ModelProvider, limit: int) -> None:
        self.provider = provider
        self.limit = limit
        self.requests = 0
        self.usage_records: list[dict] = []

    async def stream_response(self, **kwargs):
        if self.requests >= self.limit:
            raise ModelRequestLimitExceeded("Maximum model request count reached")
        self.requests += 1
        async for event in self.provider.stream_response(**kwargs):
            if isinstance(event, ProviderResponseEndEvent) and event.message.usage is not None:
                self.usage_records.append(event.message.usage.model_dump(mode="json", by_alias=False))
            yield event


async def run_task(
    task: CodingTask, *, provider: ModelProvider, model: str, cwd: Path,
    output_dir: Path, max_turns: int = DEFAULT_MAX_TURNS,
    max_model_requests: int = DEFAULT_MAX_MODEL_REQUESTS,
    timeout_seconds: float = 600, secrets: tuple[str, ...] = (),
) -> dict:
    """Consume the entire session run; completion is distinct from benchmark resolution."""
    if max_turns < 1 or max_model_requests < 1 or timeout_seconds <= 0:
        raise ValueError("Run budgets must be positive")
    cwd, output_dir = cwd.resolve(), output_dir.resolve()
    if output_dir.is_relative_to(cwd):
        raise ValueError("Artifacts must be outside the task repository")
    if git(cwd, "rev-parse", "HEAD").strip() != task.base_commit:
        raise ValueError("Repository HEAD does not match the task base_commit")
    if git(cwd, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("Task repository must be clean before execution")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "task.json", task.to_dict())
    write_json(output_dir / "evaluation_rules.json", {
        "version": "no_online_answers_prompt_v2", "network_enforced": False,
        "system_guidelines": EVALUATION_GUIDELINES,
    })
    measured = _MeasuredProvider(provider, max_model_requests)
    session = CodingSession.load(CodingSessionConfig(
        provider=measured, model=model, cwd=cwd,
        storage=JsonlSessionStorage(output_dir / "session.jsonl"), max_turns=max_turns,
        extra_guidelines=EVALUATION_GUIDELINES,
    ))
    result = {
        "instance_id": task.instance_id, "model": model, "repo": task.repo,
        "base_commit": task.base_commit, "status": "running", "errors": [],
        "grading_status": "not_run", "resolved": None,
        "budgets": {"max_turns": max_turns, "max_model_requests": max_model_requests,
                    "timeout_seconds": timeout_seconds},
    }
    started = time.monotonic()

    def redact(value: str) -> str:
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value

    fatal = False
    final_reply = False
    with (output_dir / "events.jsonl").open("w", encoding="utf-8") as events:
        try:
            async with asyncio.timeout(timeout_seconds):
                prompt = (
                    "Resolve the following issue in the checked-out repository. Inspect the code, "
                    "make the necessary changes, and run relevant tests. Do not commit changes.\n\n"
                    + task.problem_statement
                )
                async for event in session.prompt(prompt):
                    events.write(redact(event.model_dump_json(by_alias=False)) + "\n")
                    events.flush()
                    if isinstance(event, ErrorEvent):
                        result["errors"].append(json.loads(redact(event.model_dump_json(by_alias=False))))
                        if not event.recoverable:
                            fatal = True
                    elif isinstance(event, MessageEndEvent):
                        final_reply = not event.message.tool_calls
                        # An overflow error followed by a final reply was recovered.
                        if final_reply:
                            fatal = False
            result["status"] = "completed" if final_reply and not fatal else "agent_error"
        except TimeoutError:
            result["status"] = "timeout"
        except asyncio.CancelledError:
            result["status"] = "cancelled"
            raise
        except ModelRequestLimitExceeded:
            result["status"] = "model_request_limit"
        except Exception as exc:
            result["status"] = "exception"
            result["exception"] = redact(f"{type(exc).__name__}: {exc}")
        finally:
            result["duration_seconds"] = round(time.monotonic() - started, 3)
            result["model_requests"] = measured.requests
            result["usage_records"] = measured.usage_records
            result["usage"] = {
                key: sum(row[key] for row in measured.usage_records)
                for key in ("input_tokens", "output_tokens", "total_tokens")
            }
            result["usage_complete"] = len(measured.usage_records) == measured.requests
            # This diagnostic snapshot can contain messages not persisted after cancellation.
            snapshot = [message.model_dump(mode="json", by_alias=False) for message in session.messages]
            (output_dir / "messages_snapshot.json").write_text(
                redact(json.dumps(snapshot, ensure_ascii=False, indent=2)), encoding="utf-8",
            )
            try:
                patch = export_patch(cwd, task.base_commit, output_dir)
                (output_dir / "patch.diff").write_bytes(patch.encode("utf-8"))
                prediction = {"instance_id": task.instance_id,
                              "model_name_or_path": "my-agent/" + model, "model_patch": patch}
                (output_dir / "predictions.jsonl").write_text(
                    json.dumps(prediction, ensure_ascii=False) + "\n", encoding="utf-8",
                )
                result["patch_bytes"] = len(patch.encode("utf-8"))
            except Exception as exc:
                result["patch_export_error"] = redact(f"{type(exc).__name__}: {exc}")
            write_json(output_dir / "result.json", result)
            events.flush()
            write_json(output_dir / "network-audit.json", audit_tool_commands(
                output_dir / "events.jsonl", target_repo=task.repo,
            ))
    return result
