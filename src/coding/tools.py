"""Local read, write, edit, and shell tools for a coding agent."""

import asyncio
import base64
import difflib
import json
import mimetypes
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.tools import AgentTool, AgentToolResult
from agent.types import JSONValue


ToolExecutor = Callable[[dict[str, JSONValue]], Awaitable[AgentToolResult]]
MAX_READ_CHARS = 20_000
MAX_BASH_CHARS = 20_000
DEFAULT_BASH_TIMEOUT_SECONDS = 120
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    prompt_snippet: str
    prompt_guidelines: tuple[str, ...]
    input_schema: dict[str, JSONValue]
    executor: ToolExecutor

    def to_agent_tool(self) -> AgentTool:
        return AgentTool(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            executor=self.executor,
            prompt_snippet=self.prompt_snippet,
            prompt_guidelines=self.prompt_guidelines,
        )


def _cwd_path(cwd: str | Path) -> Path:
    path = Path(cwd).resolve()
    if not path.is_dir():
        raise ValueError(f"Working directory does not exist: {path}")
    return path


def _file_path(cwd: Path, arguments: dict[str, JSONValue]) -> Path:
    raw = arguments.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("path must be a non-empty string")
    path = Path(raw)
    return (path if path.is_absolute() else cwd / path).resolve()


def _text_argument(arguments: dict[str, JSONValue], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _integer_argument(
    arguments: dict[str, JSONValue], name: str, default: int | None = None
) -> int | None:
    value = arguments.get(name, default)
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _result(name: str, content: str, *, data: dict[str, JSONValue] | None = None) -> AgentToolResult:
    # The loop replaces the placeholder ID with the provider's tool-call ID.
    return AgentToolResult(tool_call_id="", name=name, content=content, ok=True, data=data)


def _failure(name: str, exc: Exception) -> AgentToolResult:
    message = str(exc) or type(exc).__name__
    return AgentToolResult(
        tool_call_id="", name=name, content=message, ok=False, error=message
    )


def create_read_tool_definition(cwd: str | Path) -> ToolDefinition:
    root = _cwd_path(cwd)

    async def execute(arguments: dict[str, JSONValue]) -> AgentToolResult:
        try:
            path = _file_path(root, arguments)
            if path.suffix.lower() in IMAGE_SUFFIXES:
                raw = await asyncio.to_thread(path.read_bytes)
                mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                return _result("read", f"Image: {path} ({mime_type}, {len(raw)} bytes)", data={
                    "path": str(path),
                    "mime_type": mime_type,
                    "base64": base64.b64encode(raw).decode("ascii"),
                    "bytes": len(raw),
                })

            offset = _integer_argument(arguments, "offset", 1)
            limit = _integer_argument(arguments, "limit")
            if offset is None or offset < 0:
                raise ValueError("offset must be 0 or a positive line number")
            if limit is not None and limit < 1:
                raise ValueError("limit must be positive")

            text = await asyncio.to_thread(path.read_text, encoding="utf-8")
            lines = text.splitlines()
            start = max(offset - 1, 0)
            selected = lines[start:start + limit] if limit is not None else lines[start:]
            output = "\n".join(f"{number}: {line}" for number, line in
                               enumerate(selected, start=start + 1))
            truncated = len(output) > MAX_READ_CHARS
            if truncated:
                output = output[:MAX_READ_CHARS]
            next_line = start + len(selected) + 1
            if truncated:
                # The char limit may cut inside the selected range. Point to the
                # first line that was not fully included.
                next_line = start + output.count("\n") + 1
            has_more = truncated or start + len(selected) < len(lines)
            if has_more:
                output += f"\n[Truncated; continue with offset={next_line}]"
            return _result("read", output, data={
                "path": str(path), "total_lines": len(lines),
                "offset": start + 1, "truncated": truncated, "has_more": has_more,
                "next_offset": next_line if has_more else None,
            })
        except (OSError, UnicodeError, ValueError) as exc:
            return _failure("read", exc)

    return ToolDefinition(
        name="read",
        description="Read a UTF-8 file, optionally starting at a line and limiting lines.",
        prompt_snippet="read(path, offset?, limit?): inspect a local file.",
        prompt_guidelines=("Use offset and limit to continue long reads.",),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
        },
        executor=execute,
    )


def create_read_tool(cwd: str | Path) -> AgentTool:
    return create_read_tool_definition(cwd).to_agent_tool()


_mutation_locks: dict[Path, asyncio.Lock] = {}


def _mutation_lock(path: Path) -> asyncio.Lock:
    return _mutation_locks.setdefault(path, asyncio.Lock())


def create_write_tool_definition(cwd: str | Path) -> ToolDefinition:
    root = _cwd_path(cwd)

    async def execute(arguments: dict[str, JSONValue]) -> AgentToolResult:
        try:
            path = _file_path(root, arguments)
            content = _text_argument(arguments, "content")
            async with _mutation_lock(path):
                await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
                await asyncio.to_thread(path.write_bytes, content.encode("utf-8"))
            return _result("write", f"Wrote {path}", data={
                "path": str(path), "bytes": len(content.encode("utf-8")),
            })
        except (OSError, ValueError) as exc:
            return _failure("write", exc)

    return ToolDefinition(
        name="write",
        description="Write complete UTF-8 file content, creating parent directories.",
        prompt_snippet="write(path, content): replace a complete local file.",
        prompt_guidelines=("Provide the entire desired file content.",),
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        executor=execute,
    )


def create_write_tool(cwd: str | Path) -> AgentTool:
    return create_write_tool_definition(cwd).to_agent_tool()


def _normalize_edits(arguments: dict[str, JSONValue]) -> list[tuple[str, str]]:
    raw: Any = arguments.get("edits")
    if raw is None and "oldText" in arguments and "newText" in arguments:
        raw = [{"oldText": arguments["oldText"], "newText": arguments["newText"]}]
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list) or not raw:
        raise ValueError("edits must be a non-empty array")
    edits: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each edit must be an object")
        old, new = item.get("oldText"), item.get("newText")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError("each edit needs non-empty oldText and string newText")
        edits.append((old, new))
    return edits


def _apply_exact_edits(text: str, edits: list[tuple[str, str]]) -> tuple[str, int]:
    positions: list[tuple[int, int, str]] = []
    line_ending = "\r\n" if "\r\n" in text else ("\r" if "\r" in text else "\n")
    for old, new in edits:
        start = text.find(old)
        if start < 0 or text.find(old, start + 1) >= 0:
            raise ValueError(f"oldText must match exactly once: {old!r}")
        new = new.replace("\r\n", "\n").replace("\r", "\n").replace("\n", line_ending)
        positions.append((start, start + len(old), new))
    positions.sort()
    if any(left[1] > right[0] for left, right in zip(positions, positions[1:])):
        raise ValueError("edits must not overlap")

    first_line = text.count("\n", 0, positions[0][0]) + 1
    updated = text
    for start, end, new in reversed(positions):
        updated = updated[:start] + new + updated[end:]
    return updated, first_line


def create_edit_tool_definition(cwd: str | Path) -> ToolDefinition:
    root = _cwd_path(cwd)

    async def execute(arguments: dict[str, JSONValue]) -> AgentToolResult:
        try:
            path = _file_path(root, arguments)
            edits = _normalize_edits(arguments)
            async with _mutation_lock(path):
                raw = await asyncio.to_thread(path.read_bytes)
                has_bom = raw.startswith(b"\xef\xbb\xbf")
                original = raw[3:].decode("utf-8") if has_bom else raw.decode("utf-8")
                updated, first_line = _apply_exact_edits(original, edits)
                if updated != original:
                    encoded = (b"\xef\xbb\xbf" if has_bom else b"") + updated.encode("utf-8")
                    await asyncio.to_thread(path.write_bytes, encoded)

            before_lines = original.splitlines(keepends=True)
            after_lines = updated.splitlines(keepends=True)
            diff = "".join(difflib.ndiff(before_lines, after_lines))
            patch = "".join(difflib.unified_diff(
                before_lines, after_lines, fromfile=str(path), tofile=str(path)
            ))
            return _result("edit", f"Edited {path}\n{patch}", data={
                "path": str(path), "edits": len(edits), "diff": diff,
                "patch": patch, "first_changed_line": first_line,
            })
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            return _failure("edit", exc)

    return ToolDefinition(
        name="edit",
        description="Apply exact, non-overlapping text replacements to a UTF-8 file.",
        prompt_snippet="edit(path, edits): replace text that matches exactly once.",
        prompt_guidelines=("Check oldText is unique before editing.",),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "edits": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}},
                    "required": ["oldText", "newText"],
                }},
            },
            "required": ["path", "edits"],
        },
        executor=execute,
    )


def create_edit_tool(cwd: str | Path) -> AgentTool:
    return create_edit_tool_definition(cwd).to_agent_tool()


def create_bash_tool_definition(cwd: str | Path) -> ToolDefinition:
    root = _cwd_path(cwd)

    async def execute(arguments: dict[str, JSONValue]) -> AgentToolResult:
        try:
            command = _text_argument(arguments, "command")
            if not command:
                raise ValueError("command must not be empty")
            timeout = arguments.get("timeout")
            if timeout is None:
                timeout = DEFAULT_BASH_TIMEOUT_SECONDS
            if type(timeout) not in (int, float) or timeout <= 0:
                raise ValueError("timeout must be a positive number of seconds")
            started = time.monotonic()
            process_options: dict[str, Any] = {
                "cwd": str(root),
                "stdin": asyncio.subprocess.DEVNULL,
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.STDOUT,
            }
            if os.name == "nt":
                process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                process_options["start_new_session"] = True
            process = await asyncio.create_subprocess_shell(command, **process_options)
            timed_out = False
            try:
                output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except TimeoutError:
                timed_out = True
                await _kill_process_tree(process)
                output, _ = await process.communicate()
            duration = time.monotonic() - started
            full_text = output.decode("utf-8", errors="replace")
            truncated = len(full_text) > MAX_BASH_CHARS
            full_output_path: str | None = None
            if truncated:
                temp_dir = root / ".codex_tmp"
                await asyncio.to_thread(temp_dir.mkdir, parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", prefix="bash-output-", suffix=".txt",
                    dir=temp_dir, delete=False,
                ) as output_file:
                    output_file.write(full_text)
                    full_output_path = output_file.name
            content = full_text[:MAX_BASH_CHARS]
            if truncated:
                content += f"\n[Truncated; full output: {full_output_path}]"
            if timed_out:
                content += "\n[Command timed out]"
            exit_code = process.returncode
            if exit_code not in (None, 0) and not timed_out:
                content += f"\n[Exit code: {exit_code}]"
            return AgentToolResult(
                tool_call_id="", name="bash", content=content,
                ok=not timed_out and exit_code == 0,
                error="Command timed out" if timed_out else (
                    f"Command exited with code {exit_code}" if exit_code else None
                ),
                data={
                    "exit_code": exit_code, "timed_out": timed_out,
                    "duration_seconds": round(duration, 3), "truncated": truncated,
                    "full_output_path": full_output_path,
                },
            )
        except (OSError, ValueError) as exc:
            return _failure("bash", exc)

    return ToolDefinition(
        name="bash",
        description=("Execute a non-interactive shell command in the working directory. "
                     "Standard input is closed; default timeout is 120 seconds."),
        prompt_snippet="bash(command, timeout?): run a shell command.",
        prompt_guidelines=(
            "Use non-interactive commands; standard input is unavailable. Set timeout in seconds for longer commands.",
            *(("On Windows, commands run in cmd.exe. Use Windows syntax, such as date /t to read the date.",)
              if os.name == "nt" else ()),
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "number", "exclusiveMinimum": 0,
                            "default": DEFAULT_BASH_TIMEOUT_SECONDS},
            },
            "required": ["command"],
        },
        executor=execute,
    )


def create_bash_tool(cwd: str | Path) -> AgentTool:
    return create_bash_tool_definition(cwd).to_agent_tool()


async def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    if os.name == "nt":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/PID", str(process.pid), "/T", "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
            if killer.returncode == 0:
                return
        except OSError:
            pass
        if process.returncode is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def create_coding_tools(cwd: str | Path) -> list[AgentTool]:
    """Create the Phase 5 default set of local coding tools."""
    return [
        create_read_tool(cwd),
        create_write_tool(cwd),
        create_edit_tool(cwd),
        create_bash_tool(cwd),
    ]
