"""本地调试页面：消费 CodingSession 事件并通过 SSE 发送。"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
from json import dumps
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, model_validator

from agent.events import AgentStartEvent, MessageEndEvent, ToolExecutionEndEvent
from agent.session.entries import MessageEntry, SessionInfoEntry
from agent.session.jsonl import SessionJsonlError, entry_to_json_line
from agent.session.storage import JsonlSessionStorage
from coding.session import CodingSession, CodingSessionConfig


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["prompt", "continue"] = "prompt"
    content: str | None = None
    session_id: str | None = None

    @model_validator(mode="after")
    def check_prompt(self) -> RunRequest:
        if self.mode == "prompt" and (self.content is None or not self.content.strip()):
            raise ValueError("请输入问题")
        return self


@dataclass(slots=True)
class _RunChannel:
    """网页断开后停止排队，后台任务仍继续消费并保存消息。"""

    queue: asyncio.Queue[dict | None] = field(default_factory=asyncio.Queue)
    connected: bool = True

    def publish(self, frame: dict | None) -> None:
        if self.connected:
            self.queue.put_nowait(frame)

    def detach(self) -> None:
        self.connected = False
        while not self.queue.empty():
            self.queue.get_nowait()


@dataclass(slots=True)
class _WebUiRuntime:
    session: CodingSession
    task: asyncio.Task[None] | None = None
    last_error: str | None = None
    directory: Path = field(init=False)

    def __post_init__(self) -> None:
        self.directory = self.session.storage.path.resolve().parent

    @property
    def session_id(self) -> str | None:
        info = self.session.state.session_info
        return info.session_id if info is not None else None

    def saved_sessions(self) -> tuple[list[tuple[Path, dict]], list[str]]:
        """只扫描当前存储目录；会话 ID 与文件路径的对应由服务器确定。"""
        records = []
        errors = []
        for path in sorted(self.directory.glob("*.jsonl")):
            try:
                entries = JsonlSessionStorage(path).read_all()
                infos = [entry for entry in entries if isinstance(entry, SessionInfoEntry)]
                if len(infos) != 1:
                    raise ValueError("会话必须包含且只包含一条元信息")
                info = infos[0]
                messages = [entry for entry in entries if isinstance(entry, MessageEntry)]
                first_input = next((entry.message.content for entry in messages
                                    if entry.message.role == "user"), "")
                summary = {
                    "session_id": info.session_id,
                    "title": " ".join(first_input.split())[:80] or "新会话",
                    "created_at": info.created_at,
                    "updated_at": max([info.created_at, *(entry.timestamp for entry in messages)]),
                    "message_count": len(messages),
                }
                records.append((path, summary))
            except (OSError, UnicodeError, SessionJsonlError, ValueError) as exc:
                errors.append(f"{path.name}：{exc}")
        records.sort(key=lambda record: record[1]["updated_at"], reverse=True)
        return records, errors

    def catalog(self) -> dict:
        records, errors = self.saved_sessions()
        return {"sessions": [summary for _, summary in records],
                "active_session_id": self.session_id, "errors": errors}

    def create_session(self) -> None:
        info = SessionInfoEntry(cwd=str(self.session.config.cwd.resolve()))
        path = self.directory / f"{info.session_id}.jsonl"
        # 独占创建新文件，不覆盖既有记录；配置加载成功后才替换当前会话。
        with path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(entry_to_json_line(info))
        try:
            candidate = CodingSession.load(replace(
                self.session.config, storage=JsonlSessionStorage(path),
            ))
        except Exception:
            path.unlink()
            raise
        self.session = candidate
        self.last_error = None

    def select_session(self, session_id: str) -> None:
        records, _ = self.saved_sessions()
        matches = [path for path, summary in records if summary["session_id"] == session_id]
        if not matches:
            raise HTTPException(status_code=404, detail="找不到该历史会话")
        if len(matches) > 1:
            raise HTTPException(status_code=409, detail="会话 ID 重复，无法确定记录文件")
        candidate = CodingSession.load(replace(
            self.session.config, storage=JsonlSessionStorage(matches[0]),
        ))
        self.session = candidate
        self.last_error = None

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def snapshot(self) -> dict:
        usage = self.session.context_usage
        info = self.session.state.session_info
        return {
            "messages": [message.model_dump(mode="json", by_alias=False)
                         for message in self.session.messages],
            "context_usage": {
                **asdict(usage),
                "estimated_tokens": usage.estimated_tokens,
                "uses_provider_usage": usage.uses_provider_usage,
            },
            "running": self.running,
            "session_info": info.model_dump(mode="json") if info is not None else None,
            "error": self.last_error,
        }

    async def consume(self, body: RunRequest, channel: _RunChannel) -> None:
        """运行任务独立于 HTTP 响应，完整消费后才确认保存结束。"""
        try:
            if body.mode == "prompt":
                assert body.content is not None  # RunRequest 已校验 prompt 必须有正文。
                events = self.session.prompt(body.content)
            else:
                events = self.session.continue_()
            async for event in events:
                channel.publish({"type": "agent_event",
                                 "event": event.model_dump(mode="json", by_alias=False)})
                if isinstance(event, (AgentStartEvent, MessageEndEvent, ToolExecutionEndEvent)):
                    channel.publish({"type": "snapshot", "session": self.snapshot()})
            # CodingSession 的保存发生在异步迭代器完全结束前。
            snapshot = self.snapshot()
            snapshot["running"] = False
            channel.publish({"type": "run_finished", "session": snapshot})
        except Exception as exc:
            self.last_error = str(exc) or type(exc).__name__
            snapshot = self.snapshot()
            snapshot["running"] = False
            channel.publish({"type": "run_error", "message": self.last_error,
                             "session": snapshot})
        finally:
            channel.publish(None)


async def _stream_frames(channel: _RunChannel) -> AsyncIterator[str]:
    try:
        while True:
            frame = await channel.queue.get()
            if frame is None:
                return
            yield f"data: {dumps(frame, ensure_ascii=False)}\n\n"
    finally:
        channel.detach()


def create_app(
    session: CodingSession,
    *,
    close_provider: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    """给已有会话增加网页入口；不改变 Harness 或事件发送端。"""
    runtime = _WebUiRuntime(session)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            try:
                if runtime.running:
                    await asyncio.shield(runtime.task)
            finally:
                if close_provider is not None:
                    await close_provider()

    app = FastAPI(title="Agent 调试界面", lifespan=lifespan)
    app.state.runtime = runtime

    @app.get("/", response_class=HTMLResponse)
    async def page() -> HTMLResponse:
        content = Path(__file__).with_name("webui.html").read_text(encoding="utf-8")
        return HTMLResponse(content)

    @app.get("/api/session")
    async def get_session() -> dict:
        return runtime.snapshot()

    @app.get("/api/sessions")
    async def list_sessions() -> dict:
        return runtime.catalog()

    @app.post("/api/sessions", status_code=201)
    async def new_session() -> dict:
        if runtime.running:
            raise HTTPException(status_code=409, detail="请等当前运行结束后再新建会话")
        runtime.create_session()
        return runtime.snapshot()

    @app.post("/api/sessions/{session_id}/select")
    async def select_session(session_id: str) -> dict:
        if runtime.running:
            raise HTTPException(status_code=409, detail="请等当前运行结束后再切换会话")
        runtime.select_session(session_id)
        return runtime.snapshot()

    @app.post("/api/run")
    async def run(body: RunRequest) -> StreamingResponse:
        if runtime.running:
            raise HTTPException(status_code=409, detail="当前运行尚未结束")
        if body.session_id is not None and body.session_id != runtime.session_id:
            raise HTTPException(status_code=409, detail="当前会话已切换，请刷新后再发送")
        channel = _RunChannel()
        runtime.last_error = None
        # 检查与创建之间没有 await，同一事件循环不会启动两个运行。
        runtime.task = asyncio.create_task(runtime.consume(body, channel))
        return StreamingResponse(
            _stream_frames(channel),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def main() -> None:
    """python -m coding.webui --model <模型名> 启动本地界面。"""
    import uvicorn
    from dotenv import load_dotenv

    from ai.env import openai_compatible_config_from_env
    from ai.openai_compatible import OpenAICompatibleProvider

    parser = argparse.ArgumentParser(description="启动本地 agent 调试界面")
    parser.add_argument("--model", required=True, help="模型 API 使用的模型名称")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="工具操作的项目目录")
    parser.add_argument("--session", type=Path, default=Path(".codex_tmp/webui/session.jsonl"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--api-key-var", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-var", default="OPENAI_BASE_URL")
    parser.add_argument("--no-usage", action="store_true", help="服务不支持 stream_options 时关闭")
    args = parser.parse_args()

    load_dotenv(Path.cwd() / ".env")
    try:
        provider_config = openai_compatible_config_from_env(
            api_key_var=args.api_key_var, base_url_var=args.base_url_var,
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.no_usage:
        provider_config = replace(provider_config, include_usage=False)
    provider = OpenAICompatibleProvider(provider_config)
    session = CodingSession.load(CodingSessionConfig(
        provider=provider, model=args.model, cwd=args.cwd.resolve(),
        storage=JsonlSessionStorage(args.session.resolve()),
    ))
    app = create_app(session, close_provider=provider.aclose)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
