import argparse
import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv
from rich.table import Table
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, Static
from textual.worker import Worker

from ai.env import OpenAICompatibleConfig, openai_compatible_config_from_env
from ai.openai_compatible import OpenAICompatibleProvider
from agent.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
)
from agent.messages import AssistantMessage
from agent.session.storage import JsonlSessionStorage
from coding.tui_adapter import TuiEventAdapter
from coding.tui_state import TuiState
from coding.tui_widgets import TranscriptView
from coding.session import CodingSession, CodingSessionConfig
from coding.session_catalog import create_new_session, list_saved_sessions, resume_session


def build_demo_state() -> TuiState:
    """构造助手正在输出时的模拟显示数据。"""
    state = TuiState()
    state.add_item("user", "介绍一下 Python")
    state.assistant_buffer = "Python 是一种"
    return state


async def simulated_events(text: str) -> AsyncIterator[AgentEvent]:
    """模拟一次回答，以短暂间隔逐段发出文本事件。"""
    response = f"Simulated reply: {text}"
    yield AgentStartEvent()
    yield MessageStartEvent()
    for index in range(0, len(response), 8):
        await asyncio.sleep(0.15)
        yield MessageDeltaEvent(delta=response[index:index + 8])
    yield MessageEndEvent(message=AssistantMessage(content=response))
    yield AgentEndEvent()


class TUIApp(App[None]): # 泛型，把返回值类型留作参数，使用时可以传入类型来指定返回值的类型
    TITLE = "My Agent"
    CSS = """
    Screen {
        layout: vertical;
    }

    #status {
        height: 1;
        padding: 0 1;
    }

    #transcript {
        height: 1fr;
        border: solid $accent;
    }

    #prompt {
        margin: 0;
    }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit")] # 第三项是显示说明，第二项是方法

    def __init__(
        self,
        session: CodingSession | None = None,
        *,
        config: CodingSessionConfig | None = None,
        create_new: bool = True,
    ) -> None:
        super().__init__()
        if session is not None:
            if config is not None:
                raise ValueError("Pass either a session or a config")
            self._session_config = session.config
        elif config is not None:
            self._session_config = config
        else:
            raise ValueError("A session or a config is required")
        self.session = session
        self._create_new = create_new
        self._session_directory = self._session_config.storage.path.resolve().parent

        self.state = TuiState()
        if session is not None:
            self.state.load_entries(session.storage.read_all())
        self.adapter = TuiEventAdapter(self.state)
        self._busy = False
        self._prompt_worker: Worker[None] | None = None

    def compose(self) -> ComposeResult:
        """创建标题、状态栏、聊天区域、输入框和快捷键提示。"""
        # yield Header()
        yield Static("Ready", id="status")
        yield TranscriptView(id="transcript")
        yield Input(placeholder="Enter a prompt", id="prompt")
        # yield Footer()

    async def on_mount(self) -> None:
        """控件挂载后，让输入框获得键盘焦点。"""
        # return
        await self.update_from_state(self.state)
        self.query_one("#prompt", Input).focus()

    async def update_from_state(self, state: TuiState) -> None:
        """把状态交给聊天容器，保留已有的消息控件。"""
        await self.query_one("#transcript", TranscriptView).update_from_state(state)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """显示用户输入，并启动消费 Session 事件的异步 Worker。"""
        text = event.value.strip()
        if not text:
            event.input.value = ""
            return
        if self._busy:
            self.query_one("#status", Static).update("Working... Please wait.")
            return

        event.input.value = ""
        if text.split(maxsplit=1)[0] == "/resume":
            await self._handle_resume(text)
            return

        self._busy = True
        self.state.add_item("user", text)
        await self.update_from_state(self.state)
        self.query_one("#status", Static).update("Working...")
        self._prompt_worker = self.run_worker(self._run_prompt(text), exclusive=True)

    async def _handle_resume(self, text: str) -> None:
        """列出已有会话，或加载成功后同时替换 Session 和显示状态。"""
        status = self.query_one("#status", Static)
        parts = text.split(maxsplit=1)
        try:
            if len(parts) == 1:
                sessions, errors = list_saved_sessions(self._session_directory)
                table = Table(title="Saved sessions")
                table.add_column("Current")
                table.add_column("Session ID")
                table.add_column("File")
                for saved in sessions:
                    table.add_row(
                        "*" if self.session is not None and saved.path.resolve() == self.session.storage.path.resolve() else "",
                        saved.session_id, saved.path.name,
                    )
                transcript = self.query_one("#transcript", TranscriptView)
                await transcript.show_notice(table if sessions else "No saved sessions found.")
                if errors:
                    await transcript.show_notice("\n".join(errors))
                status.update("Use /resume <id or filename>")
                return

            config = self.session.config if self.session is not None else self._session_config
            candidate = resume_session(config, parts[1])
            state = TuiState()
            state.load_entries(candidate.storage.read_all())
        except (OSError, UnicodeError, ValueError) as exc:
            status.update(f"Error: {exc}")
            return

        self.session = candidate
        self.state = state
        self.adapter = TuiEventAdapter(state)
        self._prompt_worker = None
        await self.update_from_state(state)
        status.update(f"Resumed: {candidate.storage.path.name}")

    async def _run_prompt(self, text: str) -> None:
        """消费事件、更新 State、刷新显示，最后解除运行标记。"""
        status = self.query_one("#status", Static)
        error: str | None = None
        try:
            if self.session is None:
                self.session = (
                    create_new_session(self._session_config)
                    if self._create_new else CodingSession.load(self._session_config)
                )
            async for event in self.session.prompt(text):
                if isinstance(event, ErrorEvent):
                    error = event.message
                self.adapter.apply(event)
                await self.update_from_state(self.state)
                # 已缓冲的网络事件可能连续返回，主动让 UI 处理输入和绘制。
                await asyncio.sleep(0)
        except Exception as exc:
            error = str(exc)
        finally:
            self._busy = False
            status.update(f"Error: {error}" if error is not None else "Ready")


async def run_tui_app(
    *,
    model: str,
    cwd: Path,
    provider_config: OpenAICompatibleConfig,
    session_path: Path | None = None,
) -> None:
    """创建真实 Provider 和 Session，并在退出时关闭网络客户端。"""
    provider = OpenAICompatibleProvider(provider_config)
    try:
        config = CodingSessionConfig(
            provider=provider,
            model=model,
            cwd=cwd,
            storage=JsonlSessionStorage(session_path or cwd / ".codex_tmp/tui/session.jsonl"),
            auto_compact_enabled=False,
        )
        if session_path is not None and session_path.exists():
            app = TUIApp(CodingSession.load(config))
        else:
            app = TUIApp(config=config, create_new=session_path is None)
        await app.run_async()
    finally:
        await provider.aclose()


def main() -> None:
    """读取启动参数和环境配置，再运行 TUI。"""
    parser = argparse.ArgumentParser(description="Start the coding agent TUI")
    parser.add_argument("--model", required=True, help="Model name used by the API")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Working directory for coding tools")
    parser.add_argument("--session", type=Path, help="Restore this JSONL file, or create it on the first prompt")
    parser.add_argument("--api-key-var", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-var", default="OPENAI_BASE_URL")
    parser.add_argument("--no-usage", action="store_true", help="Disable stream_options for incompatible services")
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

    asyncio.run(run_tui_app(
        model=args.model,
        cwd=args.cwd.resolve(),
        provider_config=provider_config,
        session_path=args.session.resolve() if args.session is not None else None,
    ))


if __name__ == "__main__":
    main()
