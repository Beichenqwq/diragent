"""会话切换恢复历史，并保持命令与模型输入、持久化记录独立。"""

import asyncio
from io import StringIO
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from textual.widgets import Input, Static

from ai.fake import FakeProvider
from agent.session.entries import CompactionEntry, MessageEntry, SessionInfoEntry
from agent.session.storage import JsonlSessionStorage
from agent.messages import AssistantMessage, UserMessage
from coding.tui_app import TUIApp
from coding.session import CodingSession, CodingSessionConfig
from coding.session_catalog import list_saved_sessions, resume_session


class ResumeFixtures:
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        root.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.first = self.create_saved("first.jsonl", "aaaa1111", "first question", 10)
        self.second = self.create_saved("second.jsonl", "bbbb2222", "second question", 20)
        self.config = CodingSessionConfig(
            provider=FakeProvider(), model="fake", cwd=self.directory,
            storage=self.first, system="test", tools=[], auto_compact_enabled=False,
        )

    def create_saved(self, filename, session_id, question, timestamp):
        storage = JsonlSessionStorage(self.directory / filename)
        storage.append(SessionInfoEntry(session_id=session_id, cwd=str(self.directory), created_at=0))
        storage.append(MessageEntry(message=UserMessage(content=question), timestamp=timestamp))
        storage.append(MessageEntry(message=AssistantMessage(content="saved answer"), timestamp=timestamp))
        os.utime(storage.path, (timestamp, timestamp))
        return storage


class SessionCatalogTests(ResumeFixtures, unittest.TestCase):
    def test_catalog_sorts_sessions_and_skips_invalid_files(self) -> None:
        (self.directory / "broken.jsonl").write_text("not json", encoding="utf-8")
        (self.directory / "empty.jsonl").write_text("", encoding="utf-8")
        sessions, errors = list_saved_sessions(self.directory)
        self.assertEqual([session.session_id for session in sessions], ["bbbb2222", "aaaa1111"])
        self.assertEqual(sessions[0].updated_at, 20)
        self.assertEqual(len(errors), 2)

    def test_catalog_does_not_load_or_validate_message_bodies(self) -> None:
        with self.second.path.open("a", encoding="utf-8") as file:
            file.write("invalid trailing record\n")
        with patch.object(JsonlSessionStorage, "read_all", side_effect=AssertionError("History was read")):
            sessions, errors = list_saved_sessions(self.directory)
        self.assertEqual({session.session_id for session in sessions}, {"aaaa1111", "bbbb2222"})
        self.assertEqual(errors, [])
        with self.assertRaises(ValueError):
            resume_session(self.config, "second.jsonl")

    def test_resume_by_filename_id_and_prefix_preserves_runtime_config(self) -> None:
        for selector in ("second.jsonl", "bbbb2222", "bbbb"):
            with self.subTest(selector=selector):
                candidate = resume_session(self.config, selector)
                self.assertEqual(candidate.storage.path, self.second.path)
                self.assertIs(candidate.config.provider, self.config.provider)
                self.assertEqual(candidate.config.model, self.config.model)
                self.assertEqual(candidate.config.cwd, self.config.cwd)
                self.assertIs(candidate.config.tools, self.config.tools)
                self.assertEqual(candidate.messages[0].content, "second question")

    def test_unknown_session_is_not_created(self) -> None:
        before = {path.name for path in self.directory.iterdir()}
        with self.assertRaisesRegex(ValueError, "Session not found"):
            resume_session(self.config, "missing.jsonl")
        self.assertEqual({path.name for path in self.directory.iterdir()}, before)

    def test_duplicate_ids_are_rejected_but_filename_is_unambiguous(self) -> None:
        self.create_saved("duplicate.jsonl", "bbbb2222", "other question", 30)
        for selector in ("bbbb2222", "bbbb"):
            with self.assertRaisesRegex(ValueError, "Ambiguous session"):
                resume_session(self.config, selector)
        self.assertEqual(resume_session(self.config, "second.jsonl").storage.path, self.second.path)


class TuiResumeTests(ResumeFixtures, unittest.IsolatedAsyncioTestCase):
    async def submit(self, app, pilot, text):
        app.query_one("#prompt", Input).value = text
        await pilot.press("enter")

    async def test_first_prompt_creates_one_session_and_later_prompts_reuse_it(self) -> None:
        existing = {path.name for path in self.directory.glob("*.jsonl")}
        app = TUIApp(config=self.config)
        async with app.run_test() as pilot:
            await self.submit(app, pilot, "   ")
            await self.submit(app, pilot, "/resume")
            await self.submit(app, pilot, "/resume missing.jsonl")
            self.assertIsNone(app.session)
            self.assertEqual({path.name for path in self.directory.glob("*.jsonl")}, existing)

            await self.submit(app, pilot, "first new question")
            await asyncio.wait_for(app._prompt_worker.wait(), timeout=3)
            session = app.session
            self.assertEqual(session.storage.path.stem, session.state.session_info.session_id)
            self.assertNotIn(session.storage.path.name, existing)
            self.assertEqual(len(session.storage.read_all()), 3)

            await self.submit(app, pilot, "second new question")
            await asyncio.wait_for(app._prompt_worker.wait(), timeout=3)
            self.assertIs(app.session, session)
            self.assertEqual(len(session.storage.read_all()), 5)
            self.assertEqual(len(list(self.directory.glob("*.jsonl"))), len(existing) + 1)

    async def test_resume_before_first_prompt_does_not_create_an_extra_session(self) -> None:
        existing = {path.name for path in self.directory.glob("*.jsonl")}
        app = TUIApp(config=self.config)
        async with app.run_test() as pilot:
            await self.submit(app, pilot, "/resume second.jsonl")
            self.assertEqual(app.session.storage.path, self.second.path)
            await self.submit(app, pilot, "continue old conversation")
            await asyncio.wait_for(app._prompt_worker.wait(), timeout=3)
            self.assertEqual({path.name for path in self.directory.glob("*.jsonl")}, existing)
            self.assertEqual(len(self.second.read_all()), 5)

    async def test_explicit_new_file_is_created_on_first_prompt(self) -> None:
        storage = JsonlSessionStorage(self.directory / "named.jsonl")
        app = TUIApp(config=replace(self.config, storage=storage), create_new=False)
        async with app.run_test() as pilot:
            self.assertFalse(storage.path.exists())
            await self.submit(app, pilot, "first question")
            await asyncio.wait_for(app._prompt_worker.wait(), timeout=3)
            self.assertEqual(app.session.storage.path, storage.path)
            self.assertEqual(len(storage.read_all()), 3)

    async def test_list_and_switch_restore_full_history_then_write_to_target(self) -> None:
        recent = MessageEntry(message=UserMessage(content="recent question"))
        self.second.append(recent)
        self.second.append(MessageEntry(message=AssistantMessage(content="recent answer")))
        self.second.append(CompactionEntry(summary="old summary", first_kept_entry_id=recent.id))
        original_bytes = self.first.path.read_bytes()
        app = TUIApp(CodingSession.load(self.config))
        async with app.run_test() as pilot:
            await self.submit(app, pilot, "/resume")
            console = Console(file=StringIO(), record=True, width=100)
            for widget in app.query_one("#transcript").children:
                console.print(widget.content)
            rendered = console.export_text()
            self.assertIn("first.jsonl", rendered)
            self.assertIn("second.jsonl", rendered)
            self.assertIsNone(app._prompt_worker)
            self.assertEqual(self.first.path.read_bytes(), original_bytes)
            self.assertEqual(len(app.state.items), 2)

            await self.submit(app, pilot, "/resume bbbb")
            self.assertEqual(app.session.storage.path, self.second.path)
            self.assertIs(app.adapter.state, app.state)
            self.assertEqual(
                [item.text for item in app.state.items],
                ["second question", "saved answer", "recent question", "recent answer"],
            )
            self.assertNotIn("second question", [message.content for message in app.session.messages])
            self.assertEqual(str(app.query_one("#status", Static).content), "Resumed: second.jsonl")

            await self.submit(app, pilot, "new question")
            await asyncio.wait_for(app._prompt_worker.wait(), timeout=3)
            self.assertEqual(len(app.state.items), 6)
            self.assertEqual(app.state.items[-1].text, "Hello from FakeProvider.")
            saved_messages = [
                entry.message.content for entry in self.second.read_all() if isinstance(entry, MessageEntry)
            ]
            self.assertEqual(saved_messages[-2:], ["new question", "Hello from FakeProvider."])
            self.assertEqual(self.first.path.read_bytes(), original_bytes)
            self.assertFalse(any(content.startswith("/resume") for content in saved_messages))

            await self.submit(app, pilot, "/resume first.jsonl")
            self.assertEqual(len(app.state.items), 2)
            self.assertEqual(app.session.storage.path, self.first.path)

    async def test_failed_load_preserves_current_session_and_display(self) -> None:
        app = TUIApp(CodingSession.load(self.config))
        async with app.run_test() as pilot:
            session, state, adapter = app.session, app.state, app.adapter
            await self.submit(app, pilot, "/resume missing.jsonl")
            self.assertIn("Session not found", str(app.query_one("#status", Static).content))
            with patch("coding.session_catalog.CodingSession.load", side_effect=ValueError("invalid session")):
                await self.submit(app, pilot, "/resume second.jsonl")
            self.assertEqual(str(app.query_one("#status", Static).content), "Error: invalid session")
            self.assertIs(app.session, session)
            self.assertIs(app.state, state)
            self.assertIs(app.adapter, adapter)

    async def test_resume_is_blocked_while_prompt_is_running(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class WaitingProvider(FakeProvider):
            async def stream_response(self, **kwargs):
                started.set()
                await release.wait()
                async for event in super().stream_response(**kwargs):
                    yield event

        app = TUIApp(CodingSession.load(replace(self.config, provider=WaitingProvider())))
        async with app.run_test() as pilot:
            await self.submit(app, pilot, "question")
            await asyncio.wait_for(started.wait(), timeout=2)
            active = app.session
            await self.submit(app, pilot, "/resume second.jsonl")
            self.assertIs(app.session, active)
            self.assertEqual(app.query_one("#prompt", Input).value, "/resume second.jsonl")
            self.assertEqual(str(app.query_one("#status", Static).content), "Working... Please wait.")
            release.set()
            await asyncio.wait_for(app._prompt_worker.wait(), timeout=3)
            await pilot.press("enter")
            self.assertEqual(app.session.storage.path, self.second.path)
