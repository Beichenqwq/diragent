"""自动压缩阈值的窗口预留、配置优先级及无副作用读取。"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

from ai.provider import ModelProvider
from agent.session import JsonlSessionStorage
from coding.context_window import (
    DEFAULT_COMPACTION_KEEP_RECENT_TOKENS, DEFAULT_COMPACTION_RESERVE_TOKENS,
    DEFAULT_CONTEXT_WINDOW_TOKENS, auto_compaction_threshold_for_context_window,
)
from coding.session import CodingSession, CodingSessionConfig


class AutoCompactionThresholdTests(unittest.TestCase):
    def test_default_window_and_reserve_match_tau(self) -> None:
        self.assertEqual(DEFAULT_CONTEXT_WINDOW_TOKENS, 128000)
        self.assertEqual(DEFAULT_COMPACTION_RESERVE_TOKENS, 16384)
        self.assertEqual(auto_compaction_threshold_for_context_window(DEFAULT_CONTEXT_WINDOW_TOKENS), 111616)
        self.assertEqual(DEFAULT_COMPACTION_KEEP_RECENT_TOKENS, 20000)

    def test_window_boundaries_and_nonpositive_windows(self) -> None:
        cases = [(-1, None), (0, None), (1, 1), (16383, 1), (16384, 1),
                 (16385, 1), (16386, 2), (64000, 47616)]
        for window, expected in cases:
            with self.subTest(window=window):
                self.assertEqual(auto_compaction_threshold_for_context_window(window), expected)


class CodingSessionThresholdTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        temp_root.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(directory.cleanup)
        cwd = Path(directory.name)
        self.provider = Mock(spec=ModelProvider)
        self.config = CodingSessionConfig(provider=self.provider, model="fake", cwd=cwd, tools=[],
                                          system="coding", storage=JsonlSessionStorage(cwd / "session.jsonl"))

    def test_defaults_and_reads_do_not_request_or_append(self) -> None:
        session = CodingSession.load(self.config)
        saved = self.config.storage.path.read_bytes()
        for _ in range(2):
            self.assertTrue(session.auto_compaction_enabled)
            self.assertEqual(session.context_window_tokens, 128000)
            self.assertEqual(session.auto_compact_token_threshold, 111616)
        self.provider.stream_response.assert_not_called()
        self.assertEqual(self.config.storage.path.read_bytes(), saved)

    def test_explicit_threshold_overrides_window_even_when_zero(self) -> None:
        for threshold in (0, 12345, 200000):
            with self.subTest(threshold=threshold):
                session = CodingSession.load(replace(self.config, context_window_tokens=64000,
                                                     auto_compact_token_threshold=threshold))
                self.assertEqual(session.auto_compact_token_threshold, threshold)

    def test_disabled_overrides_explicit_and_derived_thresholds(self) -> None:
        for threshold in (None, 12345):
            with self.subTest(threshold=threshold):
                session = CodingSession.load(replace(self.config, auto_compact_enabled=False,
                                                     auto_compact_token_threshold=threshold))
                self.assertFalse(session.auto_compaction_enabled)
                self.assertIsNone(session.auto_compact_token_threshold)

    def test_configured_window_and_invalid_window_follow_tau_rules(self) -> None:
        for window, expected in ((64000, 47616), (0, None), (-1, None)):
            with self.subTest(window=window):
                session = CodingSession.load(replace(self.config, context_window_tokens=window))
                self.assertEqual(session.context_window_tokens, window)
                self.assertEqual(session.auto_compact_token_threshold, expected)
        session = CodingSession.load(replace(self.config, context_window_tokens=0,
                                             auto_compact_token_threshold=500))
        self.assertEqual(session.auto_compact_token_threshold, 500)
