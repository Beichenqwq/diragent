import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import urllib.request

from evaluation.cli_proxy import MeteredEndpoint, normalize_usage
from evaluation.cli_compare import native_calls, task_prompt, write_audit, native_terminal_status
from evaluation.cli_cost import estimate_cost


class CLIComparisonTests(unittest.TestCase):
    def test_same_billed_tokens_across_native_protocols(self):
        anthropic = normalize_usage("anthropic", {"input_tokens": 100, "cache_read_input_tokens": 900,
                                                  "cache_creation_input_tokens": 0, "output_tokens": 20})
        responses = normalize_usage("responses", {"input_tokens": 1000,
                                "input_tokens_details": {"cached_tokens": 900}, "output_tokens": 20})
        self.assertEqual(anthropic, responses)
        self.assertEqual(responses["total_tokens"], 1020)

    def test_cache_creation_is_billed_input(self):
        usage = normalize_usage("anthropic", {"input_tokens": 5, "cache_creation_input_tokens": 100,
                                              "cache_read_input_tokens": 20, "output_tokens": 3})
        self.assertEqual(usage["input_tokens"], 125)
        self.assertEqual(usage["cached_input_tokens"], 20)

    def test_claude_stream_tool_and_repeated_history_are_one_call(self):
        stream = {"protocol": "anthropic", "response_events": [
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "tool_use", "id": "call-1", "name": "Bash", "input": {}}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"partial_json": '{"command":"git status"}'}}]}
        history = {"protocol": "anthropic", "request": {"messages": [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "call-1", "name": "Bash", "input": {"command": "git status"}}]}]}}
        self.assertEqual(native_calls([stream, history, history]),
                         [{"id": "call-1", "name": "Bash", "arguments": {"command": "git status"}}])

    def test_codex_keeps_last_call_and_deduplicates_history(self):
        item = {"type": "function_call", "call_id": "call-2", "name": "shell_command",
                "arguments": '{"command":"pytest"}'}
        records = [{"protocol": "responses", "response_events": [{"type": "response.output_item.done", "item": item}]},
                   {"protocol": "responses", "request": {"input": [item]}}]
        self.assertEqual(len(native_calls(records)), 1)
        self.assertEqual(native_calls(records)[0]["arguments"], {"command": "pytest"})

    def test_task_prompt_excludes_hidden_fields(self):
        prompt = task_prompt({"problem_statement": "Fix bug", "patch": "SECRET_ANSWER", "test_patch": "HIDDEN"})
        self.assertIn("Fix bug", prompt)
        self.assertNotIn("SECRET_ANSWER", prompt)
        self.assertNotIn("HIDDEN", prompt)

    def test_cost_counts_cache_hit_once_and_includes_all_output(self):
        costs = estimate_cost({"input_tokens": 1_000_000, "cached_input_tokens": 900_000,
                               "output_tokens": 20_000})
        self.assertAlmostEqual(costs["peak"], 0.0594)
        self.assertAlmostEqual(costs["off_peak"], 0.0297)

    def test_invalid_cache_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            estimate_cost({"input_tokens": 10, "cached_input_tokens": 20})

    def test_anthropic_server_search_is_auditable(self):
        calls = native_calls([{"protocol": "anthropic", "response_events": [
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "server_tool_use", "id": "search-1", "name": "web_search", "input": {}}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"partial_json": '{"query":"python docs"}'}}]}])
        self.assertEqual(calls[0]["arguments"], {"query": "python docs"})

    def test_answer_candidate_keeps_corresponding_tool_output(self):
        records = [{"protocol": "anthropic", "request": {"messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "fetch-1", "name": "Bash",
                 "input": {"command": "curl https://github.com/django/django/pull/123.patch"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "fetch-1",
                                           "content": "downloaded diff content", "is_error": False}]}
        ]}}]
        with tempfile.TemporaryDirectory() as tmp:
            audit = write_audit(Path(tmp), records, {"repo": "django/django"})
        self.assertEqual(audit["candidate_count"], 1)
        self.assertEqual(audit["commands"][0]["output_excerpt"], "downloaded diff content")
        self.assertTrue(audit["commands"][0]["tool_ok"])
        self.assertEqual(audit["status"], "needs_review")

    def test_native_completion_is_independent_of_wrapper_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text('{"type":"item.completed"}\n{"type":"turn.completed","usage":{}}\n')
            self.assertEqual(native_terminal_status("codex", path), "completed")
            path.write_text('{"type":"turn.failed","error":{"message":"API error"}}\n')
            self.assertEqual(native_terminal_status("codex", path), "error")
            path.write_text('{"type":"result","subtype":"error_max_turns","is_error":true}\n')
            self.assertEqual(native_terminal_status("claude-code", path), "error")


class NativeProtocolMeterTests(unittest.TestCase):
    def test_split_sse_preserves_payload_and_counts_one_terminal_usage(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 7, "cache_read_input_tokens": 100}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "OK"}},
            {"type": "message_delta", "usage": {"output_tokens": 2}},
            {"type": "message_stop"},
        ]
        wire = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        chunks = [wire[:13], wire[13:91], wire[91:]]
        captured = {}

        class Response:
            status = 200

            def getheader(self, name, default=""):
                return "text/event-stream" if name == "Content-Type" else default

            def getheaders(self):
                return [("Content-Type", "text/event-stream")]

            def read1(self, _size):
                return chunks.pop(0) if chunks else b""

        class Connection:
            debuglevel = 0

            def __init__(self, *_args, **_kwargs):
                pass

            def request(self, method, path, body, headers):
                captured.update(method=method, path=path, body=json.loads(body), headers=headers)

            def getresponse(self):
                return Response()

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp, patch("evaluation.cli_proxy.http.client.HTTPSConnection", Connection):
            endpoint = MeteredEndpoint("REAL_TEST_KEY", Path(tmp))
            original = {"model": "deepseek-flash", "messages": [{"role": "user", "content": "hello"}],
                        "tools": [{"name": "Bash", "input_schema": {"type": "object"}}], "stream": True}
            try:
                request = urllib.request.Request(endpoint.base_url + "/anthropic/v1/messages",
                        data=json.dumps(original).encode(), headers={"x-api-key": endpoint.token,
                                                                   "Content-Type": "application/json"})
                with opener.open(request, timeout=5) as response:
                    self.assertEqual(response.read(), wire)
            finally:
                endpoint.close()
            self.assertEqual(captured["body"]["messages"], original["messages"])
            self.assertEqual(captured["body"]["tools"], original["tools"])
            self.assertEqual(captured["body"]["thinking"], {"type": "disabled"})
            self.assertEqual(captured["headers"]["Authorization"], "Bearer REAL_TEST_KEY")
            self.assertEqual(len(endpoint.records), 1)
            self.assertTrue(endpoint.records[0]["complete"])
            self.assertEqual(endpoint.records[0]["usage"]["input_tokens"], 107)
            self.assertEqual(endpoint.records[0]["usage"]["output_tokens"], 2)
            self.assertNotIn("REAL_TEST_KEY", (Path(tmp) / "api-ledger.jsonl").read_text())
