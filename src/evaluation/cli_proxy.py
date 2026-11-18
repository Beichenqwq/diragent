"""Meter native CLI protocols without translating messages or tool calls."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time


def normalize_usage(protocol: str, usage: dict) -> dict:
    """Anthropic input excludes cache tokens; Responses input includes them."""
    if protocol == "anthropic":
        cached = usage.get("cache_read_input_tokens", 0)
        created = usage.get("cache_creation_input_tokens", 0)
        inputs = usage.get("input_tokens", 0) + cached + created
    else:
        inputs = usage.get("input_tokens", 0)
        cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
        created = 0
    outputs = usage.get("output_tokens", 0)
    return {"input_tokens": inputs, "cached_input_tokens": cached,
            "cache_creation_input_tokens": created, "output_tokens": outputs,
            "total_tokens": inputs + outputs}


class MeteredEndpoint:
    """One authenticated localhost endpoint per attempt; real key stays root-only.

    The only request-body adjustment is explicit disabling of thinking. In
    particular, Claude Code may omit `thinking` for third-party providers even
    with MAX_THINKING_TOKENS=0, whereas DeepSeek enables it when omitted.
    Full native prompts/tool schemas and SSE bytes are preserved.
    """

    def __init__(self, key: str, directory: Path):
        self.key = key
        self.directory = directory
        self.token = "sk-evaluation-" + secrets.token_hex(24)
        self.lock = threading.Lock()
        self.records: list[dict] = []
        self.active_requests = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.send_error(404)

            def do_POST(self):
                auth = self.headers.get("Authorization", "").removeprefix("Bearer ")
                if (auth or self.headers.get("x-api-key", "")) != owner.token:
                    self.send_error(403)
                    return
                if self.path.split("?")[0] not in ("/responses", "/v1/responses",
                                                        "/anthropic/v1/messages",
                                                        "/anthropic/v1/messages/count_tokens"):
                    self.send_error(404)
                    return
                connection = None
                record = {"started_at": datetime.now(timezone.utc).isoformat(),
                          "path": self.path, "usage": None, "complete": False}
                started = time.monotonic()
                with owner.lock:
                    owner.active_requests += 1
                response_started = False
                usage = {}
                protocol = "anthropic" if "/anthropic/" in self.path else "responses"
                try:
                    body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    encoding = self.headers.get("Content-Encoding", "")
                    if encoding == "gzip":
                        body = gzip.decompress(body)
                    elif encoding == "zstd":
                        import zstandard
                        body = zstandard.ZstdDecompressor().decompress(body)
                    request = json.loads(body)
                    model = request.get("model", "")
                    if model.removesuffix("[1m]") != "deepseek-flash":
                        raise ValueError(f"Evaluation forbids model fallback: {model}")
                    record["original_thinking"] = request.get("thinking", request.get("reasoning"))
                    if protocol == "anthropic":
                        request["thinking"] = {"type": "disabled"}
                    else:
                        request["reasoning"] = {**request.get("reasoning", {}), "effort": "none"}
                    record["protocol"] = protocol
                    record["request"] = request
                    body = json.dumps(request).encode()
                    headers = {k: v for k, v in self.headers.items() if k.lower() not in
                               ("host", "content-length", "content-encoding", "authorization",
                                "x-api-key", "accept-encoding", "connection")}
                    headers.update({"Authorization": "Bearer " + owner.key,
                                    "x-api-key": owner.key, "Accept-Encoding": "identity",
                                    "Content-Type": "application/json"})
                    connection = http.client.HTTPSConnection("api.deepseek.com", timeout=120)
                    connection.request("POST", self.path, body=body, headers=headers)
                    upstream = connection.getresponse()
                    record["http_status"] = upstream.status
                    self.send_response(upstream.status)
                    for k, v in upstream.getheaders():
                        if k.lower() not in ("transfer-encoding", "content-length", "connection", "content-encoding"):
                            self.send_header(k, v)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    response_started = True
                    is_stream = "text/event-stream" in upstream.getheader("Content-Type", "")
                    buffered = b""
                    events = []
                    client_alive = True
                    while chunk := upstream.read1(65536):
                        buffered += chunk
                        if is_stream:
                            while b"\n" in buffered:
                                line, buffered = buffered.split(b"\n", 1)
                                if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                                    event = json.loads(line[6:])
                                    events.append(event)
                                    if event.get("type") == "message_start":
                                        usage.update(event.get("message", {}).get("usage", {}))
                                    elif event.get("type") == "message_delta":
                                        usage.update(event.get("usage", {}))
                                    elif event.get("type") == "message_stop":
                                        record["complete"] = True
                                    elif event.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                                        usage.update(event.get("response", {}).get("usage") or {})
                                        record["complete"] = True
                        if client_alive:
                            try:
                                self.wfile.write(chunk)
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                # Finish metering an already billed request even if CLI stops reading.
                                client_alive = False
                    if not is_stream:
                        response = json.loads(buffered)
                        record["response"] = response
                        usage.update(response.get("usage") or {})
                        record["complete"] = 200 <= upstream.status < 300
                    record["response_events"] = events
                    if usage:
                        record["raw_usage"] = usage
                        record["usage"] = normalize_usage(protocol, usage)
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    if not response_started:
                        self.send_error(502, "Evaluation endpoint error")
                finally:
                    if connection is not None:
                        connection.close()
                    record["duration_seconds"] = round(time.monotonic() - started, 3)
                    with owner.lock:
                        record["request_number"] = len(owner.records) + 1
                        owner.records.append(record)
                        with (owner.directory / "api-ledger.jsonl").open("a", encoding="utf-8") as log:
                            log.write(json.dumps(record, ensure_ascii=False) + "\n")
                        owner.active_requests -= 1

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        deadline = time.monotonic() + 130
        while self.active_requests and time.monotonic() < deadline:
            time.sleep(0.1)
