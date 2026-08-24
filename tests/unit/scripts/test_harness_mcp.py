"""Tests for scripts/_harness_mcp.py — LANE H-mcp of the full validation harness.

The load-bearing case here is `test_tools_list_matches_reply_despite_interleaved_notification`:
a fake streamable-http server that deliberately puts a server-initiated
notification (no "id") on the SSE stream BEFORE the actual tools/list reply.
The prior, uncommitted probe read only the first `data:` line and would have
reported "0 tools" against this exact fixture. `stage_mcp_handshake` must
walk the stream frame by frame, skip the notification, and return the real
reply.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest


def _harness_mcp() -> ModuleType:
    source = Path(__file__).resolve().parents[3] / "scripts" / "_harness_mcp.py"
    spec = importlib.util.spec_from_file_location("_harness_mcp", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses (used with `from __future__ import annotations`) resolves
    # string annotations via sys.modules[cls.__module__] — register before
    # exec so the module-level @dataclass class body doesn't blow up.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sse_body(events: list[dict]) -> bytes:
    parts = [f"data: {json.dumps(ev)}\n\n" for ev in events]
    return "".join(parts).encode("utf-8")


class _FakeMcpHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Class-level knobs the tests toggle per-instance-of-server via subclassing.
    session_id = "fake-session-123"
    tool_names = ["ask", "find", "act", "why", "write", "manage", "find_tools"]
    interleave_notification = True
    seen_headers: list[dict] = []  # populated per request, inspected by tests

    def log_message(self, format, *args):  # noqa: A002 — silence test server logs
        return

    def do_POST(self):  # noqa: N802 — stdlib handler method name
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        message = json.loads(raw.decode("utf-8"))
        method = message.get("method")
        type(self).seen_headers.append(dict(self.headers.items()))

        if method == "initialize":
            events = []
            if self.interleave_notification:
                events.append(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/message",
                        "params": {"data": "warming up"},
                    }
                )
            events.append(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "serverInfo": {"name": "fake-graph-os", "version": "9.9"},
                    },
                }
            )
            body = _sse_body(events)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            if self.session_id:
                self.send_header("Mcp-Session-Id", self.session_id)
            self.end_headers()
            self.wfile.write(body)
        elif method == "notifications/initialized":
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif method == "tools/list":
            events = []
            if self.interleave_notification:
                events.append(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/message",
                        "params": {"data": "still enumerating tools"},
                    }
                )
            events.append(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"tools": [{"name": n} for n in self.tool_names]},
                }
            )
            body = _sse_body(events)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


@pytest.fixture
def fake_server():
    _FakeMcpHandler.seen_headers = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeMcpHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}/mcp"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_tools_list_matches_reply_despite_interleaved_notification(fake_server):
    """THE core requirement: a notification lands before the real reply on the
    SSE stream, and the matcher must still find the reply by id and report the
    correct tool count (not 0, and not the notification)."""
    harness_mcp = _harness_mcp()
    _FakeMcpHandler.interleave_notification = True
    _FakeMcpHandler.tool_names = [
        "ask",
        "find",
        "act",
        "why",
        "write",
        "manage",
        "find_tools",
    ]

    result = harness_mcp.stage_mcp_handshake(fake_server, timeout=5.0)

    assert result.tool_count == 7
    assert "ask" in result.tool_names_sample
    assert "find_tools" in result.tool_names_sample
    assert result.protocol_version == "2025-06-18"
    assert result.session_id == "fake-session-123"
    assert result.server_name == "fake-graph-os"


def test_session_id_propagated_on_subsequent_requests(fake_server):
    harness_mcp = _harness_mcp()
    _FakeMcpHandler.interleave_notification = True

    harness_mcp.stage_mcp_handshake(fake_server, timeout=5.0)

    # initialize, notifications/initialized, tools/list == 3 requests.
    assert len(_FakeMcpHandler.seen_headers) == 3
    init_headers, initialized_headers, tools_headers = _FakeMcpHandler.seen_headers
    # The session id only exists AFTER initialize's reply, so it cannot be on
    # the initialize request itself, but must be on both that follow.
    assert "mcp-session-id" not in {k.lower() for k in init_headers}
    assert initialized_headers.get("Mcp-Session-Id") == "fake-session-123"
    assert tools_headers.get("Mcp-Session-Id") == "fake-session-123"
    assert initialized_headers.get("Mcp-Protocol-Version") == "2025-06-18"
    assert tools_headers.get("Mcp-Protocol-Version") == "2025-06-18"


def test_zero_tools_fails_even_when_matched_correctly(fake_server):
    """A matched-but-empty tools/list reply must fail closed, not silently pass."""
    harness_mcp = _harness_mcp()
    _FakeMcpHandler.interleave_notification = True
    _FakeMcpHandler.tool_names = []

    with pytest.raises(harness_mcp.McpHandshakeError, match="zero tools"):
        harness_mcp.stage_mcp_handshake(fake_server, timeout=5.0)


def test_no_interleaving_still_works(fake_server):
    """Sanity: the matcher also handles the simple (non-interleaved) case."""
    harness_mcp = _harness_mcp()
    _FakeMcpHandler.interleave_notification = False
    _FakeMcpHandler.tool_names = ["ask", "find"]

    result = harness_mcp.stage_mcp_handshake(fake_server, timeout=5.0)

    assert result.tool_count == 2


def test_unreachable_endpoint_raises_handshake_error():
    harness_mcp = _harness_mcp()

    with pytest.raises(harness_mcp.McpHandshakeError):
        harness_mcp.stage_mcp_handshake("http://127.0.0.1:1/mcp", timeout=1.0)


def test_non_loopback_http_endpoint_is_policy_rejected():
    harness_mcp = _harness_mcp()

    with pytest.raises(harness_mcp.McpHandshakeError, match="policy rejected"):
        harness_mcp.stage_mcp_handshake("http://example.com/mcp", timeout=1.0)
