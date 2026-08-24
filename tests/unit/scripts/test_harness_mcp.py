"""Tests for scripts/_harness_mcp.py — LANE H-mcp of the full validation harness.

The load-bearing case here is `test_tools_list_matches_reply_despite_interleaved_notification`:
a fake streamable-http server that deliberately puts a server-initiated
notification (no "id") on the SSE stream BEFORE the actual tools/list reply.
The prior, uncommitted probe read only the first `data:` line and would have
reported "0 tools" against this exact fixture. `stage_mcp_handshake` must
walk the stream frame by frame, skip the notification, and return the real
reply.

FIX LANE 11 additions (two defects confirmed live against a real graph-os
server):

* DEFECT A — no auth: the live MCP endpoint 401s an unauthenticated
  ``initialize``. `stage_mcp_handshake` now takes an optional ``auth_token``
  threaded through all three requests as ``Authorization: Bearer <token>``.
  See ``test_bearer_attached_to_all_three_requests_when_supplied`` /
  ``test_bearer_omitted_when_not_supplied``.
* DEFECT B — CRLF blindness: the real server emits ``\\r\\n`` line endings.
  The old blank-line-boundary framer matched only bare ``\\n``, so a
  CRLF-terminated blank line never closed the event, and a server-initiated
  notification concatenated with the real reply into one unparseable JSON
  blob — which, if the decode error is silently swallowed, masquerades as a
  false "0 tools" (the exact regression this module exists to prevent). See
  ``test_crlf_and_lf_both_match_the_real_reply`` (parameterized over both
  line endings), ``test_unparseable_frame_raises_instead_of_zero_tools``, and
  ``test_large_multiline_payload_reassembles_correctly``.
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


def _sse_body(events: list[dict], *, eol: str = "\n") -> bytes:
    """One ``data:`` line per event (compact JSON), events separated by a
    blank line. ``eol`` is the line terminator the fake server emits on the
    wire — real graph-os emits ``\\r\\n``; the default ``\\n`` matches the
    original fixture's behaviour exactly for every pre-existing test."""
    parts = [f"data: {json.dumps(ev)}{eol}{eol}" for ev in events]
    return "".join(parts).encode("utf-8")


def _sse_body_multiline(events: list[dict], *, eol: str = "\n") -> bytes:
    """Pretty-print each event (``indent=2``) and split it across MANY
    ``data:`` continuation lines, exactly as the SSE spec's multi-line-data
    reconstruction rule expects (join the ``data:`` lines with ``\\n`` to
    get back the original text). Used to simulate the real server's ~133KB
    ``tools/list`` reply spanning many ``data:`` lines for one event."""
    parts = []
    for ev in events:
        text = json.dumps(ev, indent=2)
        for physical_line in text.split("\n"):
            parts.append(f"data: {physical_line}{eol}")
        parts.append(eol)  # blank line ends the event
    return "".join(parts).encode("utf-8")


class _FakeMcpHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Class-level knobs the tests toggle per-instance-of-server via subclassing.
    session_id = "fake-session-123"
    tool_names = ["ask", "find", "act", "why", "write", "manage", "find_tools"]
    interleave_notification = True
    seen_headers: list[dict] = []  # populated per request, inspected by tests
    eol = "\n"
    multiline_tools_list = False
    send_malformed_tools_list = False

    def log_message(self, format, *args):  # noqa: A002 — silence test server logs
        return

    def _send_sse(self, body: bytes, *, extra_headers: dict[str, str] | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

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
            body = _sse_body(events, eol=self.eol)
            extra = {"Mcp-Session-Id": self.session_id} if self.session_id else {}
            self._send_sse(body, extra_headers=extra)
        elif method == "notifications/initialized":
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif method == "tools/list":
            if self.send_malformed_tools_list:
                # Reproduces the EXACT live-confirmed regression: a
                # server-initiated notification concatenated directly onto
                # the real reply with no boundary between them, because the
                # blank-line separator never matched (CRLF-blind framer).
                # This blob is not valid JSON.
                bad = (
                    '{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}'
                    '{"jsonrpc":"2.0","id":'
                    + json.dumps(message["id"])
                    + ',"result":{"tools":[{"name":"ask"}]}}'
                )
                body = f"data: {bad}{self.eol}{self.eol}".encode()
                self._send_sse(body)
                return
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
            builder = _sse_body_multiline if self.multiline_tools_list else _sse_body
            body = builder(events, eol=self.eol)
            self._send_sse(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


@pytest.fixture
def fake_server():
    _FakeMcpHandler.seen_headers = []
    _FakeMcpHandler.eol = "\n"
    _FakeMcpHandler.multiline_tools_list = False
    _FakeMcpHandler.send_malformed_tools_list = False
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


# ---------------------------------------------------------------------------
# DEFECT B — CRLF blindness (regression tests)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eol", ["\n", "\r\n"], ids=["LF", "CRLF"])
def test_crlf_and_lf_both_match_the_real_reply(fake_server, eol):
    """The real graph-os server emits CRLF (confirmed live). Parameterized
    over both LF and CRLF so neither can regress: a notification interleaved
    ahead of the real reply must still be skipped and the correct tool count
    returned, regardless of which line ending closes each SSE frame."""
    harness_mcp = _harness_mcp()
    _FakeMcpHandler.eol = eol
    _FakeMcpHandler.interleave_notification = True
    _FakeMcpHandler.tool_names = [
        "ask",
        "find",
        "act",
        "why",
        "write",
        "manage",
        "find_tools",
        "graph_task_progress_app",
        "graph_trace_waterfall_app",
    ]

    result = harness_mcp.stage_mcp_handshake(fake_server, timeout=5.0)

    assert result.tool_count == 9
    assert "graph_task_progress_app" in result.tool_names_sample


def test_unparseable_frame_raises_instead_of_zero_tools(fake_server):
    """A frame that fails to parse as JSON (e.g. the exact live-confirmed
    concatenation of a notification directly onto the real reply, with no
    boundary between them) must surface as a loud McpHandshakeError — never
    be silently skipped into a false '0 tools' result."""
    harness_mcp = _harness_mcp()
    _FakeMcpHandler.interleave_notification = False
    _FakeMcpHandler.send_malformed_tools_list = True

    with pytest.raises(harness_mcp.McpHandshakeError) as excinfo:
        harness_mcp.stage_mcp_handshake(fake_server, timeout=5.0)

    # Never masquerades as a clean "zero tools" result — the parse failure
    # itself must be visible in the raised error.
    assert "zero tools" not in str(excinfo.value)
    assert "failed to parse" in str(excinfo.value)


def test_large_multiline_payload_reassembles_correctly(fake_server):
    """A single tools/list reply event whose data: spans many lines (~100KB,
    matching the real server's observed ~133KB reply) must reassemble into
    valid JSON via the data: continuation join, not get corrupted by
    over-aggressive whitespace stripping."""
    harness_mcp = _harness_mcp()
    _FakeMcpHandler.interleave_notification = True
    _FakeMcpHandler.multiline_tools_list = True
    _FakeMcpHandler.tool_names = [f"tool_{i:05d}" for i in range(3000)]

    result = harness_mcp.stage_mcp_handshake(fake_server, timeout=15.0)

    assert result.tool_count == 3000
    # tool_names_sample is the first 10 names sorted lexicographically.
    assert result.tool_names_sample == tuple(f"tool_{i:05d}" for i in range(10))


# ---------------------------------------------------------------------------
# DEFECT A — bearer auth threading
# ---------------------------------------------------------------------------
def test_bearer_attached_to_all_three_requests_when_supplied(fake_server):
    harness_mcp = _harness_mcp()
    _FakeMcpHandler.interleave_notification = True
    _FakeMcpHandler.tool_names = ["ask", "find"]

    harness_mcp.stage_mcp_handshake(
        fake_server, timeout=5.0, auth_token="s3cr3t-token"
    )

    assert len(_FakeMcpHandler.seen_headers) == 3
    for headers in _FakeMcpHandler.seen_headers:
        lowered = {k.lower(): v for k, v in headers.items()}
        assert lowered.get("authorization") == "Bearer s3cr3t-token"


def test_bearer_omitted_when_not_supplied(fake_server):
    harness_mcp = _harness_mcp()
    _FakeMcpHandler.interleave_notification = True
    _FakeMcpHandler.tool_names = ["ask", "find"]

    harness_mcp.stage_mcp_handshake(fake_server, timeout=5.0)

    assert len(_FakeMcpHandler.seen_headers) == 3
    for headers in _FakeMcpHandler.seen_headers:
        lowered = {k.lower() for k in headers}
        assert "authorization" not in lowered
