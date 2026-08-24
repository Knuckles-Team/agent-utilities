"""MCP-handshake validation stage for the committed full_validation_harness.

LANE H-mcp of the full-validation-harness program. Exposes exactly one stage —
:func:`stage_mcp_handshake` — that performs a REAL protocol handshake against a
live graph-os MCP streamable-http endpoint:

    initialize -> notifications/initialized -> tools/list

and fails if the matched ``tools/list`` reply reports zero tools.

WHY THIS EXISTS
----------------
The previous, uncommitted probe (``full-probe.py``, never checked in — see
handoff notes) read only the *first* ``data:`` line of the streamable-http
SSE response and treated it as the reply. A streamable-http server is free to
interleave server-initiated notifications ahead of the actual JSON-RPC reply
on the same SSE stream, so that probe sometimes parsed a notification instead
of the ``tools/list`` result, reported "0 tools", and produced a false-red (or,
worse, a stale cached "0 tools" masking a false-green elsewhere). The fix is
mechanical: parse the stream frame by frame and match every reply to its
request by JSON-RPC ``id`` (see :func:`_read_matched_reply`).

REUSE
-----
Host/TLS/redirect policy is intentionally NOT reimplemented here — it is
imported from ``scripts/validate_mcp_config.py``, which already enforces:
scheme/port sanity, loopback-only for plaintext ``http://``, a bounded
response size, and hard redirect rejection (an MCP endpoint must never
silently follow a cross-origin redirect).

USAGE (as a library — this is what the harness driver imports)
----------------------------------------------------------------
    from scripts._harness_mcp import stage_mcp_handshake, McpHandshakeError

    try:
        result = stage_mcp_handshake("http://127.0.0.1:8004/mcp")
    except McpHandshakeError as exc:
        ...  # driver reports FAIL + exc, exits with this stage's number
    else:
        ...  # result.tool_count, result.tool_names_sample, ...

This module never calls ``sys.exit`` — the driver (a later lane,
``scripts/full_validation_harness.py``) owns the stage-numbering / exit-code
contract described in ``scripts/delegation_probe.py``.

USAGE (standalone, for manual/offline testing outside the pod)
------------------------------------------------------------------
    python3 scripts/_harness_mcp.py --endpoint http://127.0.0.1:8004/mcp
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from scripts.validate_mcp_config import (
    _RejectRedirects,
    _tls_context,
    _validated_probe_host,
)

# The MCP streamable-http transport's header names (lowercase per the spec;
# HTTP headers are case-insensitive on both the wire and in every dict/Message
# object involved, so the exact casing used here doesn't matter).
_SESSION_ID_HEADER = "Mcp-Session-Id"
_PROTOCOL_VERSION_HEADER = "MCP-Protocol-Version"
_AUTHORIZATION_HEADER = "Authorization"

DEFAULT_PROTOCOL_VERSION = "2025-06-18"
DEFAULT_TIMEOUT = 15.0
# tools/list for a 750-tool fleet is a few hundred KB of JSON; bound it well
# above that so a legitimate reply is never truncated, but still bounded.
DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class McpHandshakeError(RuntimeError):
    """The initialize -> notifications/initialized -> tools/list sequence failed."""


@dataclasses.dataclass(frozen=True)
class McpHandshakeResult:
    """Structured, successful outcome of :func:`stage_mcp_handshake`."""

    endpoint: str
    session_id: str | None
    protocol_version: str
    server_name: str
    server_version: str
    tool_count: int
    tool_names_sample: tuple[str, ...]


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=_tls_context()),
        _RejectRedirects(),
    )


def _ids_match(a: Any, b: Any) -> bool:
    """JSON-RPC ids round-trip as numbers OR strings depending on the server;
    compare loosely rather than assuming the exact type we sent comes back."""
    return a == b or str(a) == str(b)


def _iter_sse_data_blocks(resp: Any, max_bytes: int) -> Iterator[str]:
    """Yield each SSE event's accumulated ``data:`` payload as one string.

    Frames strictly on the blank-line event boundary (SSE spec). Multiple
    ``data:`` lines within one event are newline-joined. Non-data lines
    (``event:``, ``id:``, ``retry:``, comments starting with ``:``) are
    ignored — JSON-RPC id-matching only needs the payload.

    CRLF SAFETY (this is the fix for DEFECT B, confirmed live against a real
    graph-os server): the SSE spec permits a line to end in ``\\n``, ``\\r\\n``,
    OR a bare ``\\r``, and the real server emits ``\\r\\n``. The whole bounded
    body is read up front and every line ending is normalised to ``\\n``
    BEFORE framing on the blank-line boundary. Framing on a literal blank
    line without normalising first is exactly the bug: matching only ``\\n``
    against an unnormalised ``\\r\\n\\r\\n`` stream never finds the boundary,
    so a server-initiated notification and the real reply concatenate into
    one unparseable JSON blob — see the module docstring's "WHY THIS EXISTS"
    and the confirmed live evidence in the lane's fix notes (frames ==
    ``notifications/tools/list_changed`` + the real id-matched reply,
    TOOL COUNT=85 once normalised; TOOL COUNT=0 when the concatenated blob's
    decode failure was silently swallowed instead).

    Only a single leading space after ``data:`` is stripped, per the SSE
    spec's exact reconstruction rule (§9.2.6) — not arbitrary leading
    whitespace — so a payload that legitimately spans many ``data:`` lines
    (observed ~133KB against the real server) reassembles byte-for-byte via
    the same ``"\\n".join`` the spec itself prescribes for multi-line data.
    """
    raw = resp.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise McpHandshakeError("SSE stream exceeded the response size boundary")
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    data_lines: list[str] = []
    for line in text.split("\n"):
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith("data:"):
            value = line[len("data:") :]
            if value.startswith(" "):
                value = value[1:]  # SSE spec: strip exactly ONE leading space
            data_lines.append(value)
        # else: event:/id:/retry:/comment framing lines — irrelevant to us.
    if data_lines:
        yield "\n".join(data_lines)


def _read_matched_reply(resp: Any, expected_id: Any, max_bytes: int) -> dict:
    """Return the JSON-RPC reply whose ``id`` matches ``expected_id``.

    THIS IS THE WHOLE POINT OF THIS MODULE. The SSE stream for one POST can
    carry server-initiated notifications ahead of the actual reply; a
    notification has no ``id`` at all. Reading only the first frame — as the
    dead, uncommitted probe did — can grab a notification and misreport a
    healthy server as broken (or hide a genuinely empty tool surface).  We
    walk every frame, skip anything without a matching ``id``, and only
    return once we find the real reply.

    A frame that fails to parse as JSON is NEVER silently skipped into a
    false "0 tools" — that is precisely how DEFECT B produced a confident
    wrong answer (a CRLF-blind framer concatenated a notification and the
    real reply into one invalid blob; see ``_iter_sse_data_blocks``'s
    docstring). Each parse failure is recorded with a truncated preview of
    the offending block; if no frame ever matches ``expected_id``, every
    recorded failure is folded into the raised :class:`McpHandshakeError` so
    the root cause is loud, not invisible.
    """
    content_type = resp.headers.get("Content-Type", "") or ""
    if "text/event-stream" in content_type:
        parse_failures: list[str] = []
        for block in _iter_sse_data_blocks(resp, max_bytes):
            try:
                message = json.loads(block)
            except json.JSONDecodeError as exc:
                preview = block if len(block) <= 200 else block[:200] + "...(truncated)"
                parse_failures.append(f"{exc}: {preview!r}")
                continue
            if not isinstance(message, dict):
                continue
            if "id" in message and _ids_match(message["id"], expected_id):
                return message
            # else: a notification (no "id") or a reply to a different
            # in-flight request on this connection — ignore/skip and keep
            # reading, exactly per the lane's core requirement.
        detail = f"SSE stream closed without a reply matching id={expected_id!r}"
        if parse_failures:
            detail += (
                f" — {len(parse_failures)} frame(s) failed to parse as JSON "
                "(never silently ignored): " + " | ".join(parse_failures)
            )
        raise McpHandshakeError(detail)

    # A compliant streamable-http server may also answer with a single plain
    # JSON body (Content-Type: application/json) instead of opening an SSE
    # stream for a request that has exactly one reply.
    raw = resp.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise McpHandshakeError("response exceeded the size boundary")
    try:
        message = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise McpHandshakeError(f"non-JSON response body: {exc}") from exc
    if not isinstance(message, dict) or not _ids_match(message.get("id"), expected_id):
        got = message.get("id") if isinstance(message, dict) else message
        raise McpHandshakeError(
            f"JSON reply id mismatch: expected {expected_id!r}, got {got!r}"
        )
    return message


def _post(
    opener: urllib.request.OpenerDirector,
    endpoint: str,
    message: dict,
    *,
    session_id: str | None,
    protocol_version: str | None,
    timeout: float,
    auth_token: str | None = None,
):
    """POST one JSON-RPC message and return the still-open response object.

    Caller owns closing it (so it can stream-parse the body). Raises
    McpHandshakeError, chaining the original exception, on any transport
    failure so the driver's ``_chain()``-style reporting still sees the root
    cause.

    ``auth_token``, when supplied, is sent as ``Authorization: Bearer
    <token>`` — the live MCP endpoint requires this (DEFECT A: an
    unauthenticated ``initialize`` 401s). It is entirely optional so an
    endpoint with no auth in front of it still works unchanged, and it is
    NEVER logged or included in any exception message here.
    """
    body = json.dumps(message).encode("utf-8")
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if session_id:
        headers[_SESSION_ID_HEADER] = session_id
    if protocol_version:
        headers[_PROTOCOL_VERSION_HEADER] = protocol_version
    if auth_token:
        headers[_AUTHORIZATION_HEADER] = f"Bearer {auth_token}"
    req = urllib.request.Request(endpoint, data=body, method="POST", headers=headers)
    method = message.get("method", "?")
    try:
        return opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise McpHandshakeError(f"{method} failed: HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise McpHandshakeError(f"{method} failed: {exc}") from exc


def stage_mcp_handshake(
    endpoint: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    client_name: str = "full_validation_harness",
    client_version: str = "1",
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    auth_token: str | None = None,
) -> McpHandshakeResult:
    """Run the real MCP handshake (``initialize`` -> ``notifications/initialized``
    -> ``tools/list``) against ``endpoint`` and return a structured result.

    Parameters
    ----------
    endpoint:
        The graph-os streamable-http URL (e.g. ``http://127.0.0.1:8004/mcp``).
        Never hardcoded — always supplied by the caller.
    timeout:
        Per-request timeout in seconds, applied to every one of the three
        requests independently. Must be in ``[0.1, 120]`` (same policy floor
        as ``scripts/validate_mcp_config.py``'s ``live_probe``).
    protocol_version:
        The ``protocolVersion`` offered in ``initialize``. The server's
        negotiated version (``result.protocolVersion`` from the ``initialize``
        reply, falling back to what we offered if absent) is what is actually
        used — as the ``MCP-Protocol-Version`` header — on the two requests
        that follow, per the streamable-http transport's negotiation contract.
    auth_token:
        DEFECT A fix: optional bearer, sent as ``Authorization: Bearer
        <auth_token>`` on ALL THREE requests (``initialize``,
        ``notifications/initialized``, ``tools/list``) when supplied. The
        live MCP endpoint requires this — an unauthenticated ``initialize``
        401s. Left as ``None`` by default so an endpoint with no auth in
        front of it keeps working unchanged. Never logged, never echoed into
        any error message.

    Returns
    -------
    McpHandshakeResult
        On success: the negotiated protocol version, the session id (if the
        server issued one via the ``Mcp-Session-Id`` response header — some
        deployments are stateless and issue none), server identity, and the
        tool count/name sample from the matched ``tools/list`` reply.

    Raises
    ------
    McpHandshakeError
        On any transport failure, a JSON-RPC error reply, a reply that never
        matches the request id, or — the specific regression this lane
        exists to catch — a matched ``tools/list`` reply reporting zero
        tools.
    """
    host = _validated_probe_host(endpoint)
    if not host:
        raise McpHandshakeError(f"endpoint policy rejected: {endpoint!r}")
    if not 0.1 <= timeout <= 120:
        raise McpHandshakeError(f"timeout out of policy range: {timeout!r}")

    opener = _opener()
    ids = itertools.count(1)

    # 1. initialize
    init_id = next(ids)
    init_message = {
        "jsonrpc": "2.0",
        "id": init_id,
        "method": "initialize",
        "params": {
            "protocolVersion": protocol_version,
            "capabilities": {},
            "clientInfo": {"name": client_name, "version": client_version},
        },
    }
    resp = _post(
        opener,
        endpoint,
        init_message,
        session_id=None,
        protocol_version=None,
        timeout=timeout,
        auth_token=auth_token,
    )
    try:
        if _validated_probe_host(resp.geturl()) != host:
            raise McpHandshakeError("endpoint origin changed on redirect")
        session_id = resp.headers.get(_SESSION_ID_HEADER)
        init_reply = _read_matched_reply(resp, init_id, max_response_bytes)
    finally:
        resp.close()

    if "error" in init_reply:
        raise McpHandshakeError(
            f"initialize returned a JSON-RPC error: {init_reply['error']!r}"
        )
    init_result = init_reply.get("result")
    if not isinstance(init_result, dict):
        raise McpHandshakeError("initialize reply had no result object")
    negotiated_version = str(init_result.get("protocolVersion") or protocol_version)
    server_info = init_result.get("serverInfo") or {}

    # 2. notifications/initialized — a notification: no "id", no reply body
    # to parse. Per the transport spec the server answers 202 Accepted.
    initialized_message = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }
    resp = _post(
        opener,
        endpoint,
        initialized_message,
        session_id=session_id,
        protocol_version=negotiated_version,
        timeout=timeout,
        auth_token=auth_token,
    )
    try:
        if not (200 <= resp.status < 300):
            raise McpHandshakeError(
                f"notifications/initialized failed: HTTP {resp.status}"
            )
    finally:
        resp.close()

    # 3. tools/list — the reply that actually matters.
    tools_id = next(ids)
    tools_message = {
        "jsonrpc": "2.0",
        "id": tools_id,
        "method": "tools/list",
        "params": {},
    }
    resp = _post(
        opener,
        endpoint,
        tools_message,
        session_id=session_id,
        protocol_version=negotiated_version,
        timeout=timeout,
        auth_token=auth_token,
    )
    try:
        tools_reply = _read_matched_reply(resp, tools_id, max_response_bytes)
    finally:
        resp.close()

    if "error" in tools_reply:
        raise McpHandshakeError(
            f"tools/list returned a JSON-RPC error: {tools_reply['error']!r}"
        )
    tools_result = tools_reply.get("result")
    if not isinstance(tools_result, dict):
        raise McpHandshakeError("tools/list reply had no result object")
    tools = tools_result.get("tools")
    if not isinstance(tools, list):
        raise McpHandshakeError("tools/list result had no tools array")
    if len(tools) == 0:
        raise McpHandshakeError(
            "tools/list matched reply reports zero tools — the handshake "
            "completed but the server exposes no tool surface"
        )

    names = sorted(
        t.get("name", "") for t in tools if isinstance(t, dict) and t.get("name")
    )

    return McpHandshakeResult(
        endpoint=endpoint,
        session_id=session_id,
        protocol_version=negotiated_version,
        server_name=str(server_info.get("name", "")),
        server_version=str(server_info.get("version", "")),
        tool_count=len(tools),
        tool_names_sample=tuple(names[:10]),
    )


def _emit(stage: str, ok: bool, detail: str = "", elapsed: float | None = None) -> None:
    """Same reporting convention as ``scripts/delegation_probe.py``, for the
    standalone CLI below — the driver that later imports this module has its
    own copy and does not need this one."""
    mark = "PASS" if ok else "FAIL"
    t = f" [{elapsed:6.2f}s]" if elapsed is not None else ""
    print(f"  {mark:4s} {stage:9s}{t} {detail}"[:600], flush=True)


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point for manual/offline testing outside the pod."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--endpoint", required=True, help="graph-os streamable-http URL")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    p.add_argument("--protocol-version", default=DEFAULT_PROTOCOL_VERSION)
    p.add_argument(
        "--auth-token",
        default=None,
        help=(
            "optional bearer for the live MCP endpoint (DEFECT A: an "
            "unauthenticated initialize 401s against the real server); "
            "never echoed back in any output"
        ),
    )
    args = p.parse_args(argv)

    t0 = time.monotonic()
    try:
        result = stage_mcp_handshake(
            args.endpoint,
            timeout=args.timeout,
            protocol_version=args.protocol_version,
            auth_token=args.auth_token,
        )
    except McpHandshakeError as exc:
        _emit("mcp", False, str(exc), time.monotonic() - t0)
        return 1

    detail = (
        f"tools={result.tool_count} protocol={result.protocol_version} "
        f"server={result.server_name}/{result.server_version} "
        f"session={result.session_id!r} sample={list(result.tool_names_sample)}"
    )
    _emit("mcp", True, detail, time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
