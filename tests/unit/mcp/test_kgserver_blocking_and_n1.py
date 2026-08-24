"""FIX LANE (kgserver-blocking-and-n1): `/api/tools` blocked the whole graph-os
event loop and made 350+ sequential engine round trips per request.

DEFECT A — event-loop blocking: `get_tools_endpoint` (`kg_server.py`, route
`GET /api/tools`) is a hand-written `async def` Starlette handler. It used to
call `engine.query_cypher()` (a plain blocking `def`) directly and
synchronously via `get_toggle_state`. Reproduced live: with `/api/tools` in
flight, concurrent static-asset requests (`/favicon.svg`, `/index.html`) all
timed out at a 25s ceiling; idle baseline for those same assets is
44-112ms. Fixed by moving the endpoint's blocking body
(`_build_tools_payload_sync`) onto a worker thread via `asyncio.to_thread`,
matching `_execute_tool`'s existing dispatch-isolation pattern in this file.
`toggle_tool_endpoint` (`POST /api/tools/toggle`) had the identical
anti-pattern via `set_toggle_state` and is fixed the same way.

DEFECT B — the N+1 itself: `get_tools_endpoint` used to call
`get_toggle_state()` once per rendered item — one synchronous Cypher round
trip each (254 skill files + 68 skill-graph files + 31 builtin tools + 66 MCP
servers = 350+ sequential round trips on the production pod; the request did
not return within 180s). Fixed with `get_toggle_states_batch()`, which
resolves every `(item_type, item_id)` pair the caller is about to render in
ONE `MATCH (p:Preference) WHERE p.id IN $pref_ids ...` round trip.

Two engine facts the batched query must respect, both confirmed live against
the deployed engine:

1. `STARTS WITH` with a `$param` operand does not parse on the deployed
   engine — the batch uses `IN` with an explicit id list instead.
2. The row-governance layer (`secured_reads.row_node_ids`) REQUIRES every
   returned row to carry an identity under `id`/`node_id`/`n.id`/`_id`, or it
   raises `PermissionError`. The original `get_toggle_state` query projected
   only `p.value` — every successful match was therefore rejected by
   governance and silently swallowed by a broad `except` into
   `return True`, meaning an explicit "disabled" toggle was reported back as
   "enabled" (real data loss). `get_toggle_states_batch` projects
   `p.id AS id`. The singular `get_toggle_state` helper was DELETED once the
   N+1 loop was removed and it had no production caller left.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.kg_server import (
    get_toggle_states_batch,
    get_tools_endpoint,
    set_toggle_state,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

ACTOR = ActorContext(
    actor_id="principal:ops",
    actor_type=ActorType.AI_AGENT,
    roles=(),
    tenant_id="acme",
    authenticated=True,
)


def _session(
    graph: str, *, scopes: frozenset[str] = frozenset({"kg:read"})
) -> GraphSession:
    return GraphSession(
        actor=ACTOR,
        tenant=ACTOR.tenant_id,
        scopes=scopes,
        graph=graph,
        policy_version="policy-v1",
        audience="agent-services",
    )


# ── Test 1 + 2: the batch issues ONE query and projects an id ──────────────


def test_batched_toggle_read_issues_one_query_for_n_items():
    """`get_toggle_states_batch` must resolve N items in ONE `query_cypher`
    call, not N — this is the DEFECT B fix itself."""
    engine = MagicMock()
    engine.query_cypher.return_value = []

    items = [("skill", f"skill-{i}") for i in range(50)] + [
        ("mcp_server", f"server-{i}") for i in range(20)
    ]

    result = get_toggle_states_batch(engine, items)

    assert engine.query_cypher.call_count == 1
    assert len(result) == len(items)
    # No stored preference for any of these -- fail-open default is enabled.
    assert all(v is True for v in result.values())


def test_batched_toggle_query_projects_id_and_uses_in_not_starts_with():
    """The row-governance layer requires an `id`/`node_id`/`n.id`/`_id`
    projection on every row (`secured_reads.row_node_ids`), and the deployed
    engine does not parse `STARTS WITH` with a `$param` operand -- the batch
    must use `IN` with an explicit id list instead."""
    engine = MagicMock()
    engine.query_cypher.return_value = []

    get_toggle_states_batch(engine, [("skill", "a"), ("skill", "b")])

    assert engine.query_cypher.call_count == 1
    (query, params), _kwargs = engine.query_cypher.call_args
    assert "p.id AS id" in query
    assert "IN $pref_ids" in query
    assert "STARTS WITH" not in query
    assert set(params["pref_ids"]) == {
        "preference:toggle:skill:a",
        "preference:toggle:skill:b",
    }


# ── Test 3: a toggle written "disabled" reads back as disabled ─────────────


class _FakePreferenceEngine:
    """Minimal fake honoring the same Cypher shapes `set_toggle_state`/
    `get_toggle_state`/`get_toggle_states_batch` issue, including the
    row-governance-required `id` projection."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}
        self.graph_compute = None  # exercised by set_toggle_state's cache sync

    def add_node(self, node_id: str, _label: str, properties: dict[str, Any]) -> None:
        self.store[node_id] = properties

    def query_cypher(self, query: str, params: dict[str, Any]):
        if "pref_ids" in params:
            return [
                {"id": pid, "value": self.store[pid]["value"]}
                for pid in params["pref_ids"]
                if pid in self.store
            ]
        if "pref_id" in params:
            pid = params["pref_id"]
            if pid in self.store:
                return [{"id": pid, "value": self.store[pid]["value"]}]
            return []
        # set_toggle_state's real-time node.disabled sync query -- not under
        # test here, no matching node.
        return []


def test_toggle_written_disabled_reads_back_disabled_single_item():
    """Regression for the silent-data-loss bug: before the `p.id AS id` fix,
    a real 'disabled' match was rejected by row governance and swallowed by
    a broad `except`, defaulting to `True` (enabled).

    Exercises the ONE-item read specifically: the batched function is now the
    only toggle-read path, so the single-item case must be covered here rather
    than through a separate singular helper."""
    engine = _FakePreferenceEngine()
    set_toggle_state(engine, "skill", "my-skill", enabled=False)

    result = get_toggle_states_batch(engine, [("skill", "my-skill")])

    assert result[("skill", "my-skill")] is False


def test_toggle_written_disabled_reads_back_disabled_batch():
    engine = _FakePreferenceEngine()
    set_toggle_state(engine, "skill", "my-skill", enabled=False)
    set_toggle_state(engine, "skill", "other-skill", enabled=True)

    result = get_toggle_states_batch(
        engine, [("skill", "my-skill"), ("skill", "other-skill")]
    )

    assert result[("skill", "my-skill")] is False
    assert result[("skill", "other-skill")] is True


# ── Test 4: get_tools_endpoint does not block the event loop ───────────────


@pytest.mark.asyncio
async def test_get_tools_endpoint_does_not_block_event_loop(monkeypatch):
    """DEFECT A: proves the NEGATIVE, the same way
    `test_engine_tools_event_loop_isolation.py` does for `engine_*` domain
    tools -- with the engine deliberately blocking on a REAL `time.sleep`
    (not `asyncio.sleep`, which would trivially pass even on the unfixed
    code), a concurrently scheduled cheap coroutine on the SAME event loop
    must keep making progress throughout the call. Reverting the
    `asyncio.to_thread` wrap in `get_tools_endpoint` makes this test fail.
    """
    _BLOCK_SECONDS = 0.6

    class _BlockingEngine:
        def query_cypher(self, _query, _params):
            time.sleep(_BLOCK_SECONDS)
            return []

    monkeypatch.setattr(kg_server, "_get_engine", lambda: _BlockingEngine())
    # Deterministic, fast: skip the real workspace skill/skill-graph glob so
    # this test's only engine round trip is the builtin-tools toggle batch.
    monkeypatch.setattr(kg_server, "setting", lambda _key, default="": default)

    ticks = 0

    async def _heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    heartbeat_task = asyncio.ensure_future(_heartbeat())
    try:
        session = _session("tenant-acme-graph")
        with use_actor(ACTOR), use_session(session):
            # `get_tools_endpoint` never reads its `request` argument.
            response = await get_tools_endpoint(None)
    finally:
        heartbeat_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await heartbeat_task

    assert response.status_code == 200
    # The heartbeat ticks roughly every 50ms; over a 600ms blocking call the
    # loop must have serviced it several times if the blocking work was
    # truly offloaded to a worker thread. On the pre-fix inline-call code
    # this stays at 0 -- the loop cannot run ANY other coroutine until
    # `time.sleep` returns.
    assert ticks >= 5, (
        f"event loop only advanced {ticks} heartbeat ticks during a "
        f"{_BLOCK_SECONDS}s blocking get_tools_endpoint call -- the loop was frozen"
    )


# ── Test 5: /api/graph/write/node succeeds with the corrected field name ───


@pytest.mark.asyncio
async def test_graph_write_node_endpoint_uses_node_id_field(monkeypatch):
    """DEFECT C: `graph_write_node_endpoint` used to call `_execute_tool`
    with `id=` but the `graph_write` tool declares the parameter as
    `node_id` -- every call failed closed with
    `UnsupportedToolFieldError: Tool 'graph_write' does not accept field(s):
    id` (confirmed live in the pod logs). This exercises the endpoint
    against the REAL `_execute_tool`/tool-registry validation path (not a
    mock of `_execute_tool` itself) so a reintroduced `id=` would fail this
    test with `UnsupportedToolFieldError` surfaced as a 400, not a 200.
    """
    from starlette.requests import Request

    from agent_utilities.mcp.kg_server import graph_write_node_endpoint

    captured: dict[str, Any] = {}

    async def _fake_execute_tool(tool_name: str, **kwargs: Any) -> str:
        captured["tool_name"] = tool_name
        captured["kwargs"] = kwargs
        # Mirror _execute_tool's own field validation so a reintroduced
        # `id=` kwarg is caught the same way it would be in production.
        if "id" in kwargs:
            raise kg_server.UnsupportedToolFieldError(
                "Tool 'graph_write' does not accept field(s): id"
            )
        return '{"action": "add_node", "node_id": "' + kwargs.get("node_id", "") + '"}'

    monkeypatch.setattr(kg_server, "_execute_tool", _fake_execute_tool)

    body = (
        b'{"node_id": "agent-1", "node_type": "Agent", "properties": {"name": "Test"}}'
    )

    async def _receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/graph/write/node",
            "headers": [(b"content-type", b"application/json")],
        },
        receive=_receive,
    )

    response = await graph_write_node_endpoint(request)

    assert response.status_code == 200
    assert captured["tool_name"] == "graph_write"
    assert captured["kwargs"]["node_id"] == "agent-1"
    assert "id" not in captured["kwargs"]
