"""U-86/U-91/BUG-171: a slow ``engine_*`` domain-tool call must not freeze the
gateway's one asyncio event loop.

Root cause: ``_make_domain_tool``'s returned coroutine (registered as
``engine_<domain>``, both the MCP tool and the ``/engine/<domain>`` REST
handler via ``kg_server.REGISTERED_TOOLS``) used to call the SYNCHRONOUS
``_dispatch`` function *inline* -- ``return _dispatch(...)`` inside an
``async def``. ``_dispatch`` does blocking native engine I/O (a real
socket RPC, sometimes a multi-attempt retry loop tens of seconds long for a
slow admin/lifecycle call such as ``tenants.create``). Because the tool
function is declared ``async def``, ``kg_server._execute_tool``'s dispatch-
isolation check (``inspect.iscoroutinefunction(tool_func)``) treats it as
"already async" and awaits it inline on the ONE gateway event-loop thread
instead of routing it through ``asyncio.to_thread`` the way it does for
ordinary synchronous tool functions -- so a slow ``engine_*`` call froze
the entire event loop, including ``/health``/``/health/ready`` (which MUST
stay schedulable so kubelet's liveness probe does not restart an otherwise-
healthy process -- ``observability/runtime_health.py``). Confirmed live:
r21 was OOM-killed by kubelet during exactly this scenario before the fix.

This file proves the NEGATIVE: with ``_dispatch`` deliberately blocking (a
real, non-yielding ``time.sleep`` -- not ``asyncio.sleep``, which would
trivially "pass" even on the unfixed code because it yields the loop
itself), a concurrently scheduled cheap coroutine on the SAME event loop
must keep making progress throughout the blocking call. Reverting the
``asyncio.to_thread`` wrap in ``engine_tools._make_domain_tool`` makes this
test fail (the heartbeat coroutine cannot advance until the blocking call
returns, so its tick count stays at 0 for the whole duration).
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_tools
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

ACTOR = ActorContext(
    actor_id="principal:ops",
    actor_type=ActorType.AI_AGENT,
    roles=(),
    tenant_id="acme",
    authenticated=True,
)


def _session(graph: str) -> GraphSession:
    return GraphSession(
        actor=ACTOR,
        tenant=ACTOR.tenant_id,
        scopes=frozenset({"kg:admin", "kg:read", "kg:write"}),
        graph=graph,
        policy_version="policy-v1",
        audience="agent-services",
    )


_BLOCK_SECONDS = 0.6


@pytest.mark.asyncio
async def test_slow_domain_tool_call_does_not_freeze_the_event_loop(monkeypatch):
    kg_server.ensure_tools_registered()
    tool = engine_tools._make_domain_tool("tenants", ["list"])

    def _blocking_dispatch(domain, methods, action, params_json, graph):
        # A REAL blocking sleep -- proves the caller's OS thread, not just an
        # asyncio task, is what's occupied. asyncio.sleep would yield the
        # loop on its own and could not distinguish the fixed code from the
        # regression this test guards against.
        time.sleep(_BLOCK_SECONDS)
        return json.dumps({"ok": True})

    monkeypatch.setattr(engine_tools, "_dispatch", _blocking_dispatch)

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
            out = await tool(action="list", params_json="{}", graph="")
    finally:
        heartbeat_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await heartbeat_task

    assert json.loads(out) == {"ok": True}
    # The heartbeat ticks roughly every 50ms; over a 600ms blocking call the
    # loop must have serviced it several times if the blocking work was
    # truly offloaded to a worker thread. On the pre-fix inline-call code
    # this stays at 0 -- the loop cannot run ANY other coroutine until
    # `time.sleep` returns.
    assert ticks >= 5, (
        f"event loop only advanced {ticks} heartbeat ticks during a "
        f"{_BLOCK_SECONDS}s blocking domain-tool call -- the loop was frozen"
    )


@pytest.mark.asyncio
async def test_offloaded_dispatch_still_sees_the_calling_tasks_session(monkeypatch):
    """The thread-offload must propagate contextvars (actor/session) --
    correctness, not just concurrency. asyncio.to_thread copies the current
    context, so `_dispatch` running on the worker thread must observe the
    SAME ambient session the calling task set up."""
    kg_server.ensure_tools_registered()
    tool = engine_tools._make_domain_tool("nodes", ["has"])

    from agent_utilities.knowledge_graph.core.session import current_session

    observed: dict[str, str] = {}

    def _observing_dispatch(domain, methods, action, params_json, graph):
        session = current_session()
        observed["graph"] = session.graph if session else ""
        return json.dumps({"ok": True})

    monkeypatch.setattr(engine_tools, "_dispatch", _observing_dispatch)

    session = _session("tenant-acme-graph")
    with use_actor(ACTOR), use_session(session):
        out = await tool(action="has", params_json='{"node_id": "n1"}', graph="")

    assert json.loads(out) == {"ok": True}
    assert observed["graph"] == "tenant-acme-graph"
