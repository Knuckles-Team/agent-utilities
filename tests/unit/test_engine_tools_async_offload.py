"""U-86: the generated ``engine_<domain>`` MCP tool surface (``engine_tools.
_make_domain_tool``) is declared ``async def`` but used to call the
synchronous ``_dispatch`` function INLINE -- so a slow/blocking native engine
RPC (the concrete trigger: ``engine_tenants(action="create")`` provisioning the
tenant ontology graph, which retried a slow native ``CreateGraph`` three times
for 242+ seconds) ran directly on the server's SOLE asyncio event-loop thread.
``_dispatch`` is a real coroutine function (``inspect.iscoroutinefunction``
returns True for it), so the central dispatcher's own "offload synchronous
tool functions" protection never applied to this generated surface -- health/
readiness requests and every other concurrent MCP/REST caller froze until the
slow call finished, and Kubernetes eventually killed the process on failed
liveness (exit 137).

The fix threads the call through ``asyncio.to_thread`` so the blocking I/O
runs on a worker thread while the event loop stays schedulable.

This test reproduces the freeze mechanically rather than by timing heuristic
(GOC-70: no timing-dependent assertions -- construct contention
deterministically, fail loudly rather than vacuously):

* The fake ``tenants.create`` blocks on a real ``threading.Event``
  (``release_dispatch``) until the test explicitly releases it -- exactly the
  shape of a slow native RPC.
* Before the fix, ``_dispatch`` runs INLINE as part of the single synchronous
  step that resumes the ``create_task`` coroutine -- so the event loop thread
  itself becomes the thread blocked inside ``release_dispatch.wait()``. Since
  nothing else can run on that one thread, the test's own driving coroutine
  can never resume to call ``release_dispatch.set()`` -- a genuine deadlock,
  not a slow pass. It is bounded by a generous outer timeout so the test fails
  loudly (not by hanging the suite).
* After the fix, ``asyncio.to_thread`` runs the blocking call on a worker
  thread, so the event loop thread stays free, the driving coroutine resumes
  immediately, releases the block, and the whole scenario finishes in
  milliseconds -- comfortably inside the outer timeout.
* Independently (and non-racy): the fake client records which OS thread
  actually called ``tenants.create`` -- it must differ from the thread that
  ran ``asyncio.run``, which is only true when the call was actually
  offloaded.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

# See the identical rationale in test_engine_tools_scope_policy.py /
# test_engine_tools_streaming_graph.py: `engine_tenants` only exists as a
# registered MCP tool when `epistemic_graph.client` was importable at
# `engine_tools` module-import time.
pytest.importorskip("epistemic_graph.client")

from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_tools
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

ADMIN_ACTOR = ActorContext(
    actor_id="principal:ops",
    actor_type=ActorType.AI_AGENT,
    roles=("admin",),
    tenant_id="tenant-ops",
    authenticated=True,
)


def _admin_session() -> GraphSession:
    return GraphSession(
        actor=ADMIN_ACTOR,
        tenant=ADMIN_ACTOR.tenant_id,
        scopes=frozenset({engine_tools.ENGINE_ADMIN_SCOPE}),
        graph="tenant-ops-graph",
        policy_version="policy-v1",
        audience="agent-services",
    )


@pytest.fixture(autouse=True)
def _fresh_client_pool(monkeypatch):
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)
    yield
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)


class _BlockingTenants:
    """Mirrors ``epistemic_graph.client.MultiTenantClient.create`` -- a plain
    synchronous call from ``_dispatch``'s point of view -- but blocks on a
    real ``threading.Event`` until released, standing in for a slow native
    ``CreateGraph`` RPC."""

    def __init__(
        self, dispatch_started: threading.Event, release_dispatch: threading.Event
    ):
        self._dispatch_started = dispatch_started
        self._release_dispatch = release_dispatch
        self.call_thread_id: int | None = None

    def create(self, graph_name: str, graph_type: str = "Agent") -> None:
        self.call_thread_id = threading.get_ident()
        self._dispatch_started.set()
        # Bounded, generous wait: real production timeouts for the native
        # CreateGraph retry sequence this reproduces were 242+ seconds; 8s is
        # far more than the fixed path needs (milliseconds) while keeping a
        # broken run's failure fast enough not to stall the suite.
        if not self._release_dispatch.wait(timeout=8):
            raise AssertionError(
                "release_dispatch was never set -- the driving coroutine could "
                "not resume, which only happens when this blocking call is "
                "running on the SAME thread as the event loop (the U-86 bug)"
            )


class _BlockingClient:
    def __init__(self, tenants: _BlockingTenants) -> None:
        self.tenants = tenants


def test_engine_tenants_create_offloads_blocking_native_io(monkeypatch):
    kg_server.ensure_tools_registered()
    tool = kg_server.REGISTERED_TOOLS["engine_tenants"]

    dispatch_started = threading.Event()
    release_dispatch = threading.Event()
    tenants = _BlockingTenants(dispatch_started, release_dispatch)
    client = _BlockingClient(tenants)
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    main_thread_id = threading.get_ident()

    async def scenario() -> None:
        with use_actor(ADMIN_ACTOR), use_session(_admin_session()):
            create_task = asyncio.create_task(
                tool(
                    action="create",
                    params_json=json.dumps(
                        {
                            "graph_name": "tenant__local__ontology",
                            "graph_type": "Global",
                        }
                    ),
                    graph="",
                )
            )

            # If _dispatch still ran inline on the event loop thread, this
            # await would never get a chance to run at all -- the loop thread
            # is stuck inside create_task's single, uninterrupted synchronous
            # step (which is itself blocked in `release_dispatch.wait()`).
            # `asyncio.to_thread` here runs the (non-blocking-loop) poll on a
            # SEPARATE worker thread, so it can observe `dispatch_started`
            # even while the main loop thread is frozen -- which is exactly
            # what lets this test distinguish "call started, then the loop
            # froze" from "the call never even started".
            await asyncio.wait_for(
                asyncio.to_thread(dispatch_started.wait, 8), timeout=8
            )

            # Unblock the fake native call and let the tool call finish. In
            # the broken (inline) case, control never returns here -- the
            # outer `asyncio.wait_for` below is what turns that hang into a
            # loud, bounded failure instead of stalling the suite.
            release_dispatch.set()
            await create_task

    # Generous but bounded: the fixed path completes in well under a second;
    # 3s comfortably covers a loaded CI runner without masking the deadlock
    # the broken path produces (which would otherwise run past the fake
    # client's own 8s internal bound).
    asyncio.run(asyncio.wait_for(scenario(), timeout=3))

    assert dispatch_started.is_set(), "the native call never started"
    assert tenants.call_thread_id is not None
    assert tenants.call_thread_id != main_thread_id, (
        "tenants.create() ran on the SAME thread that drove asyncio.run() -- "
        "the blocking native call is still executing inline on the event "
        "loop thread instead of being offloaded via asyncio.to_thread"
    )
