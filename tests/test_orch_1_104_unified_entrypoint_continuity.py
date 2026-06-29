#!/usr/bin/python
"""CONCEPT:ORCH-1.104 — Unified Agent Entrypoint: the streaming AG-UI surface joins the
SAME KG-backed continuity seam as ``run_agent`` (shared memory + provenance).

Two proofs:

1. The seam helpers (:func:`prime_session_context` / :func:`persist_session_turn`) read
   and write the SAME memento store ``run_agent`` uses, keyed by ``source=session_id`` —
   so memory is shared across surfaces, not siloed.
2. The AG-UI streaming fast-path (``/ag-ui``) actually REACHES that seam: it primes the
   session context before driving the graph and persists the turn afterwards, keyed to
   the caller's session/run id.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_utilities.orchestration import session_continuity


class _FakeEngine:
    def __init__(self) -> None:
        self.nodes: list[tuple[str, str]] = []
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, nid: str, label: str, properties: dict | None = None) -> None:
        self.nodes.append((nid, label))

    def add_edge(self, a: str, b: str, rel: str) -> None:
        self.edges.append((a, b, rel))


def test_prime_session_context_recalls_by_source(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def _fake_recent(engine, source, limit):  # noqa: ANN001
        seen["source"] = source
        seen["limit"] = limit
        return ["user asked about the migration plan"]

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.memory.memento_compressor.get_recent_mementos",
        _fake_recent,
    )
    ctx = session_continuity.prime_session_context(_FakeEngine(), "telegram:42")
    assert seen["source"] == "telegram:42"  # keyed by session id verbatim
    assert "migration plan" in ctx
    # Empty/None session is a no-op (anonymous one-shot, no cross-turn recall).
    assert session_continuity.prime_session_context(_FakeEngine(), None) == ""


def test_persist_session_turn_writes_runtrace_and_memento(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def _fake_record(engine, run_id, agent_name, task, status, *rest):  # noqa: ANN001
        calls["trace"] = (run_id, agent_name, status)

    def _fake_compress(engine, turn, source, refine):  # noqa: ANN001
        calls["memento_source"] = source

    def _fake_refresh(engine, source):  # noqa: ANN001
        calls["refresh_source"] = source

    monkeypatch.setattr(
        "agent_utilities.orchestration.agent_runner._record_execution_trace",
        _fake_record,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.memory.memento_compressor.compress_to_memento",
        _fake_compress,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.memory.session_memento_cache.refresh_session_memento_cache",
        _fake_refresh,
    )

    eng = _FakeEngine()
    asyncio.run(
        session_continuity.persist_session_turn(
            eng, "webui:99", "hello", "world", run_id="run:abc"
        )
    )
    # Provenance parity: RunTrace recorded + Session->HAS_RUN anchored.
    assert calls["trace"] == ("run:abc", "agent-ui", "completed")
    assert ("session:webui:99", "Session") in eng.nodes
    assert ("session:webui:99", "trace:run:abc", "HAS_RUN") in eng.edges
    # Memory parity: memento + cache refresh keyed by the SAME session source.
    assert calls["memento_source"] == "webui:99"
    assert calls["refresh_source"] == "webui:99"


def test_ag_ui_fast_path_reaches_continuity_seam(monkeypatch) -> None:
    """The /ag-ui streaming endpoint primes + persists via the shared seam."""
    from agent_utilities.core import config as core_config
    from agent_utilities.server.routers import agent_ui

    monkeypatch.setattr(
        core_config, "DEFAULT_GRAPH_DIRECT_EXECUTION", True, raising=False
    )

    primed: dict[str, Any] = {}
    persisted: dict[str, Any] = {}

    def _fake_prime(engine, session_id, **kw):  # noqa: ANN001
        primed["session_id"] = session_id
        return "RECALLED"

    async def _fake_persist(engine, session_id, query, reply, **kw):  # noqa: ANN001
        persisted["args"] = (session_id, query, reply)

    async def _fake_iter(*, graph, config, query, run_id, **kw):  # noqa: ANN001
        # Prove the recalled memory was injected into THIS turn's config.
        primed["invoker_context"] = config.get("invoker_context")
        yield {"type": "graph_complete", "run_id": run_id, "output": "the answer"}

    monkeypatch.setattr(session_continuity, "prime_session_context", _fake_prime)
    monkeypatch.setattr(session_continuity, "persist_session_turn", _fake_persist)
    monkeypatch.setattr(
        "agent_utilities.graph.protocol_agnostic_execution.execute_graph_iter",
        _fake_iter,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.engine.IntelligenceGraphEngine.get_active",
        staticmethod(lambda: _FakeEngine()),
    )

    class _Emitter:
        def translate(self, event):  # noqa: ANN001
            return [b""]

        def _format_sideband(self, ev):  # noqa: ANN001
            return [b""]

    monkeypatch.setattr(
        "agent_utilities.protocols.agui_emitter.AGUIGraphEmitter", lambda: _Emitter()
    )

    class _Graph:
        def iter(self):  # noqa: ANN001
            return None

    class _State:
        requested_model_id = None

    class _AppState:
        graph_bundle = (_Graph(), {"router_model": "x"})
        mcp_toolsets: list = []
        agent_instance = object()
        model_registry = None
        concurrency_manager = None

    class _App:
        state = _AppState()

    class _Req:
        app = _App()
        state = _State()

        async def json(self):
            return {"query": "what is the plan?", "session_id": "webui:7"}

    resp = asyncio.run(agent_ui.ag_ui_endpoint(_Req()))

    async def _drain() -> None:
        async for _ in resp.body_iterator:  # noqa: F841
            pass

    asyncio.run(_drain())

    # The streaming surface reached the unified seam, keyed to the caller's session.
    assert primed["session_id"] == "webui:7"
    assert primed["invoker_context"] == "RECALLED"  # recall injected into the turn
    assert persisted["args"] == ("webui:7", "what is the plan?", "the answer")
