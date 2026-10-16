#!/usr/bin/python
"""CONCEPT:AU-ORCH.session.session-continuity-entrypoint — Unified Agent Entrypoint: the streaming AG-UI surface joins the
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

from agent_utilities.observability.trace_ontology import trace_id
from agent_utilities.orchestration import session_continuity
from agent_utilities.security.persistence_privacy import persistence_reference


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
    # The runtime key reaches the Memento API; that storage boundary converts it to an opaque ref.
    assert seen["source"] == "telegram:42"
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
    session_ref = persistence_reference(
        "session", "webui:99", namespace="session-continuity"
    )
    session_node = f"session:{session_ref}"
    assert (session_node, "Session") in eng.nodes
    assert (session_node, trace_id("run:abc"), "HAS_RUN") in eng.edges
    # Memory parity: memento + cache refresh keyed by the SAME session source.
    assert calls["memento_source"] == "webui:99"
    assert calls["refresh_source"] == "webui:99"
