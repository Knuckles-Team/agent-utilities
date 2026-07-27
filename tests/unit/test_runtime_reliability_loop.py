"""Runtime-reliability detect→signal→gap→heal loop tests.

Proves the minimal spine wired this session: the four existing runtime detection sites
emit hot-path-safe signals (:mod:`agent_utilities.observability.runtime_signals`), and the
background analyzer (:mod:`agent_utilities.knowledge_graph.research.runtime_reliability`)
folds a recurring pattern into the SAME canonical ``:Gap`` (``SOURCE_RUNTIME``) the rest of
the flywheel uses — opening flywheel gaps, recommendation gaps, or recording resolved heals,
and deduping against already-open gaps.

@pytest.mark.concept("AU-AHE.harness.runtime-reliability-loop")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from agent_utilities.knowledge_graph.research import gaps
from agent_utilities.knowledge_graph.research.runtime_reliability import (
    _MIN_COUNT,
    runtime_reconciler,
    runtime_reliability_analyzer,
)
from agent_utilities.observability import runtime_signals

pytestmark = pytest.mark.concept("AU-AHE.harness.runtime-reliability-loop")

_REPO_ROOT = Path(__file__).resolve().parents[2]


class MockEngine:
    """In-memory KG double honoring exactly the surface the loop uses:
    ``add_node`` (persist :RuntimeSignal + submit_gap) and the label-scan / id-lookup /
    DETACH-DELETE cyphers ``read_recent_runtime_signals`` / ``open_gaps`` / ``get_gap`` /
    ``prune_old_runtime_signals`` issue. Backend-agnostic, like the real engine."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(
        self, node_id: str, node_type: str, properties: dict | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def add_edge(self, src: str, dst: str, rel_type: str, properties=None) -> None:
        self.edges.append((src, dst, rel_type))

    def query_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        params = params or {}
        if "WHERE n.id = $id" in query:  # get_gap
            node = self.nodes.get(params.get("id"))
            return [{"n": dict(node)}] if node else []
        if "DETACH DELETE" in query:  # prune_old_runtime_signals
            cutoff = params.get("cutoff", 0.0)
            drop = [
                nid
                for nid, v in self.nodes.items()
                if v.get("type") == "RuntimeSignal" and float(v.get("ts", 0.0)) < cutoff
            ]
            for nid in drop:
                del self.nodes[nid]
            return []
        m = re.search(r"MATCH \(n:(\w+)\) RETURN n", query)  # label scan
        if m:
            label = m.group(1)
            return [
                {"n": dict(v)} for v in self.nodes.values() if v.get("type") == label
            ]
        return []

    def gap_nodes(self) -> list[dict[str, Any]]:
        return [v for v in self.nodes.values() if v.get("type") == gaps.GAP_LABEL]

    def signal_nodes(self) -> list[dict[str, Any]]:
        return [v for v in self.nodes.values() if v.get("type") == "RuntimeSignal"]


@pytest.fixture(autouse=True)
def _clean_buffer():
    """Each test starts with an empty hot-path buffer."""
    runtime_signals.drain_buffered_signals()
    yield
    runtime_signals.drain_buffered_signals()


def _emit(kind: str, subject: str, n: int, **detail: Any) -> None:
    for _ in range(n):
        runtime_signals.record_runtime_signal(kind, subject, detail or None)


# ── 1) record_runtime_signal persists + is exception-isolated ────────────────


def test_record_runtime_signal_buffers_privacy_safe():
    runtime_signals.record_runtime_signal(
        runtime_signals.KIND_ENGINE_LATENCY,
        "nodes.get",
        {"duration_s": 2.5, "threshold_s": 1.0, "leak": {"secret": 1}},
    )
    buffered = runtime_signals.buffered_runtime_signals()
    assert len(buffered) == 1
    sig = buffered[0]
    assert sig["kind"] == runtime_signals.KIND_ENGINE_LATENCY
    assert sig["subject"] == "nodes.get"
    # only privacy-safe scalars survive — the nested dict is dropped
    assert sig["detail"]["duration_s"] == 2.5
    assert "leak" not in sig["detail"]
    assert isinstance(sig["ts"], float)


def test_record_runtime_signal_swallows_build_failure(monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("build failed")

    monkeypatch.setattr(runtime_signals, "_build_signal", _boom)
    # Must NOT raise on the hot path even when the internal store path throws.
    runtime_signals.record_runtime_signal(runtime_signals.KIND_LISTENER_RESTART, "tg")
    assert runtime_signals.buffered_runtime_signals() == []


def test_persist_runtime_signals_failing_store_does_not_raise():
    class RaisingEngine:
        def add_node(self, *_a, **_k):
            raise RuntimeError("store down")

    signals = [runtime_signals._build_signal("engine_latency", "op", None, "warning")]
    # A failing store is swallowed per-signal; returns 0 written, never raises.
    assert runtime_signals.persist_runtime_signals(RaisingEngine(), signals) == 0


def test_persist_and_read_roundtrip():
    engine = MockEngine()
    _emit(runtime_signals.KIND_ENGINE_LATENCY, "nodes.get", 2, duration_s=1.4)
    written = runtime_signals.persist_runtime_signals(
        engine, runtime_signals.drain_buffered_signals()
    )
    assert written == 2
    recent = runtime_signals.read_recent_runtime_signals(engine, window_s=3600)
    assert len(recent) == 2
    assert all(r["kind"] == runtime_signals.KIND_ENGINE_LATENCY for r in recent)


# ── 2) analyzer opens a SOURCE_RUNTIME gap on a crossing pattern + dedupes ────


def test_analyzer_opens_flywheel_gap_over_threshold():
    engine = MockEngine()
    kind = runtime_signals.KIND_DELEGATION_OVER_BUDGET
    _emit(kind, "agent-x", _MIN_COUNT[kind], elapsed_s=250.0, budget_s=300.0)

    report = runtime_reliability_analyzer(engine)

    assert report["gaps_opened"] == 1
    gap_id = gaps.canonical_gap_id(gaps.SOURCE_RUNTIME, f"{kind}:agent-x")
    gap = gaps.get_gap(engine, gap_id)
    assert gap is not None
    assert gap["source"] == gaps.SOURCE_RUNTIME
    assert gap["status"] == gaps.STATUS_OPEN


def test_analyzer_below_threshold_opens_nothing():
    engine = MockEngine()
    kind = runtime_signals.KIND_DELEGATION_OVER_BUDGET
    _emit(kind, "agent-y", _MIN_COUNT[kind] - 1)
    report = runtime_reliability_analyzer(engine)
    assert report["gaps_opened"] == 0
    assert engine.gap_nodes() == []


def test_analyzer_dedupes_already_open_gap():
    engine = MockEngine()
    kind = runtime_signals.KIND_DELEGATION_OVER_BUDGET
    _emit(kind, "agent-z", _MIN_COUNT[kind])
    first = runtime_reliability_analyzer(engine)
    assert first["gaps_opened"] == 1

    # A second crossing of the SAME (kind, subject) must not open a duplicate.
    _emit(kind, "agent-z", _MIN_COUNT[kind])
    second = runtime_reliability_analyzer(engine)
    assert second["gaps_opened"] == 0
    assert len(engine.gap_nodes()) == 1


def test_recommendation_class_opens_recommendation_gap():
    engine = MockEngine()
    kind = runtime_signals.KIND_ENGINE_LATENCY
    _emit(kind, "nodes.get", _MIN_COUNT[kind], duration_s=1.5)
    report = runtime_reliability_analyzer(engine)
    assert report["recommendations"] == 1
    gap_id = gaps.canonical_gap_id(gaps.SOURCE_RUNTIME, f"{kind}:nodes.get")
    gap = gaps.get_gap(engine, gap_id)
    assert gap is not None
    assert "consider batching" in gap["gap_statement"]
    assert gap["status"] == gaps.STATUS_OPEN


def test_heal_class_records_resolved_heal_and_dedupes():
    engine = MockEngine()
    kind = runtime_signals.KIND_LISTENER_RESTART
    _emit(kind, "telegram", _MIN_COUNT[kind], delay_s=2.0)
    report = runtime_reliability_analyzer(engine)
    assert report["heals"] == 1
    gap_id = gaps.canonical_gap_id(gaps.SOURCE_RUNTIME, f"{kind}:telegram")
    gap = gaps.get_gap(engine, gap_id)
    assert gap is not None
    # Listener restart is auto-healed by the supervisor → recorded as RESOLVED (closed loop).
    assert gap["status"] == gaps.STATUS_RESOLVED

    # A resolved heal is not re-recorded on the next pass.
    _emit(kind, "telegram", _MIN_COUNT[kind])
    second = runtime_reliability_analyzer(engine)
    assert second["heals"] == 0
    assert len(engine.gap_nodes()) == 1


def test_runtime_gap_carries_code_reference():
    """Every runtime gap carries a drift-free code reference to its known fix site —
    the anchor the standardized evolution path (spec → implement) starts from."""
    engine = MockEngine()
    kind = runtime_signals.KIND_DELEGATION_OVER_BUDGET
    _emit(kind, "agent-x", _MIN_COUNT[kind])
    runtime_reliability_analyzer(engine)
    gap_id = gaps.canonical_gap_id(gaps.SOURCE_RUNTIME, f"{kind}:agent-x")
    gap = gaps.get_gap(engine, gap_id)
    refs = gap.get("evidence_refs") or []
    assert any(r.startswith("code:") and "agent_runner.py" in r for r in refs), refs


def test_runtime_gap_links_kg_resolved_code_anchor(monkeypatch):
    """When the fix-site symbol / subject resolves to an ingested :Code node, the gap gets a
    precise file:line reference AND a traversable (:Code)-[:EVIDENCES]->(:Gap) edge — the
    'golden egg': the gap points at real ingested code with line numbers."""
    import agent_utilities.knowledge_graph.retrieval.code_context as cc

    anchor = {
        "id": "code:au:agent_runner._execute_single_server",
        "symbol": "_execute_single_server",
        "file": "agent_utilities/orchestration/agent_runner.py",
        "line": 2100,
        "kind": "function",
    }
    monkeypatch.setattr(cc, "resolve_anchors", lambda engine, **kw: [anchor])

    engine = MockEngine()
    kind = runtime_signals.KIND_DELEGATION_OVER_BUDGET
    _emit(kind, "agent-x", _MIN_COUNT[kind])
    runtime_reliability_analyzer(engine)

    gap_id = gaps.canonical_gap_id(gaps.SOURCE_RUNTIME, f"{kind}:agent-x")
    gap = gaps.get_gap(engine, gap_id)
    refs = gap.get("evidence_refs") or []
    assert any("agent_runner.py:2100" in r for r in refs), refs
    # the ingested code node is linked to the gap by the existing EVIDENCES convention
    assert (anchor["id"], gap_id, "EVIDENCES") in engine.edges


def test_reconciler_standalone_reads_and_disposes():
    engine = MockEngine()
    kind = runtime_signals.KIND_RETRIEVAL_DEGRADED
    _emit(kind, "model_context_compile", _MIN_COUNT[kind], reason="timeout")
    # Persist the signals so the standalone reconciler (which reads the window) sees them.
    runtime_signals.persist_runtime_signals(
        engine, runtime_signals.drain_buffered_signals()
    )
    result = runtime_reconciler(engine)
    assert result["recommendations"] == 1


# ── 3) the four emission call sites are reached (wire test) ───────────────────

_EMISSION_SITES = {
    "agent_utilities/knowledge_graph/core/engine_breaker.py": "KIND_ENGINE_LATENCY",
    "agent_utilities/messaging/router.py": "KIND_LISTENER_RESTART",
    "agent_utilities/core/contextual_model.py": "KIND_RETRIEVAL_DEGRADED",
    "agent_utilities/orchestration/agent_runner.py": "KIND_DELEGATION_OVER_BUDGET",
}


@pytest.mark.parametrize("rel_path,kind_const", _EMISSION_SITES.items())
def test_emission_site_is_wired(rel_path: str, kind_const: str):
    src = (_REPO_ROOT / rel_path).read_text(encoding="utf-8")
    assert "record_runtime_signal" in src, f"{rel_path} does not emit a runtime signal"
    assert kind_const in src, f"{rel_path} does not reference {kind_const}"


def test_engine_breaker_slow_call_emits_functionally():
    """The engine_breaker slow-call path actually reaches the buffer (functional wire)."""
    from agent_utilities.knowledge_graph.core.engine_breaker import _observe_latency

    _observe_latency("nodes.get", 2.0, "uds:///engine")  # >= _SLOW_ENGINE_CALL_S
    buffered = runtime_signals.buffered_runtime_signals()
    assert any(
        s["kind"] == runtime_signals.KIND_ENGINE_LATENCY and s["subject"] == "nodes.get"
        for s in buffered
    )


def test_engine_breaker_fast_call_emits_nothing():
    from agent_utilities.knowledge_graph.core.engine_breaker import _observe_latency

    _observe_latency("nodes.get", 0.01, "uds:///engine")  # below threshold
    assert runtime_signals.buffered_runtime_signals() == []
