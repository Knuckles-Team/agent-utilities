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
    ``prune_old_runtime_signals`` (SELECT page + DELETE-by-ids) issue.
    Backend-agnostic, like the real engine."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(
        self, node_id: str, node_type: str, properties: dict | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def add_edge(self, src: str, dst: str, rel_type: str, properties=None) -> None:
        self.edges.append((src, dst, rel_type))

    def _expired_signals(self, cutoff: float) -> list[str]:
        return [
            nid
            for nid, v in self.nodes.items()
            if v.get("type") == "RuntimeSignal" and float(v.get("ts", 0.0)) < cutoff
        ]

    def query_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        params = params or {}
        if "WHERE n.id = $id" in query:  # get_gap
            node = self.nodes.get(params.get("id"))
            return [{"n": dict(node)}] if node else []
        if "RETURN count(n) AS expired" in query:  # count_expired_runtime_signals
            return [{"expired": len(self._expired_signals(params.get("cutoff", 0.0)))}]
        if "RETURN n.id AS id" in query:  # retention SELECT page
            expired = self._expired_signals(params.get("cutoff", 0.0))
            limit = int(params.get("batch_size") or len(expired))
            return [{"id": nid} for nid in expired[:limit]]
        if "WHERE n.id IN $ids DETACH DELETE n" in query:  # retention DELETE page
            for nid in params.get("ids") or []:
                self.nodes.pop(nid, None)
            return []
        m = re.search(r"MATCH \(n:(\w+)\)", query)  # label scan
        if m and "RETURN n" in query:
            label = m.group(1)
            cutoff = params.get("cutoff")
            return [
                {"n": dict(v)}
                for v in self.nodes.values()
                if v.get("type") == label
                and (cutoff is None or float(v.get("ts", 0.0)) >= float(cutoff))
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


# ── 4) retention: it runs, it removes expired rows, and its FAILURE surfaces ──
#
# The :RuntimeSignal population is telemetry with no edges, and retention is its only
# growth bound. It was structurally broken from the day it was written: an unbatched
# `DETACH DELETE` issued through `engine.query_cypher` — the READ chokepoint, which
# always sends the engine wire `mode="read"` and is rejected before execution because
# `eg_query::classify_cypher` classifies a DELETE as a Write — wrapped in
# `contextlib.suppress(Exception)` that discarded the rejection. A nominal 2-hour TTL
# therefore accumulated an unbounded population while reporting nothing. These tests pin
# the three properties that were missing.


def _persist_signal_at(engine: MockEngine, *, ts: float, subject: str = "op") -> str:
    """Persist ONE :RuntimeSignal with an explicit ts (bypasses the hot-path buffer)."""
    signal = runtime_signals._build_signal(
        runtime_signals.KIND_ENGINE_LATENCY, subject, None, "warning"
    )
    signal["ts"] = ts
    runtime_signals.persist_runtime_signals(engine, [signal])
    return subject


def test_retention_removes_expired_rows_and_keeps_fresh_ones():
    engine = MockEngine()
    now = runtime_signals._now()
    _persist_signal_at(engine, ts=now - 10_000.0, subject="stale")
    _persist_signal_at(engine, ts=now - 10.0, subject="fresh")
    assert len(engine.signal_nodes()) == 2

    report = runtime_signals.prune_old_runtime_signals(
        engine, retention_s=7200.0, delete=True
    )

    assert report.deleted == 1
    assert report.dry_run is False
    remaining = engine.signal_nodes()
    assert [n["subject"] for n in remaining] == ["fresh"]


def test_retention_dry_run_counts_but_deletes_nothing():
    """The shipped default is a dry run — it reports the backlog, it does not drop it."""
    engine = MockEngine()
    now = runtime_signals._now()
    for i in range(3):
        _persist_signal_at(engine, ts=now - 10_000.0, subject=f"stale-{i}")

    report = runtime_signals.prune_old_runtime_signals(engine, retention_s=7200.0)

    assert report.dry_run is True
    assert report.deleted == 0
    assert report.expired == 3
    assert len(engine.signal_nodes()) == 3


def test_retention_is_batched_not_one_rpc_per_node(monkeypatch):
    """Retention must never scale RPCs with row count — 2 per page, not 1 per node."""
    engine = MockEngine()
    now = runtime_signals._now()
    for i in range(25):
        _persist_signal_at(engine, ts=now - 10_000.0, subject=f"stale-{i}")

    calls: list[str] = []
    inner = engine.query_cypher

    def counting(query: str, params: dict | None = None):
        calls.append(query)
        return inner(query, params)

    monkeypatch.setattr(engine, "query_cypher", counting)
    monkeypatch.setattr(runtime_signals, "_RETENTION_BATCH_SIZE", 10)

    report = runtime_signals.prune_old_runtime_signals(
        engine, retention_s=7200.0, delete=True
    )

    assert report.deleted == 25
    # 3 pages × (1 SELECT + 1 DELETE) = 6; the count is a function of PAGES, not rows.
    assert len(calls) == 6, calls


def test_retention_failure_raises_instead_of_being_suppressed():
    """A retention failure must be loud. It used to be `contextlib.suppress(Exception)`."""

    class RefusingEngine(MockEngine):
        def query_cypher(self, query: str, params: dict | None = None):
            if "RETURN n.id AS id" in query:
                raise RuntimeError(
                    "Cypher error: declared mode does not match the parsed statement"
                )
            return super().query_cypher(query, params)

    with pytest.raises(runtime_signals.RuntimeSignalRetentionError) as excinfo:
        runtime_signals.prune_old_runtime_signals(
            RefusingEngine(), retention_s=7200.0, delete=True
        )
    # The CAUSE survives — the whole point of not suppressing it.
    assert "declared mode" in str(excinfo.value)


def test_retention_failure_surfaces_in_the_analyzer_report(caplog):
    """The tick reports the failure rather than a silent, falsely-successful pass."""

    def _boom(*_a, **_k):
        raise runtime_signals.RuntimeSignalRetentionError("retention surface missing")

    engine = MockEngine()
    import agent_utilities.knowledge_graph.research.runtime_reliability as rr

    original = runtime_signals.prune_old_runtime_signals
    try:
        runtime_signals.prune_old_runtime_signals = _boom  # type: ignore[assignment]
        with caplog.at_level("ERROR", logger=rr.logger.name):
            report = runtime_reliability_analyzer(engine)
    finally:
        runtime_signals.prune_old_runtime_signals = original  # type: ignore[assignment]

    assert report["retention_error"] == "retention surface missing"
    assert any("retention did not run" in r.message for r in caplog.records)


def test_retention_runs_even_when_the_window_is_empty():
    """It used to be step 5, behind `if not recent: return report` — a quiet window
    skipped retention entirely, so the emptier the window the less retention ran."""
    engine = MockEngine()
    now = runtime_signals._now()
    _persist_signal_at(engine, ts=now - 10_000.0, subject="stale")
    assert runtime_signals.read_recent_runtime_signals(engine, window_s=900) == []

    report = runtime_reliability_analyzer(engine)

    assert report["scanned"] == 0  # nothing to analyze this tick …
    assert report["retention_error"] is None  # … but retention still ran.
    assert report["retention_dry_run_expired"] == 1


def test_window_read_pushes_the_cutoff_into_the_query():
    """The window predicate must reach the engine, not just Python.

    An unordered whole-label scan capped by LIMIT could return `limit` rows that were
    ALL outside the window once the population outgrew the cap — Python discarded every
    one, the pass saw an empty window, and (before the fix above) returned before
    retention. The bigger the population, the less likely retention was to run.
    """
    engine = MockEngine()
    seen: list[dict | None] = []
    inner = engine.query_cypher

    def capture(query: str, params: dict | None = None):
        if "RETURN n LIMIT" in query:
            seen.append(params)
        return inner(query, params)

    engine.query_cypher = capture  # type: ignore[method-assign]
    runtime_signals.read_recent_runtime_signals(engine, window_s=900)

    assert seen, "window read issued no label scan"
    assert "$cutoff" not in str(seen[0])
    assert isinstance(seen[0], dict) and "cutoff" in seen[0]


# ── 5) the persist path stays within its RPC budget ──────────────────────────


def test_persist_uses_one_batch_rpc_for_the_whole_drain():
    """The engine has ~1s of fixed overhead per call and this is a high-volume writer.

    The per-signal `engine.add_node` loop this replaced issued ONE RPC PER NODE (each
    `add_node` is itself a one-operation BatchUpdate round-trip), so a full 512-deep
    drain could cost ~512s of engine time on a tick scheduled every 180s. The batch is
    O(1) RPCs regardless of drain depth.
    """

    class BatchingEngine(MockEngine):
        def __init__(self) -> None:
            super().__init__()
            self.batches: list[list[dict]] = []
            self.add_node_calls = 0

        def batch_typed_mutations(self, mutations: list[dict]) -> bool:
            self.batches.append(list(mutations))
            for m in mutations:
                self.add_node(m["id"], m["node_type"], m["properties"])
            return True

        def add_node(self, node_id, node_type, properties=None):
            self.add_node_calls += 1
            super().add_node(node_id, node_type, properties)

    engine = BatchingEngine()
    _emit(runtime_signals.KIND_ENGINE_LATENCY, "nodes.get", 128, duration_s=1.4)
    written = runtime_signals.persist_runtime_signals(
        engine, runtime_signals.drain_buffered_signals()
    )

    assert written == 128
    assert len(engine.batches) == 1, "the drain must cost ONE batch RPC, not 128"
    assert len(engine.batches[0]) == 128


def test_persist_falls_back_to_per_node_without_batch_capability():
    """A backend with no native typed-batch capability must still persist."""
    engine = MockEngine()  # no batch_typed_mutations
    _emit(runtime_signals.KIND_ENGINE_LATENCY, "nodes.get", 3, duration_s=1.4)
    written = runtime_signals.persist_runtime_signals(
        engine, runtime_signals.drain_buffered_signals()
    )
    assert written == 3
    assert len(engine.signal_nodes()) == 3
