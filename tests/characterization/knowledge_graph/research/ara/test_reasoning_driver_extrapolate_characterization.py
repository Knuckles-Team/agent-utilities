"""Characterization tests for ``OntologyReasoningDriver.extrapolate`` (CX-AU-09).

CCN 11 at time of writing
(``agent_utilities/knowledge_graph/research/ara/reasoning_driver.py``). These
tests pin the OBSERVED, black-box behaviour before any decomposition: the
no-graph short-circuit, the best-effort exception handling, the before/after
edge diff, the persist flag's exact gating (only when there ARE topics), the
topic_filter-then-truncate ordering, the max_topics cap, and the non-dict
stats fallback.

Per the two-commit discipline, this file must be added and pass GREEN
against the UNMODIFIED ``reasoning_driver.py`` before any refactor commit,
and must not change during the refactor commit that follows.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.research.ara.reasoning_driver import (
    OntologyReasoningDriver,
)


class _Graph:
    def __init__(self):
        self._edges: list[tuple] = []

    def edges(self, data=False):
        return (
            [(s, d, p) for s, d, p in self._edges]
            if data
            else [(s, d) for s, d, _ in self._edges]
        )

    def add_edge(self, src, dst, **data):
        self._edges.append((src, dst, data))


class _Engine:
    def __init__(self, graph=None):
        self.graph = graph if graph is not None else _Graph()
        self.backend = None
        self.added: dict[str, dict] = {}

    def add_node(self, nid, ntype, properties=None):
        self.added[nid] = {"type": ntype, **(properties or {})}


class _NoGraphEngine:
    graph = None


class _StaticBridge:
    """Fake OWL bridge whose run_cycle downfeeds fixed edges, once."""

    def __init__(self, graph, edges_to_add, stats=None, called_flag=None):
        self.graph = graph
        self._edges_to_add = edges_to_add
        self._stats = stats if stats is not None else {"promoted_nodes": 1}
        self._called_flag = called_flag if called_flag is not None else []

    def run_cycle(self, lightweight=True):
        self._called_flag.append(lightweight)
        for src, dst, etype in self._edges_to_add:
            self.graph.add_edge(src, dst, type=etype, inferred=True)
        return self._stats


class _BoomBridge:
    def run_cycle(self, lightweight=True):
        raise RuntimeError("no owl backend")


def test_no_graph_returns_error_harvest_immediately() -> None:
    h = OntologyReasoningDriver(_NoGraphEngine()).extrapolate()
    assert h.error == "no graph"
    assert h.inferred_edges == []
    assert h.new_topics == []
    assert h.stats == {}


def test_reasoning_exception_is_best_effort_and_yields_empty_harvest() -> None:
    eng = _Engine()
    h = OntologyReasoningDriver(eng, bridge=_BoomBridge()).extrapolate()
    assert h.error == "no owl backend"
    assert h.inferred_edges == []
    assert h.new_topics == []


def test_preexisting_inferred_edge_is_not_recounted_as_new() -> None:
    eng = _Engine()
    eng.graph.add_edge("concept:old", "concept:older", type="broader", inferred=True)
    bridge = _StaticBridge(
        eng.graph, [("concept:harness", "service:vllm", "relates_to")]
    )
    h = OntologyReasoningDriver(eng, bridge=bridge).extrapolate(persist=False)
    keys = {(e["src"], e["dst"]) for e in h.inferred_edges}
    assert ("concept:harness", "service:vllm") in keys
    assert ("concept:old", "concept:older") not in keys


def test_cross_domain_edge_becomes_a_topic_within_domain_does_not() -> None:
    eng = _Engine()
    bridge = _StaticBridge(
        eng.graph,
        [
            ("concept:harness", "service:vllm", "relates_to"),  # cross-domain
            ("concept:a", "concept:b", "broader"),  # within research domain
        ],
    )
    h = OntologyReasoningDriver(eng, bridge=bridge).extrapolate(persist=False)
    assert len(h.inferred_edges) == 2
    assert len(h.new_topics) == 1
    assert h.new_topics[0]["kind"] == "research"
    assert h.new_topics[0]["source"] == "owl-inference"
    assert h.new_topics[0]["id"] == "loop:research:rel:concept:harness>service:vllm"


def test_persist_true_with_topics_submits_loops(monkeypatch) -> None:
    eng = _Engine()
    from agent_utilities.knowledge_graph.research import loops

    submitted: list[str] = []

    def _submit(engine, objective, *, loop_id, **_kwargs):
        submitted.append(loop_id)
        engine.add_node(loop_id, "Concept", properties={"objective": objective})
        return {"id": loop_id, "status": "submitted"}

    monkeypatch.setattr(loops, "submit_loop", _submit)
    bridge = _StaticBridge(
        eng.graph, [("concept:harness", "service:vllm", "relates_to")]
    )
    h = OntologyReasoningDriver(eng, bridge=bridge).extrapolate(persist=True)
    assert len(h.new_topics) == 1
    assert submitted == ["loop:research:rel:concept:harness>service:vllm"]


def test_persist_false_never_submits_loops_even_with_topics(monkeypatch) -> None:
    eng = _Engine()
    from agent_utilities.knowledge_graph.research import loops

    calls: list[str] = []
    monkeypatch.setattr(
        loops, "submit_loop", lambda *a, **k: calls.append(1) or {"status": "submitted"}
    )
    bridge = _StaticBridge(
        eng.graph, [("concept:harness", "service:vllm", "relates_to")]
    )
    h = OntologyReasoningDriver(eng, bridge=bridge).extrapolate(persist=False)
    assert len(h.new_topics) == 1
    assert calls == []


def test_persist_true_with_no_topics_never_calls_persist(monkeypatch) -> None:
    # OBSERVED: persist is gated on `persist and topics` -- persist=True alone
    # is not enough; there must be at least one topic.
    eng = _Engine()
    from agent_utilities.knowledge_graph.research import loops

    calls: list[str] = []
    monkeypatch.setattr(
        loops, "submit_loop", lambda *a, **k: calls.append(1) or {"status": "submitted"}
    )
    # within-domain only -> no topics
    bridge = _StaticBridge(eng.graph, [("concept:a", "concept:b", "broader")])
    h = OntologyReasoningDriver(eng, bridge=bridge).extrapolate(persist=True)
    assert h.new_topics == []
    assert calls == []


def test_topic_filter_applied_before_max_topics_truncation() -> None:
    eng = _Engine()
    bridge = _StaticBridge(
        eng.graph,
        [
            ("concept:keep", "service:a", "relates_to"),
            ("concept:drop", "service:b", "relates_to"),
        ],
    )
    h = OntologyReasoningDriver(eng, bridge=bridge).extrapolate(
        persist=False, topic_filter=lambda t: "keep" in t["id"]
    )
    assert len(h.new_topics) == 1
    assert "keep" in h.new_topics[0]["id"]


def test_filter_then_truncate_order_is_load_bearing() -> None:
    # OBSERVED: topic_filter runs BEFORE the max_topics truncation. With
    # max_topics=1 and a filter that only matches the THIRD candidate, the
    # match must still survive -- if truncation ran first (keeping only the
    # first candidate) the filter would then have nothing left to match.
    eng = _Engine()
    bridge = _StaticBridge(
        eng.graph,
        [
            ("concept:c0", "service:s0", "relates_to"),
            ("concept:c1", "service:s1", "relates_to"),
            ("concept:c2", "service:s2", "relates_to"),
        ],
    )
    h = OntologyReasoningDriver(eng, bridge=bridge, max_topics=1).extrapolate(
        persist=False, topic_filter=lambda t: "c2" in t["id"]
    )
    assert len(h.new_topics) == 1
    assert "c2" in h.new_topics[0]["id"]


def test_max_topics_caps_the_harvest() -> None:
    eng = _Engine()
    edges = [(f"concept:c{i}", f"service:s{i}", "relates_to") for i in range(5)]
    bridge = _StaticBridge(eng.graph, edges)
    h = OntologyReasoningDriver(eng, bridge=bridge, max_topics=2).extrapolate(
        persist=False
    )
    assert len(h.new_topics) == 2


def test_non_dict_stats_from_bridge_falls_back_to_empty_dict() -> None:
    eng = _Engine()

    class _WeirdBridge:
        def run_cycle(self, lightweight=True):
            return "not-a-dict"

    h = OntologyReasoningDriver(eng, bridge=_WeirdBridge()).extrapolate(persist=False)
    assert h.stats == {}
    assert h.error == ""


def test_lightweight_flag_is_forwarded_to_run_cycle() -> None:
    eng = _Engine()
    called: list[bool] = []
    bridge = _StaticBridge(eng.graph, [], called_flag=called)
    OntologyReasoningDriver(eng, bridge=bridge, lightweight=False).extrapolate(
        persist=False
    )
    assert called == [False]
