"""OntologyReasoningDriver — reasoning-as-engine harvest (CONCEPT:AU-KG.research.best-effort-lightweight-never)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.research.ara import OntologyReasoningDriver


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
    def __init__(self):
        self.graph = _Graph()
        self.backend = None
        self.added: dict[str, dict] = {}

    def add_node(self, nid, ntype, properties=None):
        self.added[nid] = {"type": ntype, **(properties or {})}


class _Bridge:
    """Fake OWL bridge: run_cycle downfeeds inferred edges into the graph."""

    def __init__(self, graph):
        self.graph = graph

    def run_cycle(self, lightweight=True):
        self.graph.add_edge(
            "concept:harness", "service:vllm", type="relates_to", inferred=True
        )
        self.graph.add_edge("concept:a", "concept:b", type="broader", inferred=True)
        return {"promoted_nodes": 2, "inferred": 2, "downfed": 2, "mode": "lightweight"}


def test_extrapolate_harvests_new_inferences_and_cross_domain_topics():
    eng = _Engine()
    # a PRE-EXISTING inferred edge must not be re-harvested as "new"
    eng.graph.add_edge("concept:old", "concept:older", type="broader", inferred=True)

    h = OntologyReasoningDriver(eng, bridge=_Bridge(eng.graph)).extrapolate(
        persist=True
    )

    assert h.error == ""
    keys = {(e["src"], e["dst"]) for e in h.inferred_edges}
    assert ("concept:harness", "service:vllm") in keys  # newly extrapolated
    assert ("concept:old", "concept:older") not in keys  # pre-existing → not new
    # cross-domain (research↔ecosystem) becomes a research topic; within-domain does not
    assert len(h.new_topics) == 1
    assert (
        h.new_topics[0]["kind"] == "research" and "harness" in h.new_topics[0]["name"]
    )
    # the harvested relationship is persisted as a fresh research Loop (closed loop)
    assert any(nid.startswith("loop:research:rel:") for nid in eng.added)


def test_reasoning_failure_is_best_effort():
    class _BoomBridge:
        def run_cycle(self, lightweight=True):
            raise RuntimeError("no owl backend")

    eng = _Engine()
    h = OntologyReasoningDriver(eng, bridge=_BoomBridge()).extrapolate()
    assert h.error and not h.inferred_edges  # degrades, never raises into the loop


def test_no_graph_yields_error_harvest():
    class _E:
        graph = None

    h = OntologyReasoningDriver(_E()).extrapolate()
    assert h.error == "no graph"
