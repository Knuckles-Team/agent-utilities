#!/usr/bin/python
"""Tests for shortcut-resistant search-task synthesis (KG-2.70/2.71/2.72)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.search_synthesis import (
    EvidenceFact,
    EvidenceGraph,
    build_evidence_subgraph,
    diagnose,
    evidence_co_coverage,
    exposed_constants,
    formulate,
    prior_knowledge_binding,
    refine,
    single_clue_selectivity,
    synthesize,
)


def _clean_graph() -> EvidenceGraph:
    """Three generic, distinctly-sourced, jointly-identifying clues. No shortcuts."""
    return EvidenceGraph(
        answer_id="Q-target",
        answer_aliases=("Ada Botanist",),
        facts=[
            EvidenceFact("f1", "described a fern species", "docA", standalone_pool=40),
            EvidenceFact(
                "f2", "advised by a doctoral mentor", "docB", standalone_pool=30
            ),
            EvidenceFact("f3", "led a botanical society", "docC", standalone_pool=25),
        ],
        root_popularity=0.1,
    )


def test_clean_graph_trips_nothing():
    eg = _clean_graph()
    q = formulate(eg).question
    report = diagnose(eg, q)
    assert report.clear
    assert not single_clue_selectivity(eg).tripped


def test_single_clue_selectivity_detected():
    eg = _clean_graph()
    eg.facts[0].standalone_pool = 1  # this clue alone identifies the answer
    finding = single_clue_selectivity(eg)
    assert finding.tripped
    assert "f1" in finding.offenders


def test_evidence_co_coverage_detected():
    eg = _clean_graph()
    eg.facts[1].source_document_id = "docA"  # f1 and f2 now share one source
    finding = evidence_co_coverage(eg)
    assert finding.tripped
    assert set(finding.offenders) == {"f1", "f2"}


def test_exposed_constants_detected_for_answer_and_intermediate():
    eg = _clean_graph()
    q_answer_leak = "Who is Ada Botanist, the fern describer?"
    assert exposed_constants(eg, q_answer_leak).tripped
    assert exposed_constants(eg, q_answer_leak).score == 1.0

    eg.facts[0].referenced_names = ("Mount Kenya",)
    q_inter_leak = "describes a fern from Mount Kenya"
    f = exposed_constants(eg, q_inter_leak)
    assert f.tripped
    assert "Mount Kenya" in f.offenders


def test_prior_knowledge_binding_popularity_and_probe():
    eg = _clean_graph()
    eg.root_popularity = 0.9
    assert prior_knowledge_binding(eg, "q").tripped

    eg.root_popularity = 0.1
    assert not prior_knowledge_binding(eg, "q").tripped
    # closed-book probe that already names the answer → bound
    bound = prior_knowledge_binding(eg, "q", probe=lambda _q: "Ada Botanist obviously")
    assert bound.tripped and bound.score == 1.0


def test_refine_converges_on_redundant_co_covered_clue():
    eg = _clean_graph()
    # add a redundant clue co-covered with f1 → refinement should prune it.
    eg.facts.append(
        EvidenceFact("f4", "extra detail", "docA", standalone_pool=50, required=False)
    )
    assert evidence_co_coverage(eg).tripped
    task = refine(eg)
    assert task.risk_report.clear
    assert "f4" not in task.evidence_fact_ids  # redundant co-covered clue pruned
    assert eg.facts[-1].id == "f4"  # original graph not mutated


def test_refine_withholds_exposed_intermediate_name():
    eg = _clean_graph()
    eg.facts[0].clue = "described a fern from Mount Kenya"
    eg.facts[0].referenced_names = ("Mount Kenya",)
    # before refinement the raw clue leaks the intermediate name
    assert exposed_constants(eg, formulate(eg).question).tripped
    task = refine(eg)
    assert task.risk_report.clear
    assert "Mount Kenya" not in task.question
    assert "the related entity" in task.question


def test_refine_generalizes_required_overselective_clue():
    eg = _clean_graph()
    eg.facts[0].standalone_pool = 1  # required + over-selective → must be generalized
    assert eg.facts[0].required
    task = refine(eg)
    assert task.risk_report.clear
    assert (
        "f1" in task.evidence_fact_ids
    )  # required clue kept (generalized, not dropped)


class _FakeKG:
    """Minimal graph reader returning a small two-hop neighborhood."""

    def __init__(self) -> None:
        self._edges = {
            "Q-target": [
                {
                    "a": {"id": "Q-target", "name": "Ada Botanist"},
                    "r": {"type": "DESCRIBED", "source": "docA"},
                    "b": {"id": "Q-fern", "name": "Athyrium kenyae", "source": "docA"},
                },
                {
                    "a": {"id": "Q-target", "name": "Ada Botanist"},
                    "r": {"type": "ADVISED_BY", "source": "docB"},
                    "b": {"id": "Q-mentor", "name": "Dr. Mentor", "source": "docB"},
                },
            ],
        }

    def query(self, cypher, params=None):  # noqa: ARG002
        return self._edges.get((params or {}).get("id"), [])


def test_build_evidence_subgraph_from_live_reader():
    eg = build_evidence_subgraph(_FakeKG(), "Q-target", hops=1, fanout=8)
    assert eg.answer_id == "Q-target"
    assert "Ada Botanist" in eg.answer_aliases
    assert len(eg.facts) == 2
    sources = {f.source_document_id for f in eg.facts}
    assert sources == {"docA", "docB"}  # distinct provenance preserved
    assert all(f.required for f in eg.facts)  # 1-hop clues are required


def test_synthesize_end_to_end_returns_clean_task():
    task = synthesize(_FakeKG(), "Q-target", hops=1)
    assert task.answer_id == "Q-target"
    assert task.risk_report.clear
    assert "Ada Botanist" not in task.question  # answer never leaked
