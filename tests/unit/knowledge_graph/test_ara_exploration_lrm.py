"""ARA /trace producer + Live Research Manager (CONCEPT:KG-2.80)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.research.ara import (
    ExplorationGraphBuilder,
    LiveResearchManager,
    ResearchArtifact,
)


class _Engine:
    def __init__(self):
        self.nodes: dict[str, dict] = {}

    def add_node(self, nid, ntype, properties=None):
        self.nodes[nid] = {"type": ntype, **(properties or {})}


# ── A2: exploration graph producer ──────────────────────────────────────────


def test_builder_marks_dead_ends_from_failures_and_pivots_from_rejects():
    builder = ExplorationGraphBuilder("p")
    traj = builder.build(
        "How to ground claims to the ecosystem?",
        decisions=["use grounded_in transitive"],
        experiments=["reason over 5 papers"],
        failure_clusters=[{"summary": "embedding recall stalled at 502"}],
        matcher_rejects=["unrelated: kafka queue"],
    )
    assert len(traj.dead_ends()) == 1
    assert traj.dead_ends()[0].text == "embedding recall stalled at 502"
    assert len(traj.pivots()) == 1
    # non-root nodes default their parent to the root question
    root = traj.root_id
    assert all(n.parent_id == root for n in traj.nodes if n.id != root)


def test_attach_extends_artifact_trace_layer():
    art = ResearchArtifact(article_id="p", title="P")
    traj = ExplorationGraphBuilder("p").build("Q", decisions=["d"])
    n = ExplorationGraphBuilder.attach(art, traj)
    assert n == len(art.exploration) == 2  # question + decision


# ── A3: Live Research Manager ───────────────────────────────────────────────


def test_router_classifies_and_stamps_provenance():
    lrm = LiveResearchManager("p")
    q = lrm.capture("Why does grounding help?", provenance="user")
    obs = lrm.capture("we find 93.7% paper-QA", provenance="ai_executed")
    assert q.type == "question" and q.provenance == "user"
    assert obs.type == "observation"


def test_crystallize_folds_events_onto_layers_idempotently():
    lrm = LiveResearchManager("p")
    lrm.capture("ARA improves reproduction", type="claim")
    lrm.capture("we find 64.4% reproduction", type="observation")
    lrm.capture("pivot to ecosystem grounding", type="pivot")
    lrm.capture("we hypothesize X", type="hypothesis")  # stays uncommitted

    art = ResearchArtifact(article_id="p", title="P")
    counts = lrm.crystallize(art)
    assert counts == {"claims": 1, "evidence": 1, "exploration": 1}
    assert len(art.claims) == 1 and len(art.evidence) == 1
    # idempotent: a second pass crystallizes nothing more
    assert lrm.crystallize(art) == {"claims": 0, "evidence": 0, "exploration": 0}


def test_flush_promotes_events_with_provenance():
    eng = _Engine()
    lrm = LiveResearchManager("p")
    lrm.capture(
        "decide to use OWL reasoning", provenance="ai_suggested", actor="agent:planner"
    )
    n = lrm.flush(eng)
    assert n == 1
    (node,) = eng.nodes.values()
    assert node["type"] == "exploration_node"
    assert node["provenance"] == "ai_suggested" and node["actor"] == "agent:planner"
