#!/usr/bin/python
"""Tests for the MLEvolve graph-search evolution module.

CONCEPT:AU-KG.retrieval.monte-carlo-graph-search — Progressive Monte-Carlo Graph Search (MLEvolve, arXiv:2606.06473).

Uses deterministic stub coder/evaluator callables — no LLM, no network.
"""

from __future__ import annotations

import pytest

from agent_utilities.harness.graph_search_evolution import (
    ColdStartKB,
    GlobalCodeMemory,
    GraphSearchEvolver,
    MemRecord,
    SearchNode,
    Stage,
    exploration_schedule,
    select_coding_mode,
)

pytestmark = pytest.mark.concept("AU-KG.retrieval.monte-carlo-graph-search")


# ── deterministic fake generation / evaluation ──────────────────────────────


def make_stubs():
    """A coder that appends a token per call and an evaluator that improves with
    the token count up to a plateau; one branch produces a buggy node.

    ``coder_fn(plan, prior_code)`` appends a unit token to the prior code; the
    plan text echoes the running token count. ``evaluate_fn(code)`` scores by
    token count (capped at a plateau) and flags a node buggy when the code
    carries the ``BUG`` marker.
    """

    def coder_fn(plan: str, prior_code: str | None) -> tuple[str, str]:
        base = prior_code or ""
        # A "fusion" plan injects a marker so the evaluator can reward it; a
        # "branch 2" draft injects a BUG marker to exercise the buggy path.
        marker = ""
        if "fuse(" in plan:
            marker = " FUSED"
        elif "branch 2:" in plan:
            marker = " BUG"
        new_code = (base + " t" + marker).strip()
        token_count = new_code.count("t") + new_code.count("FUSED")
        return f"plan@{token_count}: {plan[:30]}", new_code

    def evaluate_fn(code: str) -> tuple[float, bool]:
        if "BUG" in code:
            return (0.0, True)
        # Metric grows with token count then plateaus at 5 tokens (so branches
        # stagnate, triggering fusion). FUSED code gets a bonus so fusion can win.
        tokens = min(code.count(" t") + (1 if code.startswith("t") else 0), 5)
        bonus = 0.5 if "FUSED" in code else 0.0
        return (float(tokens) + bonus, False)

    return coder_fn, evaluate_fn


# ── exploration_schedule ────────────────────────────────────────────────────


def test_exploration_schedule_piecewise_shape():
    total = 100
    # Constant region (<= t1=30).
    assert exploration_schedule(0, total) == pytest.approx(1.4)
    assert exploration_schedule(30, total) == pytest.approx(1.4)
    # Linear-decay region (30..70): midpoint halfway between 1.4 and 0.2.
    mid = exploration_schedule(50, total)
    assert 0.2 < mid < 1.4
    assert mid == pytest.approx(0.2 + (1.4 - 0.2) * 0.5, rel=1e-6)
    # Constant lower bound (>= t2=70).
    assert exploration_schedule(70, total) == pytest.approx(0.2)
    assert exploration_schedule(100, total) == pytest.approx(0.2)


def test_exploration_schedule_monotone_non_increasing():
    total = 50
    values = [exploration_schedule(s, total) for s in range(total + 1)]
    for earlier, later in zip(values, values[1:], strict=False):
        assert later <= earlier + 1e-12


def test_exploration_schedule_degenerate_total():
    assert exploration_schedule(0, 0) == pytest.approx(0.2)


# ── SearchNode.uct ──────────────────────────────────────────────────────────


def test_uct_unvisited_is_infinite():
    node = SearchNode("n1", "code", "plan", None, Stage.DRAFT, branch_id=1)
    assert node.uct(parent_visits=10, c=1.4) == float("inf")


def test_uct_exploitation_plus_exploration_ordering():
    # Same visits, higher reward → higher UCT (exploitation dominates).
    high = SearchNode(
        "h", "c", "p", 1.0, Stage.IMPROVE, branch_id=1, visits=2, total_reward=2.0
    )
    low = SearchNode(
        "l", "c", "p", 1.0, Stage.IMPROVE, branch_id=1, visits=2, total_reward=0.5
    )
    assert high.uct(10, 1.4) > low.uct(10, 1.4)

    # Same reward mean, fewer visits → higher exploration bonus.
    rare = SearchNode(
        "r", "c", "p", 1.0, Stage.IMPROVE, branch_id=1, visits=1, total_reward=1.0
    )
    common = SearchNode(
        "o", "c", "p", 1.0, Stage.IMPROVE, branch_id=1, visits=8, total_reward=8.0
    )
    assert rare.uct(10, 1.4) > common.uct(10, 1.4)


# ── GlobalCodeMemory ────────────────────────────────────────────────────────


def _rec(
    rid: str, plan: str, *, label: int = 1, stage: Stage = Stage.IMPROVE
) -> MemRecord:
    return MemRecord(
        record_id=rid, plan=plan, code_summary=f"sum:{rid}", stage=stage, label=label
    )


def test_memory_save_idempotent():
    mem = GlobalCodeMemory()
    mem.save(_rec("a", "deep learning image model"))
    mem.save(_rec("a", "DIFFERENT plan but same id"))
    assert len(mem) == 1
    # First write wins.
    assert mem.records[0].plan == "deep learning image model"


def test_memory_retrieve_filtered_by_label_and_stage():
    mem = GlobalCodeMemory()
    mem.save(_rec("ok", "boost trees tabular", label=1, stage=Stage.IMPROVE))
    mem.save(_rec("bad", "boost trees tabular", label=-1, stage=Stage.IMPROVE))
    mem.save(_rec("dbg", "boost trees tabular", label=1, stage=Stage.DEBUG))

    by_label = mem.retrieve("boost trees tabular", label=1)
    assert {r.record_id for r in by_label} == {"ok", "dbg"}

    by_stage = mem.retrieve("boost trees tabular", stage=Stage.IMPROVE)
    assert {r.record_id for r in by_stage} == {"ok", "bad"}

    both = mem.retrieve("boost trees tabular", label=1, stage=Stage.IMPROVE)
    assert [r.record_id for r in both] == ["ok"]


def test_memory_retrieve_similarity_ranking():
    mem = GlobalCodeMemory()
    mem.save(_rec("close", "gradient boosted trees for tabular regression"))
    mem.save(_rec("far", "convolutional neural network for images"))
    ranked = mem.retrieve("gradient boosted trees tabular", k=2)
    assert ranked[0].record_id == "close"
    # min_similarity prunes the unrelated record entirely.
    filtered = mem.retrieve("gradient boosted trees tabular", k=5, min_similarity=0.3)
    assert [r.record_id for r in filtered] == ["close"]


def test_memory_injected_similarity_fn():
    # Injected similarity that ranks by record_id length (deterministic stand-in).
    def sim(_query: str, candidate: str) -> float:
        return len(candidate) / 100.0

    mem = GlobalCodeMemory(similarity_fn=sim)
    mem.save(_rec("a", "short"))
    mem.save(_rec("b", "a much much longer plan string"))
    ranked = mem.retrieve("anything", k=2)
    assert ranked[0].record_id == "b"


# ── ColdStartKB ─────────────────────────────────────────────────────────────


def test_coldstart_recommend_known_category():
    kb = ColdStartKB()
    recs = kb.recommend("This is an image classification task")
    assert recs
    assert any("ResNet" in r or "augmentation" in r for r in recs)


def test_coldstart_recommend_unknown_returns_empty():
    assert ColdStartKB().recommend("an entirely unrelated quokka problem") == []


def test_coldstart_custom_table():
    kb = ColdStartKB(table={"graph": ["use a GNN", "node2vec features"]})
    assert kb.recommend("a graph learning task") == ["use a GNN", "node2vec features"]


# ── select_coding_mode ──────────────────────────────────────────────────────


def test_select_coding_mode_dispatch():
    # Error/retry → stepwise (takes precedence even when large/stagnating).
    assert (
        select_coding_mode(stagnating=True, code_size=9999, has_error=True)
        == "stepwise"
    )
    # Large or stagnating working code → diff.
    assert select_coding_mode(stagnating=True, code_size=10, has_error=False) == "diff"
    assert (
        select_coding_mode(stagnating=False, code_size=5000, has_error=False) == "diff"
    )
    # Default → single (full rewrite).
    assert (
        select_coding_mode(stagnating=False, code_size=10, has_error=False) == "single"
    )


# ── GraphSearchEvolver.run ──────────────────────────────────────────────────


def test_run_returns_best_node_and_explores_branches():
    coder_fn, evaluate_fn = make_stubs()
    ev = GraphSearchEvolver(
        coder_fn,
        evaluate_fn,
        num_branches=3,
        num_steps=12,
        seed=0,
        stagnation_patience=2,
    )
    best = ev.run("an image classification task")

    assert isinstance(best, SearchNode)
    assert not best.is_buggy and best.metric is not None
    # Best is the maximum-metric valid node in the graph.
    valid = [n for n in ev.nodes.values() if not n.is_buggy and n.metric is not None]
    assert best.metric == max(n.metric for n in valid)

    # Multiple branches were explored (branch 0 is the root).
    explored_branches = {n.branch_id for n in ev.nodes.values() if n.branch_id != 0}
    assert len(explored_branches) >= 2


def test_run_triggers_cross_branch_fusion():
    coder_fn, evaluate_fn = make_stubs()
    ev = GraphSearchEvolver(
        coder_fn,
        evaluate_fn,
        num_branches=3,
        num_steps=16,
        seed=0,
        stagnation_patience=2,
    )
    ev.run("an image classification task")

    # At least one fusion node was created with non-empty cross-branch references.
    assert ev.fusion_nodes, "expected at least one fusion node when a branch stagnates"
    fusion_node = ev.nodes[ev.fusion_nodes[0]]
    assert fusion_node.stage == Stage.FUSION
    assert fusion_node.reference_ids, (
        "fusion node must carry cross-branch reference edges"
    )
    # References point at OTHER branches (cross-branch knowledge flow).
    for ref_id in fusion_node.reference_ids:
        assert ev.nodes[ref_id].branch_id != fusion_node.branch_id


def test_run_is_deterministic_under_fixed_seed():
    def run_once():
        coder_fn, evaluate_fn = make_stubs()
        ev = GraphSearchEvolver(
            coder_fn,
            evaluate_fn,
            num_branches=3,
            num_steps=14,
            seed=42,
            stagnation_patience=2,
        )
        best = ev.run("an image classification task")
        return (
            best.node_id,
            best.metric,
            sorted(ev.nodes),
            list(ev.fusion_nodes),
        )

    assert run_once() == run_once()


def test_run_buggy_branch_does_not_break_search():
    coder_fn, evaluate_fn = make_stubs()
    ev = GraphSearchEvolver(coder_fn, evaluate_fn, num_branches=3, num_steps=10, seed=1)
    best = ev.run("an image classification task")
    # Branch 2's drafts are buggy; the best node must come from a clean branch.
    assert best.is_buggy is False
    assert any(n.is_buggy for n in ev.nodes.values()), (
        "expected the buggy branch to exist"
    )
