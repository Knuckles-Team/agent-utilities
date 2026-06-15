"""Unified DSPy optimization subsystem — metric, registry, driver, targets, surface.

Covers CONCEPT:AHE-3.39 (metric), AHE-3.40 (registry/driver/dispatch), AHE-3.43 (demo
refine), AHE-3.44 (extraction), AHE-3.45 (concept-match/routing). The LLM-gated DSPy
compile itself is exercised by the live evolution path; here we test the wiring, the
real metric, and the self-supervised metrics — all offline-deterministic.
"""

from __future__ import annotations

import pytest

from agent_utilities.harness.dspy_optimization import (
    OPTIMIZABLE_TARGETS,
    get_target,
    graded_score,
    make_optimization_metric,
    refine_demos,
    run_component_optimization,
)


# ── AHE-3.39 — the real metric ───────────────────────────────────────────────


def test_graded_score_is_graded_not_exact():
    assert graded_score("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)
    near = graded_score("the cat sat on the mat", "the cat sat on a mat")
    assert 0.0 < near < 1.0  # graded — exact-match would be 0
    assert graded_score("alpha beta", "totally different words") < near


def test_metric_bool_and_reward_blend():
    class E:
        response = "deploy the service to staging"

    class P:
        response = "deploy the service to staging"

    m_bool = make_optimization_metric(return_bool=True)
    assert m_bool(E(), P()) is True

    # reward blend pulls a perfect text score down toward a low reward
    m_blend = make_optimization_metric(reward_fn=lambda ex: 0.0, reward_weight=0.5)
    assert m_blend(E(), P()) == pytest.approx(0.5, abs=0.01)


# ── AHE-3.40 — target registry ───────────────────────────────────────────────


def test_registry_has_three_builtin_targets():
    assert set(OPTIMIZABLE_TARGETS) == {"system_prompt", "tool_description", "skill"}
    assert get_target("tool_description").kg_label == "EvolvedToolDescriptionNode"
    assert get_target("skill").load_text({"sop": "do X then Y"}) == "do X then Y"
    assert get_target("nonexistent") is None


def test_system_prompt_target_reads_blueprint_identity_and_instructions():
    t = get_target("system_prompt")
    text = t.load_text({"identity": {"role": "planner"}, "instructions": "be terse"})
    assert "planner" in text and "be terse" in text


# ── AHE-3.43 — demo refinement ───────────────────────────────────────────────


def test_refine_demos_drops_dead_weight_to_min():
    class StubProgram:
        def __init__(self):
            self.demos = ["d1", "d2", "d3"]

        def __call__(self, context="", task=""):
            return type("Pred", (), {"response": "constant"})()

    class HoldEx:
        context = ""
        task = ""
        response = "constant"  # program always matches → demos are dead weight

    prog = StubProgram()
    metric = make_optimization_metric()
    kept = refine_demos(prog, [HoldEx(), HoldEx()], metric, min_demos=1)
    assert len(kept) == 1  # all redundant demos pruned to the floor


def test_refine_demos_noop_without_holdout():
    class StubProgram:
        demos = ["a", "b"]

    prog = StubProgram()
    kept = refine_demos(prog, [], make_optimization_metric())
    assert kept == ["a", "b"]


# ── AHE-3.40 — bridge persistence (generalized) ──────────────────────────────


@pytest.mark.asyncio
async def test_bridge_ingest_evolved_component(monkeypatch):
    from agent_utilities.knowledge_graph.dspy_kg_bridge import DSPyKGBridge

    calls = []

    class FakeEngine:
        async def execute_cypher(self, cypher, params):
            calls.append((cypher, params))

    bridge = DSPyKGBridge(FakeEngine(), workspace_path="/tmp")

    async def _noop(_fp):
        return None

    monkeypatch.setattr(bridge, "_async_git_sync", _noop)

    await bridge.ingest_evolved_component(
        kg_label="EvolvedToolDescriptionNode",
        component_type="tool_description",
        identifier="my_tool",
        file_path="agents/x/tool.py",
        compiled_state={"k": 1},
        optimizer="BootstrapFewShot",
        demos=[{"context": "c", "task": "t", "response": "r"}],
    )
    # the component MERGE
    merge = next(c for c in calls if "OptimizedComponentNode" in c[0])
    assert "EvolvedToolDescriptionNode" in merge[0]
    assert merge[1]["component_type"] == "tool_description"
    assert merge[1]["identifier"] == "my_tool"
    assert merge[1]["demo_count"] == 1
    # the demo was attached as a trajectory node
    assert any("OptimizationTrajectoryNode" in c[0] for c in calls)


# ── AHE-3.44 — self-supervised extraction metric ─────────────────────────────


def _onehot(text: str) -> list[float]:
    v = [0.0] * 4096
    v[hash((text or "").lower().strip()) % 4096] = 1.0
    return v


def test_extraction_quality_rewards_clean_over_messy():
    from agent_utilities.knowledge_graph.extraction.extraction_optimizer import (
        extraction_quality,
    )

    clean = [
        {"subject": "Acme", "predicate": "makes", "object": "Widgets"},
        {"subject": "Acme", "predicate": "located_in", "object": "Ohio"},
        {"subject": "Bob", "predicate": "works_at", "object": "Acme"},
    ]
    messy = [
        {"subject": "Acme", "predicate": "makes", "object": "Widgets"},
        {"subject": "Acme", "predicate": "makes", "object": "Widgets"},  # duplicate
        {"subject": "acme", "predicate": "located_in", "object": "Ohio"},  # fragmented
    ]
    cq = extraction_quality(clean, embed_fn=_onehot)
    mq = extraction_quality(messy, embed_fn=_onehot)
    assert cq["score"] == pytest.approx(1.0)
    assert mq["score"] < cq["score"]
    assert extraction_quality([], embed_fn=_onehot)["score"] == 0.0


def test_canonical_consistency_penalizes_fragmentation():
    from agent_utilities.knowledge_graph.extraction.extraction_optimizer import (
        canonical_consistency,
    )

    assert canonical_consistency([{"subject": "Acme", "object": "Ohio"}]) == 1.0
    frag = canonical_consistency(
        [{"subject": "Acme", "object": "x"}, {"subject": "acme", "object": "x"}]
    )
    assert frag < 1.0


# ── AHE-3.45 — policy metrics ────────────────────────────────────────────────


def test_classification_accuracy_and_routing_success():
    from agent_utilities.harness.policy_optimization import (
        classification_accuracy,
        routing_success_rate,
    )

    assert classification_accuracy([True, False, True], [True, True, True]) == pytest.approx(2 / 3)
    assert classification_accuracy([], []) == 0.0
    assert routing_success_rate(
        [{"success": True}, {"success": False}, {"success": True}]
    ) == pytest.approx(2 / 3)


def test_self_supervised_optimizers_noop_without_data():
    from agent_utilities.harness.policy_optimization import (
        optimize_concept_matcher,
        optimize_routing_policy,
    )

    assert optimize_concept_matcher([]) is None
    assert optimize_routing_policy([]) is None


# ── AHE-3.40 — the optimize-component surface dispatch ───────────────────────


def test_run_component_optimization_dispatch():
    assert "error" in run_component_optimization("bogus")
    assert run_component_optimization("system_prompt")["status"] == "registered"
    assert run_component_optimization("tool_description")["target"] == "tool_description"
    # self-supervised targets report no_data when given none
    assert run_component_optimization("extraction", {"documents": []})["status"] == (
        "no_data_or_dspy_unavailable"
    )
    assert run_component_optimization("routing")["status"] == "no_data_or_dspy_unavailable"
