"""Plan 03 Step 3: learned/adaptive routing policies surfaced via the routing
strategy package (canonical location), re-exporting the strangled
graph/adaptive_agent_router.py implementation.
"""

from __future__ import annotations

from agent_utilities.graph.routing.strategies.learned import (
    CostAwareRouter,
    RuleBasedPolicy,
    TopologicalRoutingPolicy,
    TraceLearnedPolicy,
    extract_task_features,
)


def test_single_source_identity_with_impl():
    import agent_utilities.graph.adaptive_agent_router as impl

    # The strategy package re-exports the same class objects (single source).
    assert RuleBasedPolicy is impl.RuleBasedPolicy
    assert TraceLearnedPolicy is impl.TraceLearnedPolicy
    assert CostAwareRouter is impl.CostAwareRouter
    assert TopologicalRoutingPolicy is impl.TopologicalRoutingPolicy


def test_extract_task_features_returns_signal():
    feats = extract_task_features("design a distributed scheduler with fault tolerance")
    assert isinstance(feats, dict)
    assert feats  # non-empty feature vector


def test_no_merge_markers_left_in_impl():
    import agent_utilities.graph.adaptive_agent_router as impl

    src = open(impl.__file__, encoding="utf-8").read()
    assert "# --- Merged from" not in src  # 7th botched-merge marker cleared
