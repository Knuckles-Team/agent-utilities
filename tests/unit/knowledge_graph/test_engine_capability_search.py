"""Engine-native filtered ANN for capability retrieval (CONCEPT:AU-P1-3).

Verifies capability/tenant/policy-restricted candidate selection is pushed down to
the engine's own vector index (a ``query.unified`` ``Filter``+``Rank`` plan, or a
native ``semantic_search`` + bounded post-filter) — NOT an in-process hnswlib/numpy
scan. The engine client is mocked throughout; these are pure unit tests.
"""

from __future__ import annotations

import types
from typing import Any

from agent_utilities.knowledge_graph.retrieval.engine_capability_search import (
    build_capability_filters,
    engine_filtered_search,
)


class _RecordingGraph:
    """Fake ``graph`` recording every ``query_unified`` plan it was called with."""

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.calls: list[list[dict[str, Any]]] = []

    def query_unified(self, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.calls.append(plan)
        return self.rows


class _NoUnifiedGraph:
    """A lean-tier engine surface: only the native ANN primitive, no ``query_unified``."""

    def __init__(self, rows: list[tuple[str, float]]):
        self.rows = rows
        self.semantic_search_calls: list[tuple[list[float], int]] = []

    def semantic_search(self, qvec: list[float], k: int) -> list[tuple[str, float]]:
        self.semantic_search_calls.append((qvec, k))
        return self.rows


def test_filter_plan_is_built_for_required_caps_and_tenant():
    filters = build_capability_filters(["web", "search"], "acme", ["gpu"])
    # One Filter condition per capability/policy tag, plus one for tenant.
    assert {"property": "capabilities", "op": "array_contains", "value": "web"} in filters
    assert {"property": "capabilities", "op": "array_contains", "value": "search"} in filters
    assert {"property": "tenant", "op": "eq_or_null", "value": "acme"} in filters
    assert {"property": "policy_tags", "op": "array_contains", "value": "gpu"} in filters


def test_no_filters_still_builds_empty_filter_list():
    assert build_capability_filters(None, None, None) == []


def test_engine_query_routes_to_unified_plan_with_filter_not_in_process_scan():
    """The defining AU-P1-3 assertion: a capability query calls the engine's
    ``query_unified`` with a ``Filter`` op present — the engine does the filtering,
    not a Python set-intersection scan."""
    graph = _RecordingGraph([{"id": "tool:math", "score": 0.9}])
    engine = types.SimpleNamespace(graph=graph)

    out = engine_filtered_search(
        engine, [0.1, 0.9, 0.0], k=3, required_caps=["arithmetic"]
    )

    assert out == [("tool:math", 0.9)]
    assert len(graph.calls) == 1
    plan = graph.calls[0]
    filter_ops = [op["Filter"] for op in plan if "Filter" in op]
    assert {"property": "capabilities", "op": "array_contains", "value": "arithmetic"} in (
        filter_ops
    )
    # A Rank + Limit leg is always present (the vector neighbourhood IS the engine ANN).
    assert any("Rank" in op for op in plan)
    assert any("Limit" in op for op in plan)


def test_engine_query_with_tenant_and_policy_filters():
    graph = _RecordingGraph([{"id": "agent:x", "score": 0.8}])
    engine = types.SimpleNamespace(graph=graph)

    out = engine_filtered_search(
        engine,
        [1.0, 0.0],
        k=1,
        tenant="tenant-a",
        policy_tags=["restricted"],
    )

    assert out == [("agent:x", 0.8)]
    plan = graph.calls[0]
    filter_ops = [op["Filter"] for op in plan if "Filter" in op]
    assert {"property": "tenant", "op": "eq_or_null", "value": "tenant-a"} in filter_ops
    assert {
        "property": "policy_tags",
        "op": "array_contains",
        "value": "restricted",
    } in filter_ops


def test_unfiltered_query_still_uses_unified_plan_when_available():
    graph = _RecordingGraph([{"id": "a", "score": 0.5}])
    engine = types.SimpleNamespace(graph=graph)

    out = engine_filtered_search(engine, [1.0], k=1)

    assert out == [("a", 0.5)]
    assert len(graph.calls) == 1


def test_falls_to_native_ann_when_unified_plan_unavailable():
    """A lean-tier engine (no ``query`` feature) degrades to ``semantic_search`` +
    a bounded post-filter over the returned pool — still no O(N) Python scan."""
    graph = _NoUnifiedGraph([("tool:a", 0.95), ("tool:b", 0.4)])
    engine = types.SimpleNamespace(graph=graph)

    out = engine_filtered_search(engine, [1.0, 0.0], k=5, required_caps=None)

    assert out == [("tool:a", 0.95), ("tool:b", 0.4)]
    assert len(graph.semantic_search_calls) == 1


def test_returns_none_when_no_engine_vector_surface_at_all():
    """No ``graph`` at all -> None (signal: fall back to the bounded in-process cache)."""
    engine = types.SimpleNamespace()
    assert engine_filtered_search(engine, [1.0], k=1) is None

    class _NoVectorGraph:
        pass

    engine2 = types.SimpleNamespace(graph=_NoVectorGraph())
    assert engine_filtered_search(engine2, [1.0], k=1) is None


def test_unified_plan_exception_degrades_to_native_ann():
    class _BrokenUnifiedGraph:
        def query_unified(self, plan):
            raise RuntimeError("engine build without `query` feature")

        def semantic_search(self, qvec, k):
            return [("fallback:a", 0.7)]

    engine = types.SimpleNamespace(graph=_BrokenUnifiedGraph())
    out = engine_filtered_search(engine, [1.0], k=1)
    assert out == [("fallback:a", 0.7)]


def test_native_ann_bounded_post_filter_excludes_non_matching_candidate():
    """Tier 2 (no ``query_unified``): the post-filter over the bounded returned
    pool still excludes a candidate lacking the required capability."""

    class _NoUnifiedWithProps:
        def __init__(self):
            self._props = {
                "tool:has_cap": {"capabilities": ["arithmetic"]},
                "tool:missing_cap": {"capabilities": ["web"]},
            }

        def semantic_search(self, qvec, k):
            return [("tool:has_cap", 0.9), ("tool:missing_cap", 0.85)]

        def _get_node_properties(self, nid):
            return self._props.get(nid, {})

    engine = types.SimpleNamespace(graph=_NoUnifiedWithProps())
    out = engine_filtered_search(
        engine, [1.0], k=5, required_caps=["arithmetic"]
    )
    assert out == [("tool:has_cap", 0.9)]
