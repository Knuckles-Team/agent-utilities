from __future__ import annotations

"""Tests for code_health per-repo baseline deltas.

CONCEPT:AU-KG.maintenance.baseline-vs-resolved-deltas — Baseline-aware new-vs-resolved deltas in the code-health sweep.
CONCEPT:AU-KG.maintenance.only-no-file-cache — baselines are engine-only (``:CodeHealthBaseline`` nodes on the
one engine authority, no local file cache), so these drive a real engine backend
bound to the conftest ``engine_graph`` ephemeral tenant (CONCEPT:AU-KG.memory.provides-real-ephemeral-one).
"""

import pytest

from agent_utilities.knowledge_graph.adaptation import code_health


@pytest.fixture()
def baseline_backend(engine_graph):
    """An ``EpistemicGraphBackend`` on the REAL ephemeral engine tenant."""
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )

    return EpistemicGraphBackend(graph_name=engine_graph.graph_name)


class _FakeBaselineModule:
    """Minimal stand-in for code-enhancer's analyze_baseline (snapshot/diff)."""

    @staticmethod
    def snapshot(report, now=None):
        fps = {f: {"label": f} for f in report.get("findings", [])}
        return {"fingerprints": fps}

    @staticmethod
    def diff(report, prior):
        cur = set(report.get("findings", []))
        base = set(prior.get("fingerprints", {}).keys())
        new = cur - base
        fixed = base - cur
        return {
            "counts": {"new": len(new), "fixed": len(fixed)},
            "new_debt_score": max(0, 100 - len(new) * 3),
        }


def test_first_run_has_no_delta_and_writes_baseline(baseline_backend):
    report = {"findings": ["orphan a.py", "dead foo"]}
    delta = code_health._baseline_delta(
        _FakeBaselineModule, "repoX", report, baseline_backend
    )
    assert delta == {}  # nothing to compare against on the first sweep
    # The snapshot was persisted on the engine, queryable back.
    snap = code_health._load_baseline_snapshot(baseline_backend, "repoX")
    assert snap is not None
    assert set(snap["fingerprints"]) == set(report["findings"])


def test_second_run_reports_new_and_fixed(baseline_backend):
    code_health._baseline_delta(
        _FakeBaselineModule, "repoX", {"findings": ["a", "b"]}, baseline_backend
    )
    delta = code_health._baseline_delta(
        _FakeBaselineModule, "repoX", {"findings": ["a", "c"]}, baseline_backend
    )
    assert delta["new"] == 1  # "c" is new
    assert delta["fixed"] == 1  # "b" resolved
    assert delta["new_debt_score"] == 97


def test_missing_module_degrades_gracefully(baseline_backend):
    assert (
        code_health._baseline_delta(
            None, "repoX", {"findings": ["a"]}, baseline_backend
        )
        == {}
    )


# ---------------------------------------------------------------------------
# BUG-059 — _save_baseline_snapshot is a JUSTIFIED chokepoint bypass, pinned.
# ---------------------------------------------------------------------------
#
# ``_save_baseline_snapshot`` writes straight to the raw engine-authority
# backend, never through IntelligenceGraphEngine._upsert_node/
# GraphComputeEngine.add_node, so it never reaches stamp_ownership. That is
# deliberate: the caller is the ``code_health`` maintenance daemon tick
# (opt-in via KG_CODE_HEALTH) with NO request/actor context at all, and the
# payload is a derived regression baseline, not owned content. This test
# pins that the write keeps working with zero actor bound, so nobody
# "fixes" it into calling stamp_ownership by accident and breaks the daemon.


class _FakeBaselineBackend:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props


def test_save_baseline_snapshot_works_with_no_actor_bound():
    import contextvars

    backend = _FakeBaselineBackend()

    def isolated():
        # No actor bound anywhere in this fresh context — must not raise.
        code_health._save_baseline_snapshot(backend, "repoX", {"fingerprints": {}})

    contextvars.Context().run(isolated)

    (props,) = backend.nodes.values()
    assert props["repo"] == "repoX"
    # No governance stamp was applied — this is the pinned, deliberate gap.
    assert "_owner_id" not in props
