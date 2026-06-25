from __future__ import annotations

"""Tests for code_health per-repo baseline deltas.

CONCEPT:CE-039 — Baseline-aware new-vs-resolved deltas in the code-health sweep.
CONCEPT:KG-2.248 — baselines are engine-only (``:CodeHealthBaseline`` nodes on the
one engine authority, no local file cache), so these drive a real engine backend
bound to the conftest ``engine_graph`` ephemeral tenant (CONCEPT:KG-2.238).
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
        code_health._baseline_delta(None, "repoX", {"findings": ["a"]}, baseline_backend)
        == {}
    )
