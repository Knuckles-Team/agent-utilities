from __future__ import annotations

"""Tests for code_health per-repo baseline deltas.

CONCEPT:CE-039 — Baseline-aware new-vs-resolved deltas in the code-health sweep.
"""

import json

from agent_utilities.knowledge_graph.adaptation import code_health


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


def test_first_run_has_no_delta_and_writes_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(code_health, "_BASELINE_DIR", tmp_path)
    report = {"findings": ["orphan a.py", "dead foo"]}
    delta = code_health._baseline_delta(_FakeBaselineModule, "repoX", report)
    assert delta == {}  # nothing to compare against on the first sweep
    cache = tmp_path / "repoX.json"
    assert cache.exists()
    assert set(json.loads(cache.read_text())["fingerprints"]) == set(report["findings"])


def test_second_run_reports_new_and_fixed(tmp_path, monkeypatch):
    monkeypatch.setattr(code_health, "_BASELINE_DIR", tmp_path)
    code_health._baseline_delta(_FakeBaselineModule, "repoX", {"findings": ["a", "b"]})
    delta = code_health._baseline_delta(
        _FakeBaselineModule, "repoX", {"findings": ["a", "c"]}
    )
    assert delta["new"] == 1  # "c" is new
    assert delta["fixed"] == 1  # "b" resolved
    assert delta["new_debt_score"] == 97


def test_missing_module_degrades_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr(code_health, "_BASELINE_DIR", tmp_path)
    assert code_health._baseline_delta(None, "repoX", {"findings": ["a"]}) == {}
