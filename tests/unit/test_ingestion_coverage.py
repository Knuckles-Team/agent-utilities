"""Ingestion coverage + freshness SLA assessment tests (CONCEPT:OS-5.47)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agent_utilities.knowledge_graph.ingestion.coverage import (
    assess_coverage,
    enumerate_agent_packages_repos,
    repo_symbol_counts,
)

_NOW = datetime(2026, 6, 20, tzinfo=UTC)


@pytest.mark.concept("OS-5.47")
def test_enumerate_agent_packages_repos(tmp_path):
    manifest = tmp_path / "workspace.yml"
    manifest.write_text(
        """
subdirectories:
  agent-packages:
    repositories:
      - url: https://x/agent-utilities.git
      - url: https://x/epistemic-graph
    subdirectories:
      agents:
        repositories:
          - url: https://x/servicenow-api.git
  services:
    repositories:
      - url: https://x/should-not-appear.git
""",
        encoding="utf-8",
    )
    repos = enumerate_agent_packages_repos(manifest)
    assert repos == ["agent-utilities", "epistemic-graph", "servicenow-api"]
    assert "should-not-appear" not in repos


class FakeBackend:
    def __init__(self, counts):
        self._counts = counts

    def execute(self, cypher, params):
        needle = params.get("needle", "")
        for repo, n in self._counts.items():
            if f"/{repo}/" in needle:
                return [{"n": n}]
        return []  # AGE returns no rows for a zero aggregate -> treated as 0


@pytest.mark.concept("OS-5.47")
def test_repo_symbol_counts_handles_zero_and_missing():
    backend = FakeBackend({"agent-utilities": 4200, "epistemic-graph": 0})
    counts = repo_symbol_counts(
        backend, ["agent-utilities", "epistemic-graph", "ghost"]
    )
    assert counts == {"agent-utilities": 4200, "epistemic-graph": 0, "ghost": 0}


@pytest.mark.concept("OS-5.47")
def test_assess_coverage_flags_missing_and_stale():
    repos = ["alpha-api", "bravo-mcp", "charlie-agent"]
    counts = {"alpha-api": 100, "bravo-mcp": 0, "charlie-agent": 50}
    fresh = {
        "/agent-packages/alpha-api/": (_NOW - timedelta(days=1)).isoformat(),  # fresh
        "/agent-packages/charlie-agent/": (
            _NOW - timedelta(days=30)
        ).isoformat(),  # stale
    }
    rep = assess_coverage(repos, counts, fresh, sla_days=7, now=_NOW)
    assert rep["covered"] == 2
    assert rep["missing"] == ["bravo-mcp"]
    assert [s["repo"] for s in rep["stale"]] == ["charlie-agent"]
    assert rep["total_symbols"] == 150
    assert rep["coverage_pct"] == pytest.approx(66.7, abs=0.1)


@pytest.mark.concept("OS-5.47")
def test_assess_coverage_stale_uses_most_recent_sync():
    # A repo synced 30d ago AND 1d ago (codebase + codebase_file watermarks) is
    # NOT stale — the most-recent sync governs.
    rep = assess_coverage(
        ["zeta-mcp"],
        {"zeta-mcp": 10},
        {
            "/agent-packages/zeta-mcp/old": (_NOW - timedelta(days=30)).isoformat(),
            "/agent-packages/zeta-mcp/new": (_NOW - timedelta(days=1)).isoformat(),
        },
        sla_days=7,
        now=_NOW,
    )
    assert rep["stale"] == []


@pytest.mark.concept("OS-5.47")
def test_assess_coverage_all_good():
    rep = assess_coverage(
        ["solo-api"],
        {"solo-api": 10},
        {"/solo-api/": _NOW.isoformat()},
        sla_days=7,
        now=_NOW,
    )
    assert rep["missing"] == [] and rep["stale"] == []
    assert rep["coverage_pct"] == 100.0
