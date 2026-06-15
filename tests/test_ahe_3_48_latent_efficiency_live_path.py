"""Latent-native efficiency benchmark (CONCEPT:AHE-3.48).

Unit + live-path: each latent-native mechanism (rollout latent memory KG-2.73b,
ontology-prior retrieval KG-2.44b) beats its round-tripped/flat baseline under a
fixed seed, and the measured lift is reachable through the graph_analyze surface.
"""

import json

import pytest

from agent_utilities.harness.latent_efficiency_benchmark import run_all, to_markdown
from agent_utilities.mcp import kg_server

pytestmark = pytest.mark.concept("AHE-3.48")


def test_run_all_reports_lift_for_each_mechanism():
    results = run_all(seed=0)
    assert len(results) >= 2
    for r in results:
        assert r.baseline is not None and r.ours is not None
        assert r.claim_reproduced  # ours beat the baseline in the claimed direction
        assert r.lift > 0.0
    md = to_markdown(results)
    assert "claims reproduced" in md


@pytest.mark.asyncio
async def test_latent_efficiency_benchmark_live_path(monkeypatch):
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    kg_server.ensure_tools_registered()
    res = await kg_server._execute_tool(
        "graph_analyze", action="latent_efficiency_benchmark", top_k=0
    )
    report = json.loads(res)
    assert report["total"] >= 2
    assert report["reproduced"] == report["total"]
    names = {r["name"] for r in report["results"]}
    assert any("KG-2.73b" in n for n in names)
    assert any("KG-2.44b" in n for n in names)
    for r in report["results"]:
        assert "baseline" in r and "ours" in r and "lift" in r
    assert "markdown" in report and "claims reproduced" in report["markdown"]
