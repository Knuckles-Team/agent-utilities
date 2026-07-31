"""Live-path: graph_analyze action='assimilation_benchmark' reports measured parity (AHE-3.39)."""

import json

import pytest

from agent_utilities.mcp import kg_server


@pytest.mark.asyncio
async def test_assimilation_benchmark_action(monkeypatch):
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    kg_server.ensure_tools_registered()
    res = await kg_server._execute_tool(
        "graph_analyze", action="assimilation_benchmark", top_k=0
    )
    report = json.loads(res)
    assert report["total"] >= 7
    # Every assimilated mechanism beat its baseline in the paper's claimed direction.
    assert report["reproduced"] == report["total"]
    names = {r["name"] for r in report["results"]}
    assert any("PauseRec" in n for n in names)
    assert "markdown" in report and "claims reproduced" in report["markdown"]
