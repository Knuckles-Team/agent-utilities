"""Live-path: graph_evaluate action='assimilation_benchmark' reports measured parity (AHE-3.39).

'assimilation_benchmark' lives on 'graph_evaluate' (the graph_analyze
catch-all was split into focused suite tools) with the rest of the AHE
empirical-benchmark actions.
"""

import pytest

from agent_utilities.mcp import kg_server


@pytest.mark.asyncio
async def test_assimilation_benchmark_action(monkeypatch):
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    kg_server.ensure_tools_registered()
    res = await kg_server._execute_tool(
        "graph_evaluate", action="assimilation_benchmark", top_k=0
    )
    # graph_evaluate returns the sole typed EvidenceBundle response (not a JSON
    # string). The payload's "results" list steals EvidenceBundle.from_payload's
    # claims slot, so "total"/"reproduced"/"markdown" are only recoverable from
    # the losslessly-retained original payload in reasoning_trace[0]["payload"].
    report = res.reasoning_trace[0]["payload"]
    assert report["total"] >= 7
    # Every assimilated mechanism beat its baseline in the paper's claimed direction.
    assert report["reproduced"] == report["total"]
    names = {r["name"] for r in report["results"]}
    assert any("PauseRec" in n for n in names)
    assert "markdown" in report and "claims reproduced" in report["markdown"]
