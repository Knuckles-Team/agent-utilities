"""Live-path: graph_evaluate action='assimilation_benchmark' reports measured parity (AHE-3.39)."""

import pytest

from agent_utilities.mcp import kg_server


@pytest.mark.asyncio
async def test_assimilation_benchmark_action(monkeypatch):
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    kg_server.ensure_tools_registered()
    # assimilation_benchmark is a graph_evaluate action, not graph_analyze --
    # graph_analyze is strictly the six-action operations/structural surface
    # (analyze_suite.py's own module docstring: "graph_analyze is strictly
    # the six-action operations/structural surface; it is not a
    # compatibility catch-all"). Every focused tool (graph_code/_research/
    # _evaluate/_explain) returns the sole typed EvidenceBundle, never a raw
    # JSON string.
    bundle = await kg_server._execute_tool(
        "graph_evaluate", action="assimilation_benchmark", top_k=0
    )
    assert bundle.error is None, bundle.error
    # EvidenceBundle.from_payload projects the benchmark's "results" list
    # rows into .claims and keeps the full original payload (including
    # "total"/"reproduced"/"markdown") in the first reasoning_trace entry.
    payload = bundle.reasoning_trace[0]["payload"]
    assert payload["total"] >= 7
    # Every assimilated mechanism beat its baseline in the paper's claimed direction.
    assert payload["reproduced"] == payload["total"]
    names = {c["name"] for c in bundle.claims}
    assert any("PauseRec" in n for n in names)
    assert "markdown" in payload and "claims reproduced" in payload["markdown"]
