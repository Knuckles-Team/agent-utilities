"""Live-path: graph_analyze action='night_shift' runs the vault swarm (KG-2.84).

Wire-First proof — the night-shift swarm is reachable through the real
graph_analyze tool (MCP + REST twin /graph/analyze) over a vault directory.
"""

import json

import pytest

from agent_utilities.knowledge_graph.research.night_shift import NightShiftSwarm
from agent_utilities.mcp import kg_server


@pytest.mark.asyncio
async def test_night_shift_action_runs_over_vault(tmp_path, monkeypatch):
    # Pre-populate a vault with two contradicting sources.
    swarm = NightShiftSwarm(tmp_path)
    swarm.scout(
        [
            ("s1", "lithium cost is the binding constraint on EV adoption."),
            (
                "s2",
                "sodium-ion batteries now undercut lithium on cost, so lithium is "
                "not the binding constraint on EV adoption.",
            ),
        ]
    )

    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    kg_server.ensure_tools_registered()
    res = await kg_server._execute_tool(
        "graph_analyze", action="night_shift", target=str(tmp_path)
    )
    report = json.loads(res)
    assert report["atoms_created"] >= 2
    assert "briefing_path" in report
    # The contradiction between the two sources is surfaced for human judgment.
    assert isinstance(report["frictions"], list)
    # A briefing file was actually written.
    assert (tmp_path / "briefings").exists()
