"""Source assertion: the graph_loops ``placement_control`` action is removed."""

from __future__ import annotations

from pathlib import Path

import pytest

import agent_utilities.mcp.tools.state_tools as state_tools


@pytest.mark.spec("AU-SEMANTIC-R019.6.3.1")
def test_state_tools_source_has_no_placement_control() -> None:
    source = Path(state_tools.__file__).read_text(encoding="utf-8")
    for removed in (
        "placement_control",
        "placement_scan_limit",
        "placement_canary_tolerance",
    ):
        assert removed not in source
