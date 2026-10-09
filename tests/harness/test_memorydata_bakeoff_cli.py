#!/usr/bin/python
from __future__ import annotations

"""Tests for the MemoryData bake-off CLI (CONCEPT:AU-AHE.harness.hardening-transparency-surface).

Exercises ``agent_utilities.harness.memorydata.bakeoff:main`` (also runnable as
``python -m agent_utilities.harness.memorydata.bakeoff``) end to end against the offline
``mock`` backend so the harness is proven reachable from outside its own package, not merely
importable.
"""

from agent_utilities.harness.memorydata.bakeoff import main


def test_main_runs_offline_bakeoff_and_prints_scoreboard(capsys) -> None:
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MemoryData Graph-OS Bake-off" in out
    assert "Measured results" in out
    # Router included by default -> the router-vs-best-single section renders.
    assert "Router vs best single config" in out


def test_main_no_include_router_skips_router_section(capsys) -> None:
    rc = main(["--no-include-router"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Measured results" in out
    assert "Router vs best single config" not in out
