"""AEGIS unified loop (CONCEPT:AHE-3.52): cross-round self-correction.

The integration win over HarnessX: because the Critic is the SHACL gate reasoning
over accumulated edits, a naive evolver that keeps hammering one dimension is
**stopped before** the coupling tipping point — and a landscape-aware evolver that
diversifies ships every round.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyshacl")

from agent_utilities.harness.aegis_loop import AegisLoop


def test_loop_blocks_concentration_across_rounds():
    # Naive evolver: always edits the same dimension (the HarnessX failure pattern).
    def naive_evolver(_landscape):
        return {"id": None, "dimension": "D2_context"}

    seq = 0

    def evolver(landscape):
        nonlocal seq
        seq += 1
        e = naive_evolver(landscape)
        e["id"] = f"edit:{seq}"
        return e

    loop = AegisLoop(evolver)
    decisions = loop.run(rounds=6)
    shipped = [d for d in decisions if d.shipped]
    blocked = [d for d in decisions if not d.shipped]
    # First two same-dimension edits ship; the 3rd+ are blocked by concentration —
    # the loop refuses to walk into the tipping point.
    assert len(shipped) == 2, [(_d.round, _d.shipped) for _d in decisions]
    assert blocked and all(
        "concentration" in r.lower() for d in blocked for r in d.reasons
    )


def test_landscape_aware_evolver_diversifies_and_ships_all():
    dims = ["D1_model", "D2_context", "D3_memory", "D4_tools", "D7_control", "D8_obs"]
    seq = 0

    def diversifying_evolver(landscape):
        # Pick the least-used dimension (Planner signal from the ontology).
        nonlocal seq
        seq += 1
        used = landscape["used"]
        pick = min(dims, key=lambda d: used.get(d, 0))
        return {"id": f"edit:{seq}", "dimension": pick}

    loop = AegisLoop(diversifying_evolver)
    decisions = loop.run(rounds=6)
    assert all(d.shipped for d in decisions), [(d.round, d.reasons) for d in decisions]
    # The landscape was actually consulted (no dimension exceeded the cap).
    assert max(loop.adaptation_landscape()["used"].values()) < 3


def test_manifest_verifier_rejects_before_gate():
    def evolver(_l):
        return {"id": "edit:x", "dimension": "D4_tools"}

    def verifier(_edit):
        return False, ["task:regressed"]

    loop = AegisLoop(evolver, verifier_fn=verifier)
    d = loop.run_round(1)
    assert not d.shipped and "manifest verify failed" in d.reasons
    assert loop.shipped == []
