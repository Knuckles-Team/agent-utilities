"""AEGIS unified loop (CONCEPT:AU-AHE.harness.run-aegis-loop-over): cross-round self-correction.

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


# ── Variant isolation / ensemble routing (CONCEPT:AU-AHE.harness.variant-pool) ──────────────────
def test_variant_isolation_forks_mixed_edit_instead_of_rejecting():
    """A mixed edit (fixes one cluster, regresses another) is rejected by the
    single-harness seesaw but SHIPS as a forked variant under isolation."""

    def evolver(_l):
        return {
            "id": "edit:mixed",
            "dimension": "D4_tools",
            "fixes": ["taskA"],
            "regresses": ["taskB"],
        }

    # Variant isolation ON: forks a variant scoped to {taskA}; taskB is out of
    # scope so the per-variant seesaw passes → ships.
    iso = AegisLoop(evolver, variant_capacity=2)
    d = iso.run_round(1)
    assert d.shipped and d.forked and d.variant_id
    # The forked variant's scoped regression set is empty (taskB routes elsewhere).
    assert iso.shipped[0]["regresses"] == []


def test_single_harness_seesaw_rejects_in_scope_regression():
    """With isolation OFF the mixed edit is not forked; an explicit base variant
    covering taskB makes the regression in-scope → the SHACL seesaw rejects."""
    from agent_utilities.harness.harness_gate import HarnessGate

    gate = HarnessGate()
    verdict = gate.check_facts(
        edits=[{"id": "e1", "dimension": "D4_tools", "regresses": ["taskB"]}],
        variants=[{"id": "base", "status": "accepted", "applies": ["e1"]}],
    )
    assert not verdict.passed
    assert any(
        "no-regression" in r.lower() or "seesaw" in r.lower() for r in verdict.reasons
    )


# ── Selective invocation + patience (CONCEPT:AU-AHE.harness.per-dimension-ship-outcome) ───────────────────────
def test_empty_landscape_short_circuits_as_idle():
    def evolver(_l):
        return {}  # nothing actionable

    loop = AegisLoop(evolver)
    d = loop.run_round(1)
    assert not d.shipped and "no actionable candidate" in d.reasons


def test_patience_early_stops_on_consecutive_idle():
    def evolver(_l):
        return None  # always idle

    loop = AegisLoop(evolver, patience=2)
    decisions = loop.run(rounds=10)
    # Stops after `patience` consecutive idle rounds rather than running all 10.
    assert len(decisions) == 2


# ── Reputation audit / declining-yield diversion (CONCEPT:AU-AHE.harness.per-dimension-ship-outcome) ──────────
def test_reputation_audit_discourages_declining_dimension():
    # An evolver that keeps shipping into D4 but a verifier that fails it every time
    # builds a low hit-rate ledger for D4 → it becomes discouraged with a concern.
    def evolver(_l):
        return {"id": "e", "dimension": "D4_tools"}

    def verifier(_e):
        return False, []  # manifest verify always fails → ledger records misses

    loop = AegisLoop(evolver, verifier_fn=verifier, hit_rate_floor=0.5)
    loop.run(rounds=3)
    land = loop.adaptation_landscape()
    assert "D4_tools" in land["discouraged_dimensions"]
    assert any("strategy_concern" in c for c in land["strategy_concerns"])


# ── Deterministic gate sequence: smoke + normalize (CONCEPT:AU-AHE.harness.manifest-verify) ────────
def test_smoke_test_blocks_uninstantiable_edit():
    def evolver(_l):
        return {"id": "e:bad", "dimension": "D4_tools"}

    loop = AegisLoop(evolver, smoke_fn=lambda _e: False)
    d = loop.run_round(1)
    assert not d.shipped and "smoke test failed" in d.reasons


def test_config_normalization_dedups_identical_edit():
    calls = {"n": 0}

    def evolver(_l):
        calls["n"] += 1
        # Always the same canonical edit (different id, same content).
        return {"id": f"e:{calls['n']}", "dimension": "D4_tools", "body": "X"}

    loop = AegisLoop(evolver, normalize_fn=lambda e: f"{e['dimension']}:{e['body']}")
    decisions = loop.run(rounds=3)
    shipped = [d for d in decisions if d.shipped]
    # First ships; the identical re-proposals are deduped, not counted as progress.
    assert len(shipped) == 1
    assert any("duplicate" in r for d in decisions for r in d.reasons)
