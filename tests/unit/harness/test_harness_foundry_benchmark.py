"""Harness-foundry benchmark (CONCEPT:AU-AHE.evaluation.parity-surpass-scoreboard) — all surpass claims reproduce."""

from __future__ import annotations

import pytest

pytest.importorskip("pyshacl")

from agent_utilities.harness.harness_foundry_benchmark import run_all, to_markdown


def test_all_surpass_claims_reproduce():
    results = run_all()
    assert len(results) == 5
    by_name = {r.name: r for r in results}
    # Headline: we block the τ³ concentration coupling HarnessX ships into.
    conc = by_name["concentration_tau3"]
    assert conc.claim_reproduced and conc.ours < conc.baseline
    # Held-out gate rejects the overfit variant their gate accepts.
    assert by_name["held_out_overfit_guard"].claim_reproduced
    # Cross-harness grouping recovers the cross-scaffold contrast.
    assert by_name["cross_harness_grouping"].claim_reproduced
    # Variant isolation ships a heterogeneous mixed edit the single-harness
    # seesaw rejects (CONCEPT:AU-AHE.harness.variant-pool).
    assert by_name["variant_isolation"].claim_reproduced
    # Signature-attribution refuses to credit an unattributed edit (CONCEPT:AU-AHE.evaluation.edit-claims-fix).
    assert by_name["attribution_falsifiability"].claim_reproduced
    assert all(r.claim_reproduced for r in results)


def test_markdown_renders():
    md = to_markdown(run_all())
    assert "5/5 surpass claims reproduced" in md
    assert "concentration_tau3" in md
