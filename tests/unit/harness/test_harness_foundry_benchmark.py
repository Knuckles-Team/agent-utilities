"""Harness-foundry benchmark (CONCEPT:AU-AHE.evaluation.parity-surpass-scoreboard) — all surpass claims reproduce.

EH-431: exercises ``HarnessGate``, which validates through the engine's real
``shacl_validate_ad_hoc`` surface, never local ``pyshacl`` — requires a real
engine and skips cleanly (via the standard AU engine-availability check) when
none is available in this environment.
"""

from __future__ import annotations

from agent_utilities.harness.harness_foundry_benchmark import run_all, to_markdown


def test_all_surpass_claims_reproduce(engine_graph):
    results = run_all(engine_graph)
    assert len(results) == 4
    by_name = {r.name: r for r in results}
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


def test_markdown_renders(engine_graph):
    md = to_markdown(run_all(engine_graph))
    assert "4/4 surpass claims reproduced" in md
    assert "variant_isolation" in md
