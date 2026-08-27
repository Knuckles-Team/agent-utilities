"""Characterization tests for EvolveAgent._manifest_persistence_payload (CCN 25),
agent_utilities/harness/evolve_agent.py.

tests/unit/harness/test_agent_hardening_loop.py has one existing scenario
(raw content is excluded, refs are opaque). This file targets the remaining
branches: component_ref fallback vs. reuse, conditional optional refs (diff/
commit/attribution/capability_evidence), the compiled-metadata allowlist
(agent/component/proposal refs only kept if opaque, promote/auto_apply_
eligible bool-only, baseline/candidate score finite-only, trainset_size
clamping, candidate_version_hash regex, apply_status enum, and the
program_compiled_state pass-through raising on an invalid value),
parent_round_id presence, verification_status fallback to "error", the
verification_result sub-object, and finite()'s bool/inf/out-of-range
exclusions.

Pins OBSERVED behaviour before a complexity-reduction refactor. Must stay
byte-identical across the refactor commit.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_utilities.harness.evolve_agent import EvolveAgent
from agent_utilities.harness.manifest import (
    ChangeManifest,
    ComponentEdit,
    ComponentType,
    VerificationResult,
)


def _edit(**overrides) -> ComponentEdit:
    defaults = dict(
        component_type=ComponentType.SYSTEM_PROMPT,
        file_path="prompts/system.md",
        edit_summary="tightened the tool-selection instructions",
    )
    defaults.update(overrides)
    return ComponentEdit(**defaults)


def test_component_ref_falls_back_to_file_path_when_metadata_ref_not_opaque():
    manifest = ChangeManifest()
    manifest.add_edit(_edit(metadata={"component_ref": "not-an-opaque-ref"}))
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    assert payload["edits"][0]["component_ref"].startswith("eg:component:")


def test_component_ref_reuses_an_already_opaque_metadata_ref():
    manifest = ChangeManifest()
    from agent_utilities.harness.optimization_backend import opaque_program_reference

    real_ref = opaque_program_reference("component", "some-component")
    manifest.add_edit(_edit(metadata={"component_ref": real_ref}))
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    assert payload["edits"][0]["component_ref"] == real_ref


def test_optional_refs_absent_when_source_fields_are_empty():
    manifest = ChangeManifest()
    manifest.add_edit(_edit())
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    row = payload["edits"][0]
    assert "diff_ref" not in row
    assert "commit_ref" not in row
    assert "attribution_ref" not in row
    assert "capability_evidence_refs" not in row
    assert "compiled_metadata" not in row


def test_optional_refs_present_when_source_fields_are_set():
    manifest = ChangeManifest()
    manifest.add_edit(
        _edit(
            diff_content="some diff",
            git_commit_sha="abc123",
            attribution_signature={"tool_call": "Fetch", "min_count": 1},
            capability_evidence=[{"capability": "search", "level": 2}],
        )
    )
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    row = payload["edits"][0]
    assert row["diff_ref"].startswith("eg:diff:")
    assert row["commit_ref"].startswith("eg:commit:")
    assert row["attribution_ref"].startswith("eg:attribution:")
    assert len(row["capability_evidence_refs"]) == 1
    assert row["capability_evidence_refs"][0].startswith("eg:capability_evidence:")


def test_compiled_metadata_keeps_only_opaque_agent_component_proposal_refs():
    from agent_utilities.harness.optimization_backend import opaque_program_reference

    real_agent_ref = opaque_program_reference("agent", "a1")
    manifest = ChangeManifest()
    manifest.add_edit(
        _edit(
            metadata={
                "agent_ref": real_agent_ref,
                "component_ref": "not-opaque",  # dropped from compiled_metadata
                "proposal_ref": "also-not-opaque",  # dropped
            }
        )
    )
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    compiled = payload["edits"][0]["compiled_metadata"]
    assert compiled["agent_ref"] == real_agent_ref
    assert "component_ref" not in compiled
    assert "proposal_ref" not in compiled


def test_compiled_metadata_promote_and_auto_apply_eligible_are_bool_only():
    manifest = ChangeManifest()
    manifest.add_edit(
        _edit(metadata={"promote": True, "auto_apply_eligible": "yes"})  # not a bool
    )
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    compiled = payload["edits"][0]["compiled_metadata"]
    assert compiled["promote"] is True
    assert "auto_apply_eligible" not in compiled


def test_compiled_metadata_scores_use_finite_and_reject_non_numeric():
    manifest = ChangeManifest()
    manifest.add_edit(
        _edit(
            metadata={
                "baseline_score": 0.5,
                "candidate_score": "not-a-number",
            }
        )
    )
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    compiled = payload["edits"][0]["compiled_metadata"]
    assert compiled["baseline_score"] == 0.5
    assert "candidate_score" not in compiled


def test_finite_excludes_bool_inf_nan_and_out_of_range():
    manifest = ChangeManifest()
    manifest.add_edit(
        _edit(metadata={"baseline_score": True})  # bool must NOT count as numeric
    )
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    assert "compiled_metadata" not in payload["edits"][0]

    for bad in (float("inf"), float("nan"), 2_000_000_000):
        m = ChangeManifest()
        m.add_edit(_edit(metadata={"baseline_score": bad}))
        p = EvolveAgent._manifest_persistence_payload(m)
        assert "compiled_metadata" not in p["edits"][0]


def test_trainset_size_is_clamped_to_zero_and_one_million():
    manifest = ChangeManifest()
    manifest.add_edit(_edit(metadata={"trainset_size": -5}))
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    assert payload["edits"][0]["compiled_metadata"]["trainset_size"] == 0

    manifest2 = ChangeManifest()
    manifest2.add_edit(_edit(metadata={"trainset_size": 5_000_000}))
    payload2 = EvolveAgent._manifest_persistence_payload(manifest2)
    assert payload2["edits"][0]["compiled_metadata"]["trainset_size"] == 1_000_000

    manifest3 = ChangeManifest()
    manifest3.add_edit(_edit(metadata={"trainset_size": True}))  # bool excluded
    payload3 = EvolveAgent._manifest_persistence_payload(manifest3)
    assert "compiled_metadata" not in payload3["edits"][0]


def test_candidate_version_hash_requires_exact_16_hex_chars():
    manifest = ChangeManifest()
    manifest.add_edit(
        _edit(metadata={"candidate_version_hash": "deadbeefdeadbeef"})  # 16 hex chars
    )
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    assert (
        payload["edits"][0]["compiled_metadata"]["candidate_version_hash"]
        == "deadbeefdeadbeef"
    )

    manifest2 = ChangeManifest()
    manifest2.add_edit(_edit(metadata={"candidate_version_hash": "not-hex!!"}))
    payload2 = EvolveAgent._manifest_persistence_payload(manifest2)
    assert "compiled_metadata" not in payload2["edits"][0]


def test_apply_status_only_kept_when_a_known_value():
    manifest = ChangeManifest()
    manifest.add_edit(_edit(metadata={"apply_status": "applied"}))
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    assert payload["edits"][0]["compiled_metadata"]["apply_status"] == "applied"

    manifest2 = ChangeManifest()
    manifest2.add_edit(_edit(metadata={"apply_status": "bogus"}))
    payload2 = EvolveAgent._manifest_persistence_payload(manifest2)
    assert "compiled_metadata" not in payload2["edits"][0]


def test_invalid_program_compiled_state_raises_uncaught_validation_error():
    # OBSERVED: model_validate(compiled) is NOT wrapped in try/except, so an
    # invalid program_compiled_state value crashes the whole payload build
    # rather than being dropped like the other malformed metadata fields.
    manifest = ChangeManifest()
    manifest.add_edit(
        _edit(metadata={"program_compiled_state": {"not": "a valid compiled state"}})
    )
    with pytest.raises(ValidationError):
        EvolveAgent._manifest_persistence_payload(manifest)


def test_parent_round_ref_is_none_when_manifest_has_no_parent():
    manifest = ChangeManifest()
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    assert payload["parent_round_ref"] is None


def test_parent_round_ref_is_a_reference_when_manifest_has_a_parent():
    manifest = ChangeManifest(parent_round_id="round:abc123")
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    assert payload["parent_round_ref"].startswith("eg:round:")


def test_verification_status_falls_back_to_error_for_unknown_values():
    manifest = ChangeManifest(verification_status="not-a-real-status")
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    assert payload["verification_status"] == "error"

    manifest2 = ChangeManifest(verification_status="confirmed")
    payload2 = EvolveAgent._manifest_persistence_payload(manifest2)
    assert payload2["verification_status"] == "confirmed"


def test_no_verification_key_when_verification_result_absent():
    manifest = ChangeManifest()
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    assert "verification" not in payload


def test_verification_result_projects_recommendation_and_refs():
    manifest = ChangeManifest(
        verification_result=VerificationResult(
            fix_precision=0.8,
            recommendation="confirm",
            unattributed_edits=["edit:a", "edit:a", "edit:b"],  # dup to check dedup
        )
    )
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    v = payload["verification"]
    assert v["fix_precision"] == 0.8
    assert v["recommendation"] == "confirm"
    assert len(v["unattributed_edit_refs"]) == 2  # deduplicated


def test_verification_result_unknown_recommendation_becomes_empty_string():
    manifest = ChangeManifest(
        verification_result=VerificationResult(recommendation="not-a-real-choice")
    )
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    assert payload["verification"]["recommendation"] == ""


def test_timestamp_invalid_format_becomes_empty_string():
    manifest = ChangeManifest(timestamp="not-an-iso-timestamp")
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    assert payload["timestamp"] == ""


def test_timestamp_valid_iso_format_is_preserved():
    manifest = ChangeManifest(timestamp="2026-08-27T12:00:00Z")
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    assert payload["timestamp"] == "2026-08-27T12:00:00Z"


def test_edits_over_1000_are_truncated():
    manifest = ChangeManifest()
    for i in range(1002):
        manifest.add_edit(_edit(edit_summary=f"edit {i}"))
    payload = EvolveAgent._manifest_persistence_payload(manifest)
    assert len(payload["edits"]) == 1000
