"""Characterization tests for EvolveAgent._record_proposed_change (CCN 23),
agent_utilities/harness/evolve_agent.py.

Pins OBSERVED behaviour before a complexity-reduction refactor. Must stay
byte-identical across the refactor commit.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.harness.evolve_agent import EvolveAgent
from agent_utilities.harness.manifest import (
    ChangeManifest,
    ComponentEdit,
    ComponentType,
)
from agent_utilities.harness.optimization_backend import opaque_program_reference


def _make_evolver(tmp_path, *, engine=None):
    return EvolveAgent(
        workspace_path=str(tmp_path),
        knowledge_engine=engine,
    )


def _compiled_state_dict(**overrides) -> dict:
    demo_ref = opaque_program_reference("demo", "d1")
    evidence_ref = opaque_program_reference("evidence", "e1")
    source_ref = opaque_program_reference("source", "s1")
    state = {
        "id": opaque_program_reference("candidate", "c1"),
        "program_ref": opaque_program_reference("program", "p1"),
        "optimizer": "bootstrap_few_shot",
        "execution": "native_kernel",
        "candidate_role": "proposal",
        "demonstration_refs": [demo_ref],
        "artifact_refs": [],
        "composition_refs": [],
        "instruction_ref": None,
        "tool_policy_ref": None,
        "model_profile_ref": None,
        "evidence_refs": [evidence_ref],
        "source_refs": [source_ref],
        "proof_ids": [],
        "contradiction_ids": [],
        "modalities": ["text"],
    }
    state.update(overrides)
    return state


def _edit(**overrides) -> ComponentEdit:
    metadata = overrides.pop("metadata", {})
    metadata.setdefault("program_compiled_state", _compiled_state_dict())
    defaults = dict(
        component_type=ComponentType.SYSTEM_PROMPT,
        file_path="prompts/system.md",
        edit_summary="tightened tool-selection instructions",
        evidence_references=[],
        metadata=metadata,
    )
    defaults.update(overrides)
    return ComponentEdit(**defaults)


def test_golden_path_writes_record_and_returns_proposal_id(tmp_path):
    evolver = _make_evolver(tmp_path)
    manifest = ChangeManifest()
    edit = _edit()

    proposal_id = evolver._record_proposed_change(
        edit, manifest, status="proposed", before=0.5, after=0.7, applied=False
    )

    assert proposal_id.startswith("eg:proposal:")
    assert edit.metadata["proposal_ref"] == proposal_id
    assert evolver._proposal_targets[proposal_id] == edit.file_path

    proposal_token = proposal_id.rsplit(":", 1)[-1]
    written_path = (
        tmp_path / ".specify" / "proposals" / f"prompt-proposal-{proposal_token}.json"
    )
    assert written_path.is_file()
    record = json.loads(written_path.read_text())
    assert record["status"] == "proposed"
    assert record["applied"] is False
    assert record["baseline_score"] == 0.5
    assert record["candidate_score"] == 0.7
    assert record["delta"] == 0.2
    assert record["optimizer"] == "bootstrap_few_shot"
    assert record["integrity_ref"].startswith("eg:proposal_integrity:")


def test_invalid_evidence_reference_rejected(tmp_path):
    evolver = _make_evolver(tmp_path)
    manifest = ChangeManifest()
    edit = _edit(evidence_references=["not-an-opaque-ref"])
    with pytest.raises(ValueError, match="evidence references are invalid"):
        evolver._record_proposed_change(
            edit, manifest, status="proposed", before=0.5, after=0.7, applied=False
        )


def test_invalid_status_rejected(tmp_path):
    evolver = _make_evolver(tmp_path)
    manifest = ChangeManifest()
    edit = _edit()
    with pytest.raises(ValueError, match="status is invalid"):
        evolver._record_proposed_change(
            edit, manifest, status="bogus", before=0.5, after=0.7, applied=False
        )


@pytest.mark.parametrize(
    "before,after",
    [
        (float("nan"), 0.5),
        (0.5, float("inf")),
        (-0.1, 0.5),
        (0.5, 1.1),
    ],
)
def test_unbounded_or_nonfinite_scores_rejected(tmp_path, before, after):
    evolver = _make_evolver(tmp_path)
    manifest = ChangeManifest()
    edit = _edit()
    with pytest.raises(ValueError, match="scores must be bounded"):
        evolver._record_proposed_change(
            edit, manifest, status="proposed", before=before, after=after, applied=False
        )


def test_agent_ref_reused_when_already_opaque_with_agent_namespace(tmp_path):
    evolver = _make_evolver(tmp_path)
    manifest = ChangeManifest()
    real_agent_ref = opaque_program_reference("agent", "a1")
    edit = _edit(metadata={"agent_ref": real_agent_ref})

    proposal_id = evolver._record_proposed_change(
        edit, manifest, status="proposed", before=0.5, after=0.7, applied=False
    )
    proposal_token = proposal_id.rsplit(":", 1)[-1]
    record = json.loads(
        (
            tmp_path
            / ".specify"
            / "proposals"
            / f"prompt-proposal-{proposal_token}.json"
        ).read_text()
    )
    assert record["agent_ref"] == real_agent_ref


def test_agent_ref_falls_back_to_program_ref_when_missing(tmp_path):
    evolver = _make_evolver(tmp_path)
    manifest = ChangeManifest()
    edit = _edit()  # no agent_ref in metadata

    proposal_id = evolver._record_proposed_change(
        edit, manifest, status="proposed", before=0.5, after=0.7, applied=False
    )
    proposal_token = proposal_id.rsplit(":", 1)[-1]
    record = json.loads(
        (
            tmp_path
            / ".specify"
            / "proposals"
            / f"prompt-proposal-{proposal_token}.json"
        ).read_text()
    )
    assert record["agent_ref"].startswith("eg:agent:")


def test_version_hash_reused_when_valid_else_derived_from_compiled_id(tmp_path):
    evolver = _make_evolver(tmp_path)

    manifest = ChangeManifest()
    edit_valid = _edit(metadata={"candidate_version_hash": "deadbeefdeadbeef"})
    pid1 = evolver._record_proposed_change(
        edit_valid, manifest, status="proposed", before=0.5, after=0.7, applied=False
    )
    record1 = json.loads(
        (
            tmp_path
            / ".specify"
            / "proposals"
            / f"prompt-proposal-{pid1.rsplit(':', 1)[-1]}.json"
        ).read_text()
    )
    assert record1["candidate_version_hash"] == "deadbeefdeadbeef"

    edit_invalid = _edit(metadata={"candidate_version_hash": "not-hex"})
    pid2 = evolver._record_proposed_change(
        edit_invalid, manifest, status="proposed", before=0.5, after=0.7, applied=False
    )
    record2 = json.loads(
        (
            tmp_path
            / ".specify"
            / "proposals"
            / f"prompt-proposal-{pid2.rsplit(':', 1)[-1]}.json"
        ).read_text()
    )
    compiled_id_tail = record2["program_compiled_state"]["id"].rsplit(":", 1)[-1][:16]
    assert record2["candidate_version_hash"] == compiled_id_tail


def test_trainset_size_invalid_falls_back_to_zero(tmp_path):
    evolver = _make_evolver(tmp_path)
    manifest = ChangeManifest()
    edit = _edit(metadata={"trainset_size": "not-a-number"})
    proposal_id = evolver._record_proposed_change(
        edit, manifest, status="proposed", before=0.5, after=0.7, applied=False
    )
    record = json.loads(
        (
            tmp_path
            / ".specify"
            / "proposals"
            / f"prompt-proposal-{proposal_id.rsplit(':', 1)[-1]}.json"
        ).read_text()
    )
    assert record["trainset_size"] == 0


def test_auto_apply_eligible_defaults_true_when_absent(tmp_path):
    evolver = _make_evolver(tmp_path)
    manifest = ChangeManifest()
    edit = _edit()
    proposal_id = evolver._record_proposed_change(
        edit, manifest, status="proposed", before=0.5, after=0.7, applied=False
    )
    record = json.loads(
        (
            tmp_path
            / ".specify"
            / "proposals"
            / f"prompt-proposal-{proposal_id.rsplit(':', 1)[-1]}.json"
        ).read_text()
    )
    assert record["auto_apply_eligible"] is True


def test_kg_persistence_called_with_json_string_for_compiled_state(tmp_path):
    class _FakeKG:
        def __init__(self):
            self.calls = []

        def add_node(self, node_id, node_type, properties=None):
            self.calls.append((node_id, node_type, properties))

    kg = _FakeKG()
    evolver = _make_evolver(tmp_path, engine=kg)
    manifest = ChangeManifest()
    edit = _edit()

    proposal_id = evolver._record_proposed_change(
        edit, manifest, status="proposed", before=0.5, after=0.7, applied=False
    )
    assert len(kg.calls) == 1
    node_id, node_type, props = kg.calls[0]
    assert node_id == proposal_id
    assert node_type == "ProposedPromptChange"
    assert "program_compiled_state" not in props
    assert isinstance(props["program_compiled_state_json"], str)


def test_kg_persistence_exception_is_swallowed(tmp_path):
    class _RaisingKG:
        def add_node(self, node_id, node_type, properties=None):
            raise RuntimeError("kg down")

    evolver = _make_evolver(tmp_path, engine=_RaisingKG())
    manifest = ChangeManifest()
    edit = _edit()

    # Must not raise -- persistence is best-effort.
    proposal_id = evolver._record_proposed_change(
        edit, manifest, status="proposed", before=0.5, after=0.7, applied=False
    )
    assert proposal_id.startswith("eg:proposal:")


def test_kg_without_add_node_attribute_is_skipped(tmp_path):
    class _NoAddNode:
        pass

    evolver = _make_evolver(tmp_path, engine=_NoAddNode())
    manifest = ChangeManifest()
    edit = _edit()

    # Must not raise.
    proposal_id = evolver._record_proposed_change(
        edit, manifest, status="proposed", before=0.5, after=0.7, applied=False
    )
    assert proposal_id.startswith("eg:proposal:")
