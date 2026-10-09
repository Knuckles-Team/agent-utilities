"""Focused contracts for the AU-owned Kafka enqueue-only-proof evidence check.

Closes AU-INTEGRATION-R010: AU does not run epistemic-graph's Kafka
enqueue-only-proof gate itself, but must record, in its own evidence trail,
that the exact pinned epistemic-graph version passed it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.release.check_eg_kafka_gate_evidence import (
    EvidenceError,
    verify_eg_kafka_gate_evidence,
)

_PYPROJECT = '"epistemic-graph[full]>=2.27.0,<3.0.0",\n'


def _write_pyproject(tmp_path: Path) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(_PYPROJECT, encoding="utf-8")
    return path


def _write_evidence(tmp_path: Path, **overrides: object) -> Path:
    record = {
        "epistemic_graph_version": "2.27.0",
        "epistemic_graph_commit": "5140b5c75041de6aed33921f6b8bbb1da05c4d63",
        "gate": "scripts/constrained_parallelism_gate.sh",
        "steps": ["enqueue_only_test_constrained_1"],
        "result": "passed",
        "evidence_kind": "external_dependency",
    }
    record.update(overrides)
    path = tmp_path / "eg-kafka-gate-evidence.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_matching_pin_and_passed_result_is_accepted(tmp_path: Path) -> None:
    pyproject = _write_pyproject(tmp_path)
    evidence = _write_evidence(tmp_path)

    result = verify_eg_kafka_gate_evidence(
        pyproject_path=pyproject, evidence_path=evidence
    )

    assert result["epistemic_graph_version"] == "2.27.0"
    assert result["result"] == "passed"


def test_version_mismatch_is_refused(tmp_path: Path) -> None:
    pyproject = _write_pyproject(tmp_path)
    evidence = _write_evidence(tmp_path, epistemic_graph_version="2.26.0")

    with pytest.raises(EvidenceError, match="does not match the declared pin floor"):
        verify_eg_kafka_gate_evidence(pyproject_path=pyproject, evidence_path=evidence)


def test_non_passed_result_is_refused(tmp_path: Path) -> None:
    pyproject = _write_pyproject(tmp_path)
    evidence = _write_evidence(tmp_path, result="failed")

    with pytest.raises(EvidenceError, match="not 'passed'"):
        verify_eg_kafka_gate_evidence(pyproject_path=pyproject, evidence_path=evidence)


def test_missing_steps_is_refused(tmp_path: Path) -> None:
    pyproject = _write_pyproject(tmp_path)
    evidence = _write_evidence(tmp_path, steps=[])

    with pytest.raises(EvidenceError, match="names no Kafka"):
        verify_eg_kafka_gate_evidence(pyproject_path=pyproject, evidence_path=evidence)


def test_wrong_evidence_kind_is_refused(tmp_path: Path) -> None:
    pyproject = _write_pyproject(tmp_path)
    evidence = _write_evidence(tmp_path, evidence_kind="implementation")

    with pytest.raises(EvidenceError, match="external_dependency"):
        verify_eg_kafka_gate_evidence(pyproject_path=pyproject, evidence_path=evidence)


def test_missing_evidence_file_is_refused(tmp_path: Path) -> None:
    pyproject = _write_pyproject(tmp_path)
    missing = tmp_path / "does-not-exist.json"

    with pytest.raises(EvidenceError, match="cannot read"):
        verify_eg_kafka_gate_evidence(pyproject_path=pyproject, evidence_path=missing)
