#!/usr/bin/env python3
"""Fail closed unless AU's own evidence trail records the pinned
epistemic-graph version's Kafka enqueue-only-proof gate as passed.

AU-INTEGRATION-R010: the Kafka enqueue-only-proof contract is owned and run by
epistemic-graph (``scripts/constrained_parallelism_gate.sh``), not by AU. AU
may not claim that gate's result as AU-authored proof; it must instead record,
in AU's own release/dependency-evidence trail, that the *exact* epistemic-graph
version AU pins passed that gate. This check reads the declared epistemic-graph
version floor out of ``pyproject.toml`` and the recorded evidence out of
``deploy/release/eg-kafka-gate-evidence.json``, and fails closed when the
recorded version does not match the pin, the result is not ``"passed"``, or no
step evidence is recorded — so a pin bump without a refreshed evidence record is
caught rather than silently treated as still-proven.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Final

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_PYPROJECT: Final = _REPO_ROOT / "pyproject.toml"
DEFAULT_EVIDENCE: Final = _REPO_ROOT / "deploy" / "release" / "eg-kafka-gate-evidence.json"

_PIN_RE: Final = re.compile(
    r'"epistemic-graph(?:\[[^\]]*\])?>=(?P<version>\d+\.\d+\.\d+)'
)


class EvidenceError(RuntimeError):
    """A deterministic, privacy-safe evidence-check rejection."""


def _declared_pin_floor(pyproject_path: Path) -> str:
    try:
        text = pyproject_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvidenceError(f"cannot read {pyproject_path}: {exc}") from exc
    match = _PIN_RE.search(text)
    if match is None:
        raise EvidenceError(
            f"no 'epistemic-graph>=X.Y.Z' dependency floor found in {pyproject_path}"
        )
    return match.group("version")


def _load_evidence(evidence_path: Path) -> dict[str, Any]:
    try:
        raw = evidence_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvidenceError(f"cannot read {evidence_path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"{evidence_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise EvidenceError(f"{evidence_path} must contain a JSON object")
    return data


def verify_eg_kafka_gate_evidence(
    *,
    pyproject_path: Path = DEFAULT_PYPROJECT,
    evidence_path: Path = DEFAULT_EVIDENCE,
) -> dict[str, Any]:
    """Return the verified evidence record, or raise ``EvidenceError``."""

    pin_floor = _declared_pin_floor(pyproject_path)
    evidence = _load_evidence(evidence_path)

    recorded_version = evidence.get("epistemic_graph_version")
    if recorded_version != pin_floor:
        raise EvidenceError(
            "recorded epistemic-graph version "
            f"{recorded_version!r} does not match the declared pin floor "
            f"{pin_floor!r} in {pyproject_path}; refresh the Kafka gate "
            "evidence record for the new pin before relying on it"
        )

    result = evidence.get("result")
    if result != "passed":
        raise EvidenceError(
            f"recorded Kafka enqueue-only-proof gate result is {result!r}, not 'passed'"
        )

    steps = evidence.get("steps")
    if not isinstance(steps, list) or not steps:
        raise EvidenceError(
            "recorded evidence names no Kafka enqueue-only-proof gate steps"
        )

    if evidence.get("evidence_kind") != "external_dependency":
        raise EvidenceError(
            "evidence_kind must be 'external_dependency': this gate is "
            "epistemic-graph-owned, not AU-authored proof"
        )

    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        evidence = verify_eg_kafka_gate_evidence(
            pyproject_path=args.pyproject, evidence_path=args.evidence
        )
    except EvidenceError as exc:
        print(f"eg kafka gate evidence check failed: {exc}", file=sys.stderr)
        return 1
    print(
        "eg kafka gate evidence OK: epistemic-graph="
        f"{evidence['epistemic_graph_version']} result={evidence['result']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
