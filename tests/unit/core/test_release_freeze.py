"""Refusal tests for the AU release-evidence freeze manifest model.

Covers AU-FREEZE-R001 (typed manifest; refuse on a dirty tree) and
AU-FREEZE-R002 (refuse while a scanner finding stays quarantined/unresolved).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_utilities.core.release_freeze import (
    FreezeManifest,
    FreezeRefusedError,
    MandatoryTestResult,
    QuarantinedFinding,
    ScannerResult,
    refuse_if_not_generatable,
)


def _manifest(**overrides: object) -> FreezeManifest:
    fields: dict[str, object] = {
        "repository_url": "https://github.com/Knuckles-Team/agent-utilities",
        "commit_sha": "a" * 40,
        "tree_hash": "b" * 40,
        "version": "0.0.0",
        "tracked_source_digest": "c" * 64,
        "lock_hash": "d" * 64,
        "manifest_hash": "e" * 64,
        "eg_client_digest": "f" * 64,
        "build_toolchain": "uv+python3.12",
        "generated_at": "2026-10-09T00:00:00Z",
        "artifact_digest": "g" * 64,
        "scanner_results": [
            ScannerResult(name="ruff", passed=True, ci_run_url="https://ci/1")
        ],
        "test_results": [
            MandatoryTestResult(name="pytest", passed=True, ci_run_url="https://ci/2")
        ],
    }
    fields.update(overrides)
    return FreezeManifest(**fields)


@pytest.mark.spec("AU-FREEZE-R001", "AU-FREEZE-R002")
def test_manifest_round_trips_as_frozen_typed_model() -> None:
    manifest = _manifest()
    assert manifest.commit_sha == "a" * 40
    with pytest.raises(ValidationError):
        manifest.commit_sha = "changed"  # type: ignore[misc]


@pytest.mark.spec("AU-FREEZE-R001", "AU-FREEZE-R002")
def test_refuses_generation_on_dirty_tree() -> None:
    with pytest.raises(FreezeRefusedError, match="dirty"):
        refuse_if_not_generatable(dirty_tree=True, unresolved_quarantine=[])


@pytest.mark.spec("AU-FREEZE-R001", "AU-FREEZE-R002")
def test_refuses_generation_while_finding_is_quarantined_unresolved() -> None:
    finding = QuarantinedFinding(
        scanner="jscpd", finding_id="JSCPD-1", detail="duplicate block"
    )
    with pytest.raises(FreezeRefusedError, match="JSCPD-1"):
        refuse_if_not_generatable(dirty_tree=False, unresolved_quarantine=[finding])


def test_allows_generation_once_finding_is_remediated() -> None:
    finding = QuarantinedFinding(
        scanner="jscpd",
        finding_id="JSCPD-1",
        detail="duplicate block",
        remediated=True,
    )
    refuse_if_not_generatable(dirty_tree=False, unresolved_quarantine=[finding])
