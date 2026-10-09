"""Canonical AU release-evidence freeze manifest (AU-FREEZE-R001, AU-FREEZE-R002).

Typed model plus refusal behavior only. Generation and quarantine enforcement
land in later slices (AU-FREEZE-R001.2+).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ScannerResult(BaseModel):
    """A mandatory scanner's result bound to its immutable CI run."""

    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    ci_run_url: str


class MandatoryTestResult(BaseModel):
    """A mandatory test result bound to its immutable CI run."""

    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    ci_run_url: str


class QuarantinedFinding(BaseModel):
    """An unresolved, unmerged scanner finding kept out of the frozen set.

    AU-FREEZE-R002: a quarantined finding is never counted as a passing
    mandatory scanner result until it is remediated and merged.
    """

    model_config = ConfigDict(frozen=True)

    scanner: str
    finding_id: str
    detail: str
    remediated: bool = False


class FreezeManifest(BaseModel):
    """Canonical, machine-readable AU release-evidence freeze manifest.

    Binds one exact Git commit to its build artifact digest and every
    mandatory scanner/test result together with its immutable CI run URL.
    """

    model_config = ConfigDict(frozen=True)

    repository_url: str
    commit_sha: str
    tree_hash: str
    version: str
    tracked_source_digest: str
    lock_hash: str
    manifest_hash: str
    eg_client_digest: str
    build_toolchain: str
    generated_at: str
    artifact_digest: str
    scanner_results: list[ScannerResult] = Field(default_factory=list)
    test_results: list[MandatoryTestResult] = Field(default_factory=list)
    quarantined_findings: list[QuarantinedFinding] = Field(default_factory=list)


class FreezeRefusedError(RuntimeError):
    """Raised when freeze manifest generation must refuse to proceed."""


def refuse_if_not_generatable(
    *, dirty_tree: bool, unresolved_quarantine: list[QuarantinedFinding]
) -> None:
    """Refuse freeze generation on a dirty tree or an open quarantined finding.

    AU-FREEZE-R001: a dirty working tree or an unknown input fails generation.
    AU-FREEZE-R002: an in-scope candidate with an unresolved quarantined
    finding blocks the freeze.
    """
    if dirty_tree:
        raise FreezeRefusedError(
            "refusing to generate a freeze manifest from a dirty working tree"
        )
    open_findings = [f for f in unresolved_quarantine if not f.remediated]
    if open_findings:
        names = ", ".join(f.finding_id for f in open_findings)
        raise FreezeRefusedError(
            f"refusing to freeze: unresolved quarantined finding(s): {names}"
        )
