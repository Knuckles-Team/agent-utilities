#!/usr/bin/python
"""Documentation-facing workload contract loader.

The installed authority lives in :mod:`scripts.scale.workload_contract`.  This
path remains a small documentation convenience wrapper so the docs copy and
the packaged loader cannot diverge on validation, digest binding, or
mock/live-authority rules.
"""

from __future__ import annotations

from pathlib import Path

from scripts.scale.workload_contract import (
    EXPECTED_RESIDENT_POPULATION,
    ScaledWorkload,
    SloTarget,
    TenantSpec,
    WorkloadContract,
    WorkloadContractError,
    WorkloadEvidence,
    bind_evidence,
    bind_workload_evidence,
)
from scripts.scale.workload_contract import load_workload_contract as _load

_DEFAULT_CONTRACT_PATH = Path(__file__).with_name("workload_contract.yml")


def load_workload_contract(path: str | Path | None = None) -> WorkloadContract:
    """Load the documentation contract through the canonical validator."""

    return _load(path or _DEFAULT_CONTRACT_PATH)


def summarize(contract: WorkloadContract) -> str:
    """Render the canonical loader's bounded human-readable summary."""

    from scripts.scale.workload_contract import summarize as _summarize

    return _summarize(contract)


__all__ = [
    "EXPECTED_RESIDENT_POPULATION",
    "ScaledWorkload",
    "SloTarget",
    "TenantSpec",
    "WorkloadContract",
    "WorkloadContractError",
    "WorkloadEvidence",
    "bind_evidence",
    "bind_workload_evidence",
    "load_workload_contract",
    "summarize",
]


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    print(summarize(load_workload_contract()))
