"""Retirement tripwire reconciliation for the ontology-leg scaffold script."""

from __future__ import annotations

import pytest

from scripts import check_current_only_contract as contract


@pytest.mark.spec("AU-BOUNDARY-R030.9.1")
def test_scaffold_ontology_leg_is_no_longer_retired() -> None:
    assert "scripts/scaffold_ontology_leg.py" not in contract.RETIRED_PATHS


@pytest.mark.spec("AU-BOUNDARY-R030.9.1")
def test_other_retired_paths_still_trip() -> None:
    assert "scripts/autocurate_repo.py" in contract.RETIRED_PATHS
    assert "agent_utilities/exceptions.py" in contract.RETIRED_PATHS
