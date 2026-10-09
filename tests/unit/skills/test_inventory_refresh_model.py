"""Tests for the AU-DEV-R003.1 skill inventory refresh entry model."""

import pytest
from pydantic import ValidationError

from agent_utilities.skills.inventory_refresh_model import SkillInventoryRefreshEntry


def test_accepts_planned_and_verified_refresh() -> None:
    entry = SkillInventoryRefreshEntry(
        skill_id="graph-os-core",
        source_hash="abc123",
        installed_hash="abc123",
        plan_printed=True,
        mutated=True,
    )
    assert entry.installed_hash == entry.source_hash


def test_refuses_mutation_without_printed_plan() -> None:
    """AU-DEV-R003: refresh refuses to mutate before printing its plan."""
    with pytest.raises(ValidationError, match="AU-DEV-R003"):
        SkillInventoryRefreshEntry(
            skill_id="graph-os-core",
            source_hash="abc123",
            installed_hash="abc123",
            plan_printed=False,
            mutated=True,
        )


def test_refuses_completion_with_mismatched_hash() -> None:
    """AU-DEV-R003: refresh refuses to claim success with stale hashes."""
    with pytest.raises(ValidationError, match="AU-DEV-R003"):
        SkillInventoryRefreshEntry(
            skill_id="graph-os-core",
            source_hash="abc123",
            installed_hash="stale999",
            plan_printed=True,
            mutated=True,
        )
