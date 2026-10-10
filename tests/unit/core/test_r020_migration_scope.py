"""AU-SEMANTIC-R020.1: the typed migration-scope manifest stays accurate."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.core.r020_migration_scope import (
    R020_MODULE_PATHS,
    validate_migration_scope,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.spec("AU-SEMANTIC-R020.1")
def test_r020_migration_scope_matches_checkout() -> None:
    """Every listed migration-scope path currently exists in AU."""
    validate_migration_scope(REPO_ROOT)


@pytest.mark.spec("AU-SEMANTIC-R020.1")
def test_r020_migration_scope_is_nonempty() -> None:
    assert R020_MODULE_PATHS


@pytest.mark.spec("AU-SEMANTIC-R020.1")
def test_r020_migration_scope_refuses_missing_path() -> None:
    """A manifest naming an absent path fails loud instead of drifting silently."""
    with pytest.raises(ValueError, match="migration scope missing from checkout"):
        validate_migration_scope(
            REPO_ROOT, paths=("agent_utilities/does_not_exist_r020.py",)
        )
