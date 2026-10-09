"""AU-SEMANTIC-R016.1: the typed migration-scope manifest stays accurate."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.core.r016_migration_scope import (
    R016_MODULE_PATHS,
    validate_migration_scope,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_r016_migration_scope_matches_checkout() -> None:
    """Every listed migration-scope path currently exists in AU."""
    validate_migration_scope(REPO_ROOT)


def test_r016_migration_scope_is_nonempty() -> None:
    assert R016_MODULE_PATHS


def test_r016_migration_scope_refuses_missing_path() -> None:
    """A manifest naming an absent path fails loud instead of drifting silently."""
    with pytest.raises(ValueError, match="migration scope missing from checkout"):
        validate_migration_scope(
            REPO_ROOT, paths=("agent_utilities/does_not_exist_r016.py",)
        )
