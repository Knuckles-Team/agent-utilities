"""Regression test for the AU-RETIRE-R001 duplicate-module census.

Pins the reconciled disposition of every known knowledge_graph module that
duplicates the epistemic-graph engine's native OWL/SPARQL/compute authority,
so the remaining-duplicate set can only shrink: a ``RETAINED`` entry whose
file silently disappears, or a ``DELETED`` entry whose file silently
reappears, both fail loudly here instead of drifting unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.retirement_inventory import (
    RETIREMENT_INVENTORY,
    Disposition,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.spec("AU-RETIRE-R001")
def test_every_entry_has_exactly_one_disposition():
    for entry in RETIREMENT_INVENTORY:
        assert isinstance(entry.disposition, Disposition)
        assert entry.reason, f"{entry.path} is missing a disposition reason"


def test_retained_modules_still_exist():
    """A RETAINED module must not vanish without the inventory being updated."""
    for entry in RETIREMENT_INVENTORY:
        if entry.disposition is Disposition.RETAINED:
            assert (_REPO_ROOT / entry.path).exists(), (
                f"{entry.path} is marked RETAINED but is missing from the tree; "
                "update retirement_inventory.py to DELETED if it was removed."
            )


def test_deleted_modules_stay_deleted():
    """A DELETED module must never reappear without a fresh disposition decision."""
    for entry in RETIREMENT_INVENTORY:
        if entry.disposition is Disposition.DELETED:
            assert not (_REPO_ROOT / entry.path).exists(), (
                f"{entry.path} is marked DELETED but exists again in the tree; "
                "a duplicate must not be reintroduced without updating the census."
            )


def test_no_duplicate_paths_in_inventory():
    paths = [entry.path for entry in RETIREMENT_INVENTORY]
    assert len(paths) == len(set(paths))
