"""Typed inventory: pending directory relocations for AU-BOUNDARY-R036,
AU-BOUNDARY-R037, AU-BOUNDARY-R039, AU-BOUNDARY-R040, AU-BOUNDARY-R044, AU-BOUNDARY-R045, and AU-BOUNDARY-R047.

Each of these rows moves a whole directory (or a named set of files) out of
Agent Utilities to a named destination -- the epistemic graph, graph-os, or
repository-manager -- once the destination proves equivalent, durable
coverage. This module is the ``.1`` slice for each row: a typed record, read
directly out of ``requirements.md``, of exactly which paths are pending
relocation and where, plus a refusal-style regression that fails loudly if a
listed path silently disappears (the row would be marked delivered without
anyone updating this inventory) so drift cannot hide behind the gate.

This does not perform the moves. The real cutover for each row still
requires the restart, tenant-scoped-read, duplicate-event, or census test
each row's acceptance criterion names, run against the receiving repository.

AU-BOUNDARY-R039 (the former ``agent_utilities/api/`` front-end surface) is
complete: that surface left Agent Utilities with the 3.0.0 cut.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "agent_utilities"

# requirement_id -> (destination repository, AU paths named verbatim in
# specs/au-boundary-deconstruction/requirements.md as pending relocation).
RELOCATION_INVENTORY: dict[str, tuple[str, tuple[str, ...]]] = {
    "AU-BOUNDARY-R036": (
        "epistemic-graph",
        (
            "usage/backend.py",
            "usage/authorization.py",
            "usage/backends",
        ),
    ),
    "AU-BOUNDARY-R037": (
        "repository-manager",
        (
            "governance/concept_allocator.py",
            "governance/concept_hierarchy.py",
            "governance/concept_lineage.py",
        ),
    ),
    "AU-BOUNDARY-R040": (
        "epistemic-graph",
        (
            "knowledge_graph/memory/agent_context.py",
            "knowledge_graph/memory/learning_engine.py",
            "knowledge_graph/memory/memory_engine.py",
            "knowledge_graph/memory/rlm_memory.py",
            "knowledge_graph/memory/media_store.py",
            "knowledge_graph/memory/distillation.py",
            "knowledge_graph/memory/lifecycle.py",
            "knowledge_graph/memory/observer.py",
        ),
    ),
    "AU-BOUNDARY-R044": (
        "graph-os",
        (
            "orchestration/agent_activation.py",
            "orchestration/agent_digital_twin.py",
            "orchestration/action_policy.py",
        ),
    ),
    "AU-BOUNDARY-R045": (
        "epistemic-graph",
        (
            "knowledge_graph/distillation",
            "knowledge_graph/adaptation",
            "knowledge_graph/pipeline",
            "knowledge_graph/search_synthesis",
            "knowledge_graph/shapes",
            "knowledge_graph/live_artifacts",
        ),
    ),
    "AU-BOUNDARY-R047": (
        "epistemic-graph",
        (
            "kvcache",
            "numeric",
            "caching",
            "media",
        ),
    ),
}


def _relocation_entries() -> list[tuple[str, str]]:
    return [
        (requirement_id, module_path)
        for requirement_id, (_, module_paths) in RELOCATION_INVENTORY.items()
        for module_path in module_paths
    ]


@pytest.mark.parametrize("requirement_id,module_path", _relocation_entries())
def test_inventoried_relocation_still_present(
    requirement_id: str, module_path: str
) -> None:
    """Every path this inventory names as pending relocation must still
    exist under agent_utilities/. If it has already been deleted, the row's
    delivery state and this inventory have drifted out of sync -- update
    both together, in the same change that removes the path.
    """
    target = PACKAGE_ROOT / module_path
    assert target.exists(), (
        f"{requirement_id} inventory names '{module_path}' as pending "
        f"relocation to {RELOCATION_INVENTORY[requirement_id][0]}, but it no "
        "longer exists under agent_utilities/. Update this inventory (and "
        "requirements.md / coverage.md) in the same change that deletes it."
    )


def test_relocation_inventory_covers_exactly_the_assigned_rows() -> None:
    """Pin the row set this slice covers so a future edit that adds or drops
    a row notices the change instead of silently expanding scope.
    """
    assert set(RELOCATION_INVENTORY) == {
        "AU-BOUNDARY-R036",
        "AU-BOUNDARY-R037",
        "AU-BOUNDARY-R040",
        "AU-BOUNDARY-R044",
        "AU-BOUNDARY-R045",
        "AU-BOUNDARY-R047",
    }
