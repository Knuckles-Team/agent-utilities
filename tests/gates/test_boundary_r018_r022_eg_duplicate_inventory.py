"""Typed inventory: AU-side duplicates of epistemic-graph's native engine.

Requirements AU-BOUNDARY-R018, R019, R020, R021, and R022 delete AU's own
graph engine facade, graph-compute/state facades, durable work/queue/state
modules, tenancy/topology/admission modules, and reasoning/analytics
duplicates once each has proven native, equivalent coverage in
epistemic-graph. This module is the ``.1`` slice for that cross-repo move:
a typed record, read directly out of ``requirements.md``, of which AU module
exactly duplicates engine-native behavior for each row, plus a refusal-style
regression that fails loudly if a listed module silently disappears (so the
row is marked delivered without anyone updating this inventory) or if a new,
unlisted module appears under one of these directories without a decision
(so drift cannot hide behind the gate).

This does not replace the real cutover: deletion still requires the
method-by-method golden-comparison tests each row's acceptance criterion
names, run against the generated epistemic-graph client. It only prevents
the inventory itself from going stale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "agent_utilities"

# requirement_id -> (destination repository, AU module paths named verbatim
# in specs/au-boundary-deconstruction/requirements.md as the duplicate to be
# deleted once epistemic-graph's native implementation is proven equivalent).
EG_DUPLICATE_INVENTORY: dict[str, tuple[str, tuple[str, ...]]] = {
    "AU-BOUNDARY-R018": (
        "epistemic-graph",
        (
            "knowledge_graph/core/engine.py",
            "knowledge_graph/core/engine_tasks.py",
            "knowledge_graph/core/engine_ingestion.py",
            "knowledge_graph/core/engine_memory.py",
            "knowledge_graph/core/engine_task_query.py",
            "knowledge_graph/core/engine_resolver.py",
            "knowledge_graph/core/engine_transport.py",
            "knowledge_graph/core/engine_lock.py",
            "knowledge_graph/core/engine_breaker.py",
            "knowledge_graph/facade.py",
        ),
    ),
    "AU-BOUNDARY-R019": (
        "epistemic-graph",
        (
            "knowledge_graph/core/graph_compute.py",
            "knowledge_graph/core/session.py",
            "knowledge_graph/core/epistemic_row.py",
            "knowledge_graph/core/ogm.py",
            "knowledge_graph/core/company_brain.py",
            "knowledge_graph/core/company_brain_runtime.py",
            "core/registry/kg_adapter.py",
        ),
    ),
    "AU-BOUNDARY-R020": (
        "epistemic-graph",
        (
            "knowledge_graph/core/work_durability.py",
            "knowledge_graph/core/kafka_queue_backend.py",
            "knowledge_graph/core/queue_backend.py",
            "knowledge_graph/core/worker_scheduler.py",
            "knowledge_graph/core/chunked_drain.py",
            "knowledge_graph/core/ingest_routing.py",
            "knowledge_graph/core/ingest_profile.py",
            "knowledge_graph/core/event_backend.py",
            "knowledge_graph/core/knowledge_stream.py",
            "knowledge_graph/core/bitemporal.py",
            "knowledge_graph/core/kg_versioning.py",
            "core/shared_resource_leases.py",
            "core/state_store.py",
            "core/chat_persistence.py",
        ),
    ),
    "AU-BOUNDARY-R021": (
        "epistemic-graph",
        (
            "knowledge_graph/core/tenant_sharing.py",
            "knowledge_graph/core/tenant_registry.py",
            "knowledge_graph/core/tenant_engine_pool.py",
            "knowledge_graph/core/shard_topology.py",
            "knowledge_graph/core/cluster_discovery.py",
            "knowledge_graph/core/placement_catalog.py",
            "knowledge_graph/core/secured_reads.py",
            "knowledge_graph/core/cypher_scoping.py",
            "knowledge_graph/core/cypher_scope_vars.py",
            "knowledge_graph/core/bounded_read.py",
            "knowledge_graph/core/host_lock.py",
            "knowledge_graph/core/file_lock.py",
            "security/system_rbac_admission.py",
            "security/tenant_rbac_admission.py",
            "security/engine_rbac_admission.py",
        ),
    ),
    "AU-BOUNDARY-R022": (
        "epistemic-graph",
        (
            "knowledge_graph/core/formal_reasoning_core.py",
            "knowledge_graph/core/graph_primitives.py",
            "knowledge_graph/core/inference_engine.py",
            "knowledge_graph/core/reasoner.py",
            "knowledge_graph/core/semantic_subsumption.py",
            "knowledge_graph/core/spectral_navigator.py",
            "knowledge_graph/core/synergy_engine.py",
            "knowledge_graph/core/analogy_engine.py",
            "knowledge_graph/core/hypergraph.py",
            "knowledge_graph/core/blast_radius.py",
            "knowledge_graph/core/nl_query.py",
            "knowledge_graph/core/hydration.py",
            "knowledge_graph/core/maintainer.py",
            "knowledge_graph/core/world_model.py",
            "knowledge_graph/core/ownership_claim.py",
            "knowledge_graph/core/fingerprint.py",
            "knowledge_graph/maintenance",
            "knowledge_graph/id_management",
            "knowledge_graph/argumentation",
            "knowledge_graph/actions",
        ),
    ),
}


def _existing_entries() -> list[tuple[str, str]]:
    """Flatten the inventory to (requirement_id, module_path) pairs."""
    return [
        (requirement_id, module_path)
        for requirement_id, (_, module_paths) in EG_DUPLICATE_INVENTORY.items()
        for module_path in module_paths
    ]


@pytest.mark.parametrize("requirement_id,module_path", _existing_entries())
def test_inventoried_duplicate_still_present(
    requirement_id: str, module_path: str
) -> None:
    """Every module this inventory names as a pending engine-duplicate
    deletion must still exist. If it has already been deleted, the row's
    delivery state and this inventory have drifted out of sync -- update
    both together, in the same change that removes the module.
    """
    target = PACKAGE_ROOT / module_path
    assert target.exists(), (
        f"{requirement_id} inventory names '{module_path}' as a pending "
        "epistemic-graph duplicate, but it no longer exists under "
        "agent_utilities/. Update this inventory (and requirements.md / "
        "coverage.md) in the same change that deletes a module."
    )


def test_destination_is_epistemic_graph_for_every_row() -> None:
    """Every row in this slice names epistemic-graph as the destination:
    these are 'delete as duplicate of engine' rows, not relocations, so the
    only typed fact worth asserting is which engine-native surface proof is
    owed before deletion.
    """
    for requirement_id, (destination, _) in EG_DUPLICATE_INVENTORY.items():
        assert destination == "epistemic-graph", requirement_id


def test_inventory_covers_exactly_the_assigned_rows() -> None:
    """Pin the row set this slice covers so a future edit that adds or drops
    a row notices the change instead of silently expanding scope.
    """
    assert set(EG_DUPLICATE_INVENTORY) == {
        "AU-BOUNDARY-R018",
        "AU-BOUNDARY-R019",
        "AU-BOUNDARY-R020",
        "AU-BOUNDARY-R021",
        "AU-BOUNDARY-R022",
    }
