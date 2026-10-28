"""Track the current state of the five EG-duplicate inventories."""

from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "agent_utilities"

INVENTORY: dict[str, tuple[str, ...]] = {
    "R018": (
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
        "knowledge_graph/orchestration/engine_*.py",
    ),
    "R019": (
        "knowledge_graph/core/graph_compute.py",
        "knowledge_graph/core/session.py",
        "knowledge_graph/core/epistemic_row.py",
        "knowledge_graph/core/ogm.py",
        "knowledge_graph/core/company_brain.py",
        "knowledge_graph/core/company_brain_runtime.py",
        "core/registry/kg_adapter.py",
    ),
    "R020": (
        "knowledge_graph/core/work_durability.py",
        "knowledge_graph/core/kafka_queue_backend.py",
        "knowledge_graph/core/postgres_queue_backend.py",
        "knowledge_graph/core/queue_backend.py",
        "knowledge_graph/core/worker_scheduler.py",
        "knowledge_graph/core/chunked_drain.py",
        "knowledge_graph/core/ingest_routing.py",
        "knowledge_graph/core/ingest_profile.py",
        "knowledge_graph/core/event_backend.py",
        "knowledge_graph/core/knowledge_stream.py",
        "knowledge_graph/core/bitemporal.py",
        "knowledge_graph/core/kg_versioning.py",
        "knowledge_graph/ingest_worker.py",
        "knowledge_graph/analytics_worker.py",
        "core/shared_resource_leases.py",
        "core/state_store.py",
        "core/chat_persistence.py",
    ),
    "R021": (
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
        "security/admission_authority.py",
        "security/system_admission_cli.py",
        "security/tenant_admission_cli.py",
        "security/permissions_kernel.py",
        "security/entitlements.py",
    ),
    "R022": (
        "knowledge_graph/core/formal_reasoning_core.py",
        "knowledge_graph/core/graph_primitives.py",
        "knowledge_graph/core/inference_engine.py",
        "knowledge_graph/core/reasoner.py",
        "knowledge_graph/core/semantic_subsumption.py",
        "knowledge_graph/core/topological_analysis_engine.py",
        "knowledge_graph/core/topological_partition.py",
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
}


def _matches(root: Path, patterns: tuple[str, ...]) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for pattern in patterns
        for path in root.glob(pattern)
    }


def _check(root: Path, patterns: tuple[str, ...], recorded: set[str]) -> None:
    assert _matches(root, patterns) == recorded


@pytest.mark.spec("AU-BOUNDARY-R018.1")
def test_r018_inventory_matches_tree() -> None:
    _check(PACKAGE, INVENTORY["R018"], set())


@pytest.mark.spec("AU-BOUNDARY-R019.1")
def test_r019_inventory_matches_tree() -> None:
    _check(PACKAGE, INVENTORY["R019"], set())


@pytest.mark.spec("AU-BOUNDARY-R020.1")
def test_r020_inventory_matches_tree() -> None:
    _check(PACKAGE, INVENTORY["R020"], set())


@pytest.mark.spec("AU-BOUNDARY-R021.1")
def test_r021_inventory_matches_tree() -> None:
    _check(PACKAGE, INVENTORY["R021"], set())


@pytest.mark.spec("AU-BOUNDARY-R022.1")
def test_r022_inventory_matches_tree() -> None:
    _check(PACKAGE, INVENTORY["R022"], set())


def test_inventory_refuses_unrecorded_addition_or_removal(tmp_path: Path) -> None:
    path = tmp_path / "knowledge_graph/core/engine.py"
    path.parent.mkdir(parents=True)
    path.write_text("")
    recorded = {"knowledge_graph/core/engine.py"}
    _check(tmp_path, INVENTORY["R018"], recorded)
    with pytest.raises(AssertionError):
        _check(tmp_path, INVENTORY["R018"], set())
    path.unlink()
    with pytest.raises(AssertionError):
        _check(tmp_path, INVENTORY["R018"], recorded)
