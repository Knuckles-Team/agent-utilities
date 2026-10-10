"""Typed inventory: AU-side modules named for relocation by AU-BOUNDARY-R023,
R024, R025, R027, R029, R031, R033, and R034.

Each of these requirements moves a tree of ``agent_utilities`` modules to
either the epistemic graph or the agent connector SDK (see
``specs/au-boundary-deconstruction/requirements.md``). This module is the
``.1`` slice for that cross-repo move: a typed record of exactly which
module path each row names, read directly out of ``requirements.md``, plus
a refusal-style regression that fails loudly if a listed module silently
disappears without this inventory (and the row's delivery state) being
updated in the same change.

This does not replace the real cutover: deletion still requires the parity
tests each row's acceptance criterion names, run against the destination
(epistemic graph or the SDK). It only prevents the inventory itself, and the
requirement text it mirrors, from going stale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "agent_utilities"

EPISTEMIC_GRAPH = "epistemic-graph"
AGENT_CONNECTOR_SDK = "agent-connector-sdk"

# requirement_id -> destination -> module paths named verbatim (modulo the
# "kg/" shorthand, which is agent_utilities/knowledge_graph/) in
# specs/au-boundary-deconstruction/requirements.md.
RELOCATION_INVENTORY: dict[str, dict[str, tuple[str, ...]]] = {
    "AU-BOUNDARY-R023": {
        EPISTEMIC_GRAPH: (
            "knowledge_graph/ingestion/envelope_ingest.py",
            "knowledge_graph/ingestion/change_envelope.py",
            "knowledge_graph/ingestion/evidence_spine.py",
            "knowledge_graph/ingestion/semantic_event_model.py",
            "knowledge_graph/ingestion/object_centric_derivation.py",
            "knowledge_graph/ingestion/process_conformance.py",
            "knowledge_graph/ingestion/embedding_admission.py",
            "knowledge_graph/ingestion/gpu_slot_scheduler.py",
            "knowledge_graph/ingestion/promotion.py",
            "knowledge_graph/ingestion/supersession.py",
            "knowledge_graph/ingestion/dead_letter.py",
            "knowledge_graph/ingestion/hydration_manifest.py",
            "knowledge_graph/ingestion/skill_workflow_ingest.py",
            "knowledge_graph/ingestion/governed_documentation.py",
            "knowledge_graph/ingestion/engine.py",
        ),
    },
    "AU-BOUNDARY-R024": {
        AGENT_CONNECTOR_SDK: (
            "knowledge_graph/core/source_sync.py",
            "knowledge_graph/ingestion/external_graph.py",
            "knowledge_graph/ingestion/external_graph_schema.py",
            "knowledge_graph/ingestion/graphql_connection.py",
            "knowledge_graph/ingestion/debezium_envelope.py",
            "knowledge_graph/ingestion/event_log_adapter.py",
            "knowledge_graph/ingestion/ocel_adapter.py",
            "knowledge_graph/governance_import.py",
            "ecosystem/ea_clients.py",
            "data_prep",
            "knowledge_graph/core/connection_profiler.py",
            "knowledge_graph/core/data_prep_runtime.py",
            "knowledge_graph/core/source_resolver.py",
        ),
    },
    "AU-BOUNDARY-R025": {
        AGENT_CONNECTOR_SDK: (
            "knowledge_graph/core/gitlab_indexer.py",
            "knowledge_graph/core/conversation_ingestion.py",
            "knowledge_graph/ingestion/web_fetch.py",
            "knowledge_graph/ingestion/package_install_ingest.py",
            "knowledge_graph/ingestion/repo_classifier.py",
            "knowledge_graph/ingestion/repo_split.py",
            "ingestion",
            "automation/worldmodel_pipeline.py",
            "automation/feed_sources.py",
            "automation/file_watcher.py",
            "sdd/watcher.py",
            "knowledge_graph/kb",
            "knowledge_graph/extraction/readers.py",
            "knowledge_graph/extraction/readers_media.py",
            "knowledge_graph/extraction/readers_office.py",
            "knowledge_graph/extraction/pdf.py",
            "ecosystem/media",
        ),
    },
    "AU-BOUNDARY-R027": {
        EPISTEMIC_GRAPH: (
            "knowledge_graph/enrichment/semantic.py",
            "knowledge_graph/enrichment/relation_projection.py",
            "knowledge_graph/enrichment/materialize.py",
            "knowledge_graph/enrichment/git_coupling.py",
            "knowledge_graph/enrichment/graph_collapse.py",
            "knowledge_graph/enrichment/provenance.py",
            "knowledge_graph/assimilation",
            "knowledge_graph/etl",
            "knowledge_graph/streams",
            "knowledge_graph/memory/timeseries",
        ),
    },
    "AU-BOUNDARY-R029": {
        EPISTEMIC_GRAPH: (
            "knowledge_graph/ontology/object_set.py",
            "knowledge_graph/ontology/property_types.py",
            "knowledge_graph/ontology/derived_properties.py",
            "knowledge_graph/ontology/links.py",
            "knowledge_graph/ontology/functions",
            "knowledge_graph/ontology/edits",
            "knowledge_graph/ontology/indexing",
            "knowledge_graph/ontology/classification_claims.py",
            "knowledge_graph/ontology/permissioning.py",
            "knowledge_graph/ontology/permissioning_external_sync.py",
            "knowledge_graph/ontology/document_processing.py",
            "knowledge_graph/ontology/schema_graph.py",
            "knowledge_graph/ontology/ops_causal_crosswalk.py",
            "knowledge_graph/ontology/repository_provenance.py",
            "knowledge_graph/ontology/style_lint.py",
            "knowledge_graph/ontology/finance_objects.py",
            "knowledge_graph/ontology/research_objects.py",
            "knowledge_graph/ontology/object_path.py",
        ),
        # sync_conflict and leanix_metamodel move to the SDK instead of EG.
        AGENT_CONNECTOR_SDK: (
            "knowledge_graph/ontology/sync_conflict.py",
            "knowledge_graph/ontology/leanix_metamodel.py",
        ),
    },
    "AU-BOUNDARY-R031": {
        EPISTEMIC_GRAPH: (
            "models/knowledge_graph.py",
            "models/schema_definition.py",
            "models/evidence_bundle.py",
            "models/knowledge_pack.py",
            "models/codemap.py",
            "models/graph.py",
            "models/knowledge_base.py",
            "models/schema_pack.py",
            "models/schema_pack_audit.py",
            "models/schema_pack_loader.py",
            "models/schema_packs",
            "knowledge_graph/domain_packs",
        ),
    },
    "AU-BOUNDARY-R033": {
        EPISTEMIC_GRAPH: (
            "knowledge_graph/backends/fanout_backend.py",
            "knowledge_graph/backends/postgresql_backend.py",
            "knowledge_graph/backends/cypher_transpiler.py",
            "knowledge_graph/backends/age_backend.py",
            "knowledge_graph/backends/contrib",
            "knowledge_graph/backends/mirror_target.py",
            "knowledge_graph/backends/outbox.py",
            "knowledge_graph/backends/brain_guarded_backend.py",
            "knowledge_graph/backends/trino_backend.py",
            "knowledge_graph/backends/spark_jobs.py",
        ),
    },
    "AU-BOUNDARY-R034": {
        EPISTEMIC_GRAPH: (
            "knowledge_graph/retrieval/hybrid_retriever.py",
            "knowledge_graph/retrieval/semantic_retrieval_engine.py",
            "knowledge_graph/retrieval/neural_reranker.py",
            "knowledge_graph/retrieval/hierarchical_document_retriever.py",
            "knowledge_graph/retrieval/memory_retriever.py",
            "knowledge_graph/retrieval/code_context.py",
            "knowledge_graph/retrieval/code_metrics.py",
            "knowledge_graph/retrieval/graph_engineering.py",
            "knowledge_graph/retrieval/analytics_job_registry.py",
            "knowledge_graph/retrieval/temporal_semantic_id.py",
            "knowledge_graph/retrieval/embedding_versioning.py",
            "knowledge_graph/retrieval/generative_recommender.py",
            "knowledge_graph/retrieval/iterative_expansion.py",
            "knowledge_graph/retrieval/lineage.py",
            "knowledge_graph/retrieval/direct_corpus.py",
            "knowledge_graph/retrieval/latent_topology_rag.py",
            "knowledge_graph/retrieval/reasoning_reranker.py",
            "knowledge_graph/retrieval/autocut.py",
            "knowledge_graph/neural",
            "knowledge_graph/search",
        ),
    },
}


def _entries() -> list[tuple[str, str, str]]:
    """Flatten the inventory to (requirement_id, destination, module_path)."""
    return [
        (requirement_id, destination, module_path)
        for requirement_id, by_destination in RELOCATION_INVENTORY.items()
        for destination, module_paths in by_destination.items()
        for module_path in module_paths
    ]


@pytest.mark.spec("AU-BOUNDARY-R023")
@pytest.mark.spec("AU-BOUNDARY-R024")
@pytest.mark.spec("AU-BOUNDARY-R025")
@pytest.mark.spec("AU-BOUNDARY-R027")
@pytest.mark.spec("AU-BOUNDARY-R029")
@pytest.mark.spec("AU-BOUNDARY-R031")
@pytest.mark.spec("AU-BOUNDARY-R033")
@pytest.mark.spec("AU-BOUNDARY-R034")
@pytest.mark.parametrize("requirement_id,destination,module_path", _entries())
def test_inventoried_relocation_target_still_present(
    requirement_id: str, destination: str, module_path: str
) -> None:
    """Every module this inventory names as pending relocation must still
    exist under agent_utilities/. If it has already been deleted, the row's
    delivery state and this inventory have drifted out of sync -- update
    both (and requirements.md) in the same change that removes the module.
    """
    target = PACKAGE_ROOT / module_path
    assert target.exists(), (
        f"{requirement_id} inventory names '{module_path}' as pending "
        f"relocation to {destination}, but it no longer exists under "
        "agent_utilities/. Update this inventory (and requirements.md / "
        "status.json) in the same change that deletes a module."
    )


def test_every_destination_is_a_known_target() -> None:
    """Pin the only two valid relocation targets for this slice's rows."""
    known = {EPISTEMIC_GRAPH, AGENT_CONNECTOR_SDK}
    for requirement_id, by_destination in RELOCATION_INVENTORY.items():
        for destination in by_destination:
            assert destination in known, (requirement_id, destination)


def test_inventory_covers_exactly_the_assigned_rows() -> None:
    """Pin the row set this slice covers so a future edit that adds or drops
    a row notices the change instead of silently expanding scope.
    """
    assert set(RELOCATION_INVENTORY) == {
        "AU-BOUNDARY-R023",
        "AU-BOUNDARY-R024",
        "AU-BOUNDARY-R025",
        "AU-BOUNDARY-R027",
        "AU-BOUNDARY-R029",
        "AU-BOUNDARY-R031",
        "AU-BOUNDARY-R033",
        "AU-BOUNDARY-R034",
    }
