"""CONCEPT:AU-KG.query.object-graph-mapper

Integration tests for KG lifecycle management:
- Soft-delete (ARCHIVED status) filtering convergence
- DiffEntry schema validation
- ArchiMate class schema validation
- Native WorkItem queue submit/list/immutable audit
- DocumentDeletionPipeline + QueryMixin parity
"""

import pytest

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.models.schema_definition import SCHEMA


def _isolated_engine(test_graph_name: str) -> IntelligenceGraphEngine:
    """Build an ``IntelligenceGraphEngine`` bound to the fixture's own test graph.

    ``IntelligenceGraphEngine(db_path=":memory:")`` routes through
    ``EpistemicGraphBackend(graph_name=None)``, which resolves the *ambient
    actor's tenant-default* graph (``resolve_routing_graph``) — a name that
    (a) was never registered as a tenant (only the autouse
    ``isolate_graph_compute_engine`` fixture's own redirected construction
    registers one, for ITS ``_test_graph_name``, not this derived one) and
    (b) does not match the ambient ``GraphSession.graph`` the same fixture
    sets (also ``_test_graph_name``). Depending on what else in the test had
    already constructed a ``GraphComputeEngine``, that mismatch surfaced as
    either "Graph ... not found" or the engine's own fail-closed
    "a graph-scoped view cannot retarget the verified GraphSession" guard —
    correct behavior; the bug was the fixture handing the engine the wrong
    identity, not the guard.

    Passing the exact ``test_graph_name`` the ``isolate_graph_compute_engine``
    fixture yields makes ``resolve_routing_graph`` return it verbatim, so this
    engine's graph identity matches the ambient ``GraphSession`` the rest of
    the test executes under, and the first construction (mirroring the
    fixture's own redirect logic) registers it as a real tenant.
    """
    backend = EpistemicGraphBackend(graph_name=test_graph_name)
    return IntelligenceGraphEngine(backend=backend)


# ── Fixtures ──


@pytest.fixture
def engine(isolate_graph_compute_engine):
    """Create a lightweight IntelligenceGraphEngine for testing."""
    return _isolated_engine(isolate_graph_compute_engine)


# ── Gap 1: DiffEntry Schema ──


def test_diff_entry_in_schema():
    """DiffEntry must be a registered node type in the graph schema."""
    node_names = [n.name for n in SCHEMA.nodes]
    assert "DiffEntry" in node_names, (
        "DiffEntry must be in SCHEMA.nodes for engine_tasks.py diff ingestion"
    )


def test_diff_entry_has_required_columns():
    """DiffEntry schema must include content, embedding, target_path, status."""
    diff_def = next(n for n in SCHEMA.nodes if n.name == "DiffEntry")
    required = {
        "id",
        "content",
        "embedding",
        "target_path",
        "status",
        "last_seen_timestamp",
    }
    actual = set(diff_def.columns.keys())
    missing = required - actual
    assert not missing, f"DiffEntry schema missing columns: {missing}"


# ── Gap 2: ArchiMate Classes in Schema ──


def test_archimate_business_role_in_schema():
    """BusinessRole (ArchiMate) must be registered in the graph schema."""
    node_names = [n.name for n in SCHEMA.nodes]
    assert "BusinessRole" in node_names


def test_archimate_application_component_in_schema():
    """ApplicationComponent (ArchiMate) must be registered in the graph schema."""
    node_names = [n.name for n in SCHEMA.nodes]
    assert "ApplicationComponent" in node_names


def test_archimate_business_process_in_schema():
    """BusinessProcess (ArchiMate) must be registered in the graph schema."""
    node_names = [n.name for n in SCHEMA.nodes]
    assert "BusinessProcess" in node_names


# ── Gap 3: Soft-Delete Convergence ──


def test_archived_node_excluded_from_graph_search(engine):
    """Nodes with status=ARCHIVED must be excluded from keyword search."""
    # Add an active node
    engine.graph.add_node(
        "active-node", name="TaxService", description="Handles tax", status="ACTIVE"
    )
    # Add an archived node
    engine.graph.add_node(
        "archived-node",
        name="TaxService-OLD",
        description="Handles tax",
        status="ARCHIVED",
    )

    # This test's own concern is the ARCHIVED-status filter, not ranking
    # quality; the retrieval relevance quality gate (LOW_RELEVANCE_TOPK) needs
    # a real embedding model, which the test environment doesn't configure —
    # the established convention elsewhere in the suite (e.g.
    # tests/test_backlink_boost.py) is ``skip_quality_gate=True`` for exactly
    # this case.
    results = engine.search_hybrid("TaxService", skip_quality_gate=True)
    result_ids = [r.get("id") for r in results]

    assert "active-node" in result_ids, "Active node should appear in search"
    assert "archived-node" not in result_ids, (
        "Archived node must be excluded from search"
    )


def test_archived_node_visible_via_direct_graph_query(engine):
    """ARCHIVED nodes must be accessible via direct graph query for restore/audit operations."""
    engine.graph.add_node(
        "archived-node", name="Legacy", description="Old system", status="ARCHIVED"
    )

    # Direct graph access should still see the node (bypasses search filtering)
    node_data = engine.graph.nodes.get("archived-node")
    assert node_data is not None, (
        "ARCHIVED nodes must be accessible via direct graph query"
    )
    assert node_data["status"] == "ARCHIVED"

    # But keyword search should exclude it
    results = engine._search_keyword("Legacy")
    result_ids = [r.get("id") for r in results]
    assert "archived-node" not in result_ids, (
        "ARCHIVED nodes must be excluded from keyword search"
    )


@pytest.mark.asyncio
async def test_soft_delete_pipeline_uses_archived_status(isolate_graph_compute_engine):
    """DocumentDeletionPipeline._soft_delete must set status=ARCHIVED, not is_deleted."""
    engine = _isolated_engine(isolate_graph_compute_engine)
    engine.graph.add_node("doc-001", name="TestDoc", content="test", status="ACTIVE")

    from agent_utilities.knowledge_graph.pipeline.document_deletion import (
        DocumentDeletionPipeline,
    )

    pipeline = DocumentDeletionPipeline(knowledge_graph=engine)

    await pipeline._soft_delete_from_knowledge_graph("doc-001")

    node_data = engine.graph.nodes["doc-001"]
    assert node_data.get("status") == "ARCHIVED", "Soft-delete must set status=ARCHIVED"
    assert "is_deleted" not in node_data or node_data.get("is_deleted") is not True, (
        "Legacy is_deleted flag must NOT be set"
    )


@pytest.mark.asyncio
async def test_restore_document_resets_to_active(isolate_graph_compute_engine):
    """Restoring a soft-deleted document must set status=ACTIVE."""
    engine = _isolated_engine(isolate_graph_compute_engine)
    engine.graph.add_node(
        "doc-002",
        name="TestDoc",
        content="test",
        status="ARCHIVED",
        deleted_at="2024-01-01",
    )

    from agent_utilities.knowledge_graph.pipeline.document_deletion import (
        DocumentDeletionPipeline,
    )

    pipeline = DocumentDeletionPipeline(knowledge_graph=engine)

    result = await pipeline.restore_document("doc-002")

    assert result["status"] == "restored"
    node_data = engine.graph.nodes["doc-002"]
    assert node_data.get("status") == "ACTIVE", (
        "Restored document must have status=ACTIVE"
    )


@pytest.mark.asyncio
async def test_document_update_rejects_archived(isolate_graph_compute_engine):
    """DocumentUpdatePipeline must reject updates to ARCHIVED documents."""
    engine = _isolated_engine(isolate_graph_compute_engine)
    engine.graph.add_node("doc-003", name="Archived", content="old", status="ARCHIVED")

    from agent_utilities.knowledge_graph.pipeline.document_update import (
        DocumentUpdatePipeline,
    )

    pipeline = DocumentUpdatePipeline(knowledge_graph=engine)

    with pytest.raises(ValueError, match="archived"):
        await pipeline.update_document("doc-003", new_content="new")


# ── Native WorkItem Queue Wiring ──


def test_task_submit_and_list(engine):
    """Submitting ingestion work creates one WorkItem visible via list_tasks."""
    if not engine.backend:
        pytest.skip("Requires a persistent backend for task operations")

    job_id = engine.submit_task(
        target_path=".tmp/test_repo",
        is_codebase=True,
        provenance={"source": "test"},
    )
    assert job_id.startswith("job-")

    tasks = engine.list_tasks()
    all_jobs = tasks["pending"] + tasks["running"]
    assert any(j["job_id"] == job_id for j in all_jobs), (
        "Submitted task must appear in list"
    )


# ── Gap 6: Lifecycle States in Schema ──


def test_task_schema_has_status_column():
    """Task/WorkItem schema must include a 'status' column for lifecycle tracking.

    The native ingestion task queue's node type is ``WorkItem`` (see the
    "Native WorkItem Queue Wiring" tests above and ``engine_tasks.py``'s
    ``MATCH (w:WorkItem)`` queries) — there has been no ``Task`` node type in
    ``SCHEMA.nodes`` since the task/work-item consolidation. This test's
    assertion is unchanged; only the node name it looks up is corrected to
    match the current schema authority.
    """
    task_def = next(n for n in SCHEMA.nodes if n.name == "WorkItem")
    assert "status" in task_def.columns, "WorkItem schema must have a 'status' column"


def test_article_schema_exists():
    """Article (document chunk) schema must exist for document ingestion."""
    node_names = [n.name for n in SCHEMA.nodes]
    assert "Article" in node_names, "Article schema required for document ingestion"
