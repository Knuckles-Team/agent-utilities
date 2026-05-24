"""CONCEPT:KG-2.0

Integration tests for KG lifecycle management:
- Soft-delete (ARCHIVED status) filtering convergence
- DiffEntry schema validation
- ArchiMate class schema validation
- Task queue submit/list/clear
- DocumentDeletionPipeline + QueryMixin parity
"""

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.models.schema_definition import SCHEMA

# ── Fixtures ──


@pytest.fixture
def engine():
    """Create a lightweight IntelligenceGraphEngine for testing."""
    graph = nx.MultiDiGraph()
    return IntelligenceGraphEngine(graph=graph)


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


def test_archived_node_excluded_from_networkx_search(engine):
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

    results = engine.search_hybrid("TaxService")
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
async def test_soft_delete_pipeline_uses_archived_status():
    """DocumentDeletionPipeline._soft_delete must set status=ARCHIVED, not is_deleted."""
    graph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph=graph)
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
async def test_restore_document_resets_to_active():
    """Restoring a soft-deleted document must set status=ACTIVE."""
    graph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph=graph)
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
async def test_document_update_rejects_archived():
    """DocumentUpdatePipeline must reject updates to ARCHIVED documents."""
    graph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph=graph)
    engine.graph.add_node("doc-003", name="Archived", content="old", status="ARCHIVED")

    from agent_utilities.knowledge_graph.pipeline.document_update import (
        DocumentUpdatePipeline,
    )

    pipeline = DocumentUpdatePipeline(knowledge_graph=engine)

    with pytest.raises(ValueError, match="archived"):
        await pipeline.update_document("doc-003", new_content="new")


# ── Gap 5: Task Queue Wiring ──


def test_task_submit_and_list(engine):
    """Submitting a task should create a Task node retrievable via list_tasks."""
    if not engine.backend:
        pytest.skip("Requires a persistent backend for task operations")

    job_id = engine.submit_task(
        target_path="/tmp/test_repo",
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
    """Task schema must include a 'status' column for lifecycle tracking."""
    task_def = next(n for n in SCHEMA.nodes if n.name == "Task")
    assert "status" in task_def.columns, "Task schema must have a 'status' column"


def test_article_schema_exists():
    """Article (document chunk) schema must exist for document ingestion."""
    node_names = [n.name for n in SCHEMA.nodes]
    assert "Article" in node_names, "Article schema required for document ingestion"
