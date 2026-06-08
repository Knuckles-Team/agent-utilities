"""Operational-Cypher subset interpreter for the in-memory L1 backend.

The orchestration engine's persistent Task queue drives status transitions via
``MATCH ... SET`` and polls with ``MATCH (t:Task {status: 'pending'})``. The
epistemic-graph (in-memory) backend has no Cypher engine, so
``EpistemicGraphBackend.execute`` interprets that operational subset directly.
A regression here re-introduces the infinite task re-claim loop (status writes
silently dropped → tasks never leave ``pending``). (CONCEPT:KG-2.0)
"""

from __future__ import annotations

import base64
import json

import pytest

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)


def _enc(d: dict) -> str:
    return base64.b64encode(json.dumps(d).encode()).decode()


@pytest.fixture()
def backend() -> EpistemicGraphBackend:
    b = EpistemicGraphBackend()
    meta = _enc({"target": "/repo", "type": "codebase"})
    b.add_node("job-1", node_type="Task", status="pending", metadata=meta)
    b.add_node("job-2", node_type="Task", status="pending", metadata=meta)
    b.add_node("code-1", node_type="Code", file_path="/a/b.py")
    return b


def test_poll_filters_by_label_and_status(backend):
    rows = backend.execute(
        "MATCH (t:Task {status: 'pending'}) RETURN t.id as id, t.metadata as meta LIMIT 1"
    )
    assert len(rows) == 1
    assert rows[0]["id"] in {"job-1", "job-2"}
    assert "meta" in rows[0]


def test_guarded_set_transitions_status_and_breaks_reclaim(backend):
    # Claim job-1 (the core fix: SET must actually mutate, not no-op).
    backend.execute(
        "MATCH (t:Task {id: $id, status: 'pending'}) SET t.status = 'running'",
        {"id": "job-1"},
    )
    s = backend.execute(
        "MATCH (t:Task) WHERE t.id = $id RETURN t.status as s", {"id": "job-1"}
    )
    assert s == [{"s": "running"}]

    # A re-poll must now skip the claimed task — otherwise the worker loops.
    again = backend.execute(
        "MATCH (t:Task {status: 'pending'}) RETURN t.id as id LIMIT 1"
    )
    assert again == [{"id": "job-2"}]


def test_guarded_set_is_idempotent_when_status_no_longer_pending(backend):
    backend.execute(
        "MATCH (t:Task {id: $id, status: 'pending'}) SET t.status = 'running'",
        {"id": "job-1"},
    )
    # Second claim of the same job should match nothing (status != pending).
    backend.execute(
        "MATCH (t:Task {id: $id, status: 'pending'}) SET t.status = 'running'",
        {"id": "job-1"},
    )
    s = backend.execute(
        "MATCH (t:Task) WHERE t.id = $id RETURN t.status as s", {"id": "job-1"}
    )
    assert s == [{"s": "running"}]


def test_update_status_and_metadata(backend):
    new_meta = _enc({"target": "/repo", "type": "codebase", "nodes_added": 12})
    backend.execute(
        "MATCH (t:Task {id: $id}) SET t.status = $status, t.metadata = $meta",
        {"id": "job-2", "status": "completed", "meta": new_meta},
    )
    row = backend.execute(
        "MATCH (t:Task {id: $id}) RETURN t.status as status, t.metadata as meta",
        {"id": "job-2"},
    )
    assert row[0]["status"] == "completed"
    assert json.loads(base64.b64decode(row[0]["meta"]))["nodes_added"] == 12


def test_count_with_in_filter(backend):
    backend.execute(
        "MATCH (t:Task {id: $id}) SET t.status = 'completed'", {"id": "job-1"}
    )
    pending = backend.execute(
        "MATCH (t:Task) WHERE t.status IN ['pending', 'running'] RETURN count(t) as cnt"
    )
    assert pending == [{"cnt": 1}]
    total = backend.execute("MATCH (t:Task) RETURN count(t) as count")
    assert total == [{"count": 2}]


def test_return_full_node_is_column_named(backend):
    # Cypher ``RETURN t`` yields a single column ``t`` holding the node, matching
    # engine_tasks (``res[0]["t"]``) and memory_phase hydration (``res["n"]``).
    rows = backend.execute("MATCH (t:Task {id: $id}) RETURN t", {"id": "job-1"})
    node = rows[0]["t"]
    assert node["id"] == "job-1"
    assert node["node_type"] == "Task"
    assert node["status"] == "pending"


def test_return_all_nodes_for_hydration(backend):
    # memory_phase hydrates via ``MATCH (n) RETURN n`` then reads ``res["n"]``.
    rows = backend.execute("MATCH (n) RETURN n")
    ids = {r["n"]["id"] for r in rows}
    assert {"job-1", "job-2", "code-1"} <= ids


def test_where_contains_other_category(backend):
    rows = backend.execute(
        "MATCH (c:Code) WHERE c.file_path CONTAINS $name RETURN c.id as id",
        {"name": "b.py"},
    )
    assert rows == [{"id": "code-1"}]


def test_detach_delete_terminal_tasks(backend):
    backend.execute(
        "MATCH (t:Task {id: $id}) SET t.status = 'completed'", {"id": "job-1"}
    )
    backend.execute(
        "MATCH (t:Task) WHERE t.status IN ['completed', 'failed'] DETACH DELETE t"
    )
    total = backend.execute("MATCH (t:Task) RETURN count(t) as count")
    assert total == [{"count": 1}]


def test_merge_node_upsert_persists_and_is_idempotent(backend):
    # The graph-writer daemon persists ingested nodes via MERGE ... SET. Without
    # interpreter support these silently no-op (Code nodes never land in L1).
    backend.execute(
        "MERGE (n:Code {id: $id}) SET n.file_path = $props_fp, n.type = $props_t",
        {"id": "c-9", "props_fp": "/a/b.py", "props_t": "symbol"},
    )
    rows = backend.execute("MATCH (c:Code {id: $id}) RETURN c", {"id": "c-9"})
    assert rows[0]["c"]["file_path"] == "/a/b.py"
    assert rows[0]["c"]["node_type"] == "Code"

    # Re-MERGE updates in place (no duplicate node).
    backend.execute(
        "MERGE (n:Code {id: $id}) SET n.file_path = $props_fp",
        {"id": "c-9", "props_fp": "/x/y.py"},
    )
    after = backend.execute(
        "MATCH (c:Code {id: $id}) RETURN c.file_path as fp", {"id": "c-9"}
    )
    assert after == [{"fp": "/x/y.py"}]
    # Idempotent: re-MERGE updated in place rather than inserting a duplicate
    # (fixture's code-1 + the single c-9 == 2).
    count = backend.execute("MATCH (c:Code) RETURN count(c) as c")
    assert count == [{"c": 2}]


def test_label_match_honours_type_key(backend):
    # The Rust node store normalises the label onto the ``type`` property on
    # read-back (graph_compute uses ``props.get("type", props.get("node_type"))``).
    # A label filter MUST match nodes whose label lives in ``type`` — else the
    # task worker poll finds no pending Task and ingestion stalls indefinitely.
    backend.add_node("job-typed", type="Task", status="pending")
    rows = backend.execute("MATCH (t:Task {status: 'pending'}) RETURN t.id as id")
    assert "job-typed" in {r["id"] for r in rows}


def test_legacy_id_lookup_still_works(backend):
    # Unrecognised/relationship queries fall back to the legacy reader, which
    # honours a bare id param lookup.
    rows = backend.execute("MATCH (a)-[:REL]->(b) RETURN b", {"id": "job-1"})
    assert rows and rows[0]["id"] == "job-1"
