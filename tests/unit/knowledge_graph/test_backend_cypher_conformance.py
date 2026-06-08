"""Cross-backend operational-Cypher conformance contract.

The ingestion/orchestration engine talks to *every* graph backend through the
single ``GraphBackend.execute(cypher, params)`` interface and never branches on
backend type for data ops. That abstraction only holds if each backend honours
the **same bounded Cypher subset** the engine actually emits. When a backend
silently ignores a query (the in-memory backend used to ignore the query string;
the pggraph transpiler returns ``UNKNOWN`` → ``[]``), ingestion breaks in ways
unit tests for individual backends never catch — e.g. Task status SET becoming a
no-op (infinite re-claim) or ``MERGE`` node upserts vanishing.

This module pins that contract:

* ``CONTRACT_QUERIES`` is the canonical list of operational query shapes the
  engine emits (graph-writer/sync node upserts, TaskManager lifecycle, dedupe,
  hydration). Keep it in sync with ``core/engine_tasks.py`` + ``pipeline/phases``.
* In-process backends (memory/epistemic_graph, and ladybug when installed) run
  the queries and must produce identical *semantics*.
* For the pggraph/PostgreSQL durable tier (no live DB in unit CI), we assert the
  Cypher→SQL transpiler recognises every contract query — i.e. never silently
  degrades to ``UNKNOWN``.

(CONCEPT:KG-2.0 / OS-5.0 — backend abstraction, single interface.)
"""

from __future__ import annotations

import base64
import json

import pytest

from agent_utilities.knowledge_graph.backends.cypher_transpiler import (
    QueryType,
    transpile,
)


def _enc(d: dict) -> str:
    return base64.b64encode(json.dumps(d).encode()).decode()


# The exact operational query shapes the engine emits. Each entry is the Cypher
# string the durable transpiler must recognise (not degrade to UNKNOWN).
CONTRACT_QUERIES: list[tuple[str, str, dict]] = [
    (
        "node_upsert_merge_set",
        "MERGE (n:Code {id: $id}) SET n.file_path = $props_file_path, n.type = $props_type",
        {"id": "c1", "props_file_path": "/a.py", "props_type": "symbol"},
    ),
    (
        "task_status_set",
        "MATCH (t:Task {id: $id}) SET t.status = $status, t.metadata = $meta",
        {"id": "j1", "status": "completed", "meta": _enc({})},
    ),
    (
        "task_status_read",
        "MATCH (t:Task {id: $id}) RETURN t.status as status, t.metadata as meta",
        {"id": "j1"},
    ),
    (
        "task_terminal_delete",
        "MATCH (t:Task {id: $id}) DETACH DELETE t",
        {"id": "j1"},
    ),
    # Single-hop traversal — the engine relies on this for concept↔code/feature
    # interweaving, golden-loop intake, and orchestration. A backend that
    # degrades these to UNKNOWN silently returns wrong data (CONCEPT:KG-2.8).
    (
        "traversal_count",
        "MATCH (s:Article)-[r:MENTIONS]->(t:Concept) RETURN count(r) as c",
        {},
    ),
    (
        "traversal_count_distinct",
        "MATCH (s:Article)-[r:MENTIONS]->(t:Concept) RETURN count(DISTINCT t) as c",
        {},
    ),
    (
        "traversal_projection",
        "MATCH (s:Article)-[r:MENTIONS]->(t:Concept) "
        "RETURN s.id as a, t.name as concept LIMIT 5",
        {},
    ),
]

# Tables the durable transpiler must know about for the contract to resolve.
KNOWN_TABLES = {"Code", "Task", "Article", "Skill", "Agent", "MCPServer", "Concept"}


@pytest.mark.parametrize(
    "name,cypher,params",
    CONTRACT_QUERIES,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_durable_transpiler_recognises_contract(name, cypher, params):
    """pggraph durable tier: no contract query may degrade to UNKNOWN."""
    if not isinstance(name, str):  # pragma: no cover - param id artifact
        return
    tq = transpile(cypher, params, KNOWN_TABLES)
    assert tq.query_type != QueryType.UNKNOWN, (
        f"Transpiler silently drops contract query '{name}': {cypher!r}. "
        "pggraph would no-op this ingestion write."
    )
    assert tq.sql, f"Empty SQL for contract query '{name}'"


def _inprocess_backends():
    """Backend instances that run fully in-process (no external server)."""
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )

    backends = [("epistemic_graph", EpistemicGraphBackend())]
    try:
        from agent_utilities.knowledge_graph.backends import (
            LADYBUG_AVAILABLE,
            LadybugBackend,
        )

        if LADYBUG_AVAILABLE:
            import tempfile

            path = tempfile.mktemp(suffix=".db")  # noqa: S306 - test scratch
            lb = LadybugBackend(path)
            try:
                lb.create_schema()
            except Exception:
                pass
            backends.append(("ladybug", lb))
    except Exception:
        pass
    return backends


@pytest.mark.parametrize(
    "label,backend",
    _inprocess_backends(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_inprocess_backend_honours_lifecycle_contract(label, backend):
    """Every in-process backend must honour the Task/node lifecycle semantics."""
    if not isinstance(label, str):  # pragma: no cover - param id artifact
        return

    # 1) Node upsert via MERGE ... SET actually persists + is queryable.
    backend.execute(
        "MERGE (n:Code {id: $id}) SET n.file_path = $props_fp",
        {"id": "code-1", "props_fp": "/a/b.py"},
    )
    rows = backend.execute(
        "MATCH (c:Code {id: $id}) RETURN c.file_path as fp", {"id": "code-1"}
    )
    assert rows and rows[0].get("fp") == "/a/b.py", f"{label}: MERGE node upsert lost"

    # 2) Task lifecycle: create (pending) → claim (SET running) must mutate.
    backend.execute(
        "MERGE (n:Task {id: $id}) SET n.status = $props_status",
        {"id": "job-1", "props_status": "pending"},
    )
    backend.execute(
        "MATCH (t:Task {id: $id}) SET t.status = $status",
        {"id": "job-1", "status": "running"},
    )
    st = backend.execute(
        "MATCH (t:Task {id: $id}) RETURN t.status as s", {"id": "job-1"}
    )
    assert st and st[0].get("s") == "running", (
        f"{label}: Task status SET was a no-op — would cause infinite re-claim"
    )
