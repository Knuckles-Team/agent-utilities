"""Cross-backend operational-Cypher conformance contract.

The ingestion/orchestration engine talks to *every* graph backend through the
single ``GraphBackend.execute(cypher, params)`` interface and never branches on
backend type for data ops. That abstraction only holds if each backend honours
the **same bounded Cypher subset** the engine actually emits. When a backend
silently ignores a query (the in-memory backend used to ignore the query string;
the pggraph transpiler returns ``UNKNOWN`` → ``[]``), ingestion breaks in ways
unit tests for individual backends never catch — e.g. a generic record update
becoming a no-op or ``MERGE`` node upserts vanishing.

This module pins that contract:

* ``CONTRACT_QUERIES`` is the canonical list of generic query shapes the
  graph-writer emits (node upsert/read/update/delete and traversal).
* In-process backends (memory/epistemic_graph, and ladybug when installed) run
  the queries and must produce identical *semantics*.
* For pggraph/PostgreSQL (no live DB in unit CI), we assert the
  Cypher→SQL transpiler recognises every contract query — i.e. never silently
  degrades to ``UNKNOWN``.

(CONCEPT:AU-KG.query.object-graph-mapper / OS-5.0 — backend abstraction, single interface.)
"""

from __future__ import annotations

import base64
import json
import re

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
        "record_status_set",
        "MATCH (r:Record {id: $id}) SET r.status = $status, r.metadata = $meta",
        {"id": "r1", "status": "completed", "meta": _enc({})},
    ),
    (
        "record_status_read",
        "MATCH (r:Record {id: $id}) RETURN r.status as status, r.metadata as meta",
        {"id": "r1"},
    ),
    (
        "record_delete",
        "MATCH (r:Record {id: $id}) DETACH DELETE r",
        {"id": "r1"},
    ),
    # Single-hop traversal — the engine relies on this for concept↔code/feature
    # interweaving, golden-loop intake, and orchestration. A backend that
    # degrades these to UNKNOWN silently returns wrong data (CONCEPT:EG-KG.storage.nonblocking-checkpoint).
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
KNOWN_TABLES = {"Code", "Record", "Article", "Skill", "Agent", "MCPServer", "Concept"}


@pytest.mark.parametrize(
    "name,cypher,params",
    CONTRACT_QUERIES,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_durable_transpiler_recognises_contract(name, cypher, params):
    """No pggraph contract query may degrade to UNKNOWN."""
    if not isinstance(name, str):  # pragma: no cover - param id artifact
        return
    tq = transpile(cypher, params, KNOWN_TABLES, node_tables=KNOWN_TABLES)
    assert tq.query_type != QueryType.UNKNOWN, (
        f"Transpiler silently drops contract query '{name}': {cypher!r}. "
        "pggraph would no-op this ingestion write."
    )
    assert tq.sql, f"Empty SQL for contract query '{name}'"
    # A dangling/empty WHERE is as bad as UNKNOWN: ``DELETE FROM "Record" WHERE``
    # is a syntax error (swallowed mirror failure → durable row never removed).
    # Guards the inline-``{id:$id}`` filter being dropped on deletes. (CONCEPT:AU-KG.query.vendor-agnostic-traversal)
    assert not re.search(r"\bWHERE\s*$", tq.sql), (
        f"Contract query '{name}' transpiled to a dangling empty WHERE: {tq.sql!r}"
    )


def _build_inprocess_backend(label: str):
    """Construct one in-process backend by name, at TEST-EXECUTION time.

    Deliberately NOT called from the ``parametrize`` decorator below (that
    would run at *collection* time, executed unconditionally on every single
    pytest invocation regardless of ``-k``/``-m`` selection, since parametrize
    arguments are evaluated while the module is imported for collection, not
    when a selected test actually runs). Measured with cProfile
    (2026-08-15): the ladybug branch alone -- ``LadybugBackend.create_schema()``
    issuing ~7.6k individual native DDL statements to build the full ~40-table
    schema -- cost ~267s of a ~447s profiled full-suite `--collect-only` run
    (~60%), paid on every collection of all 17k+ tests, not just the 2 that
    use it. Moving construction into the test body means it only runs for the
    2 items actually SELECTED to execute; skip semantics are unchanged (same
    reasons, same visible skip, just decided when the test runs instead of
    when the module is imported).

    The epistemic-graph backend binds a ``SyncEpistemicGraphClient`` at
    construction, which connects to the engine's Tokio service. CI runs an
    engine, but a bare dev box (or a hermetic unit run with no engine) has none,
    so the bind failure produces a skip rather than raising at *collection*
    time (which would abort the whole module). The contract still runs
    wherever an engine is up.
    """
    if label == "epistemic_graph":
        from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
            EpistemicGraphBackend,
        )

        try:
            return EpistemicGraphBackend()
        except Exception as exc:  # engine not reachable in this environment
            pytest.skip(f"epistemic-graph engine unreachable: {exc}")

    if label == "ladybug":
        try:
            from agent_utilities.knowledge_graph.backends import (
                LADYBUG_AVAILABLE,
                LadybugBackend,
            )
        except Exception as exc:
            pytest.skip(f"ladybug backend import failed: {exc}")
        if not LADYBUG_AVAILABLE:
            pytest.skip("ladybug not installed")

        import tempfile

        path = tempfile.mktemp(suffix=".db")  # noqa: S306 - test scratch
        lb = LadybugBackend(path)
        try:
            lb.create_schema()
        except Exception:
            pass
        return lb

    raise ValueError(f"unknown in-process backend label: {label!r}")  # pragma: no cover


@pytest.mark.parametrize("label", ["epistemic_graph", "ladybug"])
def test_inprocess_backend_honours_lifecycle_contract(label):
    """Every in-process backend must honour generic node mutation semantics."""
    backend = _build_inprocess_backend(label)

    # 1) Node upsert via MERGE ... SET actually persists + is queryable.
    backend.execute(
        "MERGE (n:Code {id: $id}) SET n.file_path = $props_fp",
        {"id": "code-1", "props_fp": "/a/b.py"},
    )
    rows = backend.execute(
        "MATCH (c:Code {id: $id}) RETURN c.file_path as fp", {"id": "code-1"}
    )
    assert rows and rows[0].get("fp") == "/a/b.py", f"{label}: MERGE node upsert lost"

    # 2) A declared lifecycle node: create then update must mutate. Ladybug/Kuzu
    # is schema-backed, so use the canonical DiffEntry.status column rather
    # than inventing an undeclared property on a generic Record label.
    backend.execute(
        "MERGE (n:DiffEntry {id: $id}) SET n.status = $props_status",
        {"id": "diff-1", "props_status": "pending"},
    )
    backend.execute(
        "MATCH (r:DiffEntry {id: $id}) SET r.status = $status",
        {"id": "diff-1", "status": "running"},
    )
    st = backend.execute(
        "MATCH (r:DiffEntry {id: $id}) RETURN r.status as s", {"id": "diff-1"}
    )
    assert st and st[0].get("s") == "running", f"{label}: record status SET was a no-op"
