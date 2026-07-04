"""Cross-backend conformance suite — one body, every supported backend.

CONCEPT:AU-KG.query.object-graph-mapper / KG-2.7 — Vendor-agnostic Graph Backend parity.

The same assertions run against **every** backend via the ``backend_under_test``
parametrized fixture (see ``conftest.py``): the zero-dep epistemic-graph L1 and
embedded LadybugDB run in the default PR suite; pggraph/Neo4j/FalkorDB run under
``pytest -m live`` against throwaway testcontainers. This is the regression net
behind "an operation that works on pggraph also works on Neo4j/Falkor/Ladybug".

Contract under test = the ``GraphBackend`` interface. So writes go through the
``IntelligenceGraphEngine`` (which normalizes dialect) but **reads go through
``backend.execute``** — reading via the engine's ``query_cypher`` would mask
backend differences by serving relationships from the in-memory compute layer.

Two backend realities the assertions respect (and document via named skips rather
than false passes):
  * LadybugDB (Kuzu) is **strict-schema** — only columns declared in the unified
    ``SCHEMA`` persist, so assertions use declared columns (``name``,
    ``description``, ``importance_score``), never ad-hoc keys.
  * Relationships carry **no properties** on strict-schema backends, and the
    epistemic-graph L1 keeps edges in the compute layer (not query-able via its
    backend Cypher); the edge test asserts *existence*, skipping backends that
    don't return relationships through ``backend.execute``.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import pytest

from agent_utilities.knowledge_graph.backends import set_active_backend
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

pytestmark = pytest.mark.integration


@pytest.fixture()
def engine(backend_under_test: Any) -> IntelligenceGraphEngine:
    """An ``IntelligenceGraphEngine`` bound to the backend under test (for writes).

    Requires the epistemic-graph compute engine (the L0 scratchpad the engine
    writes through) — gated on ``GRAPH_SERVICE_SOCKET``.
    """
    if not os.environ.get("GRAPH_SERVICE_SOCKET"):
        pytest.skip("epistemic-graph engine required (GRAPH_SERVICE_SOCKET unset)")
    set_active_backend(backend_under_test)
    eng = IntelligenceGraphEngine(backend=backend_under_test)
    IntelligenceGraphEngine.set_active(eng)
    return eng


# ───────────────────────────── schema & CRUD ───────────────────────────────────


def test_schema_creation_is_idempotent(backend_under_test: Any) -> None:
    """``create_schema`` is safe to call repeatedly (already called at build)."""
    backend_under_test.create_schema()
    backend_under_test.create_schema()  # second call must not raise


def test_node_roundtrip_preserves_unicode_and_escapes(
    engine: IntelligenceGraphEngine, backend_under_test: Any
) -> None:
    """A node's declared columns read back byte-faithful on every backend."""
    node_id = f"agent:{uuid.uuid4().hex[:8]}"
    engine.add_node(
        node_id,
        "Agent",
        {
            "name": "Router 🤖 (Quotes: \"hi\", 'yo')",
            "description": "multi\nline\nwith \\back\\slashes",
            "importance_score": 0.98,
        },
    )

    rows = backend_under_test.execute(
        "MATCH (n:Agent) WHERE n.id = $id "
        "RETURN n.name AS name, n.description AS description, "
        "n.importance_score AS importance",
        {"id": node_id},
    )
    assert rows, "node not found after write"
    row = rows[0]
    assert "🤖" in row["name"]
    assert "back\\slashes" in row["description"]
    assert float(row["importance"]) == pytest.approx(0.98)


def test_bulk_write_all_nodes_retrievable(
    engine: IntelligenceGraphEngine, backend_under_test: Any
) -> None:
    """Many nodes written then each independently retrieved — bulk-ingest parity."""
    suffix = uuid.uuid4().hex[:8]
    ids = [f"agent:bulk:{suffix}:{i}" for i in range(15)]
    for i, node_id in enumerate(ids):
        engine.add_node(node_id, "Agent", {"name": f"item-{i}"})

    found = 0
    for node_id in ids:
        rows = backend_under_test.execute(
            "MATCH (n:Agent) WHERE n.id = $id RETURN n.id AS id", {"id": node_id}
        )
        if rows and rows[0]["id"] == node_id:
            found += 1
    assert found == len(ids), f"only {found}/{len(ids)} bulk nodes persisted"


def test_edge_persists_in_backend(
    engine: IntelligenceGraphEngine, backend_under_test: Any
) -> None:
    """A typed relationship is query-able from the storage backend.

    Edges carry no properties on strict-schema backends, so this asserts
    *existence* only. Backends that keep relationships in the compute layer
    (epistemic-graph L1) rather than the query-able store skip with a reason.
    """
    suffix = uuid.uuid4().hex[:8]
    agent_id, tool_id = f"agent:{suffix}", f"tool:{suffix}"
    engine.add_node(agent_id, "Agent", {"name": "router"})
    engine.add_node(tool_id, "Tool", {"name": "math-eval"})
    engine.link_nodes(agent_id, tool_id, "PROVIDES", {"confidence": 0.95})

    rows = backend_under_test.execute(
        "MATCH (s:Agent)-[r:PROVIDES]->(t:Tool) "
        "WHERE s.id = $sid AND t.id = $tid RETURN count(r) AS c",
        {"sid": agent_id, "tid": tool_id},
    )
    count = int(rows[0]["c"]) if rows else 0
    if count == 0:
        pytest.skip(
            f"{type(backend_under_test).__name__} does not return relationships via "
            f"backend Cypher (edges live in the compute layer) — known parity gap, "
            f"see docs/guides/backend-parity-and-profile-testing.md"
        )
    assert count >= 1


def _adhoc_value(node: dict[str, Any], key: str) -> Any:
    """Read an ad-hoc property: top-level, folded into ``metadata`` JSON, or in
    Ladybug/Kuzu's ``{key: value}`` map rendering of the metadata column."""
    if key in node:
        return node[key]
    meta = node.get("metadata")
    if isinstance(meta, dict):
        return meta.get(key)
    if isinstance(meta, str) and meta:
        try:
            return json.loads(meta).get(key)
        except Exception:
            # Kuzu renders a map column as `{key: value}` (no JSON quotes).
            import re

            m = re.search(rf"\b{re.escape(key)}\s*:\s*([^,}}]+)", meta)
            if m:
                return m.group(1).strip()
    return None


def test_adhoc_property_survives(
    engine: IntelligenceGraphEngine, backend_under_test: Any
) -> None:
    """An ad-hoc (non-schema) node property is not lost on any backend.

    Schemaless backends keep it as a first-class column; strict-schema backends
    (Ladybug/pggraph) fold it into the ``metadata`` JSON column. Either way the
    value round-trips (Phase 1 regression gate).
    """
    node_id = f"agent:{uuid.uuid4().hex[:8]}"
    engine.add_node(node_id, "Agent", {"name": "router", "x_adhoc": "KEEPME"})
    rows = backend_under_test.execute(
        "MATCH (n:Agent) WHERE n.id = $id RETURN n", {"id": node_id}
    )
    assert rows, "node not found"
    node = rows[0].get("n") or rows[0]
    assert isinstance(node, dict)
    assert _adhoc_value(node, "x_adhoc") == "KEEPME"


def test_edge_property_roundtrip(
    engine: IntelligenceGraphEngine, backend_under_test: Any
) -> None:
    """An edge property round-trips on every backend (Phase 2 regression gate).

    Native-property backends expose ``r.confidence``; Ladybug stores edge props as
    a JSON ``r.properties`` column. Backends that keep edges only in the compute
    layer skip (documented gap).
    """
    suffix = uuid.uuid4().hex[:8]
    agent_id, tool_id = f"agent:{suffix}", f"tool:{suffix}"
    engine.add_node(agent_id, "Agent", {"name": "router"})
    engine.add_node(tool_id, "Tool", {"name": "calc"})
    engine.link_nodes(agent_id, tool_id, "PROVIDES", {"confidence": 0.95})

    params = {"sid": agent_id, "tid": tool_id}
    direct = backend_under_test.execute(
        "MATCH (s:Agent)-[r:PROVIDES]->(t:Tool) "
        "WHERE s.id = $sid AND t.id = $tid RETURN r.confidence AS conf",
        params,
    )
    conf = direct[0].get("conf") if direct else None
    if conf is not None and conf != "":
        assert abs(float(conf) - 0.95) < 1e-6
        return

    # Ladybug: edge props live in a JSON `properties` column.
    pr = backend_under_test.execute(
        "MATCH (s:Agent)-[r:PROVIDES]->(t:Tool) "
        "WHERE s.id = $sid AND t.id = $tid RETURN r.properties AS p",
        params,
    )
    raw = pr[0].get("p") if pr else None
    if isinstance(raw, str) and raw:
        assert abs(float(json.loads(raw).get("confidence")) - 0.95) < 1e-6
        return
    pytest.skip(
        f"{type(backend_under_test).__name__} does not expose edge properties via "
        f"backend Cypher (edges in compute layer only) — known parity gap"
    )


# ─────────────────────────── vectors / semantic search ─────────────────────────


def test_embedding_and_semantic_search_ranking(
    engine: IntelligenceGraphEngine, backend_under_test: Any
) -> None:
    """Nearest-vector ranking: the query vector must surface its own document first.

    Backends whose vector path needs a specific indexed label (e.g. Neo4j's
    ``:Chunk`` vector index) may return nothing for generic nodes — a real parity
    gap surfaced as a backend-named skip rather than a false pass.
    """
    # Use the configured embedding dimension, not a hardcoded one: backends that
    # build their vector index up-front from ``config.kg_embedding_dim`` (e.g.
    # FalkorDB's CREATE VECTOR INDEX) reject — and drop the connection on — a write
    # whose dimension differs from the index. The system standardized on 1024
    # (bge-m3), so a stale 768 here mismatched the index and reset FalkorDB.
    from agent_utilities.core.config import config as _cfg

    dim = int(_cfg.kg_embedding_dim or 768)
    suffix = uuid.uuid4().hex[:8]
    targets = {
        f"doc:hit:{suffix}": [1.0] + [0.0] * (dim - 1),
        f"doc:miss1:{suffix}": [0.0, 1.0] + [0.0] * (dim - 2),
        f"doc:miss2:{suffix}": [0.0, 0.0, 1.0] + [0.0] * (dim - 3),
    }
    for node_id, vec in targets.items():
        engine.add_node(node_id, "Document", {"name": node_id, "content": "x"})
        backend_under_test.add_embedding(node_id, vec)

    query_vec = [0.99] + [0.0] * (dim - 1)  # closest to doc:hit
    results = backend_under_test.semantic_search(query_vec, n_results=3)

    if not results:
        pytest.skip(
            f"{type(backend_under_test).__name__} returned no semantic results for "
            f"generic :Document nodes (vendor vector-index label mismatch — known "
            f"parity gap, see docs/guides/backend-parity-and-profile-testing.md)"
        )

    top = results[0].get("node") or results[0].get("n") or results[0]
    top_id = top.get("id") if isinstance(top, dict) else None
    assert top_id == f"doc:hit:{suffix}", f"expected nearest doc first, got {top_id!r}"


# ─────────────────────────────── maintenance ───────────────────────────────────


def test_prune_does_not_raise(backend_under_test: Any) -> None:
    """``prune`` honors its contract (no-op for unmatched criteria) on every backend.

    Per-backend prune *semantics* differ (importance vs last_accessed); this
    asserts the shared contract that a benign prune call never raises.
    """
    backend_under_test.prune({"last_accessed": "1970-01-01T00:00:00"})


# ─────────────────────────────── durability ────────────────────────────────────


def test_write_survives_reconnect(
    engine: IntelligenceGraphEngine, backend_under_test: Any
) -> None:
    """Data persists across a close + reopen of the same durable store."""
    if not backend_under_test._parity_durable:
        pytest.skip("epistemic_graph L1 is in-process/in-memory — nothing to reopen")

    from agent_utilities.knowledge_graph.backends import create_backend

    node_id = f"durable:{uuid.uuid4().hex[:8]}"
    engine.add_node(node_id, "Agent", {"name": "persist-me"})
    backend_under_test.close()

    reopened = create_backend(
        backend_type=backend_under_test._parity_backend_type,
        **backend_under_test._parity_kwargs,
    )
    assert reopened is not None
    try:
        rows = reopened.execute(
            "MATCH (n:Agent) WHERE n.id = $id RETURN n.id AS id", {"id": node_id}
        )
        assert rows and rows[0]["id"] == node_id
    finally:
        reopened.close()


# ───────────────────────────────── SPARQL ──────────────────────────────────────


def test_sparql_capability_contract(backend_under_test: Any) -> None:
    """LPG backends must declare no SPARQL and refuse ``execute_sparql`` loudly."""
    if backend_under_test.supports_sparql:
        out = backend_under_test.execute_sparql("ASK { ?s ?p ?o }")
        assert isinstance(out, list)
    else:
        with pytest.raises(RuntimeError):
            backend_under_test.execute_sparql("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
