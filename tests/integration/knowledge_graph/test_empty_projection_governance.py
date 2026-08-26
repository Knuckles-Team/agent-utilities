"""Regression coverage for the empty-projection governance defect.

Symptom, measured against the live running service (eg 2.27.0): a non-aggregate Cypher
projection with no ``id`` column answered ``[]`` on a graph that demonstrably
had matching rows —

    MATCH (n:Skill) RETURN n.name AS name LIMIT 5   -> []      (should be 5 rows)
    MATCH (a)-[r]->(b) RETURN a.id AS s, b.id AS o   -> []      (should be N rows)

— while the identical projection WITH an aggregate (``count(*)``) worked.
This was believed to be a Rust engine defect; it was reproduced NOT to be one
(the engine returns correct rows for both shapes when called directly through
``eg-query``'s ``exec_cypher_params_indexed``, the same call the server makes).

Root cause, isolated to two AU-side post-hoc row classifiers that raised
``PermissionError`` on ANY row lacking a governed node id
(``knowledge_graph.core.secured_reads.row_node_ids``/``filter_rows`` and
``knowledge_graph.ontology.permissioning.restricted_view``) — because a
plain projection like ``RETURN n.name AS name`` (or
``RETURN a.id AS s, b.id AS o`` — note the ALIASED column names; the
classifier looks for a literal ``id``/``node_id``/``n.id``/``_id`` key, so
even a projection that DOES select ids under a different alias triggers this)
never carries an ``id`` column at all. Failing the WHOLE read this way meant
`agent_webui.api_extensions._read_union_cypher` → `tenant_sharing.read_union`
caught the `PermissionError` per accessible graph and logged it at `DEBUG`,
silently contributing zero rows — which is why the REST-visible symptom was
an empty list rather than a visible error.

Fix: both classifiers accept a ``trust_pushdown`` flag. ``QueryMixin
.query_cypher``/``KnowledgeGraph.query`` set it only when owner/scope
visibility was DEMONSTRABLY pushed into the query text for that specific call
(``tenant_sharing.push_down_visibility`` — best-effort, never fails open).
When true, a row with no governed id is trusted and kept rather than
rejecting the whole read or being silently dropped — the exact same
trade-off already made for aggregate rows, and the same shape as
``filter_commons_catalog``'s pre-existing ``trust_pushdown`` escape.

These tests exercise the REAL non-aggregate projection through the REAL
governed read chokepoints — ``QueryMixin.query_cypher`` and
``KnowledgeGraph.query`` (the facade) — against a real (in-memory)
LadybugBackend, under the SAME non-privileged ambient actor/session the rest
of this suite uses (`tests/conftest.py::isolate_graph_compute_engine`), not a
mocked/stubbed governance layer. A cross-tenant row with no id is also
asserted to still be excluded by the query-level tenant boundary, proving
this fix widens nothing: `trust_pushdown` never bypasses tenant scope, only
the fine-grained per-node classification ACL for a row it structurally
cannot identify.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.backends.contrib.ladybug_backend import (
    LadybugBackend,
)
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.facade import KnowledgeGraph


@pytest.fixture
def graph_engine():
    backend = LadybugBackend(db_path=":memory:")
    backend.create_schema()
    engine = IntelligenceGraphEngine(
        graph=GraphComputeEngine(backend_type="rust"), backend=backend
    )
    yield engine
    backend.close()


def _seed_skills(backend: LadybugBackend, count: int = 5) -> None:
    for i in range(count):
        backend.execute(
            "CREATE (n:Skill {id: $id, name: $name})",
            {"id": f"skill:{i}", "name": f"skill-{i}"},
        )


class TestQueryCypherNonAggregateProjection:
    """`QueryMixin.query_cypher` — the first AU-owned governed chokepoint."""

    def test_plain_projection_with_no_id_column_returns_rows_not_empty(
        self, graph_engine
    ):
        """The exact live-reproduction query: `RETURN n.name AS name LIMIT 5`."""
        _seed_skills(graph_engine.backend, count=5)

        rows = graph_engine.query_cypher("MATCH (n:Skill) RETURN n.name AS name LIMIT 5")

        assert len(rows) == 5
        assert {r["name"] for r in rows} == {f"skill-{i}" for i in range(5)}

    def test_aliased_id_columns_also_return_rows(self, graph_engine):
        """`RETURN a.id AS s, b.id AS o` — ids ARE selected, just not under a
        literal `id` key. The row classifier only recognizes literal
        `id`/`node_id`/`n.id`/`_id` keys, so this shape is affected exactly
        like a column-less projection (`get_graph_relationships`'s real
        query: `RETURN a.id as source, type(r) as type, b.id as target`)."""
        backend = graph_engine.backend
        backend.execute(
            "CREATE (a:Skill {id: $aid, name: 'a'})-[:DEPENDS_ON]->"
            "(b:Skill {id: $bid, name: 'b'})",
            {"aid": "skill:a", "bid": "skill:b"},
        )

        # `type(r)` is deliberately NOT projected here (Ladybug/Kuzu's Cypher
        # dialect does not implement it) -- irrelevant to the point under
        # test, which is the `a.id AS s` / `b.id AS o` ALIASING alone.
        rows = graph_engine.query_cypher(
            "MATCH (a:Skill)-[r:DEPENDS_ON]->(b:Skill) RETURN a.id AS s, b.id AS o"
        )

        assert rows == [{"s": "skill:a", "o": "skill:b"}]

    def test_cross_tenant_row_stays_excluded_despite_the_fix(
        self, graph_engine, monkeypatch
    ):
        """`trust_pushdown` must widen NOTHING: the query-level tenant
        boundary (`secured_reads.scope()`, mandatory and unconditional)
        still excludes another tenant's rows before the relaxed row-id
        classifier ever sees them. Proves this fix is a governance
        RELOCATION (post-hoc raise -> query-level pushdown), not a
        relaxation of what data is reachable.
        """
        from _test_engine import TEST_TENANT

        backend = graph_engine.backend
        # Own-tenant rows (visible under the ambient actor's own tenant).
        _seed_skills(backend, count=2)
        # A row stamped for a DIFFERENT tenant.
        backend.execute(
            "CREATE (n:Skill {id: $id, name: $name, tenant_id: $tenant})",
            {"id": "skill:other", "name": "other-tenant-skill", "tenant": "some-other-tenant"},
        )
        assert TEST_TENANT != "some-other-tenant"

        rows = graph_engine.query_cypher("MATCH (n:Skill) RETURN n.name AS name")

        names = {r["name"] for r in rows}
        assert "other-tenant-skill" not in names
        assert names == {"skill-0", "skill-1"}


class TestFacadeQueryNonAggregateProjection:
    """`KnowledgeGraph.query` (`facade.py`) — the second, independent AU-owned
    governed chokepoint; has its own copy of the same fix (`push_down_visibility`
    + `filter_rows(..., trust_pushdown=...)` + `permissioning.enforce(...,
    trust_pushdown=...)`)."""

    def test_plain_projection_with_no_id_column_returns_rows_not_empty(
        self, graph_engine
    ):
        _seed_skills(graph_engine.backend, count=3)
        kg = KnowledgeGraph.from_engine(graph_engine)

        rows = kg.query("MATCH (n:Skill) RETURN n.name AS name LIMIT 5")

        assert len(rows) == 3
        assert {r["name"] for r in rows} == {"skill-0", "skill-1", "skill-2"}
