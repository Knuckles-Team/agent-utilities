"""Label-less node UNIONs must span only node-shaped tables (CONCEPT:KG-2.9).

A bare ``MATCH (n {id: …})`` projects ``properties`` across a UNION of node
tables. Typed/ontology tables (e.g. ``Account``) lack a ``properties`` column, so
including them makes the whole UNION fail with ``column "properties" does not
exist`` — which silently broke the CAS L3-mirror (a label-less ``MATCH (n {id})
SET``). The transpiler must fan out only over ``node_tables`` (those that carry the
universal node shape). Pure-string transpile; no DB.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.backends.cypher_transpiler import transpile

_KNOWN = {"Task", "Schedule", "Account", "Action"}  # Account/Action: no properties
_NODE = {"Task", "Schedule"}  # node-shaped subset


def test_labelless_property_union_spans_only_node_tables():
    sql = transpile(
        "MATCH (n {id: $id}) RETURN n", {"id": "x"}, _KNOWN, node_tables=_NODE
    ).sql
    assert "Account" not in sql and "Action" not in sql
    assert "Task" in sql and "Schedule" in sql


def test_labelless_set_union_excludes_non_node_tables():
    # The CAS L3-mirror shape that was failing on Account/Action.
    sql = transpile(
        "MATCH (n {id: $id}) SET n.status = $s",
        {"id": "x", "s": "cancelled"},
        _KNOWN,
        node_tables=_NODE,
    ).sql
    assert "Account" not in sql and "Action" not in sql


def test_defaults_to_known_tables_when_node_tables_unset():
    # Back-compatible: callers that don't pass node_tables get the old behavior.
    sql = transpile("MATCH (n {id: $id}) RETURN n", {"id": "x"}, _KNOWN).sql
    assert "Account" in sql  # spans all known tables when node subset not supplied


def test_labelless_where_union_param_count_matches_branches():
    """Regression: the WHERE-clause label-less UNION repeats the placeholder once
    per branch — branches span node_tables, so bound params must repeat
    len(node_tables) times, NOT len(known_tables). A mismatch raises psycopg
    'the query has N placeholders but M parameters were passed' (CONCEPT:KG-2.9)."""
    tq = transpile(
        "MATCH (n) WHERE n.id = $id RETURN n", {"id": "x"}, _KNOWN, node_tables=_NODE
    )
    placeholders = tq.sql.count("%s")
    assert placeholders == len(_NODE), (placeholders, tq.sql)
    assert len(tq.params) == placeholders, (len(tq.params), placeholders)
