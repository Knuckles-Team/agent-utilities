from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_timeseries_cypher_literal_cannot_close_string() -> None:
    from agent_utilities.knowledge_graph.memory.timeseries.engine_backend import (
        _cypher_string,
    )

    rendered = _cypher_string("x' RETURN s //")
    assert rendered == "'x\\' RETURN s //'"
    with pytest.raises(ValueError):
        _cypher_string("x\nMATCH (n)")


def test_mcp_toggle_queries_parameterize_ids_and_values() -> None:
    from agent_utilities.mcp.kg_server import (
        get_existing_disabled,
        get_toggle_state,
        set_toggle_state,
    )

    engine = MagicMock()
    engine.query_cypher.return_value = []
    attack = "x' SET n.admin = true //"

    get_existing_disabled(engine, attack)
    query, params = engine.query_cypher.call_args.args
    assert attack not in query and params == {"node_id": attack}

    get_toggle_state(engine, "skill", attack)
    query, params = engine.query_cypher.call_args.args
    assert attack not in query and params["pref_id"].endswith(attack)

    set_toggle_state(engine, "skill", attack, enabled=False)
    query, params = engine.query_cypher.call_args.args
    assert attack not in query
    assert params["disabled"] is True


def test_sparql_iri_and_source_partition_reject_query_breakout() -> None:
    from agent_utilities.knowledge_graph.backends.sparql.source_partition import (
        graph_uri_for_source,
    )
    from agent_utilities.knowledge_graph.integrations.stardog_sync import _sparql_iri

    assert graph_uri_for_source("System:Instance") == "urn:source:system:instance"
    assert graph_uri_for_source("system> } UNION {") == "urn:source:system-union"
    with pytest.raises(ValueError, match="IRI"):
        _sparql_iri("https://example.invalid/> } UNION {")


def test_age_sql_wrapper_uses_collision_checked_delimiter_and_safe_graph() -> None:
    from agent_utilities.knowledge_graph.backends.age_backend import (
        _dollar_quote_cypher,
        _require_age_graph_name,
    )

    attack = "$ag$) SELECT pg_sleep(9); --"
    wrapped = _dollar_quote_cypher(attack)
    delimiter = wrapped[: wrapped.index("$", 1) + 1]
    assert wrapped == f"{delimiter}{attack}{delimiter}"
    assert wrapped.count(delimiter) == 2
    assert _require_age_graph_name("tenant_graph_1") == "tenant_graph_1"
    with pytest.raises(ValueError, match="graph name"):
        _require_age_graph_name("x'); DROP TABLE kg_edges; --")


def test_cypher_transpiler_rejects_catalog_identifier_injection() -> None:
    from agent_utilities.knowledge_graph.backends.cypher_transpiler import transpile

    with pytest.raises(ValueError, match="identifier"):
        transpile(
            "MATCH (n) WHERE n.id = $id RETURN label(n) AS label",
            {"id": "n1"},
            known_tables={'Node"; DROP TABLE kg_edges; --'},
            node_tables={"Node"},
        )


def test_cypher_transpiler_binds_coalesce_literals() -> None:
    from agent_utilities.knowledge_graph.backends.cypher_transpiler import transpile

    result = transpile(
        "MATCH (n:Node) WHERE coalesce(n.status, 'new') <> 'deleted' RETURN n",
        {},
        known_tables={"Node"},
        node_tables={"Node"},
    )
    assert "'new'" not in result.sql
    assert "'deleted'" not in result.sql
    assert result.params == ["new", "deleted"]


def test_database_schema_identifiers_fail_closed() -> None:
    from agent_utilities.knowledge_graph.backends.postgresql_backend import (
        _require_sql_identifier,
    )
    from agent_utilities.knowledge_graph.core.engine_tasks import (
        _require_database_identifier,
    )
    from agent_utilities.knowledge_graph.migrations import _schema_identifier

    for gate in (
        _require_sql_identifier,
        _require_database_identifier,
        _schema_identifier,
    ):
        assert gate("Node_1") == "Node_1"
        with pytest.raises(ValueError, match="identifier"):
            gate('Node"; DROP TABLE kg_edges; --')
