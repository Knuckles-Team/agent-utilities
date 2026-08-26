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
        get_toggle_states_batch,
        set_toggle_state,
    )

    engine = MagicMock()
    engine.query_cypher.return_value = []
    attack = "x' SET n.admin = true //"

    get_toggle_states_batch(engine, [("skill", attack)])
    query, params = engine.query_cypher.call_args.args
    assert attack not in query
    assert params["pref_ids"] == [f"preference:toggle:skill:{attack}"]

    set_toggle_state(engine, "skill", attack, enabled=False)
    query, params = engine.query_cypher.call_args.args
    assert attack not in query
    assert params["disabled"] is True


_UNLABELED_DISABLED_QUERY_BY_ID = (
    "MATCH (n) WHERE n.id = $id RETURN n.id AS id, n.disabled AS disabled"
)


def test_get_existing_disabled_batch_issues_one_labeled_round_trip() -> None:
    from agent_utilities.mcp.kg_server import get_existing_disabled_batch

    engine = MagicMock()
    engine.graph_compute = None
    engine.query_cypher.return_value = []
    ids = ["resource:skill:a", "resource:skill:b"]

    result = get_existing_disabled_batch(engine, ids)

    assert engine.query_cypher.call_count == 1
    query, params = engine.query_cypher.call_args.args
    assert "MATCH (n:CallableResource)" in query
    assert params == {"node_ids": ids}
    # Genuinely-new ids (query ran fine, found nothing) are absent, not a
    # failure — the caller's own default (False, "not disabled") applies.
    assert result == {}


def test_get_existing_disabled_batch_custom_label_is_validated_and_scoped() -> None:
    """The ``label`` kwarg (added for the ``_ingest_capabilities`` MCP-config/
    native-tool loops, which pass ``"MCPServer"``/``"NativeTool"``) must be
    validated through ``validate_identifier`` like every other interpolated
    label, and must actually scope the query -- not silently fall back to
    the ``CallableResource`` default or an unlabeled scan."""
    from agent_utilities.mcp.kg_server import get_existing_disabled_batch
    from agent_utilities.security.identifiers import InvalidIdentifierError

    engine = MagicMock()
    engine.graph_compute = None
    engine.query_cypher.return_value = []

    get_existing_disabled_batch(engine, ["native_tool_x"], label="NativeTool")

    assert engine.query_cypher.call_count == 1
    query = engine.query_cypher.call_args.args[0]
    assert "MATCH (n:NativeTool)" in query

    with pytest.raises(InvalidIdentifierError):
        get_existing_disabled_batch(
            engine, ["x"], label='NativeTool"; DROP TABLE kg_edges; --'
        )


def test_get_existing_disabled_batch_fails_closed_on_query_error() -> None:
    """Every unresolved id must be marked disabled=True in the returned
    mapping on failure — never merely omitted, since the caller
    (``disabled_by_resource.get(resource_id, False)``) reads a missing key as
    "not disabled"."""
    from agent_utilities.mcp.kg_server import get_existing_disabled_batch

    engine = MagicMock()
    engine.graph_compute = None
    engine.query_cypher.side_effect = RuntimeError("engine unavailable")
    ids = ["resource:skill:a", "resource:skill:b"]

    result = get_existing_disabled_batch(engine, ids)

    assert result == {"resource:skill:a": True, "resource:skill:b": True}


def test_source_sync_existing_disabled_tries_labeled_queries_before_unlabeled_scan() -> (
    None
):
    from agent_utilities.knowledge_graph.core.source_sync import _existing_disabled

    engine = MagicMock()
    engine.graph_compute = None
    engine.query_cypher.return_value = []

    _existing_disabled(engine, "tool_demo_thing")

    first_query = engine.query_cypher.call_args_list[0].args[0]
    assert first_query != _UNLABELED_DISABLED_QUERY_BY_ID
    assert "MATCH (n:MCPServer)" in first_query


def test_source_sync_existing_disabled_fails_closed_on_query_error() -> None:
    from agent_utilities.knowledge_graph.core.source_sync import _existing_disabled

    engine = MagicMock()
    engine.graph_compute = None
    engine.query_cypher.side_effect = RuntimeError("engine unavailable")

    assert _existing_disabled(engine, "tool_demo_thing") is True


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
