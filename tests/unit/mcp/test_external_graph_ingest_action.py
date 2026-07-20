from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class _MockMCP:
    def __init__(self) -> None:
        self.funcs = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.funcs[fn.__name__] = fn
            return fn

        return decorator

    def custom_route(self, *args, **kwargs):
        def decorator(fn):
            self.funcs[fn.__name__] = fn
            return fn

        return decorator


@pytest.fixture
def registered_tools(tmp_path, monkeypatch):
    # Server construction may ask optional native backends for a scratch file.
    # Keep that side effect outside the repository under the pytest-owned root.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    mock_mcp = _MockMCP()
    engine = MagicMock()
    engine.backend = MagicMock()
    engine.backend.read_only = False
    with patch(
        "agent_utilities.mcp.server_factory.create_mcp_server",
        return_value=(None, mock_mcp, []),
    ):
        with patch("agent_utilities.mcp.kg_server._get_engine", return_value=engine):
            from agent_utilities.mcp.kg_server import _build_server

            _build_server(bootstrap=False)
    return mock_mcp.funcs


@pytest.mark.asyncio
@pytest.mark.concept("AU-KG.ingest.external-graph-federation")
async def test_graph_configure_ingest_connection_live_path(
    registered_tools, monkeypatch
) -> None:
    from agent_utilities.mcp import kg_server

    registry = MagicMock()
    authority = MagicMock()
    registry.get_engine.return_value = authority
    registry.backend_kind.return_value = "neo4j"
    monkeypatch.setattr(kg_server, "get_connection_registry", lambda: registry)
    monkeypatch.setattr(
        "agent_utilities.security.secrets_client.create_secrets_client",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "agent_utilities.mcp.tools.analysis_tools._configured_external_graph_declaration",
        lambda _name: {
            "ingest_page_size": 125,
            "ingest_max_pages": 8,
            "ingest_max_row_bytes": 4_096,
            "ingest_max_total_bytes": 32_768,
            "ingest_max_nesting_depth": 8,
            "ingest_max_collection_items": 2_000,
            "sync_mode": "snapshot",
            "reconcile_deletions": False,
            "allow_empty_snapshot": True,
        },
    )

    captured = {}

    def fake_ingest(engine, received_registry, request):
        captured.update(
            {"engine": engine, "registry": received_registry, "request": request}
        )
        return {
            "status": "dry_run",
            "source_alias": request.source_alias,
            "planned_nodes": 2,
            "planned_edges": 1,
        }

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_registered_graph",
        fake_ingest,
    )
    raw = await kg_server._execute_tool(
        "graph_configure",
        action="ingest_connection",
        config_key="external-catalog",
        config_value=json.dumps(
            {
                "source_alias": "business-graph",
                "dry_run": True,
            }
        ),
    )
    result = json.loads(raw)

    assert result == {
        "status": "dry_run",
        "source_alias": "business-graph",
        "planned_nodes": 2,
        "planned_edges": 1,
    }
    assert captured["engine"] is authority
    assert captured["registry"] is registry
    assert captured["request"].connection == "external-catalog"
    assert captured["request"].dry_run is True
    assert captured["request"].profile_ref == ""
    assert captured["request"].variables == {}
    assert len(captured["request"].runtime_policy_digest) == 64
    assert captured["request"].page_size == 125
    assert captured["request"].max_pages == 8
    assert captured["request"].max_row_bytes == 4_096
    assert captured["request"].max_total_bytes == 32_768
    assert captured["request"].max_nesting_depth == 8
    assert captured["request"].max_collection_items == 2_000
    assert captured["request"].sync_mode == "snapshot"
    assert captured["request"].reconcile_deletions is False
    assert captured["request"].allow_empty_snapshot is True


def test_property_graph_sync_policy_is_bound_into_mapping_approval() -> None:
    from agent_utilities.mcp.tools.analysis_tools import (
        _resolved_external_mapping_policy,
    )

    store = MagicMock()
    first_policy, first_digest = _resolved_external_mapping_policy(
        store,
        {
            "ingest_page_size": 250,
            "ingest_max_pages": 4,
            "sync_mode": "snapshot",
            "reconcile_deletions": True,
            "allow_empty_snapshot": False,
        },
    )
    second_policy, second_digest = _resolved_external_mapping_policy(
        store,
        {
            "ingest_page_size": 250,
            "ingest_max_pages": 4,
            "sync_mode": "snapshot",
            "reconcile_deletions": False,
            "allow_empty_snapshot": False,
        },
    )

    assert first_policy == second_policy == {}
    assert first_digest != second_digest
    store.resolve_ref.assert_not_called()


@pytest.mark.asyncio
async def test_property_graph_semantic_mapping_uses_verified_governed_enricher(
    registered_tools, monkeypatch
) -> None:
    from agent_utilities.knowledge_graph.ingestion import external_graph_schema
    from agent_utilities.mcp import kg_server

    external_source = MagicMock()
    authority = MagicMock()
    registry = MagicMock()
    registry.backend_kind.return_value = "neo4j"
    registry.get_engine.side_effect = lambda name: (
        external_source if name == "external-catalog" else authority
    )
    monkeypatch.setattr(kg_server, "get_connection_registry", lambda: registry)
    monkeypatch.setattr(
        "agent_utilities.mcp.tools.analysis_tools._configured_external_graph_declaration",
        lambda _name: {
            "source_alias": "business-graph",
            "semantic_mapping": True,
        },
    )
    monkeypatch.setattr(
        "agent_utilities.security.secrets_client.create_secrets_client",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.connection_profiler._our_ontology_vocabulary",
        lambda _authority, _scope: ["Service", "Document"],
    )
    verified_session = object()
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.session.resolve_session",
        lambda **_kwargs: verified_session,
    )
    semantic_enricher = object()
    monkeypatch.setattr(
        external_graph_schema,
        "governed_semantic_mapping_enricher",
        semantic_enricher,
    )
    captured: dict[str, object] = {}

    def fake_proposal(engine, **kwargs):
        captured.update(engine=engine, **kwargs)
        return {"status": "proposed"}

    monkeypatch.setattr(
        external_graph_schema,
        "propose_mapping_profile",
        fake_proposal,
    )

    raw = await kg_server._execute_tool(
        "graph_configure",
        action="propose_connection_mapping",
        config_key="external-catalog",
        config_value="{}",
    )

    assert json.loads(raw) == {"status": "proposed"}
    assert captured["engine"] is external_source
    assert captured["semantic_enricher"] is semantic_enricher
    assert captured["context_session"] is verified_session


@pytest.mark.asyncio
@pytest.mark.concept("AU-KG.ingest.external-graph-federation")
async def test_graph_configure_rejects_inline_endpoint_or_query(
    registered_tools,
) -> None:
    from agent_utilities.mcp import kg_server

    raw = await kg_server._execute_tool(
        "graph_configure",
        action="ingest_connection",
        config_key="external-catalog",
        config_value=json.dumps(
            {
                "source_alias": "business-graph",
                "endpoint": "https://private.example.test/graphql",
                "node_query": "MATCH (n) RETURN n LIMIT $limit",
            }
        ),
    )
    result = json.loads(raw)

    assert "error" in result
    assert "private.example.test" not in raw
    assert "MATCH" not in raw


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    [
        "custom_ontology",
        "email",
        "local_path",
        "profile",
        "sync_mode",
        "variables",
    ],
)
async def test_graph_configure_rejects_all_inline_source_material(
    registered_tools, monkeypatch, field
) -> None:
    from agent_utilities.mcp import kg_server

    registry = MagicMock()
    registry.backend_kind.return_value = "neo4j"
    monkeypatch.setattr(kg_server, "get_connection_registry", lambda: registry)

    raw = await kg_server._execute_tool(
        "graph_configure",
        action="ingest_connection",
        config_key="external-catalog",
        config_value=json.dumps({field: "must-not-enter-the-contract"}),
    )

    assert "error" in json.loads(raw)
    assert "must-not-enter-the-contract" not in raw


@pytest.mark.asyncio
@pytest.mark.concept("AU-KG.ingest.external-graph-federation")
async def test_graph_configure_routes_graphql_document_through_native_lifecycle(
    registered_tools, monkeypatch
) -> None:
    from agent_utilities.knowledge_graph.ingestion.graphql_connection import (
        GraphQLSourceAdapter,
    )
    from agent_utilities.mcp import kg_server

    source = GraphQLSourceAdapter(
        connection="external-source",
        source_alias="external-source",
        connection_profile_ref="secret://source/connection",
        mapping_policy_ref="secret://source/policy",
        resolver=lambda _ref: None,
    )
    authority = MagicMock()
    registry = MagicMock()
    registry.backend_kind.return_value = "graphql"
    registry.get_engine.side_effect = lambda name: (
        source if name == "external-source" else authority
    )
    monkeypatch.setattr(kg_server, "get_connection_registry", lambda: registry)
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.graphql_connection.ingest_registered_graphql",
        lambda _authority, _source, **kwargs: {
            "status": "dry_run",
            "connection": kwargs["connection"],
            "source_alias": _source.source_alias,
            "nodes_created": 0,
            "edges_created": 0,
        },
    )
    monkeypatch.setattr(
        "agent_utilities.security.secrets_client.create_secrets_client",
        lambda: MagicMock(),
    )

    raw = await kg_server._execute_tool(
        "graph_configure",
        action="ingest_connection",
        config_key="external-source",
        config_value=json.dumps(
            {
                "operation": "document_read",
                "variables_ref": "secret://source/variables",
                "dry_run": True,
            }
        ),
    )
    result = json.loads(raw)

    assert result["status"] == "dry_run"
    assert result["connection"] == "external-source"
    assert "secret://" not in raw


@pytest.mark.asyncio
async def test_graph_configure_graphql_rejects_inline_variables(
    registered_tools, monkeypatch
) -> None:
    from agent_utilities.mcp import kg_server

    registry = MagicMock()
    registry.backend_kind.return_value = "graphql"
    monkeypatch.setattr(kg_server, "get_connection_registry", lambda: registry)
    raw = await kg_server._execute_tool(
        "graph_configure",
        action="ingest_connection",
        config_key="external-source",
        config_value=json.dumps({"variables": {"identity": "runtime-value"}}),
    )

    assert "error" in json.loads(raw)
    assert "runtime-value" not in raw


@pytest.mark.asyncio
async def test_graph_configure_connection_key_cannot_be_overridden_in_payload(
    registered_tools, monkeypatch
) -> None:
    from agent_utilities.mcp import kg_server

    registry = MagicMock()
    monkeypatch.setattr(kg_server, "get_connection_registry", lambda: registry)
    raw = await kg_server._execute_tool(
        "graph_configure",
        action="add_connection",
        config_key="external-source",
        config_value=json.dumps(
            {
                "name": "different-source",
                "backend": "graphql",
                "role": "read",
                "source_alias": "external-source",
                "connection_profile_ref": "secret://source/connection",
                "mapping_policy_ref": "secret://source/policy",
            }
        ),
    )

    assert "error" in json.loads(raw)
    registry.register.assert_not_called()


def test_graph_os_verbose_manifest_exposes_registered_graph_ingestion() -> None:
    from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS

    actions = {
        row["action"] for row in GRAPHOS_ACTIONS if row["tool"] == "graph_configure"
    }
    assert {
        "approve_connection_mapping",
        "connection_mapping_status",
        "discover_connection_schema",
        "external_graph_doctor",
        "ingest_connection",
        "propose_connection_mapping",
    } <= actions
