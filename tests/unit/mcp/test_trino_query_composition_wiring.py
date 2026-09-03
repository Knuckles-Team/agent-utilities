"""Real caller and import-direction gates for the Trino query slice."""

from __future__ import annotations

import ast
from pathlib import Path

from starlette.requests import Request

from agent_utilities.mcp import kg_server


class _Result:
    def __init__(self):
        self._done = False

    def keys(self):
        return ("id",)

    def fetchmany(self, _size):
        if self._done:
            return []
        self._done = True
        return [("row-1",)]


class _Connection:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, _statement):
        return _Result()


class _Engine:
    def __init__(self):
        self.disposed = False

    def connect(self):
        return _Connection()

    def dispose(self):
        self.disposed = True


def _registered_tabular_query():
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.query_tools import register_query_tools

    register_query_tools(FastMCP("trino-wiring-test"))
    return kg_server.REGISTERED_TOOLS["tabular_query"]


def test_registered_tabular_query_reaches_composed_service_and_injected_backend(
    monkeypatch,
):
    built = {}
    audiences = []

    def create_engine(url, **kwargs):
        built["url"] = url
        built["kwargs"] = kwargs
        built["engine"] = _Engine()
        return built["engine"]

    monkeypatch.setattr("sqlalchemy.create_engine", create_engine)
    monkeypatch.setattr(
        kg_server,
        "_trino_composition_settings",
        lambda: ("https://trino.apps.svc:8443", "lakehouse", "graph-services"),
    )
    monkeypatch.setattr(
        "agent_utilities.mcp.delegated_auth.get_delegated_token",
        lambda audience: audiences.append(audience) or "delegated-token",
    )
    monkeypatch.setattr(kg_server, "_trino_principal_ref", lambda: "principal-ref")
    monkeypatch.setattr(kg_server, "_TABULAR_QUERY_SERVICE", None)
    monkeypatch.setattr(kg_server, "_TABULAR_QUERY_SERVICE_KEY", None)

    result = _registered_tabular_query()(
        sql="SELECT id FROM analytics.agents"
    ).model_dump()

    assert result["claims"] == [{"id": "row-1"}]
    assert audiences == ["graph-services"]
    assert built["url"].host == "trino.apps.svc"
    assert built["url"].port == 8443
    assert built["url"].username == "principal-ref"
    assert built["url"].query["access_token"] == "delegated-token"
    assert built["kwargs"]["connect_args"] == {"http_scheme": "https"}


def _import_names(node):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [node.module or ""]
    return []


def _module_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        imports.extend(_import_names(node))
    return imports


def test_lower_tabular_layers_do_not_import_mcp_or_runtime_config():
    root = Path(__file__).parents[3]
    for relative in (
        "agent_utilities/knowledge_graph/backends/trino_backend.py",
        "agent_utilities/knowledge_graph/core/tabular_query_service.py",
    ):
        imports = _module_imports(root / relative)
        assert not any("agent_utilities.mcp" in name for name in imports)
        assert not any("core.config" in name for name in imports)


def test_rest_adapter_binds_and_resets_verified_bearer():
    from agent_utilities.mcp.delegated_auth import get_user_token
    from agent_utilities.security.brain_context import ActorContext, use_actor

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/query/tabular",
            "headers": [(b"authorization", b"Bearer verified-rest-token")],
        }
    )
    assert get_user_token() is None
    with use_actor(
        ActorContext(
            actor_id="caller-1",
            tenant_id="tenant-1",
            authenticated=True,
        )
    ):
        with kg_server._rest_tabular_delegated_identity(request):
            assert get_user_token() == "verified-rest-token"
    assert get_user_token() is None
