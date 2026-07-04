"""Tests for the database environment provisioner (Stardog + pg-age wiring).

Covers each composed step in isolation plus a live-path test exercising the
``graph_configure`` MCP action (``setup_databases`` / ``verify_databases``), which
is what the CLI and the ``database-environment-setup`` skill drive.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph import setup as dbsetup
from agent_utilities.knowledge_graph.setup import database_environment as de


class _MockMCP:
    def __init__(self):
        self.funcs = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.funcs[fn.__name__] = fn
            return fn

        return deco

    def custom_route(self, *a, **k):
        def deco(fn):
            self.funcs[fn.__name__] = fn
            return fn

        return deco


@pytest.fixture
def registered_tools():
    """Build the KG server so the real graph_configure tool is registered."""
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

            _build_server()
    return mock_mcp.funcs


# ── verify_postgres ────────────────────────────────────────────────────────
class _FakePG:
    def __init__(self, *, age=True, vector=True, search=True, **kw):
        self._a, self._v, self._s = age, vector, search

    pggraph_available = property(lambda self: self._a)
    pgvector_available = property(lambda self: self._v)
    paradedb_available = property(lambda self: self._s)


def _patch_pg(monkeypatch, **flags):
    import agent_utilities.knowledge_graph.backends.postgresql_backend as pgmod

    monkeypatch.setattr(
        pgmod, "PostgreSQLBackend", lambda dsn=None, **kw: _FakePG(**flags)
    )


def test_verify_postgres_all_present(monkeypatch):
    _patch_pg(monkeypatch, age=True, vector=True, search=True)
    out = de.verify_postgres("postgresql://agent:secret@h:5432/db")
    assert out["status"] == "success"
    assert out["ready"] is True
    assert out["missing"] == []
    # Password must be redacted in the echoed DSN.
    assert "secret" not in out["dsn"]
    assert "***" in out["dsn"]


def test_verify_postgres_reports_missing(monkeypatch):
    _patch_pg(monkeypatch, age=False, vector=True, search=False)
    out = de.verify_postgres("postgresql://h/db")
    assert out["ready"] is False
    assert set(out["missing"]) == {"age", "pg_search"}
    assert "hint" in out


def test_verify_postgres_connection_error(monkeypatch):
    import agent_utilities.knowledge_graph.backends.postgresql_backend as pgmod

    def _boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(pgmod, "PostgreSQLBackend", _boom)
    out = de.verify_postgres("postgresql://h/db")
    assert out["status"] == "error"
    assert "connection refused" in out["error"]


# ── configure_backend ──────────────────────────────────────────────────────
def test_configure_backend_persists_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path))
    # Avoid touching the real active backend registry.
    import agent_utilities.knowledge_graph.backends as backends_mod

    monkeypatch.setattr(backends_mod, "set_active_backend", lambda b: None)

    out = de.configure_backend("postgresql://agent:pw@h:5432/agent_kg")
    assert out["status"] == "success"
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["GRAPH_DB_URI"] == "postgresql://agent:pw@h:5432/agent_kg"
    assert cfg["GRAPH_PG_AGE"] == "1"
    assert cfg["GRAPH_BACKEND"] == "fanout"
    assert json.loads(cfg["GRAPH_MIRROR_TARGETS"]) == ["age"]
    # Live process env is set too (cross-process signalling).
    import os

    assert os.environ["GRAPH_PG_AGE"] == "1"
    # The reported DSN is redacted.
    assert "pw" not in out["applied"]["GRAPH_DB_URI"]


def test_configure_backend_mirror_targets(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path))
    import agent_utilities.knowledge_graph.backends as backends_mod

    monkeypatch.setattr(backends_mod, "set_active_backend", lambda b: None)
    de.configure_backend("postgresql://h/db", mirror_targets=["neo4j", "falkordb"])
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert json.loads(cfg["GRAPH_MIRROR_TARGETS"]) == ["neo4j", "falkordb"]


# ── publish_ontology ───────────────────────────────────────────────────────
def test_publish_ontology_builtin(monkeypatch):
    import agent_utilities.knowledge_graph.core.ontology_publisher as op

    monkeypatch.setattr(op, "collect_bundled_ontology_graph", lambda: [1, 2, 3])
    out = de.publish_ontology("builtin")
    assert out["status"] == "success"
    assert out["target"] == "builtin"
    assert out["triple_count"] == 3


def test_publish_ontology_stardog_delegates(monkeypatch):
    import agent_utilities.knowledge_graph.core.ontology_publisher as op

    monkeypatch.setattr(op, "collect_bundled_ontology_graph", lambda: [1])
    captured = {}

    def _push(self, graph, endpoint=None, database=None, named_graph=None):
        captured["endpoint"] = endpoint
        return {"status": "success", "triple_count": len(graph)}

    monkeypatch.setattr(op.OntologyPublisher, "push_to_stardog", _push)
    out = de.publish_ontology("stardog", endpoint="http://sd:5820", database="kg")
    assert out["status"] == "success"
    assert out["target"] == "stardog"
    assert captured["endpoint"] == "http://sd:5820"


# ── backfill_to_age ────────────────────────────────────────────────────────
class _FakeTiered:
    def reconcile_to_durable(self):
        return {"nodes": 5, "edges": 4, "nodes_missing": 0, "edges_missing": 0}

    def durability_stats(self):
        return {"l3_writes": 9, "l3_failures": 0}


def test_backfill_to_age_consistent(monkeypatch):
    import agent_utilities.knowledge_graph.backends as backends_mod

    monkeypatch.setattr(backends_mod, "get_active_backend", lambda: _FakeTiered())
    out = de.backfill_to_age()
    assert out["status"] == "success"
    assert out["consistent"] is True
    assert out["reconcile"]["nodes"] == 5
    assert out["durability"]["l3_writes"] == 9


def test_backfill_to_age_no_backend(monkeypatch):
    import agent_utilities.knowledge_graph.backends as backends_mod

    monkeypatch.setattr(backends_mod, "get_active_backend", lambda: object())
    monkeypatch.setattr(backends_mod, "create_backend", lambda *a, **k: object())
    monkeypatch.setattr(backends_mod, "set_active_backend", lambda b: None)
    out = de.backfill_to_age()
    assert out["status"] == "error"


# ── verify_sparql ──────────────────────────────────────────────────────────
class _FakeBridge:
    # SPARQLEndpoint.execute now dispatches via query_sparql (CONCEPT:AU-KG.compute.native-sparql-owl-shacl —
    # engine-native first, rdflib only as a no-engine last resort).
    def query_sparql(self, query):
        return [{"s": "http://example.org/Agent"}]

    def _sparql_via_rdflib(self, query):
        return [{"s": "http://example.org/Agent"}]


def test_verify_sparql_builtin(monkeypatch):
    import agent_utilities.gateway.graph_api as gapi

    monkeypatch.setattr(gapi, "_get_sparql_bridge", lambda: _FakeBridge())
    out = de.verify_sparql("builtin")
    assert out["status"] == "success"
    assert out["rows"] == 1
    assert out["url"] == "/api/sparql"


def test_verify_sparql_builtin_no_bridge(monkeypatch):
    import agent_utilities.gateway.graph_api as gapi

    monkeypatch.setattr(gapi, "_get_sparql_bridge", lambda: None)
    out = de.verify_sparql("builtin")
    assert out["status"] == "error"


# ── setup_environment (orchestration) ──────────────────────────────────────
def test_setup_environment_dev_partial_on_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        de,
        "verify_postgres",
        lambda dsn: {
            "status": "success",
            "extensions": {"age": False},
            "missing": ["age", "pg_search"],
        },
    )
    monkeypatch.setattr(de, "configure_backend", lambda *a, **k: {"status": "success"})
    monkeypatch.setattr(de, "publish_ontology", lambda *a, **k: {"status": "success"})
    monkeypatch.setattr(
        de, "backfill_to_age", lambda: {"status": "error", "error": "no age"}
    )
    monkeypatch.setattr(de, "verify_sparql", lambda *a, **k: {"status": "success"})

    out = de.setup_environment(profile="dev", postgres_mode="existing")
    assert out["sparql_target"] == "builtin"
    assert out["status"] == "partial"  # backfill failed
    assert "warnings" in out  # missing extensions surfaced


def test_setup_environment_prod_targets_stardog(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path))
    for name in (
        "verify_postgres",
        "configure_backend",
        "publish_ontology",
        "verify_sparql",
    ):
        monkeypatch.setattr(
            de, name, lambda *a, **k: {"status": "success", "extensions": {"age": True}}
        )
    monkeypatch.setattr(de, "backfill_to_age", lambda: {"status": "success"})
    out = de.setup_environment(profile="prod")
    assert out["sparql_target"] == "stardog"
    assert out["status"] == "success"


# ── live path: graph_configure MCP action ──────────────────────────────────
@pytest.mark.asyncio
async def test_graph_configure_verify_databases_live_path(
    monkeypatch, registered_tools
):
    from agent_utilities.mcp import kg_server

    monkeypatch.setattr(
        dbsetup,
        "verify_postgres",
        lambda dsn: {"status": "success", "dsn": dsn, "ready": True},
    )
    raw = await kg_server._execute_tool(
        "graph_configure",
        action="verify_databases",
        config_value=json.dumps({"dsn": "postgresql://h/db"}),
    )
    out = json.loads(raw)
    assert out["status"] == "success"
    assert out["ready"] is True


@pytest.mark.asyncio
async def test_graph_configure_setup_databases_live_path(monkeypatch, registered_tools):
    from agent_utilities.mcp import kg_server

    seen = {}

    def _fake_setup(**kwargs):
        seen.update(kwargs)
        return {"status": "success", "profile": kwargs.get("profile")}

    monkeypatch.setattr(dbsetup, "setup_environment", _fake_setup)
    raw = await kg_server._execute_tool(
        "graph_configure",
        action="setup_databases",
        config_key="prod",
        config_value=json.dumps(
            {"postgres_mode": "existing", "dsn": "postgresql://h/db"}
        ),
    )
    out = json.loads(raw)
    assert out["status"] == "success"
    assert seen["profile"] == "prod"
    assert seen["postgres_mode"] == "existing"
