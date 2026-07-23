"""Universal multi-DB / GraphQL DataConnector tests (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Uses **sqlite3** (stdlib, always available) as the concrete backend so the
bounded read/introspection surface is exercised with no external services.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.enrichment.registry import write_batch
from agent_utilities.protocols.universal_connector import (
    UniversalConnector,
    infer_kind,
)
from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


@pytest.fixture
def sqlite_db(tmp_path):
    """A tmp sqlite db with 2 tables + a foreign key, pre-seeded with rows."""
    import sqlite3

    db_path = tmp_path / "app.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE authors (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            author_id INTEGER,
            FOREIGN KEY (author_id) REFERENCES authors(id)
        );
        INSERT INTO authors(id, name) VALUES (1, 'Ada');
        INSERT INTO books(id, title, author_id) VALUES (1, 'Notes', 1);
        """
    )
    conn.commit()
    conn.close()
    return f"sqlite:///{db_path}"


# ---------------------------------------------------------------------- #
# DSN scheme → kind inference.
# ---------------------------------------------------------------------- #
def test_infer_kind_from_scheme():
    assert infer_kind("postgresql://u:p@host/db") == "postgresql"
    assert infer_kind("postgres://host/db") == "postgresql"
    assert infer_kind("mysql://host/db") == "mysql"
    assert infer_kind("mssql://host/db") == "mssql"
    assert infer_kind("oracle://host/db") == "oracle"
    assert infer_kind("sqlite:///foo.db") == "sqlite"
    assert infer_kind("mongodb://host/db") == "mongodb"
    with pytest.raises(ValueError):
        infer_kind("/tmp/local.db")


def test_graphql_uses_the_governed_source_adapter():
    with pytest.raises(ValueError):
        infer_kind("https://example.invalid/graphql")


def test_constructor_infers_kind(sqlite_db):
    conn = UniversalConnector(sqlite_db)
    assert conn.kind == "sqlite"
    assert conn.name.startswith("universal:sqlite:")
    assert conn.source_ref.startswith("pref_datasource_")
    assert sqlite_db not in conn.name


def test_unsupported_kind_raises():
    with pytest.raises(ValueError):
        UniversalConnector("sqlite:///x.db", kind="cassandra")


# ---------------------------------------------------------------------- #
# bounded reads and mutation denial against sqlite.
# ---------------------------------------------------------------------- #
def test_read_returns_rows(sqlite_db):
    conn = UniversalConnector(sqlite_db)
    rows = conn.read("SELECT id, name FROM authors ORDER BY id")
    assert rows == [{"id": 1, "name": "Ada"}]


def test_read_with_params(sqlite_db):
    conn = UniversalConnector(sqlite_db)
    rows = conn.read("SELECT name FROM authors WHERE id = ?", (1,))
    assert rows[0]["name"] == "Ada"


def test_mutation_methods_are_not_exposed(sqlite_db):
    conn = UniversalConnector(sqlite_db)
    assert not hasattr(conn, "write")
    assert not hasattr(conn, "update")


def test_read_enforces_row_cap(sqlite_db):
    conn = UniversalConnector(sqlite_db)
    assert len(conn.read("SELECT id FROM authors", max_rows=1)) == 1
    with pytest.raises(ValueError):
        conn.read("SELECT id FROM authors", max_rows=10_001)


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM authors",
        "SELECT 1; SELECT 2",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT * FROM authors FOR UPDATE",
    ],
)
def test_read_rejects_unsafe_sql(sqlite_db, query):
    with pytest.raises(PermissionError):
        UniversalConnector(sqlite_db).read(query)


def test_health_check_ok(sqlite_db):
    conn = UniversalConnector(sqlite_db)
    assert conn.health_check() is True


def test_health_check_bad_path():
    conn = UniversalConnector("sqlite:////nonexistent/dir/missing.db")
    # connecting to a path in a missing dir fails -> unhealthy
    assert conn.health_check() is False


# ---------------------------------------------------------------------- #
# Schema introspection → KG nodes/edges.
# ---------------------------------------------------------------------- #
def test_introspect_emits_kg_schema(sqlite_db):
    conn = UniversalConnector(sqlite_db)
    batch = conn.introspect()

    types = {n.type for n in batch.nodes}
    assert "DataSource" in types
    assert "Table" in types
    assert "Column" in types

    table_names = {n.props["name"] for n in batch.nodes if n.type == "Table"}
    assert {"authors", "books"} <= table_names

    rel_types = {e.rel_type for e in batch.edges}
    assert "HAS_TABLE" in rel_types
    assert "HAS_COLUMN" in rel_types
    assert "FOREIGN_KEY" in rel_types

    # FK edge: books.author_id -> authors.id; identifiers are opaque refs.
    nodes = {node.id: node for node in batch.nodes}
    fk_edges = [e for e in batch.edges if e.rel_type == "FOREIGN_KEY"]
    assert any(
        nodes[e.source].props.get("name") == "author_id"
        and nodes[e.target].props.get("name") == "id"
        for e in fk_edges
    )
    assert all(sqlite_db not in node.id for node in batch.nodes)


def test_introspect_persists_via_write_batch(sqlite_db):
    conn = UniversalConnector(sqlite_db)
    batch = conn.introspect()

    backend = FakeBackend()
    n, e = write_batch(backend, batch)

    assert n == len(batch.nodes)
    assert e == len(batch.edges)
    assert n >= 1 + 2 + 4  # DataSource + 2 tables + >=4 columns
    assert e >= 1  # at least one HAS_TABLE/HAS_COLUMN/FOREIGN_KEY edge

    # DataSource node persisted with correct type/props. ``write_batch`` maps
    # ``GraphNode.type`` onto the canonical ``node_type`` key — plain ``type``
    # is a retired key that ``write_entities`` rejects outright.
    ds_nodes = {
        nid: props
        for nid, props in backend.nodes.items()
        if props.get("node_type") == "DataSource"
    }
    assert len(ds_nodes) == 1
    (ds_props,) = ds_nodes.values()
    assert ds_props["kind"] == "sqlite"
    assert ds_props["source_ref"].startswith("pref_datasource_")

    # HAS_TABLE edges from the datasource were persisted. ``write_batch`` maps
    # ``EnrichmentEdge.rel_type`` onto the canonical ``relationship`` key (a
    # bare ``type`` key is a retired alias ``write_entities`` rejects), so
    # read it from the recorded row properties rather than the backend's
    # ``.edges`` tuple (whose third element is derived from the retired key).
    assert any(
        props.get("relationship") == "HAS_TABLE" for props in backend.edge_props
    )


# ---------------------------------------------------------------------- #
# Non-sqlite kinds: missing driver → clear RuntimeError.
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "dsn,kind,module",
    [
        ("postgresql://u@h/db", "postgresql", "psycopg"),
        ("mysql://u@h/db", "mysql", "pymysql"),
        ("oracle://u@h/db", "oracle", "oracledb"),
        ("mongodb://h/db", "mongodb", "pymongo"),
    ],
)
def test_missing_driver_raises_runtime_error(dsn, kind, module):
    import importlib.util

    if importlib.util.find_spec(module) is not None:
        pytest.skip(f"driver {module} is installed")
    conn = UniversalConnector(dsn, kind=kind)
    with pytest.raises(RuntimeError):
        conn._driver()


def test_postgres_connection_uses_agent_config_tls_profile(monkeypatch):
    trust = MagicMock()
    trust.psycopg_kwargs.return_value = {
        "sslmode": "verify-full",
        "sslrootcert": "/runtime/trust.pem",
    }
    driver = MagicMock()
    connector = UniversalConnector(
        "postgresql://runtime-user@db.example.test/domain",
        kind="postgresql",
        tls_profile_ref="secret://database/domain/tls",
    )
    monkeypatch.setattr(connector, "_driver", lambda: driver)
    monkeypatch.setattr(
        "agent_utilities.core.transport_security.resolve_configured_tls_profile",
        lambda *_args, **_kwargs: trust,
    )

    connector._connect()

    driver.connect.assert_called_once_with(
        connector.dsn,
        sslmode="verify-full",
        sslrootcert="/runtime/trust.pem",
    )
    trust.cleanup.assert_called_once()
