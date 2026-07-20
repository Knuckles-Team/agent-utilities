"""Tests for native database traversal tools (CONCEPT:AU-ECO.toolkit.database-traversal-tools).

Offline + deterministic: exercised against a temp SQLite DB (UniversalConnector
needs no driver for sqlite), so no external database is required.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from agent_utilities.tools.db_tools import db_query, db_schema, db_tables, db_tools


@pytest.fixture
def sqlite_dsn(tmp_path):
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE users(id INTEGER, name TEXT);"
        "CREATE TABLE orders(id INTEGER, user_id INTEGER, total REAL);"
        "INSERT INTO users VALUES (1,'alice'),(2,'bob');"
    )
    con.commit()
    con.close()
    return f"sqlite:///{db}"


@pytest.fixture
def sqlite_ref(sqlite_dsn, monkeypatch):
    from agent_utilities.security import secrets_client

    class _Secrets:
        def resolve_ref(self, reference):
            assert reference == "env://TEST_DATABASE_DSN"
            return sqlite_dsn

    monkeypatch.setattr(secrets_client, "create_secrets_client", lambda: _Secrets())
    return "env://TEST_DATABASE_DSN"


@pytest.mark.concept("AU-ECO.toolkit.database-traversal-tools")
def test_db_tools_registered():
    assert {t.__name__ for t in db_tools} == {"db_tables", "db_schema", "db_query"}


@pytest.mark.concept("AU-ECO.toolkit.database-traversal-tools")
def test_db_tables(sqlite_ref):
    out = json.loads(asyncio.run(db_tables(None, sqlite_ref)))
    assert set(out["tables"]) == {"users", "orders"}


@pytest.mark.concept("AU-ECO.toolkit.database-traversal-tools")
def test_db_schema(sqlite_ref):
    out = json.loads(asyncio.run(db_schema(None, sqlite_ref)))
    assert out["schema"]["users"] == ["id", "name"]
    assert set(out["schema"]["orders"]) == {"id", "user_id", "total"}


@pytest.mark.concept("AU-ECO.toolkit.database-traversal-tools")
def test_db_query_read(sqlite_ref):
    out = json.loads(asyncio.run(db_query(None, sqlite_ref, "SELECT * FROM users")))
    assert out["row_count"] == 2
    assert {r["name"] for r in out["rows"]} == {"alice", "bob"}


@pytest.mark.concept("AU-ECO.toolkit.database-traversal-tools")
def test_db_query_blocks_writes(sqlite_ref):
    for stmt in ("DELETE FROM users", "DROP TABLE users", "UPDATE users SET name='x'"):
        out = json.loads(asyncio.run(db_query(None, sqlite_ref, stmt)))
        assert out["error"] == "interactive database tools are read-only"


@pytest.mark.concept("AU-ECO.toolkit.database-traversal-tools")
def test_literal_connection_is_rejected(sqlite_dsn):
    out = json.loads(asyncio.run(db_tables(None, sqlite_dsn)))
    assert out["error"] == "database introspection failed"


@pytest.mark.concept("AU-ECO.toolkit.database-traversal-tools")
def test_empty_query_rejected(sqlite_ref):
    out = json.loads(asyncio.run(db_query(None, sqlite_ref, "   ")))
    assert out["error"] == "empty query"
