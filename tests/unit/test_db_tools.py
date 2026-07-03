"""Tests for native database traversal tools (CONCEPT:ECO-4.33).

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


@pytest.mark.concept("ECO-4.33")
def test_db_tools_registered():
    assert {t.__name__ for t in db_tools} == {"db_tables", "db_schema", "db_query"}


@pytest.mark.concept("ECO-4.33")
def test_db_tables(sqlite_dsn):
    out = json.loads(asyncio.run(db_tables(None, sqlite_dsn)))
    assert set(out["tables"]) == {"users", "orders"}


@pytest.mark.concept("ECO-4.33")
def test_db_schema(sqlite_dsn):
    out = json.loads(asyncio.run(db_schema(None, sqlite_dsn)))
    assert out["schema"]["users"] == ["id", "name"]
    assert set(out["schema"]["orders"]) == {"id", "user_id", "total"}


@pytest.mark.concept("ECO-4.33")
def test_db_query_read(sqlite_dsn):
    out = json.loads(asyncio.run(db_query(None, sqlite_dsn, "SELECT * FROM users")))
    assert out["row_count"] == 2
    assert {r["name"] for r in out["rows"]} == {"alice", "bob"}


@pytest.mark.concept("ECO-4.33")
def test_db_query_blocks_writes_by_default(sqlite_dsn):
    for stmt in ("DELETE FROM users", "DROP TABLE users", "UPDATE users SET name='x'"):
        out = json.loads(asyncio.run(db_query(None, sqlite_dsn, stmt)))
        assert "blocked" in out.get("error", "")


@pytest.mark.concept("ECO-4.33")
def test_db_query_allows_write_when_opted_in(sqlite_dsn, monkeypatch):
    monkeypatch.setenv("DB_TOOLS_ALLOW_WRITE", "1")
    out = json.loads(
        asyncio.run(db_query(None, sqlite_dsn, "DELETE FROM users WHERE id=1"))
    )
    assert "error" not in out or "blocked" not in out.get("error", "")
    remaining = json.loads(
        asyncio.run(db_query(None, sqlite_dsn, "SELECT * FROM users"))
    )
    assert remaining["row_count"] == 1


@pytest.mark.concept("ECO-4.33")
def test_dsn_alias_resolves_from_env(monkeypatch, sqlite_dsn):
    monkeypatch.setenv("WAREHOUSE_DSN", sqlite_dsn)
    out = json.loads(asyncio.run(db_tables(None, "warehouse")))
    assert set(out["tables"]) == {"users", "orders"}


@pytest.mark.concept("ECO-4.33")
def test_empty_query_rejected(sqlite_dsn):
    out = json.loads(asyncio.run(db_query(None, sqlite_dsn, "   ")))
    assert out["error"] == "empty query"
