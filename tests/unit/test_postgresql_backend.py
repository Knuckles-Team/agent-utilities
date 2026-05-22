"""Unit tests for the PostgreSQL + pgGraph backend (mocked — no database required)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.backends.postgresql_backend import (
    PostgreSQLBackend,
    _SingleConnPool,
)


@pytest.fixture
def mock_backend():
    """Create a PostgreSQLBackend with mocked pool."""
    backend = PostgreSQLBackend.__new__(PostgreSQLBackend)
    backend._dsn = "postgresql://test:test@localhost:5432/test"
    backend._graph_name = "test_graph"
    backend._pool_min = 1
    backend._pool_max = 2
    backend._pggraph_schema = "public"
    backend._known_tables = {"Agent", "Tool", "Memory"}
    backend._pggraph_available = False
    backend._pgvector_available = False
    backend._paradedb_available = False

    # Mock connection and cursor
    mock_cur = MagicMock()
    mock_cur.description = [
        MagicMock(name="id"),
        MagicMock(name="name"),
        MagicMock(name="type"),
    ]
    # Set .name attribute on description mocks
    mock_cur.description[0].name = "id"
    mock_cur.description[1].name = "name"
    mock_cur.description[2].name = "type"
    mock_cur.fetchall.return_value = [("agent-1", "TestAgent", "agent")]
    mock_cur.fetchone.return_value = ("agent-1", "TestAgent", "agent")
    mock_cur.rowcount = 1

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    mock_pool = _SingleConnPool(mock_conn)
    backend._pool = mock_pool

    return backend, mock_cur, mock_conn


class TestExecute:
    def test_select_by_id(self, mock_backend):
        backend, mock_cur, _ = mock_backend
        result = backend.execute(
            "MATCH (n:Agent) WHERE n.id = $id RETURN n",
            {"id": "agent-1"},
        )
        assert mock_cur.execute.called
        assert isinstance(result, list)

    def test_create_node(self, mock_backend):
        backend, mock_cur, _ = mock_backend
        mock_cur.description = None
        mock_cur.fetchall.return_value = []
        result = backend.execute(
            "CREATE (n:Agent {id: $id, name: $name})",
            {"id": "agent-2", "name": "New"},
        )
        assert mock_cur.execute.called
        assert isinstance(result, list)

    def test_unknown_pattern_returns_empty(self, mock_backend):
        backend, _, _ = mock_backend
        result = backend.execute("CALL something()")
        assert result == []

    def test_internal_params_excluded(self, mock_backend):
        backend, mock_cur, _ = mock_backend
        backend.execute(
            "MATCH (n:Agent) WHERE n.id = $id RETURN n",
            {"id": "a1", "_clearance_level": 999},
        )
        # Verify the SQL call didn't include clearance level
        call_args = mock_cur.execute.call_args
        if call_args:
            sql_params = call_args[0][1] if len(call_args[0]) > 1 else []
            assert 999 not in sql_params


class TestExecuteBatch:
    def test_batch_insert(self, mock_backend):
        backend, mock_cur, _ = mock_backend
        mock_cur.description = None
        batch = [{"id": f"agent-{i}", "name": f"Agent{i}"} for i in range(3)]
        result = backend.execute_batch("CREATE (n:Agent {id: $id, name: $name})", batch)
        assert isinstance(result, list)
        assert mock_cur.execute.call_count >= 3

    def test_empty_batch(self, mock_backend):
        backend, _, _ = mock_backend
        result = backend.execute_batch("CREATE (n:Agent {id: $id})", [])
        assert result == []


class TestSemanticSearch:
    def test_no_pgvector_returns_empty(self, mock_backend):
        backend, _, _ = mock_backend
        backend._pgvector_available = False
        result = backend.semantic_search([0.1] * 768)
        assert result == []


class TestAddEmbedding:
    def test_no_pgvector_skips(self, mock_backend):
        backend, mock_cur, _ = mock_backend
        backend._pgvector_available = False
        backend.add_embedding("agent-1", [0.1] * 768)
        # Should not execute any SQL
        assert not mock_cur.execute.called


class TestPrune:
    def test_prune_by_importance(self, mock_backend):
        backend, mock_cur, _ = mock_backend
        backend.prune({"min_importance": 0.3})
        assert mock_cur.execute.called


class TestHealthCheck:
    def test_healthy(self, mock_backend):
        backend, _, _ = mock_backend
        assert backend.health_check() is True

    def test_unhealthy(self, mock_backend):
        backend, _, mock_conn = mock_backend
        mock_conn.cursor.side_effect = Exception("connection refused")
        assert backend.health_check() is False


class TestGetStats:
    def test_stats_structure(self, mock_backend):
        backend, mock_cur, _ = mock_backend
        mock_cur.fetchone.return_value = (42,)
        stats = backend.get_stats()
        assert stats["backend"] == "postgresql"
        assert "tables" in stats
        assert stats["pggraph"] is False


class TestPgGraphOperations:
    def test_traverse_without_pggraph(self, mock_backend):
        backend, _, _ = mock_backend
        backend._pggraph_available = False
        result = backend.graph_traverse("Agent", "a1")
        assert result == []

    def test_shortest_path_without_pggraph(self, mock_backend):
        backend, _, _ = mock_backend
        backend._pggraph_available = False
        result = backend.graph_shortest_path("Agent", "a1", "Tool", "t1")
        assert result == []

    def test_graph_build_without_pggraph(self, mock_backend):
        backend, _, _ = mock_backend
        backend._pggraph_available = False
        result = backend.graph_build()
        assert result["status"] == "skipped"

    def test_graph_status_without_pggraph(self, mock_backend):
        backend, _, _ = mock_backend
        backend._pggraph_available = False
        result = backend.graph_status()
        assert result["available"] is False


class TestClose:
    def test_close_pool(self, mock_backend):
        backend, _, mock_conn = mock_backend
        backend.close()
        assert backend._pool is None


class TestSingleConnPool:
    def test_shim_close(self):
        mock_conn = MagicMock()
        pool = _SingleConnPool(mock_conn)
        pool.close()
        mock_conn.close.assert_called_once()
