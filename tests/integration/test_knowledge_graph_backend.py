from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.backends import create_backend
from agent_utilities.knowledge_graph.backends.base import GraphBackend
from agent_utilities.knowledge_graph.backends.falkordb_backend import FalkorDBBackend
from agent_utilities.knowledge_graph.backends.neo4j_backend import Neo4jBackend

try:
    from agent_utilities.knowledge_graph.backends.ladybug_backend import (
        LADYBUG_AVAILABLE,
        LadybugBackend,
    )
except ImportError:
    LADYBUG_AVAILABLE = False


class TestFalkorDBBackend:
    def test_initialization(self):
        backend = FalkorDBBackend(host="localhost", port=6379, db_name="test_graph")
        assert backend.db_name == "test_graph"

    def test_execute_stub(self):
        backend = FalkorDBBackend()
        res = backend.execute("MATCH (n) RETURN n")
        assert res == []


class TestNeo4jBackend:
    def test_initialization(self):
        backend = Neo4jBackend(uri="bolt://localhost:7687")
        assert backend is not None

    def test_execute_stub(self):
        backend = Neo4jBackend()
        res = backend.execute("MATCH (n) RETURN n")
        assert res == []


@pytest.mark.skipif(not LADYBUG_AVAILABLE, reason="LadybugDB not available")
class TestLadybugBackend:
    @patch("agent_utilities.knowledge_graph.backends.ladybug_backend.ladybug.Database")
    @patch(
        "agent_utilities.knowledge_graph.backends.ladybug_backend.ladybug.Connection"
    )
    def test_initialization(self, mock_conn, mock_db):
        LadybugBackend("test.db")
        mock_db.assert_called_with("test.db")
        mock_conn.assert_called_once()

    @patch("agent_utilities.knowledge_graph.backends.ladybug_backend.ladybug.Database")
    @patch(
        "agent_utilities.knowledge_graph.backends.ladybug_backend.ladybug.Connection"
    )
    def test_execute(self, mock_conn, mock_db):
        mock_conn_instance = mock_conn.return_value

        # Mock the rows_as_dict().get_all() chain used by the actual backend
        mock_result = MagicMock()
        mock_rows = MagicMock()
        mock_rows.get_all.return_value = [{"id": "1"}, {"id": "2"}]
        mock_result.rows_as_dict.return_value = mock_rows
        mock_conn_instance.execute.return_value = mock_result

        backend = LadybugBackend("test.db")
        res = backend.execute("MATCH (n) RETURN n")
        assert len(res) == 2
        assert res[0]["id"] == "1"


class TestBackendFactory:
    """Test the create_backend() factory function."""

    @pytest.mark.skipif(not LADYBUG_AVAILABLE, reason="LadybugDB not available")
    def test_default_creates_ladybug(self, tmp_path):
        """Default backend type should be LadybugDB."""
        db_file = tmp_path / "test_factory.db"
        backend = create_backend(db_path=str(db_file))
        assert isinstance(backend, LadybugBackend)

    def test_explicit_falkordb(self):
        """Explicitly requesting FalkorDB should return FalkorDBBackend."""
        backend = create_backend(backend_type="falkordb")
        assert isinstance(backend, FalkorDBBackend)

    def test_explicit_neo4j(self):
        """Explicitly requesting Neo4j should return Neo4jBackend."""
        backend = create_backend(backend_type="neo4j")
        assert isinstance(backend, Neo4jBackend)

    def test_neo4j_with_params(self):
        """Neo4j backend should accept custom connection parameters."""
        backend = create_backend(
            backend_type="neo4j",
            uri="bolt://custom-host:7688",
            user="admin",
            password="secret",
        )
        assert isinstance(backend, Neo4jBackend)

    def test_falkordb_with_params(self):
        """FalkorDB backend should accept custom connection parameters."""
        backend = create_backend(
            backend_type="falkordb",
            host="redis-host",
            port=6380,
            db_name="custom_graph",
        )
        assert isinstance(backend, FalkorDBBackend)
        assert backend.db_name == "custom_graph"

    def test_unknown_backend_returns_none(self):
        """Unknown backend type should return None."""
        backend = create_backend(backend_type="nonexistent_db")
        assert backend is None

    def test_env_var_override(self, monkeypatch):
        """GRAPH_BACKEND env var should override the default."""
        monkeypatch.setenv("GRAPH_BACKEND", "falkordb")
        backend = create_backend()
        assert isinstance(backend, FalkorDBBackend)

    def test_env_var_db_path(self, monkeypatch, tmp_path):
        """GRAPH_DB_PATH env var should be used for LadybugDB."""
        if not LADYBUG_AVAILABLE:
            pytest.skip("LadybugDB not available")
        db_file = tmp_path / "custom_path.db"
        monkeypatch.setenv("GRAPH_DB_PATH", str(db_file))
        backend = create_backend(backend_type="ladybug")
        assert isinstance(backend, LadybugBackend)
        assert backend.db_path == str(db_file)

    def test_case_insensitive(self):
        """Backend type should be case-insensitive."""
        backend = create_backend(backend_type="FalkorDB")
        assert isinstance(backend, FalkorDBBackend)

    def test_all_backends_implement_abc(self):
        """All backends must implement the GraphBackend ABC."""
        assert issubclass(FalkorDBBackend, GraphBackend)
        assert issubclass(Neo4jBackend, GraphBackend)
        if LADYBUG_AVAILABLE:
            assert issubclass(LadybugBackend, GraphBackend)
