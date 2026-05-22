#!/usr/bin/python
"""CONCEPT:KG-2.0"""

from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.backends.ladybug_backend import LadybugBackend


@pytest.fixture
def temp_db_dir(tmp_path):
    db_dir = tmp_path / "test_db"
    db_dir.mkdir()
    return db_dir


def test_ladybug_backend_init_and_backup(temp_db_dir):
    db_path = str(temp_db_dir / "test.db")

    # Create a dummy db file
    with open(db_path, "w") as f:
        f.write("dummy data")

    # Mock ladybug.Database and connection
    with patch("ladybug.Database"), patch("ladybug.Connection"):
        backend = LadybugBackend(db_path)

        # Manually trigger backup
        backend._backup_db()

        # Check if backup exists
        backups = list(temp_db_dir.glob("test.db.*.bak"))
        assert len(backups) == 1


def test_ladybug_backend_backup_rotation(temp_db_dir):
    db_path = str(temp_db_dir / "test.db")
    with open(db_path, "w") as f:
        f.write("dummy data")

    with (
        patch("ladybug.Database"),
        patch("ladybug.Connection"),
        patch("agent_utilities.core.config.DEFAULT_KG_BACKUPS", 2),
    ):
        backend = LadybugBackend(db_path)

        # Trigger backup 3 times
        import time

        backend._backup_db()
        time.sleep(1.1)
        backend._backup_db()
        time.sleep(1.1)
        backend._backup_db()

        # Should only keep 2 backups
        backups = list(temp_db_dir.glob("test.db.*.bak"))
        assert len(backups) == 2


def test_ladybug_backend_cleanup_corrupted(temp_db_dir):
    db_path = temp_db_dir / "corrupt.db"
    wal_path = temp_db_dir / "corrupt.db.wal"
    shm_path = temp_db_dir / "corrupt.db.shm"

    db_path.write_text("corrupt data")
    wal_path.write_text("wal data")
    shm_path.write_text("shm data")

    with patch("ladybug.Database") as mock_db, patch("ladybug.Connection"):
        # Force corruption error on first call, then success
        mock_db.side_effect = [Exception("invalid wal record"), MagicMock()]

        LadybugBackend(str(db_path))

        # Files should be gone after cleanup (except the main DB)
        assert db_path.exists()
        assert not wal_path.exists()
        assert not shm_path.exists()


def test_ladybug_backend_disabled_backup(temp_db_dir):
    db_path = str(temp_db_dir / "test.db")
    with open(db_path, "w") as f:
        f.write("dummy data")

    with (
        patch("ladybug.Database"),
        patch("ladybug.Connection"),
        patch("agent_utilities.core.config.DEFAULT_KG_BACKUPS", 0),
    ):
        backend = LadybugBackend(db_path)
        backend._backup_db()

        # No backups should be created
        backups = list(temp_db_dir.glob("test.db.*.bak"))
        assert len(backups) == 0


def test_ladybug_backend_get_lock(temp_db_dir):
    db_path = str(temp_db_dir / "test.db")
    with patch("ladybug.Database"), patch("ladybug.Connection"):
        backend = LadybugBackend(db_path)

        # Test memory database lock (should be a nullcontext/no-op)
        backend.db_path = ":memory:"
        mem_lock = backend._get_lock()
        with mem_lock:
            pass  # Should not raise any error

        # Test file database lock (should be filelock.FileLock)
        backend.db_path = db_path
        file_lock = backend._get_lock()
        from filelock import FileLock

        assert isinstance(file_lock, FileLock)
        assert file_lock.lock_file == f"{db_path}.lock"

        # Verify it actually locks
        with file_lock:
            # Try to acquire lock again with timeout=0, should raise Timeout
            from filelock import Timeout

            another_lock = FileLock(f"{db_path}.lock", timeout=0)
            with pytest.raises(Timeout):
                another_lock.acquire()


def test_ladybug_backend_escaped_properties(temp_db_dir):
    import networkx as nx

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    # Mock LadybugBackend and initialize IntelligenceGraphEngine
    mock_backend = MagicMock()
    mock_backend.__class__.__name__ = "LadybugBackend"

    engine = IntelligenceGraphEngine(nx.MultiDiGraph(), backend=mock_backend)
    mock_backend.reset_mock()

    # 1. Test upsert node with reserved keyword property
    # Reserved keyword 'order'
    engine.add_node("node1", "MemoryNode", {"order": 5, "name": "Test Node"})

    # Check what query was executed
    # It should have called mock_backend.execute twice: once for MATCH/SET (update) and once for CREATE (insert) if MATCH returned empty
    assert mock_backend.execute.call_count >= 1

    # Let's inspect the arguments passed to mock_backend.execute
    # First call: Match query
    # MATCH (n:MemoryNode) WHERE n.id = $id SET n.`order` = $order, n.`name` = $name RETURN n.id
    calls = mock_backend.execute.call_args_list
    match_query = calls[0][0][0]
    assert "n.`order` = $order" in match_query
    assert "n.`name` = $name" in match_query

    # 2. Test create query formatting when not found
    mock_backend.execute.return_value = []  # MATCH returns empty, triggering CREATE
    engine.add_node("node2", "MemoryNode", {"order": 10})

    # The last execute call should be the CREATE query:
    # CREATE (n:MemoryNode {id: $id, `order`: $order})
    create_query = mock_backend.execute.call_args_list[-1][0][0]
    assert "CREATE (n:MemoryNode" in create_query
    assert "`order`: $order" in create_query
