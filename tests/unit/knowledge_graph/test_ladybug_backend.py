#!/usr/bin/python
"""CONCEPT:AU-KG.query.object-graph-mapper"""

from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.backends.contrib.ladybug_backend import (
    LadybugBackend,
)


@pytest.fixture
def temp_db_dir(tmp_path):
    db_dir = tmp_path / "test_db"
    db_dir.mkdir()
    return db_dir


def test_ladybug_auto_creates_tables_for_arbitrary_labels(temp_db_dir):
    """CONCEPT:AU-KG.backend.mirror-health-repair — an arbitrary KG (labels/rels beyond the declared SCHEMA)
    mirrors into Kuzu losslessly: unknown node labels auto-create a generic table,
    unknown rel types auto-create / extend their REL table, and ad-hoc props fold
    into the ``metadata`` column. Without this, every undeclared-label node/edge is
    dropped (Kuzu has fixed typed tables)."""
    from agent_utilities.knowledge_graph.migration import _portable_writer

    be = LadybugBackend(db_path=str(temp_db_dir / "arb.db"))
    be.create_schema()
    w = _portable_writer(be)

    # two labels NOT in SCHEMA, each with an ad-hoc prop, plus an edge between them
    w._upsert_node(
        "idea_block", "a", {"id": "a", "type": "idea_block", "trusted_answer": "ans"}
    )
    w._upsert_node("zonkey", "b", {"id": "b", "type": "zonkey", "weird_prop": 7})
    w._upsert_edge(
        "a",
        "b",
        "LINKS",
        {"confidence": 0.9},
        source_label="idea_block",
        target_label="zonkey",
    )

    assert be.execute("MATCH (n) RETURN count(n) AS c")[0]["c"] == 2
    assert be.execute("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"] == 1
    # ad-hoc prop preserved in metadata (lossless)
    rows = be.execute("MATCH (n:idea_block {id:'a'}) RETURN n.metadata AS m")
    assert rows and "trusted_answer" in (rows[0]["m"] or "")
    be.close()


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

        backend = LadybugBackend(str(db_path))
        backend._ensure_connection()

        # Files should be gone after cleanup (except the main DB)
        assert db_path.exists()
        assert not wal_path.exists()
        assert not shm_path.exists()


def test_ladybug_backend_self_heal_invalid_lbug(temp_db_dir):
    db_path = temp_db_dir / "invalid.db"
    db_path.write_text("invalid data")

    with patch("ladybug.Database") as mock_db, patch("ladybug.Connection"):
        # Force 'not a valid lbug' error on first 3 calls, then success
        # The first 2 calls will attempt WAL cleanup, the 3rd will trigger renaming the file
        mock_db.side_effect = [
            Exception(
                "Unable to open database. The file is not a valid Lbug database file!"
            ),
            Exception(
                "Unable to open database. The file is not a valid Lbug database file!"
            ),
            Exception(
                "Unable to open database. The file is not a valid Lbug database file!"
            ),
            MagicMock(),
        ]

        backend = LadybugBackend(str(db_path))
        backend._ensure_connection()

        # The original db file should be renamed to .corrupted
        corrupted_path = db_path.with_suffix(".corrupted")
        assert corrupted_path.exists()
        assert corrupted_path.read_text() == "invalid data"


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

        # Test file database lock (should be CombinedLock wrapping FileLock)
        backend.db_path = db_path
        combined_lock = backend._get_lock()
        from filelock import FileLock

        from agent_utilities.knowledge_graph.backends.contrib.ladybug_backend import (
            CombinedLock,
        )

        assert isinstance(combined_lock, CombinedLock)
        assert isinstance(combined_lock.file_lock, FileLock)
        assert combined_lock.file_lock.lock_file == f"{db_path}.lock"

        # Verify it actually locks
        with combined_lock:
            # Try to acquire lock again with timeout=0, should raise Timeout
            from filelock import Timeout

            another_lock = FileLock(f"{db_path}.lock", timeout=0)
            with pytest.raises(Timeout):
                another_lock.acquire()


def test_ladybug_backend_escaped_properties(temp_db_dir):
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    # Mock LadybugBackend and initialize IntelligenceGraphEngine
    mock_backend = MagicMock()
    mock_backend.__class__.__name__ = "LadybugBackend"

    engine = IntelligenceGraphEngine(
        graph=GraphComputeEngine(backend_type="rust"), backend=mock_backend
    )
    mock_backend.reset_mock()

    # 1. Upsert a node with an UNKNOWN label (MemoryNode is not in SCHEMA). On Kuzu
    # (fixed typed tables) an undeclared property like ``order`` has no column, so it
    # MUST fold into the ``metadata`` JSON column — a bare ``n.`order` = $order``
    # would Binder-error and drop the node. Declared/generic columns (``name``) are
    # still SET, backtick-escaped. (CONCEPT:AU-KG.backend.mirror-health-repair)
    engine.add_node("node1", "MemoryNode", {"order": 5, "name": "Test Node"})
    assert mock_backend.execute.call_count >= 1

    calls = mock_backend.execute.call_args_list
    merge_query, merge_params = calls[0][0][0], calls[0][0][1]
    assert "MERGE (n:MemoryNode {id: $id})" in merge_query
    assert "n.`name` = $name" in merge_query  # generic column, backtick-escaped
    assert "n.`order`" not in merge_query  # undeclared → not a bare column SET
    assert "n.`metadata` = $metadata" in merge_query
    assert '"order": 5' in merge_params.get("metadata", "")  # folded into metadata

    # 2. A node whose only property is undeclared → it lives in metadata; the write
    # never references a non-existent column.
    mock_backend.reset_mock()
    engine.add_node("node2", "MemoryNode", {"order": 10})
    upsert_query = mock_backend.execute.call_args_list[-1][0][0]
    upsert_params = mock_backend.execute.call_args_list[-1][0][1]
    assert "MERGE (n:MemoryNode {id: $id})" in upsert_query
    assert "n.`order`" not in upsert_query
    assert '"order": 10' in upsert_params.get("metadata", "")


def test_relative_db_path_resolves_under_data_dir_not_cwd():
    """E2: a relative db_path must NOT anchor to the cwd (the workspace-root
    `knowledge_graph.db.corrupted` incident) — it resolves under the data dir."""
    import os

    from agent_utilities.core.paths import data_dir

    with patch("ladybug.Database"), patch("ladybug.Connection"):
        backend = LadybugBackend("knowledge_graph.db")
    assert os.path.isabs(backend.db_path), backend.db_path
    assert str(data_dir()) in backend.db_path
    # Never the cwd-relative form.
    assert backend.db_path != os.path.join(os.getcwd(), "knowledge_graph.db")


def test_absolute_db_path_is_left_untouched(tmp_path):
    """An explicit absolute path is honored as-is."""
    abs_path = str(tmp_path / "explicit.db")
    with patch("ladybug.Database"), patch("ladybug.Connection"):
        backend = LadybugBackend(abs_path)
    assert backend.db_path == abs_path


def test_lock_backoff_is_capped():
    """The file-lock retry backoff must be bounded so a transiently-blocked drainer
    recovers promptly instead of over-sleeping into a multi-minute stall."""
    from agent_utilities.knowledge_graph.backends.contrib.ladybug_backend import (
        _LOCK_BACKOFF_MAX_S,
    )

    assert 0 < _LOCK_BACKOFF_MAX_S <= 60.0
