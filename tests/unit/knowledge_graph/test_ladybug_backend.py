#!/usr/bin/python
import os
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

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

    with patch("ladybug.Database"), patch("ladybug.Connection"), \
         patch("agent_utilities.config.DEFAULT_KG_BACKUPS", 2):

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

        # Files should be gone after cleanup
        assert not db_path.exists()
        assert not wal_path.exists()
        assert not shm_path.exists()

def test_ladybug_backend_disabled_backup(temp_db_dir):
    db_path = str(temp_db_dir / "test.db")
    with open(db_path, "w") as f:
        f.write("dummy data")

    with patch("ladybug.Database"), patch("ladybug.Connection"), \
         patch("agent_utilities.config.DEFAULT_KG_BACKUPS", 0):

        backend = LadybugBackend(db_path)
        backend._backup_db()

        # No backups should be created
        backups = list(temp_db_dir.glob("test.db.*.bak"))
        assert len(backups) == 0
