"""CONCEPT:OS-5.0"""

import os
from pathlib import Path
import tempfile
import pytest

from agent_utilities.core import paths


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_path_defaults():
    # Verify that paths resolve dynamically and aren't empty
    assert paths.config_dir() is not None
    assert paths.data_dir() is not None
    assert paths.cache_dir() is not None
    assert paths.log_dir() is not None
    assert isinstance(paths.config_dir(), Path)
    assert isinstance(paths.data_dir(), Path)
    assert isinstance(paths.cache_dir(), Path)
    assert isinstance(paths.log_dir(), Path)


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_path_overrides(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path / "config"))
        monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("AGENT_UTILITIES_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv("AGENT_UTILITIES_LOG_DIR", str(tmp_path / "log"))

        assert paths.config_dir() == tmp_path / "config"
        assert paths.data_dir() == tmp_path / "data"
        assert paths.cache_dir() == tmp_path / "cache"
        assert paths.log_dir() == tmp_path / "log"


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_ensure_dirs(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path / "config"))
        monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("AGENT_UTILITIES_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv("AGENT_UTILITIES_LOG_DIR", str(tmp_path / "log"))

        # Directories should not exist yet
        assert not (tmp_path / "config").exists()
        assert not (tmp_path / "data" / "kg").exists()
        assert not (tmp_path / "cache").exists()
        assert not (tmp_path / "log").exists()

        # Run ensure_dirs()
        paths.ensure_dirs()

        # Now they must exist
        assert (tmp_path / "config").exists()
        assert (tmp_path / "data" / "kg").exists()
        assert (tmp_path / "cache").exists()
        assert (tmp_path / "log").exists()


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_kg_db_path_resolution(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.delenv("GRAPH_DB_PATH", raising=False)

        # Should resolve to standard XDG data directory structure
        db_path = paths.kg_db_path()
        assert db_path == tmp_path / "data" / "kg" / "knowledge_graph.db"



@pytest.mark.concept("CONCEPT:OS-5.0")
def test_kg_db_path_explicit_override(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        monkeypatch.setenv("GRAPH_DB_PATH", str(tmp_path / "custom_db.db"))

        # Explicit override takes priority
        db_path = paths.kg_db_path()
        assert db_path == tmp_path / "custom_db.db"
