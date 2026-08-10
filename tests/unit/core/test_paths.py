"""CONCEPT:AU-OS.safety.doom-loop-detection"""

import os
import tempfile
from pathlib import Path

import pytest

from agent_utilities.core import paths


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
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


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
def test_path_overrides(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path / "config"))
        monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("AGENT_UTILITIES_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv("AGENT_UTILITIES_LOG_DIR", str(tmp_path / "log"))

        assert paths.config_dir() == tmp_path / "config"
        assert paths.runtime_secrets_path() == (
            tmp_path / "config" / "runtime-secrets.json"
        )
        assert paths.data_dir() == tmp_path / "data"
        assert paths.cache_dir() == tmp_path / "cache"
        assert paths.log_dir() == tmp_path / "log"


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
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


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
@pytest.mark.skipif(
    not hasattr(os, "getuid") or os.getuid() == 0,
    reason="permission-denial only reproduces for a non-root process",
)
def test_ensure_dirs_fails_loudly_naming_the_setting_when_not_writable(monkeypatch):
    # BUG-ROFS-1 reproduction: a directory that already EXISTS (so a bare
    # mkdir(exist_ok=True) reports nothing wrong) but is not writable by this
    # process -- e.g. kubelet auto-creating an intermediate mount-point
    # directory with different ownership. ensure_dirs() must name the exact
    # setting at fault, not surface a bare PermissionError.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        locked_cache = tmp_path / "cache"
        locked_cache.mkdir()
        locked_cache.chmod(0o555)  # read+execute only, no write
        try:
            monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path / "config"))
            monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path / "data"))
            monkeypatch.setenv("AGENT_UTILITIES_CACHE_DIR", str(locked_cache))
            monkeypatch.setenv("AGENT_UTILITIES_LOG_DIR", str(tmp_path / "log"))

            with pytest.raises(paths.RuntimeDirectoryNotWritableError) as exc_info:
                paths.ensure_dirs()
            message = str(exc_info.value)
            assert "AGENT_UTILITIES_CACHE_DIR" in message
            assert str(locked_cache) in message
        finally:
            locked_cache.chmod(0o755)  # restore so TemporaryDirectory cleanup succeeds


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
def test_kg_db_path_resolution(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path / "data"))
        # Should resolve to standard XDG data directory structure
        db_path = paths.kg_db_path()
        assert db_path == tmp_path / "data" / "kg" / "knowledge_graph.db"
