"""Docker/swarm secret bridge: /run/secrets/<NAME> -> os.environ (ECO-4.82)."""

from __future__ import annotations

import os

from agent_utilities.core.config import _load_docker_secrets


def test_secret_file_bridged_into_environ(tmp_path, monkeypatch):
    (tmp_path / "OIDC_CLIENT_SECRET").write_text("s3cr3t\n")
    monkeypatch.delenv("OIDC_CLIENT_SECRET", raising=False)
    _load_docker_secrets(str(tmp_path))
    assert os.environ["OIDC_CLIENT_SECRET"] == "s3cr3t"  # stripped of trailing newline


def test_explicit_env_wins_over_secret(tmp_path, monkeypatch):
    (tmp_path / "OIDC_CLIENT_SECRET").write_text("from-secret")
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "from-env")
    _load_docker_secrets(str(tmp_path))
    assert os.environ["OIDC_CLIENT_SECRET"] == "from-env"  # spec/config env wins


def test_missing_dir_is_noop(monkeypatch):
    monkeypatch.delenv("NOPE_SECRET", raising=False)
    _load_docker_secrets("/no/such/secrets/dir")  # must not raise
    assert "NOPE_SECRET" not in os.environ


def test_empty_secret_file_skipped(tmp_path, monkeypatch):
    (tmp_path / "EMPTY_SECRET").write_text("   \n")
    monkeypatch.delenv("EMPTY_SECRET", raising=False)
    _load_docker_secrets(str(tmp_path))
    assert "EMPTY_SECRET" not in os.environ


def test_subdir_ignored(tmp_path, monkeypatch):
    (tmp_path / "adir").mkdir()
    monkeypatch.delenv("adir", raising=False)
    _load_docker_secrets(str(tmp_path))  # directories under /run/secrets are skipped
    assert "adir" not in os.environ
