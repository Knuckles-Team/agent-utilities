"""Tests for engine HMAC secret resolution (CONCEPT:OS-5.14).

The launcher must be secure by default: a per-install secret is generated
once, persisted with 0600 perms under the XDG data dir, shared by every
process, and passed to spawned engines — with KG_ENGINE_INSECURE as the
explicit dev opt-out.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import (
    _load_or_create_engine_secret,
    resolve_engine_auth,
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path))
    return tmp_path


def _cfg(**overrides):
    cfg = MagicMock()
    cfg.kg_engine_insecure = overrides.get("kg_engine_insecure", False)
    cfg.graph_service_auth_secret = overrides.get("graph_service_auth_secret", None)
    return cfg


@pytest.mark.concept("CONCEPT:OS-5.14")
def test_secret_generated_persisted_with_0600(data_dir):
    secret = _load_or_create_engine_secret()
    path = data_dir / "engine_secret"
    assert path.exists()
    assert path.read_text().strip() == secret
    assert len(secret) == 64  # token_hex(32)
    assert (os.stat(path).st_mode & 0o777) == 0o600


@pytest.mark.concept("CONCEPT:OS-5.14")
def test_secret_is_stable_across_calls(data_dir):
    first = _load_or_create_engine_secret()
    second = _load_or_create_engine_secret()
    assert first == second


@pytest.mark.concept("CONCEPT:OS-5.14")
def test_existing_secret_file_wins(data_dir):
    (data_dir / "engine_secret").write_text("pre-shared-secret\n")
    assert _load_or_create_engine_secret() == "pre-shared-secret"


@pytest.mark.concept("CONCEPT:OS-5.14")
def test_resolve_uses_configured_secret_verbatim(data_dir):
    secret, insecure = resolve_engine_auth(_cfg(graph_service_auth_secret="configured"))
    assert (secret, insecure) == ("configured", False)
    # Nothing is generated when the secret is configured explicitly.
    assert not (data_dir / "engine_secret").exists()


@pytest.mark.concept("CONCEPT:OS-5.14")
def test_resolve_generates_persisted_secret_by_default(data_dir):
    secret, insecure = resolve_engine_auth(_cfg())
    assert insecure is False
    assert secret
    assert (data_dir / "engine_secret").read_text().strip() == secret


@pytest.mark.concept("CONCEPT:OS-5.14")
def test_insecure_opt_out(data_dir):
    secret, insecure = resolve_engine_auth(_cfg(kg_engine_insecure=True))
    assert (secret, insecure) == (None, True)
    assert not (data_dir / "engine_secret").exists()
