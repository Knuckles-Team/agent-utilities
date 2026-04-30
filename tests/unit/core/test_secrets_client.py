"""Tests for the SecretsClient (CONCEPT:AU-011 Secrets & Authentication).

CONCEPT:AU-011 — Secrets & Authentication

Covers:
- InMemoryBackend: encryption round-trip, isolation, CRUD
- SQLiteBackend: persistence, encryption, key auto-generation
- SecretsClient: env fallback, URI resolution, typed retrieval
- Factory: backend selection from config
"""

import os
from unittest import mock

import pytest

from agent_utilities.security.secrets_client import (
    InMemoryBackend,
    SecretsClient,
    SecretsConfig,
    SQLiteBackend,
    create_secrets_client,
)

# CONCEPT:AU-011 Secrets & Authentication


# ---------------------------------------------------------------------------
# InMemoryBackend
# ---------------------------------------------------------------------------


class TestInMemoryBackend:
    """Tests for the Fernet-encrypted in-memory backend."""

    @pytest.mark.concept("AU-011")
    def test_set_and_get(self):
        backend = InMemoryBackend()
        backend.set("my/key", "super-secret")
        assert backend.get("my/key") == "super-secret"

    @pytest.mark.concept("AU-011")
    def test_get_nonexistent_returns_none(self):
        backend = InMemoryBackend()
        assert backend.get("does-not-exist") is None

    @pytest.mark.concept("AU-011")
    def test_delete(self):
        backend = InMemoryBackend()
        backend.set("ephemeral", "data")
        assert backend.delete("ephemeral") is True
        assert backend.get("ephemeral") is None

    @pytest.mark.concept("AU-011")
    def test_delete_nonexistent_returns_false(self):
        backend = InMemoryBackend()
        assert backend.delete("nope") is False

    @pytest.mark.concept("AU-011")
    def test_list_keys(self):
        backend = InMemoryBackend()
        backend.set("b_key", "1")
        backend.set("a_key", "2")
        assert backend.list_keys() == ["a_key", "b_key"]

    @pytest.mark.concept("AU-011")
    def test_values_are_encrypted_at_rest(self):
        """Stored values should not be plaintext."""
        backend = InMemoryBackend()
        backend.set("test", "plaintext-value")
        raw = backend._store["test"]
        assert raw != b"plaintext-value"
        assert isinstance(raw, bytes)

    @pytest.mark.concept("AU-011")
    def test_isolation_between_instances(self):
        """Two InMemoryBackend instances share nothing."""
        b1 = InMemoryBackend()
        b2 = InMemoryBackend()
        b1.set("shared_key", "from_b1")
        assert b2.get("shared_key") is None


# ---------------------------------------------------------------------------
# SQLiteBackend
# ---------------------------------------------------------------------------


class TestSQLiteBackend:
    """Tests for the persistent SQLite + Fernet backend."""

    @pytest.mark.concept("AU-011")
    def test_set_and_get(self, tmp_path):
        db = tmp_path / "test.db"
        backend = SQLiteBackend(db_path=db)
        backend.set("gitlab/token", "glpat-123")
        assert backend.get("gitlab/token") == "glpat-123"

    @pytest.mark.concept("AU-011")
    def test_persistence_across_instances(self, tmp_path):
        """Re-opening with the same key file should retrieve old secrets."""
        db = tmp_path / "persist.db"
        b1 = SQLiteBackend(db_path=db)
        b1.set("persisted", "value")
        # The key file is auto-created alongside the DB
        b2 = SQLiteBackend(db_path=db)
        assert b2.get("persisted") == "value"

    @pytest.mark.concept("AU-011")
    def test_overwrite(self, tmp_path):
        db = tmp_path / "overwrite.db"
        backend = SQLiteBackend(db_path=db)
        backend.set("k", "v1")
        backend.set("k", "v2")
        assert backend.get("k") == "v2"

    @pytest.mark.concept("AU-011")
    def test_delete(self, tmp_path):
        db = tmp_path / "del.db"
        backend = SQLiteBackend(db_path=db)
        backend.set("rmme", "bye")
        assert backend.delete("rmme") is True
        assert backend.get("rmme") is None
        assert backend.delete("rmme") is False

    @pytest.mark.concept("AU-011")
    def test_list_keys(self, tmp_path):
        db = tmp_path / "list.db"
        backend = SQLiteBackend(db_path=db)
        backend.set("z", "1")
        backend.set("a", "2")
        assert backend.list_keys() == ["a", "z"]

    @pytest.mark.concept("AU-011")
    def test_key_file_created(self, tmp_path):
        db = tmp_path / "keyfile.db"
        SQLiteBackend(db_path=db)
        key_file = db.with_suffix(".key")
        assert key_file.exists()


# ---------------------------------------------------------------------------
# SecretsClient
# ---------------------------------------------------------------------------


class TestSecretsClient:
    """Tests for the high-level SecretsClient wrapper."""

    @pytest.mark.concept("AU-011")
    def test_get_or_env_prefers_backend(self):
        client = SecretsClient()
        client.set("mykey", "from-backend")
        with mock.patch.dict(os.environ, {"MY_ENV": "from-env"}):
            assert client.get_or_env("mykey", "MY_ENV") == "from-backend"

    @pytest.mark.concept("AU-011")
    def test_get_or_env_falls_back_to_env(self):
        client = SecretsClient()
        with mock.patch.dict(os.environ, {"FALLBACK_VAR": "env-value"}):
            assert client.get_or_env("missing", "FALLBACK_VAR") == "env-value"

    @pytest.mark.concept("AU-011")
    def test_get_or_env_returns_none_when_both_missing(self):
        client = SecretsClient()
        assert client.get_or_env("nope", "DOES_NOT_EXIST") is None

    @pytest.mark.concept("AU-011")
    def test_get_secret_returns_pydantic_model(self):
        client = SecretsClient()
        client.set("typed", "secret-value")
        sv = client.get_secret("typed")
        assert sv is not None
        assert sv.value.get_secret_value() == "secret-value"

    @pytest.mark.concept("AU-011")
    def test_get_secret_returns_none_for_missing(self):
        client = SecretsClient()
        assert client.get_secret("missing") is None

    @pytest.mark.concept("AU-011")
    def test_list_keys(self):
        client = SecretsClient()
        client.set("k1", "v1")
        client.set("k2", "v2")
        assert "k1" in client.list_keys()
        assert "k2" in client.list_keys()


# ---------------------------------------------------------------------------
# URI Resolution
# ---------------------------------------------------------------------------


class TestResolveRef:
    """Tests for resolve_ref() with various URI schemes."""

    @pytest.mark.concept("AU-011")
    def test_env_scheme(self):
        client = SecretsClient()
        with mock.patch.dict(os.environ, {"MY_TOKEN": "tok123"}):
            assert client.resolve_ref("env://MY_TOKEN") == "tok123"

    @pytest.mark.concept("AU-011")
    def test_vault_scheme(self):
        client = SecretsClient()
        client.set("agents/mcp/github/token", "ghp_xxx")
        assert client.resolve_ref("vault://agents/mcp/github/token") == "ghp_xxx"

    @pytest.mark.concept("AU-011")
    def test_secret_scheme(self):
        client = SecretsClient()
        client.set("my/path", "val")
        assert client.resolve_ref("secret://my/path") == "val"

    @pytest.mark.concept("AU-011")
    def test_sqlite_scheme(self):
        client = SecretsClient()
        client.set("sqlite/key", "sqlite-val")
        assert client.resolve_ref("sqlite://sqlite/key") == "sqlite-val"

    @pytest.mark.concept("AU-011")
    def test_plain_key_fallback(self):
        client = SecretsClient()
        client.set("plain", "simple")
        assert client.resolve_ref("plain") == "simple"

    @pytest.mark.concept("AU-011")
    def test_missing_ref_returns_none(self):
        client = SecretsClient()
        assert client.resolve_ref("vault://does/not/exist") is None
        assert client.resolve_ref("env://NONEXISTENT_VAR_XYZ") is None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestSecretsFactory:
    """Tests for create_secrets_client() backend selection."""

    @pytest.mark.concept("AU-011")
    def test_default_inmemory(self):
        client = create_secrets_client()
        assert isinstance(client.backend, InMemoryBackend)

    @pytest.mark.concept("AU-011")
    def test_explicit_inmemory(self):
        config = SecretsConfig(backend="inmemory")
        client = create_secrets_client(config)
        assert isinstance(client.backend, InMemoryBackend)

    @pytest.mark.concept("AU-011")
    def test_sqlite_backend(self, tmp_path):
        config = SecretsConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / "factory.db"),
        )
        client = create_secrets_client(config)
        assert isinstance(client.backend, SQLiteBackend)
        client.set("test", "works")
        assert client.get("test") == "works"

    @pytest.mark.concept("AU-011")
    def test_vault_backend_requires_hvac(self):
        """Vault backend should raise ImportError if hvac is not installed."""
        config = SecretsConfig(backend="vault")
        # hvac may or may not be installed; we test the factory doesn't crash
        # selecting the backend when hvac IS missing.
        try:
            client = create_secrets_client(config)
            # If hvac is installed, the client is created successfully
            from agent_utilities.security.secrets_client import VaultBackend

            assert isinstance(client.backend, VaultBackend)
        except ImportError:
            # Expected if hvac is not installed
            pass

    @pytest.mark.concept("AU-011")
    def test_env_driven_config(self, tmp_path):
        db_path = str(tmp_path / "env.db")
        with mock.patch.dict(
            os.environ,
            {"SECRETS_BACKEND": "sqlite", "SECRETS_SQLITE_PATH": db_path},
        ):
            client = create_secrets_client()
            assert isinstance(client.backend, SQLiteBackend)
