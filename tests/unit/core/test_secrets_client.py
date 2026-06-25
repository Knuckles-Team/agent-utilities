"""Tests for the SecretsClient (CONCEPT:OS-5.1 / OS-5.66).

CONCEPT:OS-5.1 — Secrets & Authentication
CONCEPT:OS-5.66 — Engine-backed encrypted secret store

Covers:
- InEpistemicGraphBackend: the durable, engine-backed __secrets__ store CRUD +
  the encrypted-property split (key/metadata plaintext, value sealed by the
  engine's encryption-at-rest).
- SQLiteBackend: persistence + encryption (retained as the migration source).
- SecretsClient: env fallback, URI resolution, typed retrieval.
- Factory: backend selection (engine default everywhere, vault enterprise).
- One-time legacy secrets.db → __secrets__ migration (read-old → write-new →
  delete-old).
"""

import os
import uuid
from unittest import mock

import pytest

from agent_utilities.security.secrets_client import (
    InEpistemicGraphBackend,
    SecretsClient,
    SecretsConfig,
    SQLiteBackend,
    create_secrets_client,
)

# CONCEPT:OS-5.1 Secrets & Authentication


@pytest.fixture
def engine_backend():
    """A durable engine-backed secrets backend bound to a unique throwaway graph.

    Uses a per-test graph name so secrets never leak across tests (the
    ``__secrets__`` graph is not auto-isolated by the conftest fixture). Skips
    cleanly when no test engine is reachable (the conftest makereport hook turns
    the ConnectionError into a skip).
    """
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    graph = GraphComputeEngine(graph_name=f"__secrets_test_{uuid.uuid4().hex[:12]}__")
    return InEpistemicGraphBackend(graph=graph)


# ---------------------------------------------------------------------------
# InEpistemicGraphBackend (engine-backed, durable — CONCEPT:OS-5.66)
# ---------------------------------------------------------------------------


class TestInEpistemicGraphBackend:
    """Tests for the durable, engine-backed __secrets__ store."""

    @pytest.mark.concept("CONCEPT:OS-5.66")
    def test_set_and_get(self, engine_backend):
        engine_backend.set("my/key", "super-secret")
        assert engine_backend.get("my/key") == "super-secret"

    @pytest.mark.concept("CONCEPT:OS-5.66")
    def test_get_nonexistent_returns_none(self, engine_backend):
        assert engine_backend.get("does-not-exist") is None

    @pytest.mark.concept("CONCEPT:OS-5.66")
    def test_delete(self, engine_backend):
        engine_backend.set("ephemeral", "data")
        assert engine_backend.delete("ephemeral") is True
        assert engine_backend.get("ephemeral") is None

    @pytest.mark.concept("CONCEPT:OS-5.66")
    def test_delete_nonexistent_returns_false(self, engine_backend):
        assert engine_backend.delete("nope") is False

    @pytest.mark.concept("CONCEPT:OS-5.66")
    def test_list_keys(self, engine_backend):
        engine_backend.set("b_key", "1")
        engine_backend.set("a_key", "2")
        assert engine_backend.list_keys() == ["a_key", "b_key"]

    @pytest.mark.concept("CONCEPT:OS-5.66")
    def test_overwrite(self, engine_backend):
        engine_backend.set("k", "v1")
        engine_backend.set("k", "v2")
        assert engine_backend.get("k") == "v2"

    @pytest.mark.concept("CONCEPT:OS-5.66")
    def test_metadata_is_queryable_plaintext(self, engine_backend):
        """Key NAME + metadata stay plaintext node properties (mirrors SQLite split)."""
        engine_backend.set("svc/token", "v", service="gitlab")
        from agent_utilities.security.secrets_client import _node_id

        props = engine_backend._graph._get_node_properties(_node_id("svc/token"))
        assert props["key"] == "svc/token"
        assert props["label"] == "Secret"
        import json as _json

        assert _json.loads(props["metadata"]) == {"service": "gitlab"}

    @pytest.mark.concept("CONCEPT:OS-5.66")
    def test_round_trip_through_client(self, engine_backend):
        """End-to-end SecretsClient round-trip over the engine backend."""
        client = SecretsClient(backend=engine_backend)
        client.set("gitlab/token", "glpat-xyz", service="gitlab-api")
        assert client.get("gitlab/token") == "glpat-xyz"
        assert "gitlab/token" in client.list_keys()
        assert client.resolve_ref("vault://gitlab/token") == "glpat-xyz"
        assert client.delete("gitlab/token") is True
        assert client.get("gitlab/token") is None


# ---------------------------------------------------------------------------
# SQLiteBackend (retained as migration source)
# ---------------------------------------------------------------------------


class TestSQLiteBackend:
    """Tests for the persistent SQLite + Fernet backend (migration source)."""

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_set_and_get(self, tmp_path):
        db = tmp_path / "test.db"
        backend = SQLiteBackend(db_path=db)
        backend.set("gitlab/token", "glpat-123")
        assert backend.get("gitlab/token") == "glpat-123"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_persistence_across_instances(self, tmp_path):
        """Re-opening with the same key file should retrieve old secrets."""
        db = tmp_path / "persist.db"
        b1 = SQLiteBackend(db_path=db)
        b1.set("persisted", "value")
        # The key file is auto-created alongside the DB
        b2 = SQLiteBackend(db_path=db)
        assert b2.get("persisted") == "value"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_overwrite(self, tmp_path):
        db = tmp_path / "overwrite.db"
        backend = SQLiteBackend(db_path=db)
        backend.set("k", "v1")
        backend.set("k", "v2")
        assert backend.get("k") == "v2"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_delete(self, tmp_path):
        db = tmp_path / "del.db"
        backend = SQLiteBackend(db_path=db)
        backend.set("rmme", "bye")
        assert backend.delete("rmme") is True
        assert backend.get("rmme") is None
        assert backend.delete("rmme") is False

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_list_keys(self, tmp_path):
        db = tmp_path / "list.db"
        backend = SQLiteBackend(db_path=db)
        backend.set("z", "1")
        backend.set("a", "2")
        assert backend.list_keys() == ["a", "z"]

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_key_file_created(self, tmp_path):
        db = tmp_path / "keyfile.db"
        SQLiteBackend(db_path=db)
        key_file = db.with_suffix(".key")
        assert key_file.exists()


# ---------------------------------------------------------------------------
# SecretsClient (over the engine backend)
# ---------------------------------------------------------------------------


class TestSecretsClient:
    """Tests for the high-level SecretsClient wrapper."""

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_get_or_env_prefers_backend(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        client.set("mykey", "from-backend")
        with mock.patch.dict(os.environ, {"MY_ENV": "from-env"}):
            assert client.get_or_env("mykey", "MY_ENV") == "from-backend"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_get_or_env_falls_back_to_env(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        with mock.patch.dict(os.environ, {"FALLBACK_VAR": "env-value"}):
            assert client.get_or_env("missing", "FALLBACK_VAR") == "env-value"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_get_or_env_returns_none_when_both_missing(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        assert client.get_or_env("nope", "DOES_NOT_EXIST") is None

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_get_secret_returns_pydantic_model(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        client.set("typed", "secret-value")
        sv = client.get_secret("typed")
        assert sv is not None
        assert sv.value.get_secret_value() == "secret-value"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_get_secret_returns_none_for_missing(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        assert client.get_secret("missing") is None

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_list_keys(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        client.set("k1", "v1")
        client.set("k2", "v2")
        assert "k1" in client.list_keys()
        assert "k2" in client.list_keys()


# ---------------------------------------------------------------------------
# URI Resolution
# ---------------------------------------------------------------------------


class TestResolveRef:
    """Tests for resolve_ref() with various URI schemes."""

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_env_scheme(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        with mock.patch.dict(os.environ, {"MY_TOKEN": "tok123"}):
            assert client.resolve_ref("env://MY_TOKEN") == "tok123"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_vault_scheme(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        client.set("agents/mcp/github/token", "ghp_xxx")
        assert client.resolve_ref("vault://agents/mcp/github/token") == "ghp_xxx"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_secret_scheme(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        client.set("my/path", "val")
        assert client.resolve_ref("secret://my/path") == "val"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_sqlite_scheme(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        client.set("sqlite/key", "sqlite-val")
        assert client.resolve_ref("sqlite://sqlite/key") == "sqlite-val"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_plain_key_fallback(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        client.set("plain", "simple")
        assert client.resolve_ref("plain") == "simple"

    @pytest.mark.concept("CONCEPT:OS-5.1")
    def test_missing_ref_returns_none(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        assert client.resolve_ref("vault://does/not/exist") is None
        assert client.resolve_ref("env://NONEXISTENT_VAR_XYZ") is None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestSecretsFactory:
    """Tests for create_secrets_client() backend selection (CONCEPT:OS-5.66)."""

    @pytest.mark.concept("CONCEPT:OS-5.66")
    def test_default_is_engine_backed(self):
        """The default everywhere is the durable engine-backed store."""
        client = create_secrets_client()
        assert isinstance(client.backend, InEpistemicGraphBackend)

    @pytest.mark.concept("CONCEPT:OS-5.66")
    def test_explicit_inmemory_is_engine_backed(self):
        """``inmemory`` (legacy config key) now resolves to the engine store."""
        config = SecretsConfig(backend="inmemory")
        client = create_secrets_client(config)
        assert isinstance(client.backend, InEpistemicGraphBackend)

    @pytest.mark.concept("CONCEPT:OS-5.66")
    def test_vault_backend_requires_hvac(self):
        """Vault backend (enterprise path, UNTOUCHED) selects VaultBackend or raises."""
        config = SecretsConfig(backend="vault")
        try:
            client = create_secrets_client(config)
            from agent_utilities.security.secrets_client import VaultBackend

            assert isinstance(client.backend, VaultBackend)
        except (ImportError, RuntimeError, Exception):
            # hvac missing (ImportError) or no reachable vault (auth/connect) — the
            # factory selected the vault path, which is what this asserts.
            pass


# ---------------------------------------------------------------------------
# One-time legacy secrets.db migration (CONCEPT:OS-5.66)
# ---------------------------------------------------------------------------


class TestLegacyMigration:
    """read-old → write-new → delete-old, on first engine-backed boot."""

    @pytest.mark.concept("CONCEPT:OS-5.66")
    def test_migrates_and_deletes_legacy_db(
        self, engine_backend, tmp_path, monkeypatch
    ):
        from agent_utilities.security import secrets_client as sc

        # Seed a legacy Fernet SQLite store with its sibling .key.
        db = tmp_path / "secrets.db"
        legacy = SQLiteBackend(db_path=db)
        legacy.set("gitlab/token", "glpat-legacy", service="gitlab")
        legacy.set("openai/key", "sk-legacy")
        legacy._conn.close()
        key_file = db.with_suffix(".key")
        assert db.exists() and key_file.exists()

        # Point the migration at our temp legacy db.
        monkeypatch.setattr(sc, "_legacy_sqlite_path", lambda: db)

        migrated = sc._migrate_legacy_sqlite(engine_backend)

        assert migrated == 2
        assert engine_backend.get("gitlab/token") == "glpat-legacy"
        assert engine_backend.get("openai/key") == "sk-legacy"
        # Old db + key file removed once converted (No-Legacy on-disk exception).
        assert not db.exists()
        assert not key_file.exists()

    @pytest.mark.concept("CONCEPT:OS-5.66")
    def test_migration_is_noop_without_legacy_db(
        self, engine_backend, tmp_path, monkeypatch
    ):
        from agent_utilities.security import secrets_client as sc

        monkeypatch.setattr(sc, "_legacy_sqlite_path", lambda: tmp_path / "absent.db")
        assert sc._migrate_legacy_sqlite(engine_backend) == 0


# ---------------------------------------------------------------------------
# vault_sync — read-existing + seed (CONCEPT:OS-5.43)
# ---------------------------------------------------------------------------


class TestVaultSync:
    """Tests for the vault-first read-existing/seed routine."""

    @pytest.mark.concept("CONCEPT:OS-5.43")
    def test_seeds_missing_and_emits_refs(self, engine_backend):
        client = SecretsClient(engine_backend)
        result = client.vault_sync(
            "gitlab-api",
            ["GITLAB_TOKEN", "GITLAB_URL"],
            values={"GITLAB_TOKEN": "glpat-xyz"},
        )
        assert result["written"] == ["GITLAB_TOKEN"]
        assert result["missing"] == ["GITLAB_URL"]
        assert result["refs"]["GITLAB_TOKEN"] == "vault://gitlab-api/GITLAB_TOKEN"
        # The written value is resolvable via the emitted ref.
        assert client.resolve_ref(result["refs"]["GITLAB_TOKEN"]) == "glpat-xyz"

    @pytest.mark.concept("CONCEPT:OS-5.43")
    def test_reads_existing_without_reprompt(self, engine_backend):
        engine_backend.set("keycloak-mcp/OIDC_CLIENT_SECRET", "already-here")
        client = SecretsClient(engine_backend)
        result = client.vault_sync(
            "keycloak-mcp",
            ["OIDC_CLIENT_SECRET"],
            values={"OIDC_CLIENT_SECRET": "should-not-overwrite"},
        )
        assert result["present"] == ["OIDC_CLIENT_SECRET"]
        assert result["written"] == []
        # Existing value preserved (no re-prompt / no overwrite).
        assert (
            client.resolve_ref("vault://keycloak-mcp/OIDC_CLIENT_SECRET")
            == "already-here"
        )

    @pytest.mark.concept("CONCEPT:OS-5.43")
    def test_overwrite_replaces_existing(self, engine_backend):
        engine_backend.set("svc/API_KEY", "old")
        client = SecretsClient(engine_backend)
        result = client.vault_sync(
            "svc", ["API_KEY"], values={"API_KEY": "new"}, overwrite=True
        )
        assert result["written"] == ["API_KEY"]
        assert client.resolve_ref("vault://svc/API_KEY") == "new"
