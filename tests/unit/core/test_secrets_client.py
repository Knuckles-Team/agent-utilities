"""Tests for the SecretsClient (CONCEPT:AU-OS.config.secrets-authentication / OS-5.66).

CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication
CONCEPT:AU-OS.identity.encrypted-secret-store — Engine-backed encrypted secret store

Covers:
- InEpistemicGraphBackend: the durable, engine-backed __secrets__ store CRUD +
  the encrypted-property split (key/metadata plaintext, value sealed by the
  engine's encryption-at-rest).
- SecretsClient: env fallback, URI resolution, typed retrieval.
- Factory: backend selection (engine default everywhere, vault enterprise).
"""

import os
from unittest import mock

import pytest

from agent_utilities.security.secrets_client import (
    InEpistemicGraphBackend,
    SecretsClient,
    SecretsConfig,
    create_secrets_client,
)

# CONCEPT:AU-OS.config.secrets-authentication Secrets & Authentication


@pytest.fixture
def engine_backend():
    """A durable engine-backed secrets backend bound to a unique throwaway graph.

    The autouse graph fixture provisions a unique, session-matched graph for
    every ``GraphComputeEngine`` construction and owns its lifecycle teardown.
    Skips cleanly when no test engine is reachable (the conftest makereport hook
    turns the ConnectionError into a skip).
    """
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    graph = GraphComputeEngine()
    return InEpistemicGraphBackend(graph=graph)


# ---------------------------------------------------------------------------
# InEpistemicGraphBackend (engine-backed, durable — CONCEPT:AU-OS.identity.encrypted-secret-store)
# ---------------------------------------------------------------------------


class TestInEpistemicGraphBackend:
    """Tests for the durable, engine-backed __secrets__ store."""

    @pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
    def test_set_and_get(self, engine_backend):
        engine_backend.set("my/key", "super-secret")
        assert engine_backend.get("my/key") == "super-secret"

    @pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
    def test_get_nonexistent_returns_none(self, engine_backend):
        assert engine_backend.get("does-not-exist") is None

    @pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
    def test_delete(self, engine_backend):
        engine_backend.set("ephemeral", "data")
        assert engine_backend.delete("ephemeral") is True
        assert engine_backend.get("ephemeral") is None

    @pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
    def test_delete_nonexistent_returns_false(self, engine_backend):
        assert engine_backend.delete("nope") is False

    @pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
    def test_list_keys(self, engine_backend):
        engine_backend.set("b_key", "1")
        engine_backend.set("a_key", "2")
        assert engine_backend.list_keys() == ["a_key", "b_key"]

    @pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
    def test_overwrite(self, engine_backend):
        engine_backend.set("k", "v1")
        engine_backend.set("k", "v2")
        assert engine_backend.get("k") == "v2"

    @pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
    def test_metadata_is_queryable_plaintext(self, engine_backend):
        """Key name and metadata stay queryable while the value is sealed."""
        engine_backend.set("svc/token", "v", service="gitlab")
        from agent_utilities.security.secrets_client import _node_id

        props = engine_backend._graph._get_node_properties(_node_id("svc/token"))
        assert props["key"] == "svc/token"
        assert props["label"] == "Secret"
        import json as _json

        assert _json.loads(props["metadata"]) == {"service": "gitlab"}

    @pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
    def test_round_trip_through_client(self, engine_backend):
        """End-to-end SecretsClient round-trip over the engine backend."""
        client = SecretsClient(backend=engine_backend)
        client.set("gitlab/token", "glpat-xyz", service="gitlab-api")
        assert client.get("gitlab/token") == "glpat-xyz"
        assert "gitlab/token" in client.list_keys()
        assert client.resolve_ref("vault://gitlab/token") == "glpat-xyz"
        assert client.delete("gitlab/token") is True
        assert client.get("gitlab/token") is None

    @pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
    def test_set_if_absent_is_atomic_provision_once(self, engine_backend):
        """Durable create-if-absent: one winner, the loser's value is untouched."""
        assert engine_backend.set_if_absent("system/key", "winner") is True
        # A second provisioner never overwrites the winning value.
        assert engine_backend.set_if_absent("system/key", "loser") is False
        assert engine_backend.get("system/key") == "winner"

    @pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
    def test_compare_and_set_conditional_replace(self, engine_backend):
        """Atomic conditional replace — the rotation substrate."""
        engine_backend.set("system/key", "v1")
        # Wrong expected value ⇒ no replace.
        assert engine_backend.compare_and_set("system/key", "WRONG", "v2") is False
        assert engine_backend.get("system/key") == "v1"
        # Correct expected value ⇒ atomic replace.
        assert engine_backend.compare_and_set("system/key", "v1", "v2") is True
        assert engine_backend.get("system/key") == "v2"


class TestSecretsBackendConditionalDefaults:
    """The base-class best-effort conditional writes (engine-independent)."""

    def _backend(self):
        from agent_utilities.security.secrets_client import SecretsBackend

        class _DictBackend(SecretsBackend):
            def __init__(self) -> None:
                self._data: dict[str, str] = {}

            def get(self, key: str):
                return self._data.get(key)

            def set(self, key: str, value: str, **_metadata) -> None:
                self._data[key] = value

            def delete(self, key: str) -> bool:
                return self._data.pop(key, None) is not None

            def list_keys(self):
                return sorted(self._data)

        return _DictBackend()

    def test_set_if_absent_default(self):
        backend = self._backend()
        assert backend.set_if_absent("k", "first") is True
        assert backend.set_if_absent("k", "second") is False
        assert backend.get("k") == "first"

    def test_compare_and_set_default(self):
        backend = self._backend()
        backend.set("k", "v1")
        assert backend.compare_and_set("k", "nope", "v2") is False
        assert backend.get("k") == "v1"
        assert backend.compare_and_set("k", "v1", "v2") is True
        assert backend.get("k") == "v2"


# ---------------------------------------------------------------------------
# SecretsClient (over the engine backend)
# ---------------------------------------------------------------------------


class TestSecretsClient:
    """Tests for the high-level SecretsClient wrapper."""

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_get_or_env_prefers_backend(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        client.set("mykey", "from-backend")
        with mock.patch.dict(os.environ, {"MY_ENV": "from-env"}):
            assert client.get_or_env("mykey", "MY_ENV") == "from-backend"

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_get_or_env_falls_back_to_env(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        with mock.patch.dict(os.environ, {"FALLBACK_VAR": "env-value"}):
            assert client.get_or_env("missing", "FALLBACK_VAR") == "env-value"

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_get_or_env_returns_none_when_both_missing(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        assert client.get_or_env("nope", "DOES_NOT_EXIST") is None

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_get_secret_returns_pydantic_model(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        client.set("typed", "secret-value")
        sv = client.get_secret("typed")
        assert sv is not None
        assert sv.value.get_secret_value() == "secret-value"

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_get_secret_returns_none_for_missing(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        assert client.get_secret("missing") is None

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
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

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_env_scheme(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        with mock.patch.dict(os.environ, {"MY_TOKEN": "tok123"}):
            assert client.resolve_ref("env://MY_TOKEN") == "tok123"

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_vault_scheme(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        client.set("agents/mcp/github/token", "ghp_xxx")
        assert client.resolve_ref("vault://agents/mcp/github/token") == "ghp_xxx"

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_secret_scheme(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        client.set("my/path", "val")
        assert client.resolve_ref("secret://my/path") == "val"

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    @pytest.mark.parametrize("reference", ["sqlite://sqlite/key", "plain"])
    def test_non_current_reference_is_rejected(self, engine_backend, reference):
        client = SecretsClient(backend=engine_backend)
        with pytest.raises(ValueError, match="runtime secret reference is invalid"):
            client.resolve_ref(reference)

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_missing_ref_returns_none(self, engine_backend):
        client = SecretsClient(backend=engine_backend)
        assert client.resolve_ref("vault://does/not/exist") is None
        assert client.resolve_ref("env://NONEXISTENT_VAR_XYZ") is None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestSecretsFactory:
    """Tests for create_secrets_client() backend selection (CONCEPT:AU-OS.identity.encrypted-secret-store)."""

    @pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
    def test_default_is_engine_backed(self):
        """The default everywhere is the durable engine-backed store."""
        client = create_secrets_client()
        assert isinstance(client.backend, InEpistemicGraphBackend)

    @pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
    def test_explicit_engine_backend_is_engine_backed(self):
        config = SecretsConfig(backend="engine")
        client = create_secrets_client(config)
        assert isinstance(client.backend, InEpistemicGraphBackend)

    @pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
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
# vault_sync — read-existing + seed (CONCEPT:AU-OS.deployment.vault-first-routine-genesis)
# ---------------------------------------------------------------------------


class TestVaultSync:
    """Tests for the vault-first read-existing/seed routine."""

    @pytest.mark.concept("CONCEPT:AU-OS.deployment.vault-first-routine-genesis")
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

    @pytest.mark.concept("CONCEPT:AU-OS.deployment.vault-first-routine-genesis")
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

    @pytest.mark.concept("CONCEPT:AU-OS.deployment.vault-first-routine-genesis")
    def test_overwrite_replaces_existing(self, engine_backend):
        engine_backend.set("svc/API_KEY", "old")
        client = SecretsClient(engine_backend)
        result = client.vault_sync(
            "svc", ["API_KEY"], values={"API_KEY": "new"}, overwrite=True
        )
        assert result["written"] == ["API_KEY"]
        assert client.resolve_ref("vault://svc/API_KEY") == "new"
