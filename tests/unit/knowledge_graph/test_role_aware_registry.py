"""Role-aware multi-database registry + live config mutation (CONCEPT:AU-KG.backend.connection-registry).

Covers: connection roles + write-guard, secret-ref resolution, mirror derivation
from role=mirror, durable config write-back + restart classifier, and the doctor
connections check.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.core.connection_registry import (
    DEFAULT_ROLE,
    ConnectionRegistry,
    _resolve_secret,
)

pytestmark = pytest.mark.concept("AU-KG.backend.connection-registry")


def test_role_default_validation_and_writability():
    r = ConnectionRegistry()
    r.register("src", {"backend": "neo4j", "uri": "bolt://h", "role": "read"})
    r.register("mir", {"backend": "falkordb", "host": "h", "role": "mirror"})
    r.register("plain", {"backend": "neo4j", "uri": "bolt://h"})  # default role

    assert r.role("plain") == DEFAULT_ROLE == "read"
    assert r.is_writable("src") is False  # data source
    assert r.is_writable("mir") is False  # written only via the outbox
    assert r.is_writable(None) is True  # default/authority always writable
    with pytest.raises(ValueError):
        r.register("rw", {"backend": "neo4j", "role": "read_write"})
    with pytest.raises(ValueError):
        r.register("bad", {"backend": "neo4j", "role": "nope"})


def test_status_and_export_carry_role():
    r = ConnectionRegistry()
    r.register(
        "aa",
        {
            "backend": "neo4j",
            "connection_profile_ref": "secret://graphs/source/profile",
            "role": "read",
        },
    )
    roles = {c["name"]: c.get("role") for c in r.status()["connections"]}
    assert roles["default"] == "authority"
    assert roles["aa"] == "read"
    assert r.export_specs() == [
        {
            "name": "aa",
            "backend_type": "neo4j",
            "connection_profile_ref": "secret://graphs/source/profile",
            "role": "read",
        }
    ]


def test_export_rejects_transient_literal_connection_material():
    r = ConnectionRegistry()
    r.register("a", {"backend": "neo4j", "uri": "bolt://runtime-only"})

    with pytest.raises(ValueError, match="secret reference"):
        r.export_specs()


def test_resolve_secret_env_and_literal(monkeypatch):
    from agent_utilities.knowledge_graph.core import connection_registry
    from agent_utilities.security.secrets_client import (
        SecretsBackend,
        SecretsClient,
    )

    class _NoStore(SecretsBackend):
        def get(self, key):
            return None

        def set(self, key, value, **metadata):
            return None

        def delete(self, key):
            return False

        def list_keys(self):
            return []

    # The default client is the engine-backed store; an env:// reference never
    # reads it, so the test supplies a store-free client instead of an engine.
    monkeypatch.setattr(
        connection_registry, "_SECRETS_CLIENT", SecretsClient(backend=_NoStore())
    )
    monkeypatch.setenv("MY_PW", "s3cret")
    assert _resolve_secret("env://MY_PW") == "s3cret"
    assert _resolve_secret("literalpw") == "literalpw"  # raw passes through
    assert _resolve_secret(1234) == 1234  # non-str passthrough


def test_mirror_set_derived_from_role(monkeypatch):
    import agent_utilities.knowledge_graph.backends as backends
    from agent_utilities.core.config import config as cfg
    from agent_utilities.knowledge_graph.backends import _build_mirror_set

    monkeypatch.delenv("GRAPH_MIRROR_TARGETS", raising=False)
    monkeypatch.setattr(cfg, "graph_mirror_targets", None, raising=False)
    monkeypatch.setattr(
        cfg,
        "kg_connections",
        [
            {"name": "external", "backend": "neo4j", "role": "mirror"},
            {"name": "src", "backend": "neo4j", "role": "read"},
        ],
        raising=False,
    )
    projection = object()
    monkeypatch.setattr(backends, "_build_member", lambda _spec: projection)
    # Only role=mirror external connections become projections.
    assert _build_mirror_set() == {"external": projection}


def test_save_config_item_persists_canonical_xdg_document(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path))
    from agent_utilities.core.config import _xdg_config_file, save_config_item

    save_config_item("kg_connections", [{"name": "x", "role": "read"}])
    cf = _xdg_config_file()
    assert cf.exists()
    assert json.loads(cf.read_text())["KG_CONNECTIONS"][0]["name"] == "x"


def test_save_config_item_rejects_literal_connection_material(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path))
    from agent_utilities.core.config import save_config_item

    with pytest.raises(ValueError, match="secret reference"):
        save_config_item(
            "kg_connections",
            [{"name": "x", "backend": "neo4j", "uri": "bolt://runtime-only"}],
        )


def test_restart_required_classifier():
    from agent_utilities.deployment import is_restart_required

    assert is_restart_required("GRAPH_MIRROR_TARGETS") is True
    assert is_restart_required("GRAPH_DB_CONNECTION_PROFILE_REF") is True
    assert is_restart_required("EPISTEMIC_GRAPH_STARTUP_TIMEOUT_SECS") is True
    assert is_restart_required("AUTH_JWT_ISSUER") is True  # AUTH_ prefix
    assert is_restart_required("KG_LLM_CONCURRENCY") is False
