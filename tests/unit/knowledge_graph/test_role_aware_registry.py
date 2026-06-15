"""Role-aware multi-database registry + live config mutation (CONCEPT:KG-2.89).

Covers: connection roles + write-guard, secret-ref resolution, mirror derivation
from role=mirror, durable config write-back + restart classifier, and the doctor
connections check.
"""

from __future__ import annotations

import json
import os

import pytest

from agent_utilities.knowledge_graph.core.connection_registry import (
    DEFAULT_ROLE,
    ConnectionRegistry,
    _resolve_secret,
)

pytestmark = pytest.mark.concept("KG-2.89")


def test_role_default_validation_and_writability():
    r = ConnectionRegistry()
    r.register("src", {"backend": "neo4j", "uri": "bolt://h", "role": "read"})
    r.register("rw", {"backend": "neo4j", "uri": "bolt://h", "role": "read_write"})
    r.register("mir", {"backend": "falkordb", "host": "h", "role": "mirror"})
    r.register("plain", {"backend": "neo4j", "uri": "bolt://h"})  # default role

    assert r.role("plain") == DEFAULT_ROLE == "read"
    assert r.is_writable("src") is False  # data source
    assert r.is_writable("rw") is True
    assert r.is_writable("mir") is False  # written only via the outbox
    assert r.is_writable(None) is True  # default/authority always writable
    with pytest.raises(ValueError):
        r.register("bad", {"backend": "neo4j", "role": "nope"})


def test_status_and_export_carry_role():
    r = ConnectionRegistry()
    r.register("a", {"backend": "neo4j", "uri": "bolt://h", "role": "read"})
    roles = {c["name"]: c.get("role") for c in r.status()["connections"]}
    assert roles["default"] == "read_write"
    assert roles["a"] == "read"
    assert r.export_specs() == [
        {"name": "a", "backend_type": "neo4j", "uri": "bolt://h", "role": "read"}
    ]


def test_resolve_secret_env_and_literal(monkeypatch):
    monkeypatch.setenv("MY_PW", "s3cret")
    assert _resolve_secret("env://MY_PW") == "s3cret"
    assert _resolve_secret("literalpw") == "literalpw"  # raw passes through
    assert _resolve_secret(1234) == 1234  # non-str passthrough


def test_mirror_set_derived_from_role(monkeypatch):
    from agent_utilities.core.config import config as cfg
    from agent_utilities.knowledge_graph.backends import _build_mirror_set

    monkeypatch.delenv("GRAPH_MIRROR_TARGETS", raising=False)
    monkeypatch.setattr(cfg, "graph_mirror_targets", None, raising=False)
    monkeypatch.setattr(
        cfg,
        "kg_connections",
        [
            {"name": "memmir", "backend": "memory", "role": "mirror"},
            {"name": "src", "backend": "memory", "role": "read"},
            {"name": "rw", "backend": "memory", "role": "read_write"},
        ],
        raising=False,
    )
    # only role=mirror connections become mirrors (memory backend = no network).
    assert sorted(_build_mirror_set()) == ["memmir"]


def test_save_config_item_persists_and_sets_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path))
    from agent_utilities.core.config import _xdg_config_file, save_config_item

    save_config_item("kg_connections", [{"name": "x", "role": "read"}])
    cf = _xdg_config_file()
    assert cf.exists()
    assert json.loads(cf.read_text())["kg_connections"][0]["name"] == "x"
    assert json.loads(os.environ["KG_CONNECTIONS"])[0]["name"] == "x"


def test_restart_required_classifier():
    from agent_utilities.deployment import is_restart_required

    assert is_restart_required("GRAPH_BACKEND") is True
    assert is_restart_required("GRAPH_DB_URI") is True
    assert is_restart_required("AUTH_JWT_ISSUER") is True  # AUTH_ prefix
    assert is_restart_required("KG_LLM_CONCURRENCY") is False


def test_doctor_has_connections_check():
    from agent_utilities.deployment import CHECKS

    assert "graph_connections" in CHECKS
    res = CHECKS["graph_connections"]()
    assert res["name"] == "graph_connections"
    assert res["status"] in ("ok", "warn", "skip")
