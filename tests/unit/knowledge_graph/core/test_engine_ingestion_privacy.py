from __future__ import annotations

import json

from agent_utilities.knowledge_graph.core.engine_ingestion import (
    _mcp_persistence_resources,
    _neutral_mcp_alias,
)
from agent_utilities.security import persistence_privacy


def test_mcp_resources_exclude_source_path_and_environment_values(monkeypatch) -> None:
    monkeypatch.setattr(
        persistence_privacy, "_persistence_identity_key", lambda: b"test-key"
    )
    resources = _mcp_persistence_resources(
        "/home/local-account/private/mcp_config.json",
        {
            "ACCESS_TOKEN": "plain-secret",
            "CA_REFERENCE": "vault://runtime/ca",
            "NORMAL_SETTING": "environment-specific-value",
        },
    )
    rendered = json.dumps(resources, sort_keys=True)

    assert resources["source_ref"].startswith("pref_mcp_configuration_")
    assert resources["environment_keys"] == [
        "ACCESS_TOKEN",
        "CA_REFERENCE",
        "NORMAL_SETTING",
    ]
    assert resources["environment_secret_refs"] == {
        "CA_REFERENCE": "vault://runtime/ca"
    }
    for forbidden in (
        "local-account",
        "plain-secret",
        "environment-specific-value",
    ):
        assert forbidden not in rendered


def test_non_neutral_mcp_alias_becomes_opaque() -> None:
    alias = _neutral_mcp_alias("/mnt/c/Users/agent-user/server", config_hash="a" * 64)
    assert alias == "external-aaaaaaaaaaaa"
