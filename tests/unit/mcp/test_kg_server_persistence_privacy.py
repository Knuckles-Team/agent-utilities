from __future__ import annotations

import json

from agent_utilities.mcp import kg_server
from agent_utilities.security import persistence_privacy


def _fixed_persistence_key(monkeypatch) -> None:
    monkeypatch.setattr(
        persistence_privacy, "_persistence_identity_key", lambda: b"test-key"
    )


def test_provenance_uses_only_opaque_identity_refs(monkeypatch) -> None:
    _fixed_persistence_key(monkeypatch)
    props = kg_server._provenance_props("local-personal-agent")
    rendered = json.dumps(props)

    assert set(props) == {"agent_ref", "session_ref", "timestamp", "source"}
    assert props["agent_ref"].startswith("pref_agent_")
    assert props["session_ref"].startswith("pref_session_")
    assert "local-personal-agent" not in rendered
    assert "workspace" not in rendered.lower()


def test_mcp_declaration_never_persists_command_args_env_or_endpoint(
    monkeypatch,
) -> None:
    _fixed_persistence_key(monkeypatch)
    details = {
        "command": "/home/local-account/bin/server",
        "args": ["--token", "secret-value", "https://private.invalid"],
        "env": {"ACCESS_TOKEN": "secret-value"},
        "capabilities": ["search", "invalid capability"],
    }

    node_id, declaration = kg_server._mcp_capability_declaration(
        "/home/local-account/custom-server", details
    )
    rendered = json.dumps(declaration, sort_keys=True)

    assert node_id.startswith("mcp_server:pref_mcp_server_")
    assert set(declaration) == {
        "name",
        "server_ref",
        "configuration_ref",
        "capabilities",
        "synonyms",
    }
    assert declaration["name"].startswith("external-")
    assert declaration["capabilities"] == ["search"]
    for forbidden in (
        "local-account",
        "secret-value",
        "private.invalid",
        "command",
        "args",
        "env",
    ):
        assert forbidden not in rendered
