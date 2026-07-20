from __future__ import annotations

import json
from types import SimpleNamespace

from agent_utilities.deployment import doctor as doctor_module


class _Registry:
    def __init__(self, *, probe_error: bool = False) -> None:
        self.probe_error = probe_error
        self.probed: list[str] = []

    def status(self) -> dict[str, object]:
        return {
            "connections": [
                {"name": "default", "role": "authority"},
                {
                    "name": "external-source",
                    "role": "read",
                    "backend_type": "graphql",
                },
            ]
        }

    def probe(self, name: str) -> bool:
        self.probed.append(name)
        if self.probe_error:
            raise ValueError(
                "secret://source/auth https://private.example.test/graphql"
            )
        return True


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        external_graph_connectors=[],
        kg_connections=[
            {
                "name": "external-source",
                "source_alias": "external-source",
                "backend": "graphql",
                "role": "read",
                "connection_profile_ref": "secret://source/connection",
                "mapping_policy_ref": "secret://source/mapping",
                "auth_profile_ref": "secret://source/auth",
                "tls_profile_ref": "secret://source/tls",
            }
        ],
    )


def test_doctor_probes_kg_connections_only_and_returns_aggregate_metadata(
    monkeypatch,
) -> None:
    registry = _Registry()
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", _config)
    monkeypatch.setattr(
        "agent_utilities.mcp.kg_server.get_connection_registry", lambda: registry
    )

    result = doctor_module._check_graph_connections(live=True)
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "ok"
    assert registry.probed == ["external-source"]
    assert result["data"] == {
        "configured_count": 1,
        "registered_count": 1,
        "ready_count": 1,
        "probe_failed_count": 0,
        "invalid_declaration_count": 0,
        "duplicate_declaration_count": 0,
        "missing_declaration_count": 0,
        "stalled_mirror_count": 0,
        "roles": {"read": 1},
        "redacted": True,
        "live_probed": True,
    }
    assert "external-source" not in rendered
    assert "secret://" not in rendered
    assert "private.example.test" not in rendered


def test_doctor_fails_closed_without_leaking_connection_probe_errors(
    monkeypatch,
) -> None:
    registry = _Registry(probe_error=True)
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", _config)
    monkeypatch.setattr(
        "agent_utilities.mcp.kg_server.get_connection_registry", lambda: registry
    )

    result = doctor_module._check_graph_connections(live=True)
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "fail"
    assert result["data"]["probe_failed_count"] == 1
    assert result["data"]["ready_count"] == 0
    assert "external-source" not in rendered
    assert "secret://" not in rendered
    assert "private.example.test" not in rendered


def test_doctor_static_check_never_opens_connection_transport(monkeypatch) -> None:
    registry = _Registry(probe_error=True)
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", _config)
    monkeypatch.setattr(
        "agent_utilities.mcp.kg_server.get_connection_registry", lambda: registry
    )

    result = doctor_module._check_graph_connections(live=False)

    assert result["status"] == "ok"
    assert result["data"]["live_probed"] is False
    assert result["data"]["ready_count"] == 0
    assert registry.probed == []
