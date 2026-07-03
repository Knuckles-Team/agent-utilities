"""A client-role process must NOT spawn a centralized KG server (task #8).

Regression: the messaging daemon / served graph-os run KG_DAEMON_ROLE=client and reach the
KG via the shared engine + graph-os, but ``get_kg_client`` still tried to spawn a local
HTTP KG server via ``uv run`` — which a slim serving container lacks, so every turn logged
"Failed to spawn background centralized KG server: ... No such file or directory: 'uv'".
"""

from __future__ import annotations

import pytest

from agent_utilities.mcp import kg_coordinator
from agent_utilities.mcp.kg_coordinator import KGCoordinator


@pytest.fixture
def _spy(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, int] = {"spawn": 0}
    monkeypatch.setattr(
        KGCoordinator, "is_server_healthy", classmethod(lambda cls, *a, **k: False)
    )
    monkeypatch.setattr(
        KGCoordinator,
        "spawn_server",
        classmethod(
            lambda cls, *a, **k: calls.__setitem__("spawn", calls["spawn"] + 1)
        ),
    )
    return calls


def test_client_role_does_not_spawn(monkeypatch: pytest.MonkeyPatch, _spy) -> None:
    monkeypatch.setattr(kg_coordinator, "setting", lambda *a, **k: "client")
    KGCoordinator.get_kg_client()
    assert _spy["spawn"] == 0  # no spawn, no `uv` shell-out


def test_host_role_spawns_when_unhealthy(monkeypatch: pytest.MonkeyPatch, _spy) -> None:
    monkeypatch.setattr(kg_coordinator, "setting", lambda *a, **k: "host")
    KGCoordinator.get_kg_client()
    assert _spy["spawn"] == 1  # host owns the server → spawns when unhealthy
