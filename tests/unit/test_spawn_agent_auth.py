"""A spawned agent binds remote toolsets only through the graph-os host port.

CONCEPT:AU-ORCH.adapter.transport-toolset-factory — `execute_agent` resolves a
fleet server and binds its remote toolset. agent-utilities carries no MCP
client and no outbound credentials: graph-os registers a
:class:`~agent_utilities.host_ports.RemoteToolsetProvider` that builds the
toolset with its own service-account auth. These tests pin that the harness
hands the resolved endpoint and toolset id to that port, and fails closed when
no host is attached.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities import host_ports
from agent_utilities.orchestration import agent_runner


def _remote_meta() -> dict[str, Any]:
    return {
        "type": "server",
        "toolset_id": "repository-manager-mcp",
        "tools": [],
        "capabilities": [],
    }


def _configured_model() -> SimpleNamespace:
    return SimpleNamespace(
        id="synthetic-standard",
        provider="openai",
        base_url=None,
        api_key_ref=None,
    )


@pytest.fixture
def remote_port() -> Iterator[list[tuple[str, str | None, float]]]:
    calls: list[tuple[str, str | None, float]] = []

    def _build(url: str, toolset_id: str | None, timeout: float) -> Any:
        calls.append((url, toolset_id, timeout))
        return SimpleNamespace(id=toolset_id, url=url)

    def _load(_path: Path) -> list[Any]:
        return []

    host_ports.register_remote_toolset_provider(
        host_ports.RemoteToolsetProvider(
            build_url_toolset=_build, load_config_toolsets=_load
        )
    )
    try:
        yield calls
    finally:
        host_ports.register_remote_toolset_provider(None)


def _patch_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_runner,
        "_fleet_server_url",
        lambda _server: "https://fleet.example.invalid/mcp",
    )
    monkeypatch.setattr(
        agent_runner, "_configured_model_for_class", lambda _class: _configured_model()
    )


def test_remote_toolset_is_built_by_the_host_port(
    monkeypatch: pytest.MonkeyPatch,
    remote_port: list[tuple[str, str | None, float]],
) -> None:
    _patch_resolution(monkeypatch)

    config = agent_runner._build_execution_config(
        object(), "code-enhancer", _remote_meta()
    )

    assert config["mcp_toolsets"], "a remote toolset should be bound"
    assert remote_port == [
        ("https://fleet.example.invalid/mcp", "repository-manager-mcp", 60.0)
    ]
    assert config["mcp_toolsets"][0].id == "repository-manager-mcp"


def test_remote_toolset_fails_closed_without_a_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_resolution(monkeypatch)
    host_ports.register_remote_toolset_provider(None)

    with pytest.raises(RuntimeError, match="graph-os host"):
        agent_runner._build_execution_config(object(), "code-enhancer", _remote_meta())
