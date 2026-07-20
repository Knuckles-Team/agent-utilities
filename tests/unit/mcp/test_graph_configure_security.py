"""Adversarial source-boundary tests for ``graph_configure``.

These tests deliberately use synthetic values.  They prove registration never
persists inline endpoints, credentials, identities, or local paths and that the
write boundary cannot be redirected through a symlink.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from agent_utilities.mcp.tools import analysis_tools


class _FakeMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name: str, description: str = "", tags=None):
        def _decorator(fn):
            self.tools[name] = fn
            return fn

        return _decorator


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "https://sensitive.invalid/mcp"),
        ("AUTH_TOKEN", "synthetic-credential-material"),
        ("AGENT_ID", "synthetic-person"),
        ("CUSTOM_SETTING", "unclassified-inline-material"),
        ("REQUESTS_CA_BUNDLE", "/private/config/ca.pem"),
    ],
)
def test_mcp_registration_rejects_sensitive_literals(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        analysis_tools._validate_mcp_server_definition(
            {"command": "uvx", "env": {field: value}}
        )


def test_mcp_registration_accepts_runtime_references() -> None:
    definition = {
        "command": "uvx",
        "args": ["--from", "agent-utilities", "graph-os"],
        "url": "${GRAPH_OS_URL}",
        "env": {
            "AGENT_ID": "${GRAPH_OS_AGENT_ID}",
            "AUTH_TOKEN": "vault://apps/graph-os/token",
            "MCP_CLIENT_AUTH": "oidc-client-credentials",
            "MCP_TOOL_MODE": "intent",
            "REQUESTS_CA_BUNDLE": "${GRAPH_OS_CA_BUNDLE}",
        },
    }

    assert analysis_tools._validate_mcp_server_definition(definition) is definition


@pytest.mark.parametrize(
    "args",
    [
        ["server", "--token", "synthetic-inline-token"],
        ["server", "--endpoint=https://sensitive.invalid/mcp"],
        ["server", "--config", "relative-private-config.json"],
    ],
)
def test_mcp_registration_rejects_sensitive_command_arguments(args: list[str]) -> None:
    with pytest.raises(ValueError):
        analysis_tools._validate_mcp_server_definition(
            {"command": "uvx", "args": args}
        )


def test_register_mcp_writes_atomically_with_private_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "mcp_config.json"
    monkeypatch.setattr(
        analysis_tools, "_workspace_mcp_config_path", lambda: config_path
    )

    analysis_tools._register_mcp_server(
        "graph-os",
        json.dumps(
            {
                "command": "uvx",
                "args": ["--from", "agent-utilities", "graph-os"],
                "env": {"AUTH_TOKEN": "${GRAPH_OS_TOKEN}"},
            }
        ),
    )

    stored = json.loads(config_path.read_text(encoding="utf-8"))
    assert stored["mcpServers"]["graph-os"]["env"]["AUTH_TOKEN"] == (
        "${GRAPH_OS_TOKEN}"
    )
    if os.name == "posix":
        assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".mcp_config.json.*.tmp"))


def test_register_mcp_rejects_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text('{"mcpServers": {}}', encoding="utf-8")
    linked = tmp_path / "mcp_config.json"
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    monkeypatch.setattr(
        analysis_tools, "_workspace_mcp_config_path", lambda: linked
    )

    with pytest.raises(PermissionError):
        analysis_tools._register_mcp_server(
            "graph-os", json.dumps({"command": "uvx"})
        )

    assert outside.read_text(encoding="utf-8") == '{"mcpServers": {}}'


def test_register_mcp_rejects_existing_inline_material_without_rewriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "mcp_config.json"
    original = json.dumps(
        {
            "mcpServers": {
                "external": {"url": "https://sensitive.invalid/service"}
            }
        }
    )
    config_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        analysis_tools, "_workspace_mcp_config_path", lambda: config_path
    )

    with pytest.raises(ValueError):
        analysis_tools._register_mcp_server(
            "graph-os", json.dumps({"command": "uvx"})
        )

    assert config_path.read_text(encoding="utf-8") == original


def test_atomic_write_failure_preserves_existing_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "mcp_config.json"
    original = '{"mcpServers": {"existing": {"command": "uvx"}}}\n'
    config_path.write_text(original, encoding="utf-8")

    def _replace_failure(_source, _destination) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(analysis_tools.os, "replace", _replace_failure)
    with pytest.raises(OSError):
        analysis_tools._atomic_private_json_write(
            config_path, {"mcpServers": {"new": {"command": "uvx"}}}
        )

    assert config_path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".mcp_config.json.*.tmp"))


def test_workspace_mcp_path_rejects_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_utilities.core import config, workspace

    monkeypatch.setattr(workspace, "get_agent_workspace", lambda: tmp_path)
    monkeypatch.setattr(
        config, "setting", lambda key, default="": "../outside.json"
    )

    with pytest.raises(PermissionError):
        analysis_tools._workspace_mcp_config_path()


def test_graph_configure_registration_error_is_source_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMCP()
    analysis_tools.register_analysis_tools(fake)
    sensitive_detail = "https://sensitive.invalid/mcp?token=synthetic"

    def _fail(_name: str, _definition: str) -> None:
        raise RuntimeError(sensitive_detail)

    monkeypatch.setattr(analysis_tools, "_register_mcp_server", _fail)
    result = fake.tools["graph_configure"](
        action="register_mcp",
        config_key="graph-os",
        config_value=json.dumps({"command": "uvx"}),
    )

    assert sensitive_detail not in result
    assert json.loads(result) == {
        "error": "MCP registration rejected",
        "error_type": "RuntimeError",
    }


def test_graph_configure_preserves_filesystem_policy_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMCP()
    analysis_tools.register_analysis_tools(fake)

    def _deny(_name: str, _definition: str) -> None:
        raise PermissionError("synthetic private path")

    monkeypatch.setattr(analysis_tools, "_register_mcp_server", _deny)
    with pytest.raises(PermissionError) as denied:
        fake.tools["graph_configure"](
            action="register_mcp",
            config_key="graph-os",
            config_value=json.dumps({"command": "uvx"}),
        )
    assert str(denied.value) == "configuration operation denied"


def test_stardog_action_rejects_inline_connection_material() -> None:
    fake = _FakeMCP()
    analysis_tools.register_analysis_tools(fake)

    result = fake.tools["graph_configure"](
        action="push_to_stardog",
        config_key="",
        config_value=json.dumps(
            {
                "endpoint": "https://sensitive.invalid",
                "password": "synthetic-credential-material",
            }
        ),
    )

    assert "sensitive.invalid" not in result
    assert "synthetic-credential-material" not in result
    assert "inline Stardog connection material" in result


def test_database_action_rejects_inline_endpoint() -> None:
    fake = _FakeMCP()
    analysis_tools.register_analysis_tools(fake)

    result = fake.tools["graph_configure"](
        action="verify_databases",
        config_key="",
        config_value=json.dumps({"dsn": "postgresql://synthetic.invalid/db"}),
    )

    assert "synthetic.invalid" not in result
    assert "inline database endpoints" in result


def test_set_config_rejects_inline_sensitive_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.deployment as deployment
    from agent_utilities.core import config

    fake = _FakeMCP()
    analysis_tools.register_analysis_tools(fake)
    monkeypatch.setattr(
        deployment,
        "config_reference",
        lambda: [
            {
                "section": "test",
                "fields": [
                    {
                        "env": "GRAPH_SERVICE_ENDPOINTS",
                        "secret": False,
                    }
                ],
            }
        ],
    )
    persisted: list[tuple[str, object]] = []
    monkeypatch.setattr(
        config, "save_config_item", lambda key, value: persisted.append((key, value))
    )

    result = fake.tools["graph_configure"](
        action="set_config",
        config_key="GRAPH_SERVICE_ENDPOINTS",
        config_value="tcp://synthetic.invalid:9000",
    )

    assert "synthetic.invalid" not in result
    assert "cannot be persisted inline" in result
    assert persisted == []


def test_get_config_redacts_endpoint_and_path_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.deployment as deployment

    fake = _FakeMCP()
    analysis_tools.register_analysis_tools(fake)
    monkeypatch.setattr(
        deployment,
        "config_reference",
        lambda: [
            {
                "section": "test",
                "fields": [
                    {"env": "GRAPH_SERVICE_ENDPOINTS", "secret": False},
                    {"env": "WORKSPACE_PATH", "secret": False},
                ],
            }
        ],
    )
    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", "tcp://synthetic.invalid:9000")
    monkeypatch.setenv("WORKSPACE_PATH", "/private/synthetic/workspace")

    endpoint_result = json.loads(
        fake.tools["graph_configure"](
            action="get_config",
            config_key="GRAPH_SERVICE_ENDPOINTS",
            config_value="",
        )
    )
    path_result = json.loads(
        fake.tools["graph_configure"](
            action="get_config",
            config_key="WORKSPACE_PATH",
            config_value="",
        )
    )

    assert endpoint_result["value"] == "***"
    assert path_result["value"] == "***"
