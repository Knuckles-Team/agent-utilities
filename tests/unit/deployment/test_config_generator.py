"""Tests for the full-deployment config generator / validator.

Covers complete generation per profile, secret redaction, the grouped reference,
the doctor, and the live-path through the ``graph_configure`` MCP actions.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.deployment import (
    PROFILES,
    config_doctor,
    config_reference,
    generate_config,
    generate_mcp_config,
    write_config,
)


# ── generation ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("profile", PROFILES)
def test_generate_config_is_complete(profile):
    cfg = generate_config(profile)
    # The whole AgentConfig surface is present (>=250 fields).
    assert len(cfg) >= 250
    assert cfg["GRAPH_BACKEND"]  # always set


def test_generate_config_profile_presets():
    tiny = generate_config("tiny")
    snp = generate_config("single-node-prod")
    ent = generate_config("enterprise")
    assert not tiny.get("GRAPH_DB_URI")  # zero-infra
    assert snp["GRAPH_DB_URI"] and snp["GRAPH_PG_AGE"] == "1"
    assert ent["STATE_DB_URI"] and ent["KG_AUTH_REQUIRED"] == "1"
    assert ent["TASK_QUEUE_BACKEND"] == "kafka"


def test_generate_config_redacts_populated_secrets():
    from agent_utilities.deployment.config_generator import _is_secret

    cfg = generate_config("enterprise", redact_secrets=True)
    # No populated credential survives (None/"" are acceptable placeholders).
    for key, val in cfg.items():
        if _is_secret(key):
            assert val in (None, ""), f"{key} not redacted: {val!r}"
    # Non-credential config keys with 'SECRET' in the name are NOT clobbered.
    assert cfg["SECRETS_BACKEND"] == "vault"
    assert cfg["SECRETS_VAULT_URL"]  # preset URL preserved


def test_generate_config_unknown_profile():
    with pytest.raises(ValueError):
        generate_config("bogus")


def test_write_config_roundtrips(tmp_path):
    out = tmp_path / "config.json"
    res = write_config("single-node-prod", out)
    assert res["status"] == "success"
    assert res["keys"] >= 250
    loaded = json.loads(out.read_text())
    assert loaded["GRAPH_DB_URI"] == generate_config("single-node-prod")["GRAPH_DB_URI"]


# ── minimal mcp_config (OS-5.65) ────────────────────────────────────────────
def test_generate_mcp_config_includes_both_servers():
    cfg = generate_mcp_config("tiny")
    servers = cfg["mcpServers"]
    # graph-os = just the KG; mcp-multiplexer = the whole fleet — both offered.
    assert set(servers) == {"graph-os", "mcp-multiplexer"}
    assert servers["graph-os"]["command"] == "uv"
    assert servers["graph-os"]["args"] == ["run", "graph-os"]
    assert servers["mcp-multiplexer"]["args"] == ["run", "mcp-multiplexer"]


def test_generate_mcp_config_multiplexer_dynamic_mode_and_child_config():
    mux = generate_mcp_config("tiny")["mcpServers"]["mcp-multiplexer"]
    assert mux["env"]["MCP_MULTIPLEXER_MODE"] == "dynamic"
    assert mux["env"]["MCP_CONFIG"] == "${workspaceFolder}/mcp_config.json"


def test_generate_mcp_config_envs_are_minimal():
    # Only workspace/agent (+ the multiplexer's mode/MCP_CONFIG) — no model/secrets.
    servers = generate_mcp_config("tiny")["mcpServers"]
    assert set(servers["graph-os"]["env"]) == {"AGENT_ID", "WORKSPACE_PATH"}
    assert set(servers["mcp-multiplexer"]["env"]) == {
        "AGENT_ID",
        "WORKSPACE_PATH",
        "MCP_MULTIPLEXER_MODE",
        "MCP_CONFIG",
    }


def test_generate_mcp_config_no_fleet_is_graph_os_only():
    cfg = generate_mcp_config("tiny", fleet=False)
    assert set(cfg["mcpServers"]) == {"graph-os"}


def test_generate_mcp_config_unknown_profile():
    with pytest.raises(ValueError):
        generate_mcp_config("bogus")


def test_setup_config_mcp_subcommand_emits_valid_json(capsys):
    from agent_utilities.deployment.cli import main

    rc = main(["mcp"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out["mcpServers"]) == {"graph-os", "mcp-multiplexer"}

    rc = main(["mcp", "--no-fleet"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out["mcpServers"]) == {"graph-os"}


# ── reference ──────────────────────────────────────────────────────────────
def test_config_reference_groups_all_fields():
    ref = config_reference()
    sections = {s["section"] for s in ref}
    assert len(sections) >= 5  # multiple subsystems detected
    total = sum(len(s["fields"]) for s in ref)
    assert total >= 250  # every field appears exactly once
    # A known field is grouped and carries env + type.
    flat = {f["env"]: f for s in ref for f in s["fields"]}
    assert "GRAPH_BACKEND" in flat
    assert flat["GRAPH_BACKEND"]["type"]


def test_config_reference_marks_secrets():
    flat = {f["env"]: f for s in config_reference() for f in s["fields"]}
    assert flat["OPENAI_API_KEY"]["secret"] is True
    assert flat["OPENAI_API_KEY"]["default"] == "***"


# ── doctor ─────────────────────────────────────────────────────────────────
def test_doctor_tiny_healthy(tmp_path):
    out = tmp_path / "c.json"
    write_config("tiny", out)
    rep = config_doctor("tiny", out)
    assert rep["status"] == "success"
    assert rep["healthy"] is True


def test_doctor_enterprise_flags_missing(tmp_path):
    # An enterprise config with the required DSN/auth blanked must be unhealthy.
    cfg = generate_config("enterprise")
    cfg["GRAPH_DB_URI"] = ""
    cfg["STATE_DB_URI"] = ""
    cfg["AUTH_JWT_JWKS_URI"] = ""
    out = tmp_path / "c.json"
    out.write_text(json.dumps(cfg))
    rep = config_doctor("enterprise", out)
    assert rep["healthy"] is False
    req = next(c for c in rep["checks"] if c["check"] == "required_keys")
    assert "GRAPH_DB_URI" in req["missing"]


def test_doctor_unreadable_config(tmp_path):
    rep = config_doctor("tiny", tmp_path / "does-not-exist.json")
    assert rep["status"] == "error"


# ── live path: graph_configure MCP actions ─────────────────────────────────
class _MockMCP:
    def __init__(self):
        self.funcs = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.funcs[fn.__name__] = fn
            return fn

        return deco

    def custom_route(self, *a, **k):
        def deco(fn):
            self.funcs[fn.__name__] = fn
            return fn

        return deco


@pytest.fixture
def registered_tools():
    mock_mcp = _MockMCP()
    engine = MagicMock()
    engine.backend = MagicMock()
    engine.backend.read_only = False
    with patch(
        "agent_utilities.mcp.server_factory.create_mcp_server",
        return_value=(None, mock_mcp, []),
    ):
        with patch("agent_utilities.mcp.kg_server._get_engine", return_value=engine):
            from agent_utilities.mcp.kg_server import _build_server

            _build_server()
    return mock_mcp.funcs


@pytest.mark.asyncio
async def test_graph_configure_generate_config_live_path(registered_tools, tmp_path):
    from agent_utilities.mcp import kg_server

    out = tmp_path / "gen.json"
    raw = await kg_server._execute_tool(
        "graph_configure",
        action="generate_config",
        config_key="single-node-prod",
        config_value=json.dumps({"out": str(out)}),
    )
    res = json.loads(raw)
    assert res["status"] == "success"
    assert out.exists()


@pytest.mark.asyncio
async def test_graph_configure_config_reference_live_path(registered_tools):
    from agent_utilities.mcp import kg_server

    raw = await kg_server._execute_tool("graph_configure", action="config_reference")
    ref = json.loads(raw)
    assert isinstance(ref, list) and ref and "section" in ref[0]
