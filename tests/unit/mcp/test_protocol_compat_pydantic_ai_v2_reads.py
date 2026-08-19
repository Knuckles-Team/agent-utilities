"""Contract tests for AU's native Pydantic-AI MCP compatibility boundary.

Pydantic-AI 2.29.0 owns the SDK-v1/v2 field adaptation and modern FastMCP
session handling. These tests make that decision explicit: AU must not silently
apply a copied method body to a different release, and both field spellings
remain covered at the AU-owned boundary for mixed-generation resilience paths.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("pydantic_ai")
pytest.importorskip("fastmcp")

from agent_utilities.mcp import protocol_compat  # noqa: E402


def _skip_unless_contract_is_installed() -> None:
    installed = importlib.metadata.version("pydantic-ai-slim")
    if installed != protocol_compat._PYDANTIC_AI_CONTRACT_VERSION:
        pytest.skip(
            "installed pydantic-ai-slim "
            f"{installed} != the verified AU contract "
            f"{protocol_compat._PYDANTIC_AI_CONTRACT_VERSION}"
        )


def test_native_toolset_surface_is_used_without_monkeypatch() -> None:
    """The supported release owns the current and legacy MCP reads upstream."""
    _skip_unless_contract_is_installed()
    import pydantic_ai.mcp as pydantic_mcp

    original_aenter = pydantic_mcp.MCPToolset.__aenter__
    original_get_tools = pydantic_mcp.MCPToolset.get_tools

    protocol_compat._installed = False
    protocol_compat._toolset_reads_patched = False
    protocol_compat.install_mcp_v2_bridge()

    assert pydantic_mcp.MCPToolset.__aenter__ is original_aenter
    assert pydantic_mcp.MCPToolset.get_tools is original_get_tools
    assert "_v2_" not in original_aenter.__name__
    assert "_v2_" not in original_get_tools.__name__


def test_upstream_methods_retain_both_mcp_field_surfaces() -> None:
    """The real 2.29 source uses its SDK-neutral compatibility helpers."""
    _skip_unless_contract_is_installed()
    import pydantic_ai.mcp as pydantic_mcp

    aenter_source = inspect.getsource(pydantic_mcp.MCPToolset.__aenter__)
    get_tools_source = inspect.getsource(pydantic_mcp.MCPToolset.get_tools)
    assert "mcp_field" in aenter_source
    assert "mcp_validated_field" in get_tools_source
    assert "mcp_optional_field" in get_tools_source
    # A direct deprecated read would bypass the upstream compatibility helper.
    assert ".serverInfo" not in aenter_source
    assert ".inputSchema" not in get_tools_source
    assert ".outputSchema" not in get_tools_source
    assert ".taskSupport" not in get_tools_source


def test_install_bridge_refuses_an_unverified_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Version drift fails closed rather than silently disabling the contract."""
    real_version = importlib.metadata.version

    def fake_version(name: str) -> str:
        if name == "pydantic-ai-slim":
            return "0.0.0-unverified"
        return real_version(name)

    protocol_compat._toolset_reads_patched = False
    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    with pytest.raises(RuntimeError, match="unverified compatibility surface"):
        protocol_compat._install_pydantic_ai_v2_read_bridge()
    assert protocol_compat._toolset_reads_patched is False


@pytest.mark.parametrize(
    ("current_name", "legacy_name"),
    [("server_info", "serverInfo"), ("input_schema", "inputSchema")],
)
def test_mcp_field_matrix_accepts_current_and_legacy_surfaces(
    current_name: str, legacy_name: str
) -> None:
    """The AU-owned resolver prefers current names and accepts legacy names."""
    current = SimpleNamespace(**{current_name: "current"})
    both = SimpleNamespace(**{current_name: "current", legacy_name: "legacy"})
    legacy = SimpleNamespace(**{legacy_name: "legacy"})

    assert protocol_compat._read_mcp_field(current, current_name, legacy_name) == "current"
    assert protocol_compat._read_mcp_field(both, current_name, legacy_name) == "current"
    assert protocol_compat._read_mcp_field(legacy, current_name, legacy_name) == "legacy"


def test_mcp_field_matrix_rejects_an_unknown_surface() -> None:
    with pytest.raises(AttributeError, match="neither"):
        protocol_compat._read_mcp_field(SimpleNamespace(), "current", "legacy")


def test_real_toolset_reads_current_sdk_tool_fields_without_warning() -> None:
    """Exercise the real 2.29 MCPToolset against an SDK-v2 model instance."""
    _skip_unless_contract_is_installed()
    import mcp.types as mcp_types
    import pydantic_ai.mcp as pydantic_mcp

    tool = mcp_types.Tool(
        name="do_thing",
        description="does a thing",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        execution=mcp_types.ToolExecution(task_support="optional"),
    )
    toolset = pydantic_mcp.MCPToolset.__new__(pydantic_mcp.MCPToolset)
    toolset.max_retries = None
    toolset.include_return_schema = True
    toolset.prefer_tasks = True
    toolset.cache_tools = False
    toolset.list_tools = AsyncMock(return_value=[tool])
    ctx = SimpleNamespace(max_retries=1)

    async def run() -> dict[str, object]:
        return await toolset.get_tools(ctx)

    # SDK v2 stores canonical snake_case fields. The real upstream method must
    # produce the correct definition without invoking deprecated aliases.
    tools = asyncio.run(run())
    tool_def = tools["do_thing"].tool_def
    assert tool_def.parameters_json_schema == tool.input_schema
    assert tool_def.return_schema == tool.output_schema
    # FastMCP 4's SDK-v2 session owns task routing; Pydantic-AI intentionally
    # leaves the legacy client-side task flag disabled on this surface.
    assert tool_def.metadata["task"] is False


def test_version_contract_matches_manifest_lock_and_image() -> None:
    """AU-owned resolver, lock, requirements, and image use one exact version."""
    root = Path(__file__).resolve().parents[3]
    manifest = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    expected = f"=={protocol_compat._PYDANTIC_AI_CONTRACT_VERSION}"
    declared = [
        requirement
        for requirements in manifest["project"]["optional-dependencies"].values()
        for requirement in requirements
        if requirement.startswith("pydantic-ai-slim")
    ]
    assert declared
    assert all(requirement.endswith(expected) for requirement in declared)

    lock_text = (root / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "pydantic-ai-slim"' in lock_text
    assert 'version = "2.29.0"' in lock_text
    assert all(
        f'specifier = "{expected}"' in line
        for line in lock_text.splitlines()
        if '{ name = "pydantic-ai-slim"' in line and "specifier =" in line
    )

    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    image = (root / "docker/graphos-unified.Dockerfile").read_text(encoding="utf-8")
    assert f"pydantic-ai-slim[mcp,openai,anthropic,ag-ui,ui,web,cli]{expected}" in requirements
    assert f'"pydantic-ai-slim[mcp,openai,ag-ui,ui,web,cli,google,groq]{expected}"' in image
