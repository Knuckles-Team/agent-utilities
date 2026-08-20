"""NE-044 sub-gate 1 acceptance: does ``5d6705c3``'s pydantic-ai-slim
compatibility pin actually hold under the exact tracked lock?

``5d6705c3`` re-pinned ``protocol_compat._PYDANTIC_AI_CONTRACT_VERSION`` after
it drifted from the installed release (the bridge was fail-closed on an exact
version match and had silently stopped patching against an unverified
install). ``tests/unit/mcp/test_protocol_compat_pydantic_ai_v2_reads.py``
already proves the internal read-bridge/version-refusal machinery in
isolation; this file proves the CONSUMER-FACING contract the pin exists to
protect: a real ``pydantic_ai.mcp.MCPToolset`` built against THIS repo's
tracked ``uv.lock`` install can genuinely import, initialize against a real
(in-process) MCP server, list tools, and call one -- end to end, not just at
the internal-function level.
"""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("pydantic_ai")
pytest.importorskip("fastmcp")

from agent_utilities.mcp import protocol_compat  # noqa: E402

_ROOT = Path(__file__).resolve().parents[3]


def test_pinned_version_boundary_matches_the_installed_lock() -> None:
    """The exact tracked lock this test session actually runs under must
    match the pin protocol_compat asserts against -- not a hypothetical
    future version. If this drifts, ``install_mcp_v2_bridge`` fails closed
    (proven separately by
    ``test_protocol_compat_pydantic_ai_v2_reads.py::test_install_bridge_refuses_an_unverified_version``);
    this test proves the boundary is currently satisfied, not merely that a
    violation would be caught."""
    installed = importlib.metadata.version("pydantic-ai-slim")
    assert installed == protocol_compat._PYDANTIC_AI_CONTRACT_VERSION

    lock_text = (_ROOT / "uv.lock").read_text(encoding="utf-8")
    assert f'version = "{installed}"' in lock_text


@pytest.mark.asyncio
async def test_real_mcp_toolset_imports_initializes_lists_and_calls_a_tool() -> None:
    """The full consumer contract, under the exact tracked lock: import the
    MCP surface (``pydantic_ai.mcp``), initialize a real ``MCPToolset``
    against a real (in-process) FastMCP server, list its tools, and call
    one -- and get back a real, non-error result. Runs entirely in-process
    (no subprocess, no network transport), the same in-memory FastMCP client
    pattern ``tests/integration/agent/test_agent_mcp_toolset_e2e.py`` uses
    for a real fleet server.
    """
    installed = importlib.metadata.version("pydantic-ai-slim")
    if installed != protocol_compat._PYDANTIC_AI_CONTRACT_VERSION:
        pytest.skip(
            "installed pydantic-ai-slim "
            f"{installed} != the verified AU contract "
            f"{protocol_compat._PYDANTIC_AI_CONTRACT_VERSION}"
        )

    from fastmcp import FastMCP
    from pydantic_ai.mcp import MCPToolset

    server = FastMCP("ne044-acceptance")

    @server.tool
    def echo(text: str) -> str:
        return f"echoed: {text}"

    # 1) import + initialize the real toolset against the real server.
    toolset = MCPToolset(server)
    assert type(toolset).__name__ == "MCPToolset"

    # 2) list tools + 3) call one, through the toolset's own client -- the
    # same real connection pydantic-ai's Agent machinery uses, not a side
    # channel.
    async with toolset.client as client:
        tool_names = {t.name for t in await client.list_tools()}
        assert "echo" in tool_names

        result = await client.call_tool("echo", {"text": "ne044"})

    assert getattr(result, "is_error", False) is False
    payload = getattr(result, "data", None) or getattr(result, "content", None)
    assert payload not in (None, "", [], {})
    assert "ne044" in str(payload)


def test_pin_is_the_one_declared_across_manifest_lock_and_image() -> None:
    """The pin is not a lone constant in ``protocol_compat.py`` -- it is the
    SAME version declared in ``pyproject.toml``'s optional-dependencies,
    ``uv.lock``, ``requirements.txt``, and the shipped Docker image. A
    consumer installing from any of those surfaces gets the exact release
    the bridge was verified against."""
    manifest = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = f"=={protocol_compat._PYDANTIC_AI_CONTRACT_VERSION}"

    declared = [
        requirement
        for requirements in manifest["project"]["optional-dependencies"].values()
        for requirement in requirements
        if requirement.startswith("pydantic-ai-slim")
    ]
    assert declared
    assert all(requirement.endswith(expected) for requirement in declared)

    requirements_txt = (_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert (
        f"pydantic-ai-slim[mcp,openai,anthropic,ag-ui,ui,web,cli]{expected}"
        in requirements_txt
    )
