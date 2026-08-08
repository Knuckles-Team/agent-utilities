"""Server-side Prompts-over-MCP is LIVE (CONCEPT:AU-ECO.mcp.cross-process-prompt-harvest).

``tests/unit/mcp/test_prompt_provider_wiring.py`` proves
``_register_prompt_providers`` calls ``mcp.add_resource`` once per resolved
``*.json`` file, but does so against a ``MagicMock`` server — it never
touches a real FastMCP instance and would keep passing even if resource
registration silently no-oped. This asserts the real thing: a server built
by au's own ``create_mcp_server`` actually serves a fleet package's
``prompts/*.json`` file as a ``prompt://`` resource, using the genuine
fastmcp ``FileResource``, and a client can read the body back — the exact
capability :meth:`~agent_utilities.mcp.multiplexer.MCPMultiplexer
._harvest_prompt_bodies` depends on when graph-os probes this server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def demo_prompt_dir(tmp_path: Path) -> Path:
    provider = tmp_path / "au-live-provider" / "prompts"
    provider.mkdir(parents=True)
    (provider / "live-demo-prompt.json").write_text(
        '{"name": "live-demo-prompt", "description": "served over MCP", '
        '"content": "Body served as a prompt:// resource."}',
        encoding="utf-8",
    )
    return provider


@pytest.mark.asyncio
async def test_create_mcp_server_serves_prompt_resources(
    monkeypatch, demo_prompt_dir: Path
) -> None:
    """``create_mcp_server`` registers a REAL ``FileResource`` per prompt file
    and the built server serves it over a ``prompt://`` resource."""
    from fastmcp.resources import FileResource

    assert FileResource is not None

    monkeypatch.setattr(
        "agent_utilities.core.providers.resolve_prompt_provider_dirs",
        lambda: [("au-live-provider", demo_prompt_dir)],
    )

    from agent_utilities.mcp.server_factory import create_mcp_server

    _args, mcp, _middlewares = create_mcp_server(
        "Prompts Live Path Test", command_args=[]
    )

    resources = await mcp.list_resources()
    uris = {str(resource.uri) for resource in resources}

    assert "prompt://au-live-provider/live-demo-prompt" in uris, (
        "the registered FileResource must publish the prompt JSON file; "
        f"got {sorted(uris)}"
    )

    body = await mcp.read_resource("prompt://au-live-provider/live-demo-prompt")
    text = "".join(str(getattr(chunk, "content", chunk)) for chunk in body)
    assert "live-demo-prompt" in text
    assert "Body served as a prompt:// resource." in text
