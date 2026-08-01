"""Live proof that `agent_utilities.mcp.toolset_factory` works end-to-end against a
real fastmcp-4 server (CONCEPT:AU-ECO.mcp.protocol-compat-bridge).

fastmcp 4 is au's single default (`[mcp]`, `pyproject.toml`) — there is no longer a
fastmcp-3-vs-4 client/server split to guard, so this replaces the prior dual-extra
cross-version regression test (which spawned a separate fastmcp<4 interpreter to
prove server-side down-negotiation). What actually needs a live regression guard
now is `agent_utilities/mcp/protocol_compat.py`: empirically, a bare
`pydantic_ai.mcp.MCPToolset` fails outright against a real fastmcp-4 server unless
(a) the underlying `fastmcp.Client` is pinned to `mode="legacy"` (fastmcp 4 defaults
to `mode="auto"`, which negotiates the modern `server/discover` era and leaves
`Client.initialize_result` `None` — `MCPToolset.__aenter__` asserts it's non-`None`)
and (b) four MCP-SDK-v2 attribute/exception renames not covered by fastmcp's own
`_compat.py` bridge are patched (`PromptsCapability`/`ResourcesCapability`/
`ToolsCapability.listChanged`, `ToolExecution.taskSupport`,
`mcp.shared.exceptions.McpError`). This test spawns a real fastmcp-4 server (stdio
subprocess) with a `SkillProvider` resource plus a plain tool, builds the toolset
through `build_stdio_toolset` (au's actual production call path, not a hand-rolled
`MCPToolset`), and proves connect + resource read + tool call all round-trip.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_SERVER_SCRIPT = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    from fastmcp import FastMCP
    from fastmcp.server.providers.skills import SkillProvider

    skill_dir = Path(sys.argv[1])
    mcp = FastMCP("au-fastmcp4-live-test-server")
    mcp.add_provider(SkillProvider(skill_dir))

    @mcp.tool
    def add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    if __name__ == "__main__":
        mcp.run(show_banner=False)
    """
)


@pytest.fixture
def demo_skill_dir(tmp_path: Path) -> Path:
    skill_root = tmp_path / "demo_skill"
    skill_root.mkdir()
    (skill_root / "SKILL.md").write_text(
        "---\n"
        "description: fastmcp-4 live round-trip fixture\n"
        "---\n"
        "# Demo Skill\n\n"
        "Served over skill:// resources by a fastmcp-4 server, read through "
        "au's own MCPToolset construction path.\n",
        encoding="utf-8",
    )
    return skill_root


@pytest.mark.asyncio
async def test_toolset_factory_round_trips_a_real_fastmcp4_server(
    tmp_path: Path, demo_skill_dir: Path
) -> None:
    """`build_stdio_toolset` connects, lists resources/tools, and calls a tool
    against a live fastmcp-4 server — the guard protecting the `mode="legacy"` +
    MCP-SDK-v2 attribute bridges in `protocol_compat.py`."""
    import fastmcp

    if int(fastmcp.__version__.split(".")[0]) < 4:
        pytest.skip(
            f"this guard targets the fastmcp-4 default (`[mcp]`); the current "
            f"interpreter has fastmcp {fastmcp.__version__}"
        )

    from agent_utilities.mcp.toolset_factory import build_stdio_toolset

    server_script = tmp_path / "fastmcp4_live_server.py"
    server_script.write_text(_SERVER_SCRIPT, encoding="utf-8")

    toolset = build_stdio_toolset(
        sys.executable, [str(server_script), str(demo_skill_dir)]
    )
    async with toolset:
        assert toolset.client.mode == "legacy"

        resources = await toolset.list_resources()
        uris = {str(resource.uri) for resource in resources}
        assert "skill://demo_skill/SKILL.md" in uris

        skill_md = await toolset.client.read_resource("skill://demo_skill/SKILL.md")
        text = "".join(getattr(chunk, "text", "") for chunk in skill_md)
        assert "Demo Skill" in text

        tools = await toolset.get_tools(_FakeCtx())
        assert "add" in tools
        tool = tools["add"]
        result = await toolset.call_tool(
            "add", {"a": 2, "b": 3}, ctx=_FakeCtx(), tool=tool
        )
        data = result.data if hasattr(result, "data") else result
        assert data == 5


class _FakeCtx:
    max_retries = 1
