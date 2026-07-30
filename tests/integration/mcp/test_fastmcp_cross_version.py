"""Cross-version regression guard for the fastmcp-3 -> fastmcp-4 dual-extra
migration (servers-first strategy, see `[mcp-v4]` in ``pyproject.toml``).

au ships two extras side by side: `[mcp]` keeps `pydantic_ai.mcp` CLIENT code
on fastmcp 3.x, while `[mcp-v4]` lets a SERVER (anything constructing
``FastMCP(...)`` — e.g. ``agent_utilities.mcp.server_factory``) run on the
fastmcp 4 beta line. That split is only safe because a fastmcp-4 SERVER
negotiates the wire protocol DOWN to whatever (older) revision a connecting
client requests, instead of demanding the newest protocol. This test proves
that guarantee with a REAL subprocess round trip — not a mocked capability
model — so a future fastmcp-4 beta bump that drops down-negotiation is caught
here, before it silently breaks every fastmcp-3 client already deployed across
the fleet (agent/factory.py, mcp/toolset_factory.py, graph/executor.py,
core/config.py all stay on ``pydantic_ai.mcp`` / fastmcp-slim<4 permanently).

A single Python interpreter can only have one fastmcp major version imported
at a time, so the CLIENT side runs in-process (whatever fastmcp this test
venv has — normally fastmcp 3.x, matching the default `[mcp]` extra and every
real au client), and the SERVER side is spawned as a stdio subprocess using a
SEPARATE interpreter that has fastmcp>=4 installed (a `[mcp-v4]` venv). Point
``AGENT_UTILITIES_FASTMCP4_PYTHON`` at that interpreter's ``python`` binary to
run this for real; without it the test skips with an explicit reason instead
of silently passing, per CONCEPT:AU-OS.observability.no-op-without-metrics
(gates should say clearly why they didn't run, never pass without evidence).
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_FASTMCP4_PYTHON_ENV = "AGENT_UTILITIES_FASTMCP4_PYTHON"

# A minimal fastmcp-4 server exposing a SkillProvider (stable across both
# fastmcp 3 and 4 — verified directly against both wheels) plus one plain
# tool, run under the SEPARATE fastmcp>=4 interpreter this test spawns.
_SERVER_SCRIPT = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    from fastmcp import FastMCP
    from fastmcp.server.providers.skills import SkillProvider

    skill_dir = Path(sys.argv[1])
    mcp = FastMCP("au-cross-version-test-server")
    mcp.add_provider(SkillProvider(skill_dir))

    @mcp.tool
    def add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    if __name__ == "__main__":
        mcp.run()
    """
)


def _fastmcp4_python() -> str | None:
    """Resolve the sibling fastmcp>=4 interpreter, if one is configured."""
    import os

    candidate = os.environ.get(_FASTMCP4_PYTHON_ENV, "").strip()
    if not candidate:
        return None
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    path = Path(candidate)
    return str(path) if path.exists() else None


@pytest.fixture
def demo_skill_dir(tmp_path: Path) -> Path:
    skill_root = tmp_path / "demo_skill"
    skill_root.mkdir()
    (skill_root / "SKILL.md").write_text(
        "---\n"
        "description: fastmcp cross-version regression fixture\n"
        "---\n"
        "# Demo Skill\n\n"
        "Served over skill:// resources by a fastmcp-4 server, read by a "
        "fastmcp-3 client.\n",
        encoding="utf-8",
    )
    return skill_root


@pytest.mark.asyncio
async def test_fastmcp3_client_reads_fastmcp4_server_skill_and_calls_tool(
    tmp_path: Path, demo_skill_dir: Path
) -> None:
    """An older-protocol client can list/read `skill://` resources and call a
    tool on a fastmcp-4-hosted server — the guard protecting servers-first."""
    server_python = _fastmcp4_python()
    if server_python is None:
        pytest.skip(
            "cross-version proof requires a sibling fastmcp>=4 interpreter; "
            f"set {_FASTMCP4_PYTHON_ENV} to its python executable (e.g. a "
            "venv with `pip install agent-utilities[mcp-v4]`) to run this "
            "for real — see docstring for why this can't run in one venv"
        )

    import fastmcp as client_fastmcp
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    if int(client_fastmcp.__version__.split(".")[0]) >= 4:
        pytest.skip(
            "this guard asserts the fastmcp-3 CLIENT / fastmcp-4 SERVER "
            f"direction (servers-first); the current interpreter already has "
            f"fastmcp {client_fastmcp.__version__} — run it under the "
            "default `[mcp]` (fastmcp 3.x) environment instead"
        )

    server_script = tmp_path / "cross_version_server.py"
    server_script.write_text(_SERVER_SCRIPT, encoding="utf-8")

    transport = StdioTransport(
        command=server_python,
        args=[str(server_script), str(demo_skill_dir)],
    )
    client = Client(transport)
    async with client:
        negotiated = getattr(client.initialize_result, "protocolVersion", None)
        assert negotiated, "server did not report a negotiated protocol version"

        resources = await client.list_resources()
        uris = {str(resource.uri) for resource in resources}
        assert "skill://demo_skill/SKILL.md" in uris
        assert "skill://demo_skill/_manifest" in uris

        skill_md = await client.read_resource("skill://demo_skill/SKILL.md")
        text = "".join(getattr(chunk, "text", "") for chunk in skill_md)
        assert "Demo Skill" in text

        manifest = await client.read_resource("skill://demo_skill/_manifest")
        manifest_text = "".join(getattr(chunk, "text", "") for chunk in manifest)
        assert "demo_skill" in manifest_text

        result = await client.call_tool("add", {"a": 2, "b": 3})
        data = result.data if hasattr(result, "data") else result
        assert data == 5
