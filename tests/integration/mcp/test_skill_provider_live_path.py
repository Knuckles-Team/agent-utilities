"""Server-side Skills-over-MCP is LIVE on the fastmcp-4 default.

CONCEPT:AU-ECO.mcp.skills-over-mcp-provider

``tests/unit/mcp/test_skill_provider_wiring.py`` proves ``_register_skill_providers``
calls ``add_provider`` once per resolved directory, but it does so against a
``MagicMock`` server and a monkeypatched ``fastmcp.server.providers.skills``
module — it never touches the real ``SkillProvider`` and would keep passing even
if the whole path were dead. While the ``[mcp]`` extra pinned fastmcp 3 it *was*
dead: ``hasattr(mcp, "add_provider")`` was permanently ``False`` and every
supported install registered nothing.

With ``fastmcp>=4.0.0b1`` as the floor that branch is gone, so this asserts the
real thing: a server built by au's own ``create_mcp_server`` actually serves the
``skill://`` resources for the directories ``resolve_skill_provider_dirs``
resolves, using the genuine fastmcp-4 ``SkillProvider``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def demo_skill_dir(tmp_path: Path) -> Path:
    # `resolve_skill_provider_dirs()` yields ONE entry per individual skill
    # directory (each containing its own SKILL.md), which is exactly what
    # fastmcp-4's `SkillProvider(skill_path=...)` expects.
    skill = tmp_path / "au-live-provider" / "live-demo-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\n"
        "description: served over MCP by a real fastmcp-4 SkillProvider\n"
        "---\n"
        "# Live Demo Skill\n\nBody served as a skill:// resource.\n",
        encoding="utf-8",
    )
    return skill


@pytest.mark.asyncio
async def test_create_mcp_server_serves_skill_resources(
    monkeypatch, demo_skill_dir: Path
) -> None:
    """``create_mcp_server`` registers the REAL ``SkillProvider`` and the built
    server serves the skill's ``SKILL.md`` over a ``skill://`` resource."""
    # Real import — no fake module in sys.modules. If fastmcp ever loses this,
    # the failure must be loud rather than a silently skipped registration.
    from fastmcp.server.providers.skills import SkillProvider

    assert SkillProvider is not None

    monkeypatch.setattr(
        "agent_utilities.core.providers.resolve_skill_provider_dirs",
        lambda: [("au-live-provider", demo_skill_dir)],
    )

    from agent_utilities.mcp.server_factory import create_mcp_server

    _args, mcp, _middlewares = create_mcp_server(
        "Skills Live Path Test", command_args=[]
    )

    resources = await mcp.list_resources()
    templates = await mcp.list_resource_templates()
    uris = {str(resource.uri) for resource in resources}
    uris |= {str(template.uri_template) for template in templates}

    assert "skill://live-demo-skill/SKILL.md" in uris, (
        "the fastmcp-4 SkillProvider must publish the skill's SKILL.md; "
        f"got {sorted(uris)}"
    )

    body = await mcp.read_resource("skill://live-demo-skill/SKILL.md")
    text = "".join(str(getattr(chunk, "content", chunk)) for chunk in body)
    assert "Live Demo Skill" in text
