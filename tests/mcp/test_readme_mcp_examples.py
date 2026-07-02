"""Tests for the mcp_config examples generator + env-source distillation (CONCEPT:OS-5.72)."""

from __future__ import annotations

import json
from pathlib import Path

from agent_utilities.mcp import readme_mcp_examples as gen
from agent_utilities.mcp.env_sources import example_env_pairs, is_agent_only

ENV_EXAMPLE = """\
CONTAINER_MANAGER_TYPE=docker # options: docker, podman
SYSTEM_TOOLS_ENABLE=False
INFOTOOL=True
"""

PYPROJECT = """\
[project]
name = "demo-mcp"
[project.scripts]
demo-mcp = "demo_mcp.mcp_server:mcp_server"
demo-agent = "demo_mcp.agent_server:agent_server"
"""

# Code that reads a real MCP var + declares a tool registrar + reads an agent-only var.
CODE = (
    "from agent_utilities.core.config import setting\n"
    'setting("CONTAINER_MANAGER_TYPE", "docker")\n'
    'setting("AGENT_DESCRIPTION", "")\n'
    "def register_info_tools(mcp):\n    pass\n"
)


def _make_pkg(tmp_path: Path, *, readme: str = "") -> Path:
    root = tmp_path / "demo-mcp"
    (root / "demo_mcp").mkdir(parents=True)
    (root / ".env.example").write_text(ENV_EXAMPLE, encoding="utf-8")
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (root / "demo_mcp" / "mcp_server.py").write_text(CODE, encoding="utf-8")
    if readme:
        (root / "README.md").write_text(readme, encoding="utf-8")
    return root


def test_agent_only_classification() -> None:
    assert is_agent_only("AGENT_DESCRIPTION")
    assert is_agent_only("SYSTEM_TOOLS_ENABLE")  # companion suite (*_ENABLE)
    assert not is_agent_only("ENABLE_OTEL")  # prefixed, not a companion suite
    assert not is_agent_only("INFOTOOL")


def test_example_env_pairs_canonical_set(tmp_path: Path) -> None:
    root = _make_pkg(tmp_path)
    pairs = example_env_pairs(root)
    names = [n for n, _ in pairs]
    assert names[0] == "MCP_TOOL_MODE"  # always first
    assert "INFOTOOL" in names  # derived toggle
    assert "CONTAINER_MANAGER_TYPE" in names  # code-read var
    assert "AGENT_DESCRIPTION" not in names  # agent-only excluded
    assert "SYSTEM_TOOLS_ENABLE" not in names  # companion suite excluded
    # values come from .env.example
    assert dict(pairs)["CONTAINER_MANAGER_TYPE"] == "docker"


def test_render_examples_has_markers_and_tool_mode(tmp_path: Path) -> None:
    block = gen.render_examples(_make_pkg(tmp_path))
    assert gen.START in block and gen.END in block
    assert '"MCP_TOOL_MODE": "condensed"' in block
    assert "demo-mcp[mcp]" in block  # slim extra
    assert "SYSTEM_TOOLS_ENABLE" not in block  # no stale placeholder
    # the stdio JSON block parses and carries the canonical env
    first = block.split("```json", 1)[1].split("```", 1)[0]
    env = json.loads(first)["mcpServers"]["demo-mcp"]["env"]
    assert env["MCP_TOOL_MODE"] == "condensed"
    assert "AGENT_DESCRIPTION" not in env


def test_retrofit_replaces_stale_region(tmp_path: Path) -> None:
    """With no markers, the heading→additional-deployment span is replaced wholesale."""
    readme = (
        "# Demo\n\n## MCP Configuration Examples\n\n"
        "```json\n"
        '{"mcpServers": {"demo-mcp": {"env": {"SYSTEM_TOOLS_ENABLE": "x"}}}}\n'
        "```\n\n"
        "<!-- BEGIN GENERATED: additional-deployment-options -->\nkeep me\n"
    )
    root = _make_pkg(tmp_path, readme=readme)
    assert gen.sync_readme(root, root / "README.md") is True
    out = (root / "README.md").read_text(encoding="utf-8")
    assert "SYSTEM_TOOLS_ENABLE" not in out  # stale example gone
    assert gen.START in out and "keep me" in out  # markers in, tail preserved
    # idempotent second run
    assert gen.sync_readme(root, root / "README.md") is False


def test_sync_mcp_configs_rewrites_env(tmp_path: Path) -> None:
    root = _make_pkg(tmp_path)
    (root / "mcp_config.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "demo-mcp": {
                        "command": "uvx",
                        "args": ["demo-mcp"],
                        "env": {"AGENT_DESCRIPTION": "x", "SYSTEM_TOOLS_ENABLE": "True"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    changed = gen.sync_mcp_configs(root)
    assert changed  # file rewritten
    env = json.loads((root / "mcp_config.json").read_text())["mcpServers"]["demo-mcp"][
        "env"
    ]
    assert env["MCP_TOOL_MODE"] == "condensed"
    assert "AGENT_DESCRIPTION" not in env and "SYSTEM_TOOLS_ENABLE" not in env
    assert json.loads((root / "mcp_config.json").read_text())["mcpServers"]["demo-mcp"][
        "args"
    ] == ["demo-mcp"]  # command/args preserved
    # idempotent
    assert gen.sync_mcp_configs(root) == []


def test_url_only_server_not_rewritten(tmp_path: Path) -> None:
    """A remote-url server entry (no launch env) is left untouched."""
    root = _make_pkg(tmp_path)
    cfg = {"mcpServers": {"demo-mcp": {"url": "http://localhost:8000/demo-mcp/mcp"}}}
    (root / "mcp_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert gen.sync_mcp_configs(root) == []
