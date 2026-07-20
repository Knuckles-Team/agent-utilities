"""Tests for the mcp_config examples generator + env-source distillation (CONCEPT:AU-OS.config.env-var-drift-guard)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from agent_utilities.deployment.codex_registration import graphos_stdio_spec
from agent_utilities.mcp import readme_mcp_examples as gen
from agent_utilities.mcp.env_sources import example_env_pairs, is_agent_only

ENV_EXAMPLE = """\
CONTAINER_MANAGER_TYPE=docker # options: docker, podman
SYSTEM_TOOLS_ENABLE=False
INFOTOOL=True
# DEMO_TOKEN=env://DEMO_TOKEN # resolved by an alias-aware launcher
# DEMO_URL=env://DEMO_URL # resolved by an alias-aware launcher
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
    'setting("DEMO_TOKEN", "")\n'
    'setting("DEMO_URL", "")\n'
    'setting("EMPTY_RUNTIME_VALUE", "")\n'
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
    assert dict(pairs)["DEMO_TOKEN"] == "env://DEMO_TOKEN"
    assert dict(pairs)["DEMO_URL"] == "env://DEMO_URL"
    assert "EMPTY_RUNTIME_VALUE" not in dict(pairs)


def test_render_examples_has_markers_and_tool_mode(tmp_path: Path) -> None:
    block = gen.render_examples(_make_pkg(tmp_path))
    assert gen.START in block and gen.END in block
    assert '"MCP_TOOL_MODE": "intent"' in block
    assert "demo-mcp[mcp]" in block  # connector-focused extra
    assert "epistemic-graph[full]" in block
    assert "[agent-runtime]` extra additionally" in block
    assert "excludes" not in block
    assert "SYSTEM_TOOLS_ENABLE" not in block  # no stale placeholder
    # the stdio JSON block parses and carries the canonical env
    first = block.split("```json", 1)[1].split("```", 1)[0]
    env = json.loads(first)["mcpServers"]["demo-mcp"]["env"]
    assert env["MCP_TOOL_MODE"] == "intent"
    assert "AGENT_DESCRIPTION" not in env
    assert env["DEMO_TOKEN"] == "env://DEMO_TOKEN"
    assert env["DEMO_URL"] == "env://DEMO_URL"
    assert "EMPTY_RUNTIME_VALUE" not in env
    assert "Runtime references require an alias-aware launcher" in block
    assert "docker run -i --rm" in block
    assert "--read-only" in block
    assert "--cap-drop=ALL" in block
    assert "--security-opt=no-new-privileges" in block
    assert "-e TRANSPORT=stdio" in block
    assert "registry.example.invalid/demo-mcp@sha256:<digest> demo-mcp" in block
    assert "-p 127.0.0.1:8000:8000" not in block
    assert "-e HOST=0.0.0.0" not in block
    assert "exact `MCP_ALLOWED_HOSTS`" in block
    assert "authenticated TLS ingress" in block
    assert "-e DEMO_TOKEN \\" in block
    assert "-e DEMO_TOKEN=env://" not in block


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
                        "env": {
                            "AGENT_DESCRIPTION": "x",
                            "SYSTEM_TOOLS_ENABLE": "True",
                        },
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
    assert env["MCP_TOOL_MODE"] == "intent"
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


def _readme_section(heading: str) -> str:
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(
        encoding="utf-8"
    )
    return readme.split(heading, 1)[1].split("\n### ", 1)[0]


def test_self_contained_readme_uses_installed_portable_entry_point() -> None:
    section = _readme_section("### Self-hosted, self-contained installation")
    spec = graphos_stdio_spec()
    expected = f"codex mcp add graph-os -- {spec['command']} " + " ".join(spec["args"])
    assert expected in section
    assert "mcp_config.json" not in section
    assert "${workspaceFolder}" not in section


def test_shared_engine_readme_keeps_host_and_secrets_runtime_only() -> None:
    section = _readme_section("### Shared engine")
    assert "codex mcp add graph-os -- graph-os --transport stdio" in section
    launcher = section.split("The corresponding XDG AgentConfig", 1)[0]
    assert "GRAPH_SERVICE_ENDPOINTS" not in launcher
    assert "OIDC_CLIENT_SECRET" not in launcher
    assert "mcp_config.json" not in launcher

    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(
        encoding="utf-8"
    )
    assert not re.search(r"tcp://(?:\d{1,3}\.){3}\d{1,3}", readme)
    retired_codex_json = ".codex" + "/mcp_config.json"
    assert retired_codex_json not in readme
