"""Tests for the mcp_config examples generator + env-source distillation (CONCEPT:AU-OS.config.env-var-drift-guard)."""

from __future__ import annotations

import json
import re
from pathlib import Path

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
    assert "MCP_TOOL_MODE" not in names  # retired: one tool contract, no mode
    assert "INFOTOOL" in names  # derived toggle
    assert "CONTAINER_MANAGER_TYPE" in names  # code-read var
    assert "AGENT_DESCRIPTION" not in names  # agent-only excluded
    assert "SYSTEM_TOOLS_ENABLE" not in names  # companion suite excluded
    # values come from .env.example
    assert dict(pairs)["CONTAINER_MANAGER_TYPE"] == "docker"
    assert dict(pairs)["DEMO_TOKEN"] == "env://DEMO_TOKEN"
    assert dict(pairs)["DEMO_URL"] == "env://DEMO_URL"
    assert "EMPTY_RUNTIME_VALUE" not in dict(pairs)


def test_render_examples_has_markers_and_canonical_env(tmp_path: Path) -> None:
    block = gen.render_examples(_make_pkg(tmp_path))
    assert gen.START in block and gen.END in block
    assert "MCP_TOOL_MODE" not in block
    assert "demo-mcp[mcp]" in block  # connector-focused extra
    assert "epistemic-graph[full]" in block
    assert "[agent-runtime]` extra additionally" in block
    assert "excludes" not in block
    assert "SYSTEM_TOOLS_ENABLE" not in block  # no stale placeholder
    # the stdio JSON block parses and carries the canonical env
    first = block.split("```json", 1)[1].split("```", 1)[0]
    env = json.loads(first)["mcpServers"]["demo-mcp"]["env"]
    assert "MCP_TOOL_MODE" not in env
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
    assert "MCP_TOOL_MODE" not in env
    assert env["CONTAINER_MANAGER_TYPE"] == "docker"
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


def _readme_text() -> str:
    return (Path(__file__).resolve().parents[2] / "README.md").read_text(
        encoding="utf-8"
    )


def _readme_section(heading: str) -> str:
    """Body of the ``## heading`` section (up to the next ``## `` heading)."""
    return _readme_text().split(f"\n{heading}\n", 1)[1].split("\n## ", 1)[0]


def test_quick_start_readme_uses_installed_portable_entry_point() -> None:
    # The README no longer carries per-topology mcpServers JSON (that moved to
    # docs/recipes/); its Quick Start is the one onboarding path and must stay
    # portable: the published ``graph-os`` console entry point via ``uvx``.
    section = _readme_section("## Quick Start")
    assert 'uvx --from "agent-utilities[serving]" graph-os' in section
    assert "mcp_config.json" not in section
    # No host-specific data: a real absolute filesystem path or literal IP
    # would be a leak.
    assert not re.search(r"tcp://(?:\d{1,3}\.){3}\d{1,3}", section)
    assert "/home/" not in section and "/Users/" not in section


def test_readme_and_recipes_keep_host_and_secrets_runtime_only() -> None:
    # Security invariant: hosts and secrets are only ever shown as PLACEHOLDER
    # references, never real committed values, in the README or the shared-engine
    # recipes it points at.
    docs_root = Path(__file__).resolve().parents[2] / "docs" / "recipes"
    texts = {"README.md": _readme_text()} | {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(docs_root.glob("*.md"))
    }
    assert "unified-self-contained.md" in texts
    for name, text in texts.items():
        assert not re.search(r"tcp://(?:\d{1,3}\.){3}\d{1,3}", text), name
        assert "/Users/" not in text, name
        # A committed OIDC secret value would be a quoted literal that is not a
        # placeholder reference.
        for match in re.finditer(r'"OIDC_CLIENT_SECRET"\s*:\s*"([^"]*)"', text):
            assert re.fullmatch(r"<[^>]+>|\$\{[^}]+\}", match.group(1)), (
                name,
                match.group(0),
            )
    retired_codex_json = ".codex" + "/mcp_config.json"
    assert retired_codex_json not in texts["README.md"]
