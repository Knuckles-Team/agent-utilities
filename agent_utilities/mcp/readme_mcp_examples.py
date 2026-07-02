"""Auto-generate the ``mcp_config.json`` **examples** in a README + sync the real
``mcp_config*.json`` ``env`` blocks (CONCEPT:OS-5.72).

The README env-var table (:mod:`readme_env_vars`) and MCP-tools table
(:mod:`readme_tools`) are regenerated on every commit, but the ``mcp_config.json``
**example blocks** embedded in the README used to be written once at scaffold time and
never refreshed — so they rotted (stale placeholder vars, missing ``MCP_TOOL_MODE``).
This module closes that gap: it regenerates the stdio / streamable-http / remote-url /
docker examples **and** rewrites every ``mcp_config*.json`` ``env`` block from the one
authoritative set (:func:`env_sources.example_env_pairs`), so all three surfaces stay
1:1:1 with the code.

The README block lives between::

    <!-- MCP-CONFIG-EXAMPLES:START -->
    <!-- MCP-CONFIG-EXAMPLES:END -->

If the markers are absent, the region between the ``## MCP Configuration Examples``
heading and the ``<!-- BEGIN GENERATED: additional-deployment-options -->`` marker (both
emitted by the agent-package-builder scaffold) is replaced wholesale — this is how the
one-time retrofit removes the stale hand-written examples fleet-wide.

Usage (from an agent repo root)::

    python -m agent_utilities.mcp.readme_mcp_examples            # rewrite README + configs
    python -m agent_utilities.mcp.readme_mcp_examples --check     # exit 1 if out of date

Wire ``--check`` as a pre-commit hook (the agent-package-builder scaffold does).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_utilities.mcp.env_sources import example_env_pairs

START = "<!-- MCP-CONFIG-EXAMPLES:START -->"
END = "<!-- MCP-CONFIG-EXAMPLES:END -->"
HEADING = "### MCP Configuration Examples"
# End anchor for the one-time retrofit — emitted by the scaffold in every package README.
ADDL_MARKER = "<!-- BEGIN GENERATED: additional-deployment-options -->"
# Start anchors for the one-time retrofit — the earliest present one (before ADDL_MARKER)
# bounds the stale hand-written example region across the heading variants in use.
RETROFIT_ANCHORS = (
    "## MCP Configuration Examples",
    "### MCP Configuration Examples",
    "### Using as an MCP Server",
    "#### stdio Transport",
)


def _detect_package(root: Path) -> tuple[str, str, str]:
    """Return ``(package_name, mcp_command, server_name)`` from ``pyproject.toml``.

    ``mcp_command`` is the ``*-mcp`` console script; ``server_name`` is the key used in
    the ``mcpServers`` map (the command name). Falls back to the directory name.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # py<3.11
        import tomli as tomllib  # type: ignore

    pkg = root.name
    mcp_cmd = root.name
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        pkg = data.get("project", {}).get("name", pkg)
        scripts = data.get("project", {}).get("scripts", {})
        mcp_cmd = next(
            (name for name in scripts if name.endswith("-mcp")),
            next(iter(scripts), pkg),
        )
    return pkg, mcp_cmd, mcp_cmd


def _json_block(server: str, args: list[str], env: dict[str, str]) -> str:
    cfg = {"mcpServers": {server: {"command": "uvx", "args": args, "env": env}}}
    return "```json\n" + json.dumps(cfg, indent=2) + "\n```"


def render_examples(root: Path) -> str:
    """Render the full marker-wrapped MCP-config examples block for ``root``."""
    pkg, mcp_cmd, server = _detect_package(root)
    pairs = example_env_pairs(root)
    base_env = {name: value for name, value in pairs}

    stdio_args = ["--from", f"{pkg}[mcp]", mcp_cmd]
    http_args = [*stdio_args, "--transport", "streamable-http", "--port", "8000"]
    # HTTP examples lead with the transport binding, then the package env.
    http_env = {"TRANSPORT": "streamable-http", "HOST": "0.0.0.0", "PORT": "8000"}
    http_env.update(base_env)

    docker_flags = "\n".join(
        f"  -e {name}={_shell_value(value)} \\" for name, value in pairs
    )

    lines = [
        START,
        "",
        f"> **Install the slim `[mcp]` extra.** All examples install `{pkg}[mcp]` — the",
        "> MCP-server extra that pulls only the FastMCP / FastAPI tooling"
        " (`agent-utilities[mcp]`).",
        "> It deliberately **excludes** the heavy agent runtime (`pydantic-ai`, the"
        " epistemic-graph",
        "> engine, `dspy`, `llama-index`), so `uvx` / container installs are far smaller."
        " Use the",
        "> full `[agent]` extra only when you need the integrated Pydantic AI agent.",
        "",
        "#### stdio Transport (local IDEs — Cursor, Claude Desktop, VS Code)",
        "",
        _json_block(server, stdio_args, base_env),
        "",
        "#### Streamable-HTTP Transport (networked / production)",
        "",
        _json_block(server, http_args, http_env),
        "",
        "Alternatively, connect to a pre-deployed Streamable-HTTP instance by `url`:",
        "",
        "```json",
        json.dumps(
            {"mcpServers": {server: {"url": f"http://localhost:8000/{server}/mcp"}}},
            indent=2,
        ),
        "```",
        "",
        "Deploying the Streamable-HTTP server via Docker:",
        "",
        "```bash",
        "docker run -d \\",
        f"  --name {server}-mcp \\",
        "  -p 8000:8000 \\",
        "  -e TRANSPORT=streamable-http \\",
        "  -e HOST=0.0.0.0 \\",
        "  -e PORT=8000 \\",
        docker_flags,
        f"  knucklessg1/{pkg}:mcp",
        "```",
        "",
        "_Auto-generated from the code-read env surface (`MCP_TOOL_MODE` + package vars)"
        " — do not edit._",
        END,
    ]
    return "\n".join(lines)


def _shell_value(value: str) -> str:
    """Quote a docker ``-e`` value only when it contains shell-significant chars."""
    if value == "" or any(c in value for c in " \t\"'$&|;<>()"):
        return f'"{value}"'
    return value


def _splice(readme: str, block: str) -> tuple[str, bool]:
    """Insert/replace the examples block. Returns ``(new_text, retrofit_failed)``.

    Idempotent once the markers exist. Otherwise a one-time retrofit replaces the stale
    hand-written region — from the earliest example-section anchor to the
    additional-deployment marker — with a normalized ``### MCP Configuration Examples``
    heading + the generated (marker-wrapped) block.
    """
    if START in readme and END in readme:
        pre = readme[: readme.index(START)]
        post = readme[readme.index(END) + len(END) :]
        return pre + block + post, False
    # Retrofit: bound the stale region by the earliest anchor and the additional-deployment
    # marker, then replace it wholesale with a normalized heading + generated block.
    if ADDL_MARKER in readme:
        a_idx = readme.index(ADDL_MARKER)
        starts = [readme.index(a) for a in RETROFIT_ANCHORS if a in readme]
        starts = [s for s in starts if s < a_idx]
        if starts:
            line_start = readme.rfind("\n", 0, min(starts)) + 1
            replacement = f"{HEADING}\n\n{block}\n\n"
            return readme[:line_start] + replacement + readme[a_idx:], False
    # Fallback: no anchors — append with a heading and warn (leaves hand-written text).
    sep = "" if readme.endswith("\n") else "\n"
    return f"{readme}{sep}\n{HEADING}\n\n{block}\n", True


def sync_readme(root: Path, readme_path: Path, *, check: bool = False) -> bool:
    """Write the examples block into ``readme_path``. Returns True if it changed."""
    block = render_examples(root)
    current = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    updated, retrofit_failed = _splice(current, block)
    if retrofit_failed and not check:
        print(
            f"warning: {readme_path} has no MCP-CONFIG-EXAMPLES markers and no "
            f"'{HEADING}'/'{ADDL_MARKER}' anchors — inserted a fresh block; review for "
            "leftover hand-written examples.",
            file=sys.stderr,
        )
    if updated == current:
        return False
    if not check:
        readme_path.write_text(updated, encoding="utf-8")
    return True


def sync_mcp_configs(root: Path, *, check: bool = False) -> list[Path]:
    """Rewrite each ``mcp_config*.json`` ``env`` block to the canonical set.

    Preserves ``command``/``args`` and any non-env server keys. Returns the list of
    files that (would) change.
    """
    env = dict(example_env_pairs(root))
    changed: list[Path] = []
    seen: set[Path] = set()
    for cfg in [*root.glob("mcp_config*.json"), *root.rglob("mcp_config*.json")]:
        resolved = cfg.resolve()
        if ".venv" in cfg.parts or resolved in seen:
            continue
        seen.add(resolved)
        try:
            original = cfg.read_text(encoding="utf-8")
            data = json.loads(original)
        except (OSError, json.JSONDecodeError):
            continue
        servers = data.get("mcpServers")
        if not isinstance(servers, dict):
            continue
        touched = False
        for server in servers.values():
            # Only rewrite launch-style entries; leave remote-url servers untouched.
            if isinstance(server, dict) and "url" not in server:
                server["env"] = dict(env)
                touched = True
        if not touched:
            continue
        updated = json.dumps(data, indent=2) + "\n"
        if updated != original:
            changed.append(cfg)
            if not check:
                cfg.write_text(updated, encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync the mcp_config examples in README + the mcp_config*.json env blocks."
    )
    parser.add_argument("root", nargs="?", default=".", help="Package root")
    parser.add_argument("--check", action="store_true", help="Fail if out of date")
    parser.add_argument("--readme", default="README.md", help="README path")
    args = parser.parse_args(argv)

    root = Path(args.root)
    readme_changed = sync_readme(root, root / args.readme, check=args.check)
    cfgs_changed = sync_mcp_configs(root, check=args.check)

    if args.check and (readme_changed or cfgs_changed):
        targets = ", ".join(
            [args.readme] * bool(readme_changed) + [str(p) for p in cfgs_changed]
        )
        print(
            f"mcp_config examples/env blocks out of date ({targets}) — run "
            "`python -m agent_utilities.mcp.readme_mcp_examples` to refresh.",
            file=sys.stderr,
        )
        return 1
    if readme_changed or cfgs_changed:
        print(
            f"Updated mcp_config examples in {args.readme}"
            + (f" + {len(cfgs_changed)} config file(s)." if cfgs_changed else ".")
        )
    else:
        print("mcp_config examples + env blocks already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
