"""Auto-generate the MCP tools table in an agent package's README (CONCEPT:AU-ECO.mcp.tool-mode-standardization).

Every fleet connector exposes its surface through ``get_mcp_instance()`` and the
shared ``register_tool_surface`` wiring, so the live server is the single source of
truth for which tools exist, their per-domain ``<TAG>TOOL`` toggles, and their
descriptions. This module renders that into a Markdown table and writes it into
README.md between marker comments, so the docs never drift from the code.

Usage (from an agent repo root, with agent-utilities installed):

    python -m agent_utilities.mcp.readme_tools           # rewrite README table
    python -m agent_utilities.mcp.readme_tools --check    # exit 1 if out of date

Wire ``--check`` as a pre-commit hook (the agent-package-builder scaffold does).
The table goes between::

    <!-- MCP-TOOLS-TABLE:START -->
    <!-- MCP-TOOLS-TABLE:END -->

inserting the block under an "## Available MCP Tools" heading if the markers are
absent.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

START = "<!-- MCP-TOOLS-TABLE:START -->"
END = "<!-- MCP-TOOLS-TABLE:END -->"


def _toggle_env(tags: set[str]) -> str:
    """Derive the per-domain toggle env var (``<TAG>TOOL``) from a tool's tags.

    Mirrors ``register_tool_surface`` auto-discovery: a domain tag ``cmdb`` is
    gated by ``CMDBTOOL``. ``verbose`` and the service tag are not domain toggles.
    """
    candidates = sorted(t for t in tags if t and t != "verbose")
    if not candidates:
        return "—"
    # Prefer a tag that isn't the service name (kebab/with '-'); domains are snake.
    domain = next((t for t in candidates if "-" not in t), candidates[0])
    return f"`{domain.upper()}TOOL`"


def _first_line(text: str | None) -> str:
    if not text:
        return ""
    line = text.strip().splitlines()[0].strip()
    return line.replace("|", "\\|")


async def _list_tools(mcp: Any) -> list[Any]:
    # FastMCP exposes the private _list_tools across the versions we target.
    if hasattr(mcp, "_list_tools"):
        return list(await mcp._list_tools())
    if hasattr(mcp, "get_tools"):
        return list((await mcp.get_tools()).values())
    raise RuntimeError("Cannot list tools from this FastMCP instance")


def _table(rows: list[tuple[str, str, str]]) -> list[str]:
    """Render ``(name, toggle, desc)`` rows as a Markdown table body."""
    out = [
        "| MCP Tool | Toggle Env Var | Description |",
        "|----------|----------------|-------------|",
    ]
    for name, toggle, desc in rows:
        out.append(f"| `{name}` | {toggle} | {desc} |")
    return out


def render_tools_table(mcp: Any) -> str:
    """Render the FULL live tool surface — both the **condensed** action-routed
    tools and the **verbose** 1:1 API-mapped tools — as two clearly-labelled
    Markdown tables, so the README shows every tool, not just the default surface.

    The server is built in ``MCP_TOOL_MODE=both`` for generation (see
    :func:`_load_mcp_instance`) so both surfaces are registered; tools tagged
    ``"verbose"`` are the 1:1 per-operation surface, the rest are condensed.
    """
    tools = asyncio.run(_list_tools(mcp))
    # Exact tool->toggle map recorded by register_tool_surface (authoritative — it is
    # the env var that actually gates the tool). Falls back to tag-derivation only
    # for tools registered outside the central wiring.
    toggles = getattr(mcp, "_condensed_tool_toggles", None) or {}
    condensed: list[tuple[str, str, str]] = []
    verbose: list[tuple[str, str, str]] = []
    for tool in tools:
        tags = set(getattr(tool, "tags", None) or [])
        name = getattr(tool, "name", "?")
        env = toggles.get(name)
        toggle = f"`{env}`" if env else _toggle_env(tags)
        row = (name, toggle, _first_line(getattr(tool, "description", "")))
        (verbose if "verbose" in tags else condensed).append(row)
    condensed.sort(key=lambda r: r[0])
    verbose.sort(key=lambda r: r[0])

    lines = [START, ""]
    lines.append(
        "#### Condensed action-routed tools (default — `MCP_TOOL_MODE=condensed`)"
    )
    lines.append("")
    lines += _table(condensed)
    lines.append("")
    if verbose:
        # The verbose 1:1 surface can be large — keep it collapsed but complete.
        lines.append(
            "#### Verbose 1:1 API-mapped tools (`MCP_TOOL_MODE=verbose` or `both`)"
        )
        lines.append("")
        lines.append("<details>")
        lines.append(
            f"<summary>{len(verbose)} per-operation tools — one per public API "
            "method (click to expand)</summary>"
        )
        lines.append("")
        lines += _table(verbose)
        lines.append("")
        lines.append("</details>")
        lines.append("")
    lines.append(
        f"_{len(condensed)} action-routed tool(s) (default) · {len(verbose)} verbose "
        "1:1 tool(s). Each is enabled unless its `<DOMAIN>TOOL` toggle is set false; "
        "`MCP_TOOL_MODE` selects the surface (`condensed` default · `verbose` 1:1 · "
        "`both`). Auto-generated — do not edit._"
    )
    lines.append(END)
    return "\n".join(lines)


def _splice(readme: str, table: str) -> str:
    if START in readme and END in readme:
        pre = readme[: readme.index(START)]
        post = readme[readme.index(END) + len(END) :]
        return pre + table + post
    # No markers yet — insert under the heading, else append.
    heading = "## Available MCP Tools"
    block = f"\n{table}\n"
    if heading in readme:
        idx = readme.index(heading) + len(heading)
        return readme[:idx] + "\n" + block + readme[idx:]
    sep = "" if readme.endswith("\n") else "\n"
    return f"{readme}{sep}\n{heading}\n{block}"


def sync_readme(mcp: Any, readme_path: Path, *, check: bool = False) -> bool:
    """Write the tools table into ``readme_path``. Returns True if it changed.

    In ``check`` mode it does not write — it only reports whether the README is
    out of date (caller exits non-zero).
    """
    table = render_tools_table(mcp)
    current = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    updated = _splice(current, table)
    if updated == current:
        return False
    if not check:
        readme_path.write_text(updated, encoding="utf-8")
    return True


def _load_mcp_instance() -> Any:
    """Auto-detect the local agent package and return its built FastMCP instance.

    Forces ``MCP_TOOL_MODE=both`` so the built server registers BOTH the condensed
    and the verbose 1:1 surfaces — the generator documents the full tool set, not
    just the default condensed one. (Env *write* to drive the build is sanctioned;
    reads still go through the config layer.)
    """
    import os

    os.environ["MCP_TOOL_MODE"] = "both"
    from agent_utilities.mcp.server_factory import create_mcp_server  # noqa: F401

    module_path = _detect_mcp_module()
    sys.argv = [module_path]  # keep create_mcp_server's argv parsing happy
    import importlib

    mod = importlib.import_module(module_path)
    result = mod.get_mcp_instance()
    # get_mcp_instance returns a tuple; find the FastMCP (order varies per agent).
    candidates = result if isinstance(result, tuple) else (result,)
    for obj in candidates:
        if hasattr(obj, "_list_tools") or type(obj).__name__ == "FastMCP":
            return obj
    raise RuntimeError("get_mcp_instance() returned no FastMCP instance")


def _detect_mcp_module() -> str:
    """Find ``<pkg>.mcp_server`` from the repo's pyproject ``[project.scripts]``."""
    try:
        import tomllib
    except ModuleNotFoundError:  # py<3.11
        import tomli as tomllib  # type: ignore

    pyproject = Path.cwd() / "pyproject.toml"
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        scripts = data.get("project", {}).get("scripts", {})
        for name, target in scripts.items():
            if name.endswith("-mcp") and ":" in target:
                return target.split(":", 1)[0]
    raise RuntimeError(
        "Could not detect the MCP module — no '*-mcp = <pkg>.mcp_server:...' in "
        "pyproject [project.scripts]. Run from the agent repo root."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync the MCP tools table in README.")
    parser.add_argument("--check", action="store_true", help="Fail if out of date")
    parser.add_argument("--readme", default="README.md", help="README path")
    args = parser.parse_args(argv)

    mcp = _load_mcp_instance()
    changed = sync_readme(mcp, Path(args.readme), check=args.check)
    if args.check and changed:
        print(
            f"{args.readme} MCP tools table is out of date — run "
            "`python -m agent_utilities.mcp.readme_tools` to refresh.",
            file=sys.stderr,
        )
        return 1
    if changed:
        print(f"Updated MCP tools table in {args.readme}.")
    else:
        print(f"{args.readme} MCP tools table already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
