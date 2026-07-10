"""Auto-generate the Environment Variables table in an agent package's README.

A connector's full env surface = the variables it declares in its own
``.env.example`` PLUS the **inherited** agent-utilities variables every fleet
connector honours (transport, tool-mode, governance, telemetry, outbound auth,
agent-CLI). This module parses ``.env.example`` for the package's own variables
(name, example value, description) and merges in the curated inherited set, then
renders two clearly-labelled Markdown tables into README.md between markers::

    <!-- ENV-VARS-TABLE:START -->
    <!-- ENV-VARS-TABLE:END -->

inserting the block under an "## Environment Variables" heading if the markers are
absent. So the docs never drift from ``.env.example`` and always show the inherited
surface too.

Usage (from an agent repo root):

    python -m agent_utilities.mcp.readme_env_vars            # rewrite the table
    python -m agent_utilities.mcp.readme_env_vars --check     # exit 1 if out of date

Wire ``--check`` as a pre-commit hook (the agent-package-builder scaffold does).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

START = "<!-- ENV-VARS-TABLE:START -->"
END = "<!-- ENV-VARS-TABLE:END -->"
HEADING = "## Environment Variables"

# A var assignment, optionally commented-out (optional var): ``KEY=value`` or ``# KEY=value``.
_VAR = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)\s*=(.*)$")
# A section banner comment we must NOT treat as a description.
_BANNER = re.compile(r"^\s*#\s*([-=*_]{2,}|.*[-=]{3,}.*)\s*$")

# Curated INHERITED set — agent-utilities variables every connector honours.
# (name -> (example, description)). Package-declared vars take precedence; an
# inherited var already in the package's .env.example is not duplicated here.
INHERITED_ENV: dict[str, tuple[str, str]] = {
    "TRANSPORT": ("stdio", "MCP transport: `stdio` | `streamable-http` | `sse`"),
    "HOST": ("0.0.0.0", "Bind host (HTTP transports)"),
    "PORT": ("8000", "Bind port (HTTP transports)"),
    "MCP_TOOL_MODE": ("condensed", "Tool surface: `condensed` | `verbose` | `both`"),
    "MCP_ENABLED_TOOLS": ("", "Comma-separated tool allow-list"),
    "MCP_DISABLED_TOOLS": ("", "Comma-separated tool deny-list"),
    "MCP_ENABLED_TAGS": ("", "Comma-separated tag allow-list"),
    "MCP_DISABLED_TAGS": ("", "Comma-separated tag deny-list"),
    "EUNOMIA_TYPE": ("none", "Authorization mode: `none` | `embedded` | `remote`"),
    "EUNOMIA_POLICY_FILE": ("mcp_policies.json", "Embedded Eunomia policy file"),
    "EUNOMIA_REMOTE_URL": ("", "Remote Eunomia authorization server URL"),
    "ENABLE_OTEL": ("False", "Enable OpenTelemetry export"),
    "OTEL_EXPORTER_OTLP_ENDPOINT": ("", "OTLP collector endpoint"),
    "MCP_CLIENT_AUTH": (
        "",
        "Outbound MCP child auth: `oidc-client-credentials` | `basic` | `none`",
    ),
    "OIDC_CLIENT_ID": ("", "OIDC client id (service-account auth)"),
    "OIDC_CLIENT_SECRET": ("", "OIDC client secret (service-account auth)"),
    "MCP_BASIC_AUTH_USERNAME": ("", "HTTP Basic username (`MCP_CLIENT_AUTH=basic`)"),
    "MCP_BASIC_AUTH_PASSWORD": ("", "HTTP Basic password (`MCP_CLIENT_AUTH=basic`)"),
    "DEBUG": ("False", "Verbose logging"),
    "PYTHONUNBUFFERED": ("1", "Unbuffered stdout (recommended in containers)"),
    # agent CLI — the full `[agent]` runtime only
    "MCP_URL": (
        "http://localhost:8000/mcp",
        "URL of the MCP server the agent connects to",
    ),
    "PROVIDER": ("openai", "LLM provider for the agent"),
    "MODEL_ID": ("gpt-4o", "Model id for the agent"),
    "ENABLE_WEB_UI": ("True", "Serve the AG-UI web interface"),
}


def parse_env_example(text: str) -> list[tuple[str, str, str]]:
    """Parse ``.env.example`` into ``(name, example, description)`` rows.

    A description is the inline ``# ...`` after the value, else the standalone
    comment line immediately above the variable (section banners excluded).
    Commented-out assignments (``# KEY=value``) are kept as optional variables.
    """
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    pending_desc = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        m = _VAR.match(line)
        if m:
            name, rest = m.group(1), m.group(2)
            example, _, inline = rest.partition("#")
            desc = inline.strip() or pending_desc
            pending_desc = ""
            if name not in seen:
                seen.add(name)
                rows.append((name, example.strip(), desc.strip()))
            continue
        stripped = line.strip()
        if stripped.startswith("#") and not _BANNER.match(line):
            # a standalone comment -> candidate description for the next var
            pending_desc = stripped.lstrip("# ").strip()
        elif not stripped:
            pending_desc = ""
    return rows


def _table(rows: list[tuple[str, str, str]]) -> list[str]:
    out = [
        "| Variable | Example | Description |",
        "|----------|---------|-------------|",
    ]
    for name, example, desc in rows:
        ex = f"`{example}`" if example else "—"
        out.append(f"| `{name}` | {ex} | {desc or ''} |")
    return out


def render_env_table(env_example: str) -> str:
    """Render package + inherited env-var tables from ``.env.example`` text."""
    package = parse_env_example(env_example)
    pkg_names = {r[0] for r in package}
    inherited = [
        (name, ex, desc)
        for name, (ex, desc) in INHERITED_ENV.items()
        if name not in pkg_names
    ]
    lines = [START, ""]
    lines.append("#### Package environment variables")
    lines.append("")
    lines += _table(package) if package else ["_None declared in `.env.example`._"]
    lines.append("")
    if inherited:
        lines.append(
            "#### Inherited agent-utilities variables (apply to every connector)"
        )
        lines.append("")
        lines += _table(inherited)
        lines.append("")
    lines.append(
        f"_{len(package)} package + {len(inherited)} inherited variable(s). "
        "Auto-generated from `.env.example` + the shared agent-utilities set — "
        "do not edit._"
    )
    lines.append(END)
    return "\n".join(lines)


def _splice(readme: str, table: str) -> str:
    if START in readme and END in readme:
        pre = readme[: readme.index(START)]
        post = readme[readme.index(END) + len(END) :]
        return pre + table + post
    block = f"\n{table}\n"
    if HEADING in readme:
        idx = readme.index(HEADING) + len(HEADING)
        return readme[:idx] + "\n" + block + readme[idx:]
    sep = "" if readme.endswith("\n") else "\n"
    return f"{readme}{sep}\n{HEADING}\n{block}"


def sync_readme(env_path: Path, readme_path: Path, *, check: bool = False) -> bool:
    """Write the env-var table into ``readme_path``. Returns True if it changed."""
    env_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    table = render_env_table(env_text)
    current = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    updated = _splice(current, table)
    if updated == current:
        return False
    if not check:
        readme_path.write_text(updated, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync the env-vars table in README.")
    parser.add_argument("--check", action="store_true", help="Fail if out of date")
    parser.add_argument("--readme", default="README.md", help="README path")
    parser.add_argument("--env", default=".env.example", help=".env.example path")
    args = parser.parse_args(argv)

    changed = sync_readme(Path(args.env), Path(args.readme), check=args.check)
    if args.check and changed:
        print(
            f"{args.readme} env-vars table is out of date — run "
            "`python -m agent_utilities.mcp.readme_env_vars` to refresh.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Updated env-vars table in {args.readme}."
        if changed
        else f"{args.readme} env-vars table already up to date."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
