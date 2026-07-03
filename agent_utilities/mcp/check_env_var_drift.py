"""Detect env-var / config drift in an agent package — the code is the source of truth.

CONCEPT:OS-5.72 — Env-var drift guard.

A connector documents its env vars in several places: ``.env.example``, every
``mcp_config*.json`` ``env`` block, ``docker/*compose*.yml`` ``environment:``, and the
README tables. The **only** authority for *which vars actually exist* is the code that
reads them. This module computes the code-read set and diffs the declared sets against it,
emitting drift findings:

- ``DEAD`` — a var declared in config/docs that **no code reads** (e.g. a scaffolder's
  ``*_TOKEN``, or per-endpoint ``*_TOOL`` toggles the framework never honours). Remove it.
- ``UNDOCUMENTED`` — a var the code reads that is **missing from** ``.env.example``. Add it.
- ``MISSING_TOOL_MODE`` — a launch-style ``mcp_config.json`` ``env`` block (or README
  example) with **no** ``MCP_TOOL_MODE`` (users can't discover the surface). Add it.
- ``MALFORMED_VALUE`` — a config ``env`` value with a whitespace-padded substitution like
  ``"${ VAR:-True }"``. Use ``"${VAR:-True}"`` (no spaces inside the braces).
- ``AGENT_VAR_IN_MCP`` — an agent-runtime var (``AGENT_DESCRIPTION``, ``MCP_URL``, a
  ``*_ENABLE`` companion suite …) sitting in an **MCP-server** config; it launches the
  agent, never the server. Move it to the agent config.
- ``STALE_EXAMPLE`` — a README ``mcp_config`` example ``env`` key that isn't in the
  code-read surface (a leftover scaffold placeholder). Regenerate the examples.

The code-read set =
  ``setting("VAR", …)`` reads in the package
  ∪ derived tool toggles: ``register_<tag>_tools`` → ``<TAG>TOOL``
  ∪ the inherited agent-utilities surface (``readme_env_vars.INHERITED_ENV`` + framework extras)
  ∪ ``setting("VAR", …)`` reads in agent-utilities core (covers connector-base reads like
    ``{SERVICE}_SSL_VERIFY`` so they are never mis-flagged as dead).

Usage (from an agent repo root)::

    python -m agent_utilities.mcp.check_env_var_drift            # human report
    python -m agent_utilities.mcp.check_env_var_drift --check    # exit 1 on drift (pre-commit)
    python -m agent_utilities.mcp.check_env_var_drift --json     # machine-readable findings

Wire ``--check`` as a pre-commit hook (the agent-package-builder scaffold does).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from agent_utilities.mcp.readme_env_vars import INHERITED_ENV, parse_env_example

# Env reads the code performs. ``setting(...)`` is the sanctioned accessor, but bare
# ``os.getenv`` / ``os.environ.get`` / ``os.environ[...]`` (and ``from os import …`` forms)
# are also real reads — count them so a live var is never mis-flagged DEAD. A subscript
# ``os.environ["X"] = ...`` is a WRITE (cross-process signaling, not config to document),
# so the subscript branch excludes an assignment via negative lookahead.
_ENV_READ = re.compile(
    r"""(?:setting|(?:os\.)?getenv|(?:os\.)?environ\.get)\(\s*['"]([A-Z][A-Z0-9_]*)['"]"""
    r"""|(?:os\.)?environ\[\s*['"]([A-Z][A-Z0-9_]*)['"]\](?!\s*=(?!=))"""
)
# ``register_<tag>_tools`` — a condensed registrar; toggle env var is ``<TAG>TOOL``.
_REGISTRAR = re.compile(r"register_([a-z][a-z0-9_]*?)_tools\b")
# A ``- "KEY=value"`` or ``KEY: value`` line inside a compose ``environment:`` list/map.
_COMPOSE_ENV = re.compile(r"""^\s*-?\s*["']?([A-Z][A-Z0-9_]*)["']?\s*[:=]""")
# A ``${...}`` shell substitution — inner text is inspected for stray whitespace.
_SUBST = re.compile(r"\$\{([^}]*)\}")
# A fenced ```json block (README mcp_config examples).
_JSON_FENCE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)

# README markers/anchors that bound the mcp_config example region (see readme_mcp_examples).
README_START = "<!-- MCP-CONFIG-EXAMPLES:START -->"
README_END = "<!-- MCP-CONFIG-EXAMPLES:END -->"
README_HEADING = "## MCP Configuration Examples"
README_ADDL = "<!-- BEGIN GENERATED: additional-deployment-options -->"

# Framework vars read inside agent-utilities (create_agent / gateway / telemetry / connector
# base) on behalf of every connector — legitimately documentable, never "dead".
FRAMEWORK_EXTRA: frozenset[str] = frozenset(
    {
        "AUTH_TYPE",
        "AGENT_DESCRIPTION",
        "AGENT_SYSTEM_PROMPT",
        "DEFAULT_AGENT_NAME",
        "ENABLE_OTEL",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_PUBLIC_KEY",
        "OTEL_EXPORTER_OTLP_SECRET_KEY",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "PROVIDER",
        "MODEL_ID",
        "ENABLE_WEB_UI",
        "MCP_URL",
    }
)
# Connector-standard suffixes read by the agent-utilities connector base, not the package.
_SAFE_SUFFIXES: tuple[str, ...] = ("_SSL_VERIFY",)
# Generic process / library runtime vars legitimately set in a launch env but never read
# via ``setting()`` — not app config, so not "dead".
RUNTIME_ALLOWLIST: frozenset[str] = frozenset(
    {
        "TERM",
        "NO_COLOR",
        "FORCE_COLOR",
        "FASTMCP_LOG_LEVEL",
        "PYTHONUNBUFFERED",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
        "LOG_LEVEL",
        "TZ",
    }
)
# Library-owned runtime vars identified by prefix (consumed by the library, not our code).
_RUNTIME_PREFIXES: tuple[str, ...] = ("FASTMCP_",)


_HOST_SUFFIXES = ("_BASE_URL", "_URL", "_HOST")


def _stem(var: str) -> str:
    """The service/domain stem of a var, suffixes stripped (for alias/rename matching)."""
    return re.sub(
        r"(_BASE_URL|_URL|_HOST|_TOKEN|_API_KEY|_KEY|_SECRET|_VERIFY|_SSL_VERIFY|TOOL)$",
        "",
        var,
    )


def _is_host_var(var: str) -> bool:
    return var.endswith(_HOST_SUFFIXES)


def _in_string_literal(line: str, pos: int) -> bool:
    """True if ``pos`` on ``line`` sits inside a quote — i.e. the env-read keyword is
    itself part of a string literal (a code-generator template like
    ``lines.append('... os.environ.get(<VAR>) ...')``), not a real read."""
    return line.count("'", 0, pos) % 2 == 1 or line.count('"', 0, pos) % 2 == 1


def _scan_setting_calls(root: Path) -> set[str]:
    """Every env-var literal read (``setting`` or bare ``os.getenv``/``os.environ``) in
    ``*.py`` under ``root``. Skips assignment writes and reads nested inside a string
    literal (codegen templates)."""
    found: set[str] = set()
    for py in root.rglob("*.py"):
        if ".venv" in py.parts or "__pycache__" in py.parts:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            for m in _ENV_READ.finditer(line):
                if _in_string_literal(line, m.start()):
                    continue
                found.add(m.group(1) or m.group(2))
    found.discard("")
    return found


def _derive_toggle_vars(root: Path) -> set[str]:
    """``register_<tag>_tools`` → ``<TAG>TOOL`` (the framework's auto-derived toggle name)."""
    tags: set[str] = set()
    for py in root.rglob("*.py"):
        if ".venv" in py.parts or "__pycache__" in py.parts:
            continue
        try:
            tags.update(_REGISTRAR.findall(py.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    # The shared surface helpers match the pattern but are not domain registrars.
    tags -= {"verbose", "tool_surface"}
    return {f"{t.upper()}TOOL" for t in tags}


@lru_cache(maxsize=1)
def _agent_utilities_reads() -> frozenset[str]:
    """``setting(...)`` literals read inside the installed agent-utilities core."""
    import agent_utilities

    au_root = Path(agent_utilities.__file__).resolve().parent
    return frozenset(_scan_setting_calls(au_root))


def _mcp_config_env_blocks(root: Path) -> list[tuple[Path, dict[str, str]]]:
    """Every ``mcpServers.<name>.env`` block across ``mcp_config*.json`` files."""
    blocks: list[tuple[Path, dict[str, str]]] = []
    for cfg in [*root.glob("mcp_config*.json"), *root.rglob("mcp_config*.json")]:
        if ".venv" in cfg.parts:
            continue
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for server in (data.get("mcpServers") or {}).values():
            env = server.get("env")
            if isinstance(env, dict):
                blocks.append((cfg, env))
    # de-dup by resolved path while preserving the per-server granularity
    seen: set[tuple[str, frozenset[str]]] = set()
    uniq: list[tuple[Path, dict[str, str]]] = []
    for path, env in blocks:
        key = (str(path.resolve()), frozenset(env))
        if key not in seen:
            seen.add(key)
            uniq.append((path, env))
    return uniq


def _readme_example_region(text: str) -> str:
    """The README slice that holds the mcp_config examples (markers preferred, else the
    ``## MCP Configuration Examples`` heading up to the additional-deployment marker)."""
    if README_START in text and README_END in text:
        return text[text.index(README_START) : text.index(README_END)]
    if README_HEADING in text:
        start = text.index(README_HEADING)
        end = (
            text.index(README_ADDL)
            if README_ADDL in text and text.index(README_ADDL) > start
            else len(text)
        )
        return text[start:end]
    return ""


def _readme_example_env_blocks(root: Path) -> list[dict[str, str]]:
    """Every ``mcpServers.<name>.env`` dict in the README's fenced ```json examples."""
    readme = root / "README.md"
    if not readme.exists():
        return []
    region = _readme_example_region(readme.read_text(encoding="utf-8"))
    blocks: list[dict[str, str]] = []
    for m in _JSON_FENCE.finditer(region):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        for server in (data.get("mcpServers") or {}).values():
            env = server.get("env")
            if isinstance(env, dict):
                blocks.append(env)
    return blocks


def _compose_env_keys(root: Path) -> dict[str, set[str]]:
    """Env keys referenced in each ``*compose*.yml`` ``environment:`` section."""
    out: dict[str, set[str]] = {}
    candidates = [
        *root.glob("*compose*.y*ml"),
        *root.glob("docker/*compose*.y*ml"),
    ]
    for comp in candidates:
        try:
            lines = comp.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        keys: set[str] = set()
        in_env = False
        env_indent = 0
        for raw in lines:
            stripped = raw.strip()
            if re.match(r"^environment\s*:", stripped):
                in_env = True
                env_indent = len(raw) - len(raw.lstrip())
                continue
            if in_env:
                indent = len(raw) - len(raw.lstrip())
                if stripped and indent <= env_indent and not stripped.startswith("-"):
                    in_env = False
                    continue
                m = _COMPOSE_ENV.match(raw)
                if m:
                    keys.add(m.group(1))
        if keys:
            out[comp.name] = keys
    return out


def _is_framework_known(var: str, code_read: set[str]) -> bool:
    return (
        var in code_read
        or var in INHERITED_ENV
        or var in FRAMEWORK_EXTRA
        or var in RUNTIME_ALLOWLIST
        or var.startswith(_RUNTIME_PREFIXES)
        or any(var.endswith(suf) for suf in _SAFE_SUFFIXES)
    )


def analyze(root: Path) -> dict:
    """Compute the code-read set and diff the declared sets against it."""
    pkg_reads = _scan_setting_calls(root)
    toggles = _derive_toggle_vars(root)
    code_read = pkg_reads | toggles | set(_agent_utilities_reads())

    env_example = root / ".env.example"
    declared_env = (
        {r[0] for r in parse_env_example(env_example.read_text(encoding="utf-8"))}
        if env_example.exists()
        else set()
    )
    mcp_blocks = _mcp_config_env_blocks(root)
    compose = _compose_env_keys(root)

    findings: list[dict] = []

    # DEAD — declared somewhere but read by nothing and not a framework var.
    declared_sources: dict[str, set[str]] = {}
    for var in declared_env:
        declared_sources.setdefault(var, set()).add(".env.example")
    for path, env in mcp_blocks:
        for var in env:
            declared_sources.setdefault(var, set()).add(_rel(path, root))
    for name, keys in compose.items():
        for var in keys:
            declared_sources.setdefault(var, set()).add(f"docker/{name}")

    for var, sources in sorted(declared_sources.items()):
        # placeholder template keys like <YOUR_X> never appear as A-Z names; skip secrets keys read.
        if _is_framework_known(var, code_read):
            continue
        # a strong "rename" hint: a sibling the code DOES read with the same stem
        hint = _rename_hint(var, code_read)
        findings.append(
            {
                "type": "DEAD",
                "var": var,
                "sources": sorted(sources),
                "hint": hint,
            }
        )

    # UNDOCUMENTED — code reads it but it's absent from .env.example (additive, safe).
    documentable = (pkg_reads | toggles) - declared_env
    declared_host_stems = {_stem(v) for v in declared_env if _is_host_var(v)}
    for var in sorted(documentable):
        # skip framework-inherited vars (shown in the inherited table)
        if var in INHERITED_ENV or var in FRAMEWORK_EXTRA:
            continue
        # skip a legacy host alias whose canonical host sibling is already documented
        # (e.g. legacy LANGFUSE_HOST when LANGFUSE_BASE_URL is in .env.example) — but
        # never suppress a credential/toggle just because a host of the same stem exists.
        if _is_host_var(var) and _stem(var) in declared_host_stems:
            continue
        findings.append(
            {"type": "UNDOCUMENTED", "var": var, "sources": ["(code)"], "hint": ""}
        )

    # MISSING_TOOL_MODE — a launch-style mcp_config env block without MCP_TOOL_MODE.
    for path, env in mcp_blocks:
        if "MCP_TOOL_MODE" not in env:
            findings.append(
                {
                    "type": "MISSING_TOOL_MODE",
                    "var": "MCP_TOOL_MODE",
                    "sources": [_rel(path, root)],
                    "hint": 'add "MCP_TOOL_MODE": "condensed" to the env block',
                }
            )

    # env_sources imports this module, so defer the import to call time (no import cycle).
    from agent_utilities.mcp.env_sources import example_env_pairs, is_agent_only

    # MALFORMED_VALUE — a whitespace-padded substitution like "${ VAR:-True }".
    for path, env in mcp_blocks:
        for var, value in env.items():
            if any(
                m.group(1) != m.group(1).strip() for m in _SUBST.finditer(str(value))
            ):
                findings.append(
                    {
                        "type": "MALFORMED_VALUE",
                        "var": var,
                        "sources": [_rel(path, root)],
                        "hint": 'use "${VAR:-default}" (no spaces inside the braces)',
                    }
                )

    # AGENT_VAR_IN_MCP — an agent-runtime var in an MCP-server config env block.
    for path, env in mcp_blocks:
        for var in sorted(env):
            if is_agent_only(var):
                findings.append(
                    {
                        "type": "AGENT_VAR_IN_MCP",
                        "var": var,
                        "sources": [_rel(path, root)],
                        "hint": "agent-only — move to the agent config (not the MCP server)",
                    }
                )

    # README mcp_config examples — STALE_EXAMPLE + missing MCP_TOOL_MODE.
    allowed = {name for name, _ in example_env_pairs(root)} | {
        "TRANSPORT",
        "HOST",
        "PORT",
    }
    for env in _readme_example_env_blocks(root):
        if "MCP_TOOL_MODE" not in env:
            findings.append(
                {
                    "type": "MISSING_TOOL_MODE",
                    "var": "MCP_TOOL_MODE",
                    "sources": ["README.md (example)"],
                    "hint": 'add "MCP_TOOL_MODE": "condensed" to the example env block',
                }
            )
        for var in sorted(env):
            if var not in allowed:
                findings.append(
                    {
                        "type": "STALE_EXAMPLE",
                        "var": var,
                        "sources": ["README.md (example)"],
                        "hint": (
                            "agent-only — belongs in the agent config"
                            if is_agent_only(var)
                            else "not a code-read var — regenerate the examples"
                        ),
                    }
                )

    return {
        "package": root.name,
        "code_read_count": len(code_read),
        "findings": findings,
        "drift": len(findings),
    }


def _rename_hint(dead_var: str, code_read: set[str]) -> str:
    """If a read var shares this var's stem, the dead var is likely a rename of it."""
    stem = re.sub(r"(_URL|_BASE_URL|_HOST|_TOKEN|_KEY|TOOL)$", "", dead_var)
    if len(stem) < 3:
        return ""
    for read in code_read:
        if read != dead_var and read.startswith(stem):
            return f"likely rename of `{read}`"
    return ""


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _format(report: dict) -> str:
    lines = [f"env-var drift — {report['package']}: {report['drift']} finding(s)"]
    if not report["findings"]:
        lines.append("  ✓ clean (config matches the code-read env surface)")
        return "\n".join(lines)
    for f in report["findings"]:
        loc = ", ".join(f["sources"])
        hint = f"  → {f['hint']}" if f["hint"] else ""
        lines.append(f"  [{f['type']}] {f['var']}  ({loc}){hint}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect env-var/config drift.")
    parser.add_argument("root", nargs="?", default=".", help="Package root")
    parser.add_argument("--check", action="store_true", help="Exit 1 on any drift")
    parser.add_argument("--json", action="store_true", help="Emit JSON findings")
    args = parser.parse_args(argv)

    report = analyze(Path(args.root))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_format(report))

    if args.check and report["drift"]:
        print(
            "\nenv-var drift detected — config/docs disagree with the code-read "
            "surface. Fix the sources above (the code is the source of truth).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
