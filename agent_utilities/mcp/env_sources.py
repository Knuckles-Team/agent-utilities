"""The canonical env-var set for an MCP-server ``mcp_config.json`` example.

CONCEPT:OS-5.72 — Env-var single source of truth.

Three surfaces must agree 1:1:1 for every MCP-server package: the ``mcp_config*.json``
``env`` blocks, the README env-var table, and the README ``mcp_config.json`` examples.
The authority is the code the package reads. This module distils that into the set that
belongs in an **MCP-server** config's ``env`` block:

    (package code-read vars ∪ derived ``<TAG>TOOL`` toggles)
      − inherited agent-utilities infra (transport/telemetry/governance/outbound-auth)
      − agent-only vars (the ``[agent]`` runtime + companion tool suites)
      + ``MCP_TOOL_MODE`` (always — it selects the condensed/verbose/both surface)

Inherited infra (OTEL/EUNOMIA/OIDC/DEBUG) is documented in the env-var table's *Inherited*
section, not repeated in every example block. Agent-only vars (``AGENT_DESCRIPTION``,
``AGENT_SYSTEM_PROMPT``, ``DEFAULT_AGENT_NAME``, ``MCP_URL``, ``PROVIDER``, ``MODEL_ID``,
``ENABLE_WEB_UI``, and ``*_ENABLE`` companion suites) launch the *agent*, never the MCP
server, so they must not appear in an MCP-server config.

This is the single definition consumed by both :mod:`readme_mcp_examples` (the generator)
and :mod:`check_env_var_drift` (the guard).
"""

from __future__ import annotations

import re
from pathlib import Path

from agent_utilities.mcp.check_env_var_drift import (
    _RUNTIME_PREFIXES,
    _SAFE_SUFFIXES,
    FRAMEWORK_EXTRA,
    RUNTIME_ALLOWLIST,
    _derive_toggle_vars,
    _scan_setting_calls,
)
from agent_utilities.mcp.readme_env_vars import INHERITED_ENV, parse_env_example

# Vars that belong to the ``[agent]`` runtime, not the MCP server. They are legitimately
# read by agent-utilities core (and so appear in ``FRAMEWORK_EXTRA``) and may sit in a
# package's ``.env.example`` — but placing them in an *MCP-server* ``mcp_config.json``
# ``env`` block or README MCP example is drift.
AGENT_ONLY: frozenset[str] = frozenset(
    {
        "AGENT_DESCRIPTION",
        "AGENT_SYSTEM_PROMPT",
        "DEFAULT_AGENT_NAME",
        "MCP_URL",
        "PROVIDER",
        "MODEL_ID",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "ENABLE_WEB_UI",
    }
)
# Companion tool-suite toggles (``SYSTEM_TOOLS_ENABLE``, ``BROWSER_TOOLS_ENABLE`` …) bundle
# universal-skills suites into the *agent*; the suffix distinguishes them from framework
# ``ENABLE_*`` prefixed vars (``ENABLE_OTEL``, ``ENABLE_WEB_UI``).
_COMPANION_RE = re.compile(r"^[A-Z][A-Z0-9_]*_ENABLE$")


def is_agent_only(var: str) -> bool:
    """True if ``var`` belongs to the agent runtime, not the MCP server."""
    return var in AGENT_ONLY or bool(_COMPANION_RE.match(var))


def _is_infra(var: str) -> bool:
    """True if ``var`` is inherited framework/runtime infra (kept out of examples)."""
    return (
        var in INHERITED_ENV
        or var in FRAMEWORK_EXTRA
        or var in RUNTIME_ALLOWLIST
        or var.startswith(_RUNTIME_PREFIXES)
        or any(var.endswith(suf) for suf in _SAFE_SUFFIXES)
    )


def package_env_vars(root: Path) -> set[str]:
    """The package's own MCP-server env vars: code-read reads + derived toggles, minus
    inherited infra and agent-only vars. Excludes ``MCP_TOOL_MODE`` (added by callers)."""
    candidates = _scan_setting_calls(root) | _derive_toggle_vars(root)
    return {v for v in candidates if not _is_infra(v) and not is_agent_only(v)}


def example_env_pairs(root: Path) -> list[tuple[str, str]]:
    """Canonical ``(name, value)`` pairs for an MCP-server config ``env`` block.

    ``MCP_TOOL_MODE`` is always first. Values come from the package's ``.env.example``
    (so examples show real defaults), falling back to the inherited default, else empty.
    """
    env_example = root / ".env.example"
    values: dict[str, str] = {}
    if env_example.exists():
        for name, example, _desc in parse_env_example(
            env_example.read_text(encoding="utf-8")
        ):
            values[name] = example
    pairs: list[tuple[str, str]] = [("MCP_TOOL_MODE", "condensed")]
    for var in sorted(package_env_vars(root)):
        value = values.get(var) or (INHERITED_ENV[var][0] if var in INHERITED_ENV else "")
        pairs.append((var, value))
    return pairs
