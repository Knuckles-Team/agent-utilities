"""The canonical env-var set for an MCP-server ``mcp_config.json`` example.

CONCEPT:AU-OS.config.env-var-single-source — Env-var single source of truth.

Three surfaces must agree 1:1:1 for every MCP-server package: the ``mcp_config*.json``
``env`` blocks, the README env-var table, and the README ``mcp_config.json`` examples.
The authority is the code the package reads. This module distils that into the set that
belongs in an **MCP-server** config's ``env`` block:

    (package code-read vars ∪ derived ``<TAG>TOOL`` toggles)
      − inherited agent-utilities infra (transport/telemetry/governance/outbound-auth)
      − agent-only vars (the ``[agent-runtime]`` environment + companion tool suites)
      + ``MCP_TOOL_MODE`` (always — it selects intent/condensed/verbose/both)

Inherited infra (OTEL/EUNOMIA/OIDC/DEBUG) is documented in the env-var table's *Inherited*
section, not repeated in every example block. Agent-only vars (``AGENT_DESCRIPTION``,
``AGENT_SYSTEM_PROMPT``, ``DEFAULT_AGENT_NAME``, ``MCP_URL``, ``PROVIDER``, ``MODEL_ID``,
``ENABLE_WEB_UI``, and ``*_ENABLE`` companion suites) launch the *agent*, never the MCP
server, so they must not appear in an MCP-server config.

This is the single definition consumed by both :mod:`readme_mcp_examples` (the generator)
and :mod:`check_env_var_drift` (the guard).
"""

from __future__ import annotations

from pathlib import Path

from agent_utilities.mcp.check_env_var_drift import (
    _RUNTIME_PREFIXES,
    _SAFE_SUFFIXES,
    FRAMEWORK_EXTRA,
    RUNTIME_ALLOWLIST,
    _derive_toggle_vars,
    _scan_setting_calls,
)
from agent_utilities.mcp.env_policy import AGENT_ONLY, is_agent_only
from agent_utilities.mcp.readme_env_vars import INHERITED_ENV, parse_env_example

# ``AGENT_ONLY``/``is_agent_only`` moved to the dependency-free leaf
# ``agent_utilities.mcp.env_policy`` so ``check_env_var_drift`` can import the
# predicate eagerly instead of deferring it to break the cycle this module
# formed with it (BUG-CX-004 / WD10-B-004). Re-exported here: this module is
# still the documented entry point for callers and tests.
__all__ = [
    "AGENT_ONLY",
    "example_env_pairs",
    "is_agent_only",
    "package_env_vars",
]


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
    (so examples show real defaults and commented runtime references). Variables with no
    example are omitted rather than projected as empty values that shadow runtime
    injection.
    """
    env_example = root / ".env.example"
    values: dict[str, str] = {}
    if env_example.exists():
        for name, example, _desc in parse_env_example(
            env_example.read_text(encoding="utf-8")
        ):
            values[name] = example
    pairs: list[tuple[str, str]] = [("MCP_TOOL_MODE", "intent")]
    for var in sorted(package_env_vars(root)):
        value = values.get(var) or (
            INHERITED_ENV[var][0] if var in INHERITED_ENV else ""
        )
        if value:
            pairs.append((var, value))
    return pairs
