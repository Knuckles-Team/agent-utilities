"""Which env vars belong to the *agent runtime* rather than an MCP server.

CONCEPT:AU-OS.config.env-var-single-source — Env-var single source of truth.

★ WHY THIS MODULE EXISTS (BUG-CX-004 / WD10-B-004)
:mod:`agent_utilities.mcp.env_sources` imports six names from
:mod:`agent_utilities.mcp.check_env_var_drift` at module scope, and
``check_env_var_drift.analyze()`` needed ``is_agent_only`` back from
``env_sources`` — a real circular dependency that was papered over with a
function-local import carrying the comment *"env_sources imports this module,
so defer the import to call time (no import cycle)"*.

The predicate has no dependency on either side, so it belongs in a leaf that
both import. Keep this module free of intra-package imports — its whole job is
to have no outgoing edges.

``env_sources`` re-exports ``AGENT_ONLY`` and ``is_agent_only`` so existing
``from agent_utilities.mcp.env_sources import is_agent_only`` callers (and
``tests/mcp/test_readme_mcp_examples.py``) keep working unchanged.
"""

from __future__ import annotations

import re

# Vars that belong to the ``[agent-runtime]`` environment, not the MCP server. They are legitimately
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
