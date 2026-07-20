#!/usr/bin/env bash
# Day-0 Tiny-tier bootstrap for agent-utilities.
#
# Idempotent: creates a venv, installs the package, seeds the zero-infra XDG
# AgentConfig document when absent, and runs a smoke test that exercises the
# in-process knowledge graph (no model provider needed).
#
# Usage:  ./scripts/bootstrap.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }

# 1. Python check
if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found (need >= 3.11)" >&2
  exit 1
fi
log "Python: $(python3 --version)"

# 2. Install (prefer uv, fall back to venv + pip)
if command -v uv >/dev/null 2>&1; then
  log "Installing with uv..."
  uv sync >/dev/null
  RUN=(uv run)
else
  log "uv not found — using venv + pip..."
  [ -d .venv ] || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -q --upgrade pip
  pip install -q -e ".[all]"
  RUN=()
fi

# 3. Zero-infra XDG AgentConfig (only if absent — never clobber)
if "${RUN[@]}" python - <<'PY'
from agent_utilities.deployment.config_generator import _default_config_path

raise SystemExit(0 if _default_config_path().is_file() else 1)
PY
then
  log "XDG AgentConfig already exists — leaving it untouched."
else
  log "Writing zero-infra XDG AgentConfig..."
  "${RUN[@]}" setup-config generate --profile tiny >/dev/null
fi

# 4. Smoke test — exercise the KG directly (no model provider required)
log "Running knowledge-graph smoke test..."
"${RUN[@]}" python - <<'PY'
import asyncio
from agent_utilities.mcp import kg_server

async def main():
    kg_server.ensure_tools_registered()
    await kg_server._execute_tool(
        "graph_write", action="add_node",
        node_id="bootstrap:hello", node_type="Greeting",
        properties='{"msg":"it works"}',
    )
    res = await kg_server._execute_tool(
        "graph_query", cypher="MATCH (n:Greeting) RETURN n",
    )
    assert res is not None, "graph_query returned nothing"
    print("  ✓ wrote a node and queried it back — the KG works with zero infra.")

asyncio.run(main())
PY

log "Done. Next:"
log "  • Library:   from agent_utilities import create_agent"
log "  • MCP:       ${RUN[*]:-} graph-os            (stdio, for IDE agents)"
log "  • REST:      ${RUN[*]:-} graph-os-daemon     (:8100)"
log "  • Docs:      docs/start-here.md"
