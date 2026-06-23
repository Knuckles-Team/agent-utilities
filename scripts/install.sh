#!/usr/bin/env bash
# One-link self-deploy bootstrap for agent-utilities.
#
# Designed to be curl-pipe-able as well as run from a clone:
#   curl -fsSL https://knuckles-team.github.io/agent-utilities/install.sh | sh
#   ./scripts/install.sh --profile single-node-prod --component agent-webui
#
# What it does (idempotent, safe to re-run):
#   1. checks Python (>=3.11,<3.15) and ensures uv is available
#   2. installs agent-utilities (+ universal-skills for the skill/MCP installers)
#   3. runs the host dependency PREFLIGHT for the chosen profile/components
#   4. installs the skill toolkit into EVERY agent tool on the host, and wires the
#      graph-os MCP server into each (so your agent immediately has the KG + the
#      genesis skills to finish the deployment)
#   5. points you at the next step — telling your agent to "deploy agent-utilities"
#
# It deliberately does NOT fabricate the 50+ *-mcp fleet config or secrets — that
# is the job of the agent-utilities-deployment / agent-os-genesis skills, which read
# genesis.yaml and your chosen profile. This script gets your agent to the doorstep.
set -euo pipefail

PROFILE="${AU_PROFILE:-tiny}"
COMPONENTS=()
EXTRAS=""            # "" = auto (all for non-tiny, base for tiny)
DO_SKILLS=1
DO_MCP=1
EDITABLE=0
DRY_RUN=0

c_info() { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
c_warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*"; }
c_err()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; }

run() {
  if [ "$DRY_RUN" = 1 ]; then printf '  $ %s\n' "$*"; else eval "$@"; fi
}

usage() {
  cat <<'USAGE'
agent-utilities one-link installer

Usage: install.sh [options]
  --profile <p>       tiny | single-node-prod | enterprise   (default: tiny)
  --component <c>     opt-in UI: agent-webui | geniusbot | agent-terminal-ui (repeatable)
  --extras all|none   force install extras (default: auto — all for non-tiny)
  --editable          pip install -e from the current repo checkout (dev)
  --no-skills         do not install the skill toolkit into agent tools
  --no-mcp            do not wire the graph-os MCP server into agent tools
  --dry-run           print the planned actions without executing
  -h | --help         show this help
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --component) COMPONENTS+=("$2"); shift 2 ;;
    --extras) EXTRAS="$2"; shift 2 ;;
    --all-extras) EXTRAS="all"; shift ;;
    --editable) EDITABLE=1; shift ;;
    --no-skills) DO_SKILLS=0; shift ;;
    --no-mcp) DO_MCP=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) c_err "unknown option: $1"; usage; exit 2 ;;
  esac
done

# Components from env (comma-separated), merged with any --component flags.
if [ -n "${AU_COMPONENTS:-}" ]; then
  IFS=',' read -r -a _env_comps <<< "$AU_COMPONENTS"
  COMPONENTS+=("${_env_comps[@]}")
fi

# Auto extras: full integration unless the zero-infra tiny profile.
if [ -z "$EXTRAS" ]; then
  if [ "$PROFILE" = "tiny" ]; then EXTRAS="none"; else EXTRAS="all"; fi
fi

c_info "profile=$PROFILE  components=[${COMPONENTS[*]:-}]  extras=$EXTRAS  dry_run=$DRY_RUN"

# 1. Python check (the package enforces >=3.11,<3.15; fail early with guidance).
if ! command -v python3 >/dev/null 2>&1; then
  c_err "python3 not found — install Python 3.11–3.14 first."; exit 1
fi
PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
c_info "Python $PYV"
python3 - <<'PY' || { c_err "agent-utilities needs Python >=3.11,<3.15"; exit 1; }
import sys
ok = (3, 11) <= sys.version_info[:2] < (3, 15)
sys.exit(0 if ok else 1)
PY

# 2. Ensure uv (preferred installer). Auto-install if missing (official script).
if ! command -v uv >/dev/null 2>&1; then
  c_warn "uv not found — installing it (https://astral.sh/uv)…"
  run "curl -fsSL https://astral.sh/uv/install.sh | sh"
  export PATH="$HOME/.local/bin:$PATH"
fi

# Package spec — editable from a clone, else from PyPI; extras by profile.
PKG="agent-utilities"
if [ "$EXTRAS" = "all" ]; then PKG="agent-utilities[all]"; fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd || true)"
IN_REPO=0
if [ -n "$REPO_ROOT" ] && grep -q 'name = "agent-utilities"' "$REPO_ROOT/pyproject.toml" 2>/dev/null; then
  IN_REPO=1
fi

c_info "Installing $PKG + universal-skills…"
if [ "$EDITABLE" = 1 ] && [ "$IN_REPO" = 1 ]; then
  run "uv pip install --system -e \"$REPO_ROOT[${EXTRAS/none/}]\" || pip install -e \"$REPO_ROOT\""
else
  run "uv tool install \"$PKG\"" || run "pip install \"$PKG\""
  run "uv tool install universal-skills" || run "pip install universal-skills"
fi

# 3. Host dependency preflight for the chosen profile + components.
PF_ARGS="--preflight --profile $PROFILE"
for c in "${COMPONENTS[@]:-}"; do [ -n "$c" ] && PF_ARGS="$PF_ARGS --component $c"; done
c_info "Running host preflight…"
if command -v agent-utilities-doctor >/dev/null 2>&1; then
  run "agent-utilities-doctor $PF_ARGS" || c_warn "preflight reported issues — see remediations above."
else
  run "python3 -m agent_utilities.deployment.doctor $PF_ARGS" || c_warn "preflight reported issues."
fi

# 4. Install the skill toolkit into EVERY agent tool present, and wire graph-os MCP.
#    This sweep also picks up skills contributed by any installed agent-package
#    via its `agent_utilities.skill_providers` entry-point (CONCEPT:OS-5.52), so
#    re-run `install-skills --all-detected --symlink` after the *-mcp fleet is
#    deployed to wire in the freshly-installed providers' skills. Contributed
#    prompts enter the KG automatically on the next registry build.
if [ "$DO_SKILLS" = 1 ]; then
  c_info "Installing skills into every detected agent tool (--all-detected)…"
  if command -v install-skills >/dev/null 2>&1; then
    run "install-skills --all-detected --symlink" || c_warn "skill install reported issues."
  else
    c_warn "install-skills not on PATH — skipping (pip install universal-skills)."
  fi
fi

if [ "$DO_MCP" = 1 ]; then
  c_info "Wiring the graph-os MCP server into every detected agent tool…"
  MCP_SRC="$(mktemp -t graphos-mcp.XXXXXX.json)"
  cat > "$MCP_SRC" <<'JSON'
{
  "mcpServers": {
    "graph-os": {
      "command": "graph-os",
      "args": [],
      "env": {"GRAPH_BACKEND": "epistemic_graph"}
    }
  }
}
JSON
  MCP_INSTALL="$(python3 -c 'import importlib.util as u,os; s=u.find_spec("universal_skills"); print(os.path.join(os.path.dirname(s.origin),"agent-tools","mcp-installer","scripts","install.py")) if s else print("")' 2>/dev/null || true)"
  if [ -n "$MCP_INSTALL" ] && [ -f "$MCP_INSTALL" ]; then
    run "python3 \"$MCP_INSTALL\" --config \"$MCP_SRC\" --all-detected" || c_warn "MCP wiring reported issues."
  else
    c_warn "mcp-installer not found — skipping graph-os MCP wiring."
  fi
  [ "$DRY_RUN" = 1 ] || rm -f "$MCP_SRC"
fi

# 4.5 Draw the Claude Code permission fence (CONCEPT:OS-5.40) so the CLI can run
# unattended safely: settings.json allow/ask/deny + defaultMode=acceptEdits +
# .claudeignore, derived from the live ActionPolicy. Idempotent + best-effort.
if command -v setup-config >/dev/null 2>&1; then
  c_info "Writing the Claude Code permission fence into ~/.claude (harness-fence)…"
  run "setup-config harness-fence --target \"$HOME/.claude\"" \
    || c_warn "harness-fence reported issues (non-fatal)."
fi

# 4.6 Register the concepts-regen git merge driver (CONCEPT:OS-5.42) so merging
# parallel session worktrees regenerates docs/concepts.yaml from CONCEPT markers
# instead of producing a textual conflict. The reservation ledger uses the
# built-in union driver (declared in .gitattributes); this only registers the
# custom regenerate driver. Best-effort and only inside a git work tree.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  c_info "Registering the concepts-regen git merge driver…"
  run "git config merge.concepts-regen.name 'regenerate concepts.yaml from CONCEPT markers'" \
    || c_warn "could not set merge driver name (non-fatal)."
  run "git config merge.concepts-regen.driver 'python3 scripts/build_concepts_yaml.py >/dev/null 2>&1; cp docs/concepts.yaml %A'" \
    || c_warn "could not set merge driver command (non-fatal)."
fi

# 5. Hand off to the deployment skill.
SKILL="agent-utilities-deployment"
[ "$PROFILE" = "enterprise" ] && SKILL="agent-os-genesis"
cat <<EOF

$(c_info "Done. Your agent now has the KG tools + the genesis skills.")

Next — open your agent (Claude Code / Cursor / …) and say:

    deploy agent-utilities (profile=$PROFILE)

It will run the '$SKILL' skill, read genesis.yaml, generate the full config
(setup-config), wire the *-mcp fleet + any UI components, and verify with
'agent-utilities-doctor'.

Manual path:
    setup-config generate --profile $PROFILE
    graph-os                       # stdio MCP for your IDE
    graph-os-daemon                # REST gateway (:8100)
    agent-utilities-doctor         # verify

Docs: https://knuckles-team.github.io/agent-utilities/
EOF
