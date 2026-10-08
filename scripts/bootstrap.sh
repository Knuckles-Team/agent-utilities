#!/usr/bin/env bash
# One-command bootstrap for agent-utilities contributors and Tiny deployments.
#
# From a fresh clone with git and python3 (with pip) available this:
#   1. checks out the pinned `.uv-workspace-siblings/` sources listed in
#      scripts/siblings.lock (the same pins CI uses);
#   2. installs or upgrades uv to >= 0.9 when needed, then the pinned
#      Python, and syncs the locked development environment (`.venv`) with the
#      same selection the hosted CI jobs use;
#   3. fetches full history for a shallow clone and installs the pre-commit
#      and pre-push git hooks;
#   4. optionally installs the native clone scanners (--scanners), or builds the
#      native epistemic-graph engine from its pinned source and runs the Tiny
#      profile setup + knowledge-graph smoke test (--engine).
#
# Idempotent and non-interactive. Afterwards run every commit-stage gate with:
#   uvx pre-commit run --all-files --config .config/pre-commit.yaml
#
# Usage: scripts/bootstrap.sh [--siblings-only] [--engine] [--scanners] [--no-hooks]
#
#   --siblings-only  only materialize .uv-workspace-siblings/ (CI uses this)
#   --engine         also check out epistemic-graph, build it from source (Rust +
#                    maturin; a cold build takes tens of minutes on 4 CPUs), seed
#                    the zero-infra XDG AgentConfig when absent, and run the
#                    knowledge-graph smoke test. Without it the engine package
#                    is skipped exactly like CI; tests that need the real engine
#                    carry the `engine` marker.
#   --scanners       also install the pinned clone scanners (scripts/install_scanners.sh)
#   --no-hooks       do not install git hooks
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SIBLINGS_ONLY=0
ENGINE=0
SCANNERS=0
HOOKS=1
for arg in "$@"; do
  case "$arg" in
    --siblings-only) SIBLINGS_ONLY=1 ;;
    --engine) ENGINE=1 ;;
    --scanners) SCANNERS=1 ;;
    --no-hooks) HOOKS=0 ;;
    -h | --help)
      sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "bootstrap: unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

log() { printf '[bootstrap] %s\n' "$*" >&2; }

# ── 1. pinned sibling sources ───────────────────────────────────────────────
# A symlink (materialized by scripts/uv_workspace.py from a local ecosystem
# workspace) is respected as-is.
checkout_sibling() {
  local name="$1" url="$2" commit="$3"
  local dest=".uv-workspace-siblings/$name"
  if [ -L "$dest" ]; then
    log "$name: $dest is a workspace symlink; leaving it untouched"
    return 0
  fi
  if [ -d "$dest/.git" ] && [ "$(git -C "$dest" rev-parse HEAD 2>/dev/null)" = "$commit" ]; then
    log "$name: already at ${commit:0:12}"
    return 0
  fi
  if [ -e "$dest" ] && [ ! -d "$dest/.git" ]; then
    echo "bootstrap: $dest exists but is not a git checkout; remove it and re-run" >&2
    exit 1
  fi
  log "$name: checking out ${commit:0:12} from $url"
  mkdir -p "$dest"
  git -C "$dest" init -q
  git -C "$dest" remote remove origin 2>/dev/null || true
  git -C "$dest" remote add origin "$url"
  local attempt
  for attempt in 1 2 3 4; do
    if git -C "$dest" fetch -q --depth 1 origin "$commit"; then
      break
    fi
    if [ "$attempt" -eq 4 ]; then
      echo "bootstrap: could not fetch $name@$commit" >&2
      exit 1
    fi
    sleep $((2 ** attempt))
  done
  git -C "$dest" -c advice.detachedHead=false checkout -q --force FETCH_HEAD
  test "$(git -C "$dest" rev-parse HEAD)" = "$commit"
}

while read -r name url commit when; do
  case "$name" in '' | '#'*) continue ;; esac
  if [ "$when" = "always" ] || { [ "$when" = "engine" ] && [ "$ENGINE" -eq 1 ]; }; then
    checkout_sibling "$name" "$url" "$commit"
  fi
done <scripts/siblings.lock

if [ "$SIBLINGS_ONLY" -eq 1 ]; then
  exit 0
fi

# ── 2. uv, Python, locked environment ───────────────────────────────────────
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# uv older than 0.9 cannot download current CPython patch releases, so an older
# uv (or none) is upgraded from PyPI. Piping the astral installer into a shell is
# deliberately not a fallback: the supply-chain gate bans executing an
# unverified network response in an installer.
UV_MIN="0.9"
uv_is_current() {
  command -v uv >/dev/null 2>&1 || return 1
  local version
  version="$(uv --version | awk '{print $2}')"
  [ "$(printf '%s\n%s\n' "$UV_MIN" "$version" | sort -V | head -n1)" = "$UV_MIN" ]
}
if ! uv_is_current; then
  log "uv >= $UV_MIN not found; installing it"
  python3 -m pip install --user --quiet --upgrade "uv>=$UV_MIN" 2>/dev/null ||
    python3 -m pip install --user --quiet --upgrade --break-system-packages "uv>=$UV_MIN" 2>/dev/null ||
    true
  hash -r
  uv_is_current || {
    echo "bootstrap: could not install uv >= $UV_MIN with pip; install it with the" >&2
    echo "official installer (https://docs.astral.sh/uv/getting-started/installation/)" >&2
    echo "and re-run" >&2
    exit 1
  }
fi
log "$(uv --version)"

# The interpreter floor of `requires-python` is the version CI runs.
PYTHON_VERSION="$(sed -n 's/^requires-python *= *">=\([0-9]*\.[0-9]*\).*/\1/p' pyproject.toml)"
: "${PYTHON_VERSION:?could not read requires-python from pyproject.toml}"
uv python install "$PYTHON_VERSION"

# langfuse-agent depends back on agent-utilities and is a deployment component,
# never part of the development environment (CI excludes it the same way).
# The import-safety gate walks browser modules too; install their declared Python
# dependency. Browser binaries are not needed or installed by this bootstrap.
SYNC=(uv sync --frozen --python "$PYTHON_VERSION"
  --extra test --extra agent-runtime --extra browser --group guardrails
  --no-install-package langfuse-agent)
if [ "$ENGINE" -eq 1 ]; then
  log "building the native engine from source; this is the slow step"
else
  SYNC+=(--no-install-package epistemic-graph)
fi
log "${SYNC[*]}"
"${SYNC[@]}"

# ── 3. full history, git hooks ──────────────────────────────────────────────
# Some gates resolve recorded revisions and changed-function ranges, which a
# shallow clone (the cloud-session default) does not contain.
if [ -z "${CI:-}" ] && [ "$(git rev-parse --is-shallow-repository 2>/dev/null)" = "true" ]; then
  log "unshallowing the clone for history-aware gates"
  git fetch -q --unshallow origin || log "could not unshallow; history-aware gates may fail"
fi
if [ "$HOOKS" -eq 1 ] && [ -z "${CI:-}" ] && git rev-parse --git-dir >/dev/null 2>&1; then
  if [ -n "$(git config --get core.hooksPath || true)" ]; then
    log "core.hooksPath is set; skipping hook installation (unset it to install)"
  else
    uvx pre-commit install --config .config/pre-commit.yaml \
      --hook-type pre-commit --hook-type pre-push
    scripts/hook_python.sh scripts/pre_push.py --install
  fi
fi

# ── 4. optional extras ──────────────────────────────────────────────────────
if [ "$SCANNERS" -eq 1 ]; then
  bash scripts/install_scanners.sh
fi

if [ "$ENGINE" -eq 1 ]; then
  # Zero-infra XDG AgentConfig (only if absent; never clobber).
  if uv run --no-sync python - <<'PY'; then
from agent_utilities.deployment.config_generator import _default_config_path

raise SystemExit(0 if _default_config_path().is_file() else 1)
PY
    log "XDG AgentConfig already exists; leaving it untouched"
  else
    log "writing zero-infra XDG AgentConfig"
    uv run --no-sync setup-config generate --profile tiny >/dev/null
  fi

  log "running knowledge-graph smoke test"
  uv run --no-sync python - <<'PY'
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
        "graph_query", query="MATCH (n:Greeting) RETURN n",
    )
    assert res is not None, "graph_query returned nothing"
    print("  wrote a node and queried it back: the KG works with zero infra.")

asyncio.run(main())
PY
fi

log "done. Next:"
log "  gates:  uvx pre-commit run --all-files --config .config/pre-commit.yaml"
log "  tests:  uv run --no-sync pytest tests/unit/<path> -q"
