#!/usr/bin/env bash
# One-command contributor setup from a fresh clone. Idempotent and
# non-interactive; safe for people, CI and any AI agent.
#
#   scripts/bootstrap.sh
#
# Installs a current uv, the pinned Python (.python-version, else 3.14), the
# locked environment (`uv sync --locked`, or `uv sync` when the repository
# has no uv.lock yet), and the pre-commit and pre-push git hooks. Then it
# prints the gate command. Set BOOTSTRAP_SKIP_HOOKS=1 to skip hook install.
set -euo pipefail

cd "$(dirname "$0")/.."

log() { printf 'bootstrap: %s\n' "$*"; }
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

case "${1:-}" in
  -h | --help)
    sed -n '2,10p' "$0"
    exit 0
    ;;
  "") ;;
  *)
    echo "unknown argument: $1" >&2
    exit 2
    ;;
esac

# 1. uv + Python ------------------------------------------------------------
# uv older than 0.9 cannot download current CPython patch releases.
uv_minor() { uv --version 2>/dev/null | awk '{split($2, v, "."); print v[1] * 1000 + v[2]}'; }
if ! command -v uv >/dev/null 2>&1 || [[ "$(uv_minor)" -lt 9 ]]; then
  log "installing a current uv"
  python3 -m pip install --quiet --user --upgrade "uv>=0.9" \
    || python3 -m pip install --quiet --user --break-system-packages --upgrade "uv>=0.9" \
    || { echo "bootstrap: could not install uv>=0.9; see https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 2; }
fi
python_version="3.14"
if [ -f .python-version ]; then
  python_version="$(tr -d '[:space:]' < .python-version)"
fi
log "Python ${python_version} (uv $(uv --version | awk '{print $2}'))"
uv python install "$python_version"

# 2. Environment ------------------------------------------------------------
if [ -f uv.lock ]; then
  log "uv sync --locked"
  uv sync --locked --python "$python_version"
else
  log "no uv.lock: uv sync (generate and commit one)"
  uv sync --python "$python_version"
fi

# 3. Git hooks --------------------------------------------------------------
# The pre-commit config is .pre-commit-config.yaml, or .config/pre-commit.yaml
# in repositories that relocate it.
if [ "${BOOTSTRAP_SKIP_HOOKS:-0}" != 1 ]; then
  precommit_config=""
  for candidate in .pre-commit-config.yaml .config/pre-commit.yaml; do
    if [ -f "$candidate" ]; then
      precommit_config="$candidate"
      break
    fi
  done
  if [ -n "$precommit_config" ]; then
    log "installing pre-commit and pre-push hooks ($precommit_config)"
    uvx pre-commit install --config "$precommit_config" --hook-type pre-commit --hook-type pre-push >/dev/null
  fi
fi

if [ -n "${precommit_config:-}" ] && [ "$precommit_config" != .pre-commit-config.yaml ]; then
  log "done. Gate command: uvx pre-commit run --config $precommit_config --all-files"
else
  log "done. Gate command: uvx pre-commit run --all-files"
fi
