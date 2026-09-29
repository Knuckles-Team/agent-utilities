#!/usr/bin/env bash
# Install the pinned native clone scanners (dupehound, jscpd).
#
# The versions are read from [tool.agent_utilities.clone_scanners] in
# pyproject.toml, the one place they are pinned; the gate wrappers verify the
# installed versions against the same table. Hooks never install anything
# themselves: run this once (scripts/bootstrap.sh --scanners does), or let CI
# run it. Idempotent: a scanner already at its pinned version is left alone.
#
# Needs cargo (Rust) for dupehound and npm (Node.js) for jscpd.
#
# Env: SCANNER_ROOT  install prefix (default ~/.local; binaries in $SCANNER_ROOT/bin)
# Under GitHub Actions the binaries are also exported through GITHUB_ENV and
# GITHUB_PATH for later steps.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCANNER_ROOT="${SCANNER_ROOT:-$HOME/.local}"
BIN="$SCANNER_ROOT/bin"
mkdir -p "$BIN"

pin() {
  python3 - "$REPO_ROOT/pyproject.toml" "$1" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    profile = tomllib.load(handle)["tool"]["agent_utilities"]["clone_scanners"]
print(profile[sys.argv[2]])
PY
}

log() { printf '[install_scanners] %s\n' "$*" >&2; }

DUPEHOUND_VERSION="$(pin dupehound_version)"
JSCPD_VERSION="$(pin jscpd_version)"

if [ "$("$BIN/dupehound" --version 2>/dev/null || true)" = "dupehound $DUPEHOUND_VERSION" ]; then
  log "dupehound $DUPEHOUND_VERSION already installed"
else
  command -v cargo >/dev/null 2>&1 || { echo "install_scanners: cargo is required for dupehound" >&2; exit 1; }
  log "installing dupehound $DUPEHOUND_VERSION"
  cargo install dupehound --version "$DUPEHOUND_VERSION" --locked --root "$SCANNER_ROOT"
fi
test "$("$BIN/dupehound" --version)" = "dupehound $DUPEHOUND_VERSION"

if [ "$("$BIN/jscpd" --version 2>/dev/null || true)" = "cpd $JSCPD_VERSION" ]; then
  log "jscpd $JSCPD_VERSION already installed"
else
  command -v npm >/dev/null 2>&1 || { echo "install_scanners: npm is required for jscpd" >&2; exit 1; }
  log "installing jscpd $JSCPD_VERSION"
  npm install --global --prefix "$SCANNER_ROOT" --no-audit --no-fund "jscpd@$JSCPD_VERSION"
fi
test "$("$BIN/jscpd" --version)" = "cpd $JSCPD_VERSION"

if [ -n "${GITHUB_ENV:-}" ]; then
  {
    echo "DUPEHOUND_BIN=$BIN/dupehound"
    echo "JSCPD_BIN=$BIN/jscpd"
  } >>"$GITHUB_ENV"
  echo "$BIN" >>"$GITHUB_PATH"
fi
log "scanners ready in $BIN"
