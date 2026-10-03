#!/usr/bin/env bash
# Install the pinned native clone scanners (dupehound, jscpd).
#
# The versions are read from [tool.agent_utilities.clone_scanners] in
# pyproject.toml, the one place they are pinned; the gate wrappers verify the
# installed versions against the same table. jscpd additionally requires the
# source-pinned pipelines build and its verified provenance. Hooks never install anything
# themselves: run this once (scripts/bootstrap.sh --scanners does), or let CI
# run it. Cached jscpd builds are reverified on every invocation.
#
# Needs cargo for dupehound, Rust 1.97.0 for jscpd, and curl for provider source.
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

# Immutable merged provider source; cache bytes must still match before extraction.
PIPELINES_REV="c0a089c83eea9d0d08f48e6c00681eb989268d9a"
PIPELINES_ARCHIVE_SHA256="5521ddf30a7d8b09fd0a48fca1aa6251179567d1a56e76115d679516090f6d7a"
provider_archive="$SCANNER_ROOT/providers/$PIPELINES_REV.tar.gz"
provider_source="$(mktemp -d "$SCANNER_ROOT/pipelines.XXXXXXXX")"
trap 'rm -rf "$provider_source"' EXIT
mkdir -p "$SCANNER_ROOT/providers" "$provider_source/source"
if [ ! -f "$provider_archive" ]; then
  curl --fail --location --silent --show-error \
    "https://codeload.github.com/Knuckles-Team/pipelines/tar.gz/$PIPELINES_REV" \
    --output "$provider_source/provider.tar.gz"
  printf '%s  %s\n' "$PIPELINES_ARCHIVE_SHA256" "$provider_source/provider.tar.gz" | sha256sum --check >&2
  mv "$provider_source/provider.tar.gz" "$provider_archive"
fi
printf '%s  %s\n' "$PIPELINES_ARCHIVE_SHA256" "$provider_archive" | sha256sum --check >&2
tar -xzf "$provider_archive" --strip-components=1 -C "$provider_source/source"
jscpd_bin_dir="$(python3 "$provider_source/source/scripts/install_jscpd.py" --root "$SCANNER_ROOT/jscpd")"
test "$("$jscpd_bin_dir/jscpd" --version)" = "cpd $JSCPD_VERSION"
cat "$jscpd_bin_dir/jscpd.provenance.json" >&2
sha256sum "$jscpd_bin_dir/jscpd" > "$SCANNER_ROOT/jscpd.sha256"
cat "$SCANNER_ROOT/jscpd.sha256" >&2

if [ -n "${GITHUB_ENV:-}" ]; then
  {
    echo "DUPEHOUND_BIN=$BIN/dupehound"
    echo "JSCPD_BIN=$jscpd_bin_dir/jscpd"
  } >>"$GITHUB_ENV"
  printf '%s\n' "$BIN" "$jscpd_bin_dir" >>"$GITHUB_PATH"
fi
log "scanners ready in $BIN"
