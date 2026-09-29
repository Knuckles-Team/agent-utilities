#!/usr/bin/env bash
# Python interpreter for the `language: system` hooks in .config/pre-commit.yaml.
#
# A bare `python3` in a hook resolves from PATH, which depends on how pre-commit
# was launched: the git hook inherits the caller's PATH, and `uvx pre-commit`
# puts its own tool environment's python3 first. Either way the hook would miss
# this repository's locked dependencies. This launcher runs the repository's
# bootstrapped `.venv` (scripts/bootstrap.sh) instead, so a hook behaves the same
# from a git hook, from `uvx pre-commit run`, and in CI. Without a `.venv` it
# falls back to python3 on PATH.
#
# Usage: scripts/hook_python.sh [--needs-engine GATE] SCRIPT|-m MODULE [ARGS...]
#
#   --needs-engine GATE  the gate imports the native epistemic-graph engine,
#                        which only `scripts/bootstrap.sh --engine` installs.
#                        When it is not importable the gate prints
#                        `SKIPPED (GATE): <reason>` and exits 0 locally, and
#                        exits 2 (CANNOT RUN) when CI is set.
set -euo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python=python3
for candidate in "$root/.venv/bin/python" "$root/.venv/Scripts/python.exe"; do
  if [ -x "$candidate" ]; then
    python="$candidate"
    break
  fi
done

if [ "${1:-}" = "--needs-engine" ]; then
  gate="${2:?--needs-engine requires a gate name}"
  shift 2
  if ! "$python" -c "import epistemic_graph" >/dev/null 2>&1; then
    reason="the epistemic-graph engine is not installed (scripts/bootstrap.sh --engine builds it)"
    if [ -n "${CI:-}" ]; then
      echo "$gate: CANNOT RUN: $reason" >&2
      exit 2
    fi
    echo "SKIPPED ($gate): $reason"
    exit 0
  fi
fi

exec "$python" "$@"
