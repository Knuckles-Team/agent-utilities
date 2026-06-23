#!/usr/bin/env bash
# CONCEPT:ORCH-1.88 — Build a Wizer-preinitialized CPython-WASI payload for the `wasm` rung.
#
# The `wasm` sandbox (agent_utilities/rlm/sandboxes/wasm_backend.py) instantiates a fresh CPython
# every run, paying the interpreter-init + import cost cold each time. Wizer
# (https://github.com/bytecodealliance/wizer) runs a module's init ONCE at build time and snapshots
# the resulting linear memory into a new .wasm, so each later instantiation starts from the warmed
# heap — the build-time analogue of the forkserver rung's runtime warm-fork, and it works on any
# platform (incl. ARM, where Firecracker can't run). _resolve_payload() prefers `python-warm*.wasm`.
#
# Prerequisites (all out-of-band, none are core deps — detection-gated like every other rung):
#   * wizer            — `cargo install wizer --all-features`  (or a release binary on PATH)
#   * a base CPython-WASI module that exports a Wizer init hook (`wizer.initialize`). The
#     VMware-Labs / wasmlabs `python.wasm` and the cpython WASI SDK builds support this.
#
# Usage:
#   scripts/build_wasm_warm_payload.sh [BASE_WASM] [OUT_NAME] [PRELOAD ...]
#     BASE_WASM   path to the cold python.wasm        (default: $RLM_WASM_PYTHON or cache/python.wasm)
#     OUT_NAME    output filename                     (default: python-warm.wasm)
#     PRELOAD     module names to import during warm-up (default: json os io math statistics)
#
# Output lands in the same cache dir _resolve_payload() scans:
#   ${XDG_CACHE_HOME:-~/.cache}/agent-utilities/rlm-wasm/<OUT_NAME>
set -euo pipefail

CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/agent-utilities/rlm-wasm"
BASE_WASM="${1:-${RLM_WASM_PYTHON:-$CACHE_DIR/python.wasm}}"
OUT_NAME="${2:-python-warm.wasm}"
shift || true
shift || true
PRELOAD=("${@:-json os io math statistics}")

if ! command -v wizer >/dev/null 2>&1; then
  echo "error: 'wizer' not found on PATH. Install with: cargo install wizer --all-features" >&2
  exit 1
fi
if [[ ! -f "$BASE_WASM" ]]; then
  echo "error: base CPython-WASI module not found: $BASE_WASM" >&2
  echo "       set RLM_WASM_PYTHON or place python.wasm in $CACHE_DIR" >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# The init script Wizer runs once at build time: import the preload set so the post-import heap is
# captured. Keep it to stdlib + whatever the base payload actually bundles — third-party libs on
# WASI are limited, so the realistic win here is interpreter + stdlib + harness import state.
cat >"$WORK/warmup.py" <<PY
import importlib
for _m in "${PRELOAD[*]}".split():
    try:
        importlib.import_module(_m)
    except Exception as _e:  # a missing module must not abort the snapshot
        print("warm-up skip", _m, _e)
print("wizer warm-up complete")
PY

OUT_PATH="$CACHE_DIR/$OUT_NAME"
echo ">> wizer: warming $BASE_WASM  (preload: ${PRELOAD[*]})"
# --allow-wasi lets the init script use WASI (stdin/preopens); --init-func is the module's hook.
wizer "$BASE_WASM" \
  --allow-wasi \
  --init-func "wizer.initialize" \
  --mapdir "/::$WORK" \
  -- python /warmup.py \
  -o "$OUT_PATH"

echo ">> wrote warm payload: $OUT_PATH"
echo "   the wasm rung will now prefer it automatically (no config change needed)."
