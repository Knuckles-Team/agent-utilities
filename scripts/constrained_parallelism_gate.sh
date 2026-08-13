#!/usr/bin/env bash
# GOC-70 constrained-parallelism verification gate (agent-utilities).
#
# WHY: the sibling epistemic-graph repo shipped 2.25.0 with all gates verified
# green on a 64-core dev host, and CI (a 2-core runner) failed anyway -- a
# test asserted a scheduling-dependent property (every one of 400 concurrent
# writes landed through one specific counter path) that only holds with real
# many-core scheduler overlap. CI runners are the REFERENCE environment, not
# a degraded one: our build hosts are the outlier, and because we routinely
# verify on them, they systematically hide this defect class. See
# plans/graph-os-completion-program/GOC-59-67-EXPANSION-TRACKS.md, GOC-70.
#
# An au-side audit under this same edict found (and fixed, test-only, no
# production code changed) five sites where a fixed `sleep()` + exact-count/
# tight-wall-clock assertion depended on enough scheduler overlap happening
# within an arbitrary window -- the same shape, in Python/asyncio/threading
# form. This gate re-runs exactly those files (plus any future ones added to
# CONCURRENCY_SENSITIVE_TESTS below) under restricted CPU affinity so the
# class cannot silently return.
#
# SCOPE: intentionally NOT the full ~17,846-test suite (36-47 minutes even
# unconstrained) -- that would price this gate out of "routine". This is a
# curated, growing allowlist of files with timing/scheduling-sensitive
# assertions, not a claim of exhaustive coverage; CI's full run remains the
# completeness backstop. Add a file here whenever a new concurrency- or
# timing-sensitive test is introduced (spawns threads/tasks and asserts a
# count, or uses `sleep()`/`time.monotonic()` as its synchronization).
#
# WHY THE ENV SYNC ISN'T ALSO CONSTRAINED: like eg's build-vs-run split, `uv
# sync`/dependency resolution is not what this defect class is about --
# restricting it to 2 cores would only slow first-run setup for no additional
# signal. `scripts/uv_workspace.py run --all-extras` handles sync-before-exec
# itself; this script only restricts the pytest CHILD process's CPU affinity.
#
# `-n auto` (pytest-xdist, perf/au-test-runtime, in flight on a sibling
# branch) is NOT assumed live here -- this repo's checked-out `main` runs
# pytest single-process (confirmed: no `-n auto`/`--dist` in pytest.ini,
# pyproject.toml, .pre-commit-config.yaml, or the GitHub workflows). When
# xdist does land, re-verify this gate under it too: a 2-core runner has far
# fewer xdist workers than a dev host, which is exactly the kind of
# worker-count assumption this edict exists to catch (see
# `tests/unit/knowledge_graph/test_worker_scheduler.py::
# test_resolve_engine_shard_writers_uses_engine_K` for the one place this
# repo currently derives sizing from `os.cpu_count()` in a TEST -- already
# correctly deterministic there, monkeypatched rather than reading the real
# host).
#
# USAGE:
#   scripts/constrained_parallelism_gate.sh
#   EG_CONSTRAINED_CORES=0,1,2,3 scripts/constrained_parallelism_gate.sh
#
# Coordinate with `rm_gates(action=run, stage=heavy)` (feat/rm-gates,
# in-flight sibling lane) -- this belongs in that tier, not as a second,
# parallel enforcement mechanism.
set -uo pipefail
cd "$(dirname "$0")/.."

CORES="${EG_CONSTRAINED_CORES:-0,1}"

CONCURRENCY_SENSITIVE_TESTS=(
  tests/unit/core/test_resource_priority.py
  tests/unit/messaging/test_messaging_coalescer.py
  tests/unit/knowledge_graph/test_pool_adoption.py
  tests/unit/test_non_blocking_execution.py
  tests/unit/knowledge_graph/test_kafka_ingest_scaleout.py
  tests/unit/knowledge_graph/test_fanout_backend.py
  tests/unit/knowledge_graph/test_ladybug_singleton.py
  tests/unit/knowledge_graph/core/test_engine_resolver.py
  tests/unit/mcp/test_multiplexer_mount_singleflight.py
  tests/unit/core/test_provider_materialization.py
  tests/integration/core/test_parallel_engine_advanced.py
)

if ! command -v taskset >/dev/null 2>&1; then
  cat >&2 <<EOF
FAIL: taskset is not installed -- constrained-parallelism verification cannot run.

Install it (util-linux; Debian/Ubuntu: 'apt-get install -y util-linux') or run
this on a Linux host, then re-run: scripts/constrained_parallelism_gate.sh

This is a hard failure, not a skip -- we never report a pass we didn't verify.
EOF
  exit 2
fi

# Only test files that actually exist (keeps the list forward-safe if one is
# ever renamed/removed without this script also needing to change in lockstep).
existing=()
for f in "${CONCURRENCY_SENSITIVE_TESTS[@]}"; do
  if [ -f "$f" ]; then
    existing+=("$f")
  else
    echo "WARN: $f no longer exists -- remove it from CONCURRENCY_SENSITIVE_TESTS" >&2
  fi
done
if [ "${#existing[@]}" -eq 0 ]; then
  echo "FAIL: none of the CONCURRENCY_SENSITIVE_TESTS files exist -- the list is stale." >&2
  exit 2
fi

n_cores=$(($(echo "$CORES" | tr ',' '\n' | wc -l)))
echo "== GOC-70 constrained-parallelism gate (agent-utilities) =="
echo "== environment: CPU affinity restricted to cores [$CORES] ($n_cores logical cores) =="
echo "== running ${#existing[@]} concurrency-sensitive test file(s) =="

if taskset -c "$CORES" python3 scripts/uv_workspace.py run --all-extras pytest "${existing[@]}" -q --tb=short --timeout=120; then
  echo "== PASS: concurrency-sensitive tests green under $n_cores-core CPU affinity (cores $CORES) =="
else
  rc=$?
  cat >&2 <<EOF

FAIL: one or more concurrency-sensitive tests failed under $n_cores-core CPU
affinity. Per GOC-70: a test that requires a large machine is a DEFECTIVE
TEST, not a machine requirement. Do not mark it xfail/skip, raise a timeout
blindly, or delete the assertion. Instead: read the failure and check
whether it asserts a timing/scheduling-dependent property (assert what's
true regardless of scheduling instead), needs deterministic contention
construction (Event/Barrier/Lock, not sleep-and-hope), or has a margin tuned
for a fast host (widen it, with the ratio documented). See
plans/graph-os-completion-program/GOC-59-67-EXPANSION-TRACKS.md, GOC-70.
EOF
  exit "$rc"
fi
