"""DE7 — replay-determinism property test for :class:`DurableRun`.

Sibling of the Rust-side proofs for `eg-jobs`/`eg-statechart`
(`epistemic-graph` `crates/eg-jobs/tests/replay_determinism.rs` and
`crates/eg-statechart/tests/replay_determinism.rs`, lane
`w6-de7-replay-determinism`): the same input script, replayed across a
SIMULATED CRASH, must reach byte-identical results as an uninterrupted run,
and no already-completed step may be re-executed.

Distinct from the existing ``test_durable_run_resume.py`` (a single
hand-written script proving resume-after-exception): this is a
*property/fuzz* test — Hypothesis generates the script length and the exact
crash point — and it forces a GENUINE storage-layer crash rather than a
same-process object drop (see ``_evict_pooled_connection`` below).

★ GENUINE FINDING: :class:`SQLiteCheckpointStore` pools ONE ``sqlite3.Connection``
per ``db_path``, cached at class scope (``SQLiteCheckpointStore._conns``,
shared across every ``DurableExecutionManager``/``DurableRun`` in the process).
Simply dropping a ``DurableRun`` object and constructing a new one against the
same ``db_path`` does NOT sever storage the way a real process crash does —
the second instance transparently reuses the first one's still-open
connection. A naive "drop and reconstruct" test would therefore never
exercise a real crash-and-reopen at all; it would just call the same live
handle twice. This test explicitly evicts the pooled connection between the
two halves of the crash-interrupted run to make the simulation faithful.
"""

from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_utilities.orchestration.durable_execution import (
    DurableRun,
    SQLiteCheckpointStore,
)


def _evict_pooled_connection(db_path) -> None:
    """Force the NEXT :class:`SQLiteCheckpointStore` for ``db_path`` to open a
    genuinely fresh connection — mimicking an OS-level process kill (file
    descriptor closed, no cached handle) instead of a same-process object
    drop, which the class-level connection pool would otherwise paper over.
    """
    key = str(db_path)
    with SQLiteCheckpointStore._conns_lock:
        entry = SQLiteCheckpointStore._conns.pop(key, None)
    if entry is not None:
        entry[0].close()


def _step_action(counters: dict[int, int], i: int):
    def action():
        counters[i] = counters.get(i, 0) + 1
        # A pure function of the step index only — no wall clock, no random,
        # no shared mutable state beyond the exactly-once counter above.
        return {"i": i, "value": i * 3}

    return action


def _run_script(
    run: DurableRun, counters: dict[int, int], steps: range
) -> dict[str, object]:
    results: dict[str, object] = {}
    for i in steps:
        name = f"step{i}"
        results[name] = run.step(name, _step_action(counters, i))
    return results


@pytest.mark.slow  # property/fuzz suite: 25 Hypothesis examples x 2 real SQLite
# stores each — excluded from the fast pre-commit gate (`-m "not slow"`), runs
# in the full/nightly suite per this repo's existing `slow` marker convention
# (pytest.ini's default addopts filter only `live`, not `slow`, so a plain
# `pytest tests/` — e.g. the merge queue's declared, if not queue-enforced,
# `full-suite` gate — still exercises it).
@settings(
    max_examples=25,
    deadline=None,
    database=None,  # no persistent `.hypothesis/` example DB in the repo tree
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    n_steps=st.integers(min_value=1, max_value=6),
    crash_after=st.integers(min_value=0, max_value=6),
)
def test_crash_replay_reaches_identical_state(tmp_path_factory, n_steps, crash_after):
    crash_after = min(crash_after, n_steps)

    # Uninterrupted baseline.
    clean_db = tmp_path_factory.mktemp("clean") / "durable.db"
    clean_counters: dict[int, int] = {}
    clean_run = DurableRun("de7-clean", db_path=clean_db)
    clean_results = _run_script(clean_run, clean_counters, range(n_steps))
    clean_run.finish()

    # Crash-interrupted: identical script, a hard crash (pooled connection
    # evicted, no `.finish()` call) after `crash_after` steps, then resume.
    crash_db = tmp_path_factory.mktemp("crash") / "durable.db"
    crash_counters: dict[int, int] = {}
    run_a = DurableRun("de7-crash", db_path=crash_db)
    run_id_before_crash = run_a.run_id
    results = _run_script(run_a, crash_counters, range(crash_after))
    del run_a  # drop every in-process reference before the "crash"
    _evict_pooled_connection(crash_db)

    run_b = DurableRun("de7-crash", db_path=crash_db)
    # The crash must not mint a fresh run identity — a resumed run continues
    # under the SAME run_id, or every already-completed step's idempotency key
    # (`f"{run_id}:{name}"`) silently changes namespace and gets re-executed.
    assert run_b.run_id == run_id_before_crash
    assert run_b.resumed is True
    results.update(_run_script(run_b, crash_counters, range(crash_after, n_steps)))
    run_b.finish()

    assert results == clean_results
    # Exactly-once: no step's underlying action ran more than once across the
    # crash + resume, regardless of where the crash landed.
    assert all(count == 1 for count in crash_counters.values())
    assert len(crash_counters) == n_steps
