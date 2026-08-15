"""Background throttle + foreground pause (CONCEPT:AU-KG.query.vendor-agnostic-traversal)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

from agent_utilities.core.background_throttle import BackgroundThrottle


def _lease_dir(data_dir: Path) -> Path:
    return data_dir / "runtime" / "foreground-leases"


def test_foreground_flag_reentrant():
    t = BackgroundThrottle()
    assert not t.foreground_active
    t.set_foreground(True)
    t.set_foreground(True)
    assert t.foreground_active
    t.set_foreground(False)
    assert t.foreground_active  # still one holder
    t.set_foreground(False)
    assert not t.foreground_active


def test_foreground_context_manager():
    t = BackgroundThrottle()
    with t.foreground():
        assert t.foreground_active
    assert not t.foreground_active


def test_foreground_lease_coordinates_independent_throttles(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path))
    foreground = BackgroundThrottle(lease_scan_interval=0.0)
    host = BackgroundThrottle(lease_scan_interval=0.0)

    foreground.set_foreground(True)
    foreground.set_foreground(True)
    assert host.foreground_active
    assert host.should_yield_background
    foreground.set_foreground(False)
    assert host.foreground_active
    with host.background_slot(wait_foreground=False) as acquired:
        assert acquired is False
    foreground.set_foreground(False)
    assert not host.foreground_active


def test_foreground_lease_coordinates_a_separate_process(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path))
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    program = """
from agent_utilities.core.background_throttle import BackgroundThrottle
import time
t = BackgroundThrottle(lease_ttl=0.15, lease_heartbeat=0.02, lease_scan_interval=0.0)
t.set_foreground(True)
print('ready', flush=True)
time.sleep(30)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program],
        env=environment,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        host = BackgroundThrottle(lease_ttl=0.15, lease_scan_interval=0.0)
        assert host.foreground_active
        process.terminate()
        process.wait(timeout=3)
        time.sleep(0.2)
        assert not host.foreground_active
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


def test_foreground_lease_expires_after_a_crashed_process(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path))
    writer = BackgroundThrottle(
        lease_ttl=0.12, lease_heartbeat=0.02, lease_scan_interval=0.0
    )
    host = BackgroundThrottle(lease_ttl=0.12, lease_scan_interval=0.0)

    writer.set_foreground(True)
    assert host.foreground_active
    # Deliberately do not release the writer: this models a killed foreground
    # process, whose last heartbeat must not leave the host paused forever.
    assert writer._lease_stop is not None
    writer._lease_stop.set()
    time.sleep(0.16)
    assert not host.foreground_active


def test_rapid_foreground_reentry_cannot_revive_an_old_heartbeat(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path))
    throttle = BackgroundThrottle(
        lease_ttl=0.12, lease_heartbeat=0.02, lease_scan_interval=0.0
    )

    throttle.set_foreground(True)
    first_stop = throttle._lease_stop
    assert first_stop is not None
    throttle.set_foreground(False)
    throttle.set_foreground(True)
    second_stop = throttle._lease_stop
    assert second_stop is not None and second_stop is not first_stop
    assert first_stop.is_set()

    throttle.set_foreground(False)
    time.sleep(0.04)
    assert not list(_lease_dir(tmp_path).glob("*.json"))


def test_foreground_lease_scan_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path))
    throttle = BackgroundThrottle(lease_scan_interval=0.0)
    directory = _lease_dir(tmp_path)
    directory.mkdir(parents=True, mode=0o700)
    for number in range(129):
        lease = directory / f"{number:03}.json"
        lease.write_text("{}")
        lease.chmod(0o600)

    inspected: list[Path] = []

    def invalid(path: Path, now: float) -> bool:
        del now
        inspected.append(path)
        return False

    throttle._lease_is_valid = invalid  # type: ignore[method-assign]
    assert not throttle.foreground_active
    assert len(inspected) == 128


def test_foreground_lease_is_private_and_rejects_symlink_directories(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path))
    throttle = BackgroundThrottle(lease_scan_interval=0.0)
    with throttle.foreground():
        directory = _lease_dir(tmp_path)
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        leases = list(directory.glob("*.json"))
        assert len(leases) == 1
        assert stat.S_IMODE(leases[0].stat().st_mode) == 0o600
        record = json.loads(leases[0].read_text())
        assert set(record) == {"version", "expires_at"}
        assert record["version"] == 1
        assert isinstance(record["expires_at"], float)

        unsafe_record = tmp_path / "unsafe-record.json"
        unsafe_record.write_text(json.dumps(record))
        (directory / "untrusted.json").symlink_to(unsafe_record)

    host = BackgroundThrottle(lease_scan_interval=0.0)
    assert not host.foreground_active
    (directory / "untrusted.json").unlink()

    unsafe_root = tmp_path / "unsafe"
    unsafe_root.mkdir()
    runtime = tmp_path / "runtime"
    os.rmdir(runtime / "foreground-leases")
    (runtime / "foreground-leases").symlink_to(unsafe_root, target_is_directory=True)
    throttle = BackgroundThrottle(lease_scan_interval=0.0)
    with throttle.foreground():
        assert not list(unsafe_root.iterdir())


def test_background_slot_skips_when_foreground_and_no_wait():
    t = BackgroundThrottle()
    t.set_foreground(True)
    with t.background_slot(wait_foreground=False) as ok:
        assert ok is False  # caller should skip this cycle


def test_background_slot_acquires_when_idle():
    t = BackgroundThrottle(max_concurrent=1)
    with t.background_slot() as ok:
        assert ok is True


def test_background_slot_yields_until_foreground_clears():
    t = BackgroundThrottle(max_concurrent=1)
    t.set_foreground(True)
    proceeded = threading.Event()

    def worker():
        with t.background_slot(fg_poll=0.02) as ok:
            if ok:
                proceeded.set()

    th = threading.Thread(target=worker)
    th.start()
    time.sleep(0.1)
    assert not proceeded.is_set()  # blocked by foreground
    t.set_foreground(False)
    th.join(timeout=2)
    assert proceeded.is_set()  # proceeded once foreground cleared


def test_bulk_ingest_flag_reentrant():
    t = BackgroundThrottle()
    assert not t.bulk_ingest_active
    t.set_bulk_ingest(True)
    t.set_bulk_ingest(True)
    assert t.bulk_ingest_active
    t.set_bulk_ingest(False)
    assert t.bulk_ingest_active  # still one holder
    t.set_bulk_ingest(False)
    assert not t.bulk_ingest_active


def test_bulk_ingest_context_manager():
    t = BackgroundThrottle()
    with t.bulk_ingest():
        assert t.bulk_ingest_active
    assert not t.bulk_ingest_active


def test_should_yield_background_covers_foreground_and_ingest():
    t = BackgroundThrottle()
    assert not t.should_yield_background
    with t.foreground():
        assert t.should_yield_background  # interactive
    assert not t.should_yield_background
    with t.bulk_ingest():
        assert t.should_yield_background  # bulk ingest
    assert not t.should_yield_background
    # Both at once, independently reentrant.
    with t.foreground(), t.bulk_ingest():
        assert t.should_yield_background


def test_wait_while_busy_blocks_then_returns_when_cleared():
    t = BackgroundThrottle()
    t.set_bulk_ingest(True)
    returned = threading.Event()
    result = {}

    def worker():
        result["cleared"] = t.wait_while_busy(poll=0.02, max_wait=5.0)
        returned.set()

    th = threading.Thread(target=worker)
    th.start()
    time.sleep(0.1)
    assert not returned.is_set()  # still paused — ingest active
    t.set_bulk_ingest(False)
    th.join(timeout=2)
    assert returned.is_set()
    assert result["cleared"] is True  # returned because gate cleared, not timeout


def test_wait_while_busy_gives_up_after_max_wait():
    # A permanently-busy gate must never starve background work forever.
    t = BackgroundThrottle()
    t.set_foreground(True)
    t0 = time.time()
    cleared = t.wait_while_busy(poll=0.02, max_wait=0.1)
    assert cleared is False
    assert time.time() - t0 >= 0.1
    t.set_foreground(False)


# ── enrichment-lane fairness (CONCEPT:AU-KG.ontology.capability-card-backfill-lane, MUST-FIX) ────────────────────
#
# Regression guards for the dedicated OWL card-enrichment lane. The bug: the
# enrichment_backfill task was wrapped in an OUTER ``background_slot()`` at the
# task-dispatch boundary (engine_tasks ``_execute_claimed_task``) AND the
# per-batch ``_tick_enrichment`` loop acquired the slot AGAIN per batch — a
# double-acquire. Holding one outer permit for a whole 16*64=1024-symbol tick
# (a) capped the lane at ``_BACKGROUND_MAX_CONCURRENT`` no matter how many spare
# workers it could expand into, and (b) starved the other ~17 background/maint
# ticks for the full tick. The fix: enrichment is NOT wrapped at dispatch; the
# per-batch acquire/release in ``_tick_enrichment`` is the only gate, so the
# permit is yielded BETWEEN batches and other background work interleaves.
#
# These tests drive the REAL ``BackgroundThrottle`` the way the fixed code does:
# an enrichment "tick" is modelled as a loop that acquires ``background_slot()``
# per batch and releases it between batches (mirroring engine_tasks line ~1595).


def _enrichment_tick(throttle, *, batches, batch_work, progress, hold_between=0.0):
    """Model ``_tick_enrichment``: acquire a background_slot PER BATCH, do the
    batch's work, then RELEASE before looping. Mirrors the fixed dispatch where
    enrichment is NOT wrapped in an outer per-tick slot."""
    for _ in range(batches):
        with throttle.background_slot(acquire_timeout=5.0) as ok:
            assert ok, "enrichment batch should eventually get a slot"
            progress.append(1)
            if batch_work:
                time.sleep(batch_work)
        if hold_between:
            time.sleep(hold_between)


def test_two_concurrent_enrichment_dispatches_both_progress():
    """Two concurrent enrichment_backfill dispatches must BOTH make progress
    (not serialize to one) even when the shared semaphore is the binding cap.

    Because each tick releases its permit between batches, two enrichment
    workers interleave their batches rather than one running to completion
    while the other is fully starved.
    """
    # cap=1 is the worst case: the ONLY way both can progress is if each
    # releases between batches so the other can grab the freed permit.
    t = BackgroundThrottle(max_concurrent=1)
    progress_a: list[int] = []
    progress_b: list[int] = []

    def worker(progress):
        _enrichment_tick(
            t, batches=6, batch_work=0.01, progress=progress, hold_between=0.005
        )

    th_a = threading.Thread(target=worker, args=(progress_a,))
    th_b = threading.Thread(target=worker, args=(progress_b,))
    th_a.start()
    th_b.start()
    th_a.join(timeout=10)
    th_b.join(timeout=10)
    assert not th_a.is_alive() and not th_b.is_alive()
    # Both ticks completed all their batches — neither was starved to zero.
    assert sum(progress_a) == 6
    assert sum(progress_b) == 6


def test_long_enrichment_tick_does_not_block_other_background_task():
    """A long enrichment tick (many batches) must NOT hold the background
    permit for its whole duration — another background task acquires a slot
    while the enrichment tick is mid-flight (between batches).

    This is the fairness regression the MUST-FIX targets: under the old
    per-tick outer hold (cap=1), the other background task could only run AFTER
    the entire 1024-symbol tick finished.

    This does NOT time a real gap against a background task racing to land
    inside it — under the parallel suite's `-n auto` CPU contention (and,
    worse, host-level swapping) a fixed-duration gap is itself a real
    wall-clock assumption that can lose to OS scheduler jitter regardless of
    how generous the duration is (widening it from ~10ms to ~50ms here still
    left the *tick's total* runtime blowing past a 10s join budget on a
    loaded box — the exact "just raise it further" trap, on a different
    knob). Instead: release the permit after the first batch like the real
    tick does, then PARK the enrichment thread on an event with no fixed
    duration — held open until the test has *observed* the other task
    acquire the freed permit. The proof (the permit is released between
    batches, not held for the whole tick) is unconditional; only the upper
    safety bound is a wall-clock timeout, not the interleaving itself.
    """
    t = BackgroundThrottle(max_concurrent=1)
    other_acquired = threading.Event()
    enrichment_done = threading.Event()
    resume_after_first_batch = threading.Event()
    enrichment_progress: list[int] = []

    def long_enrichment():
        # Batch 1: acquire + release exactly like a real tick batch, so the
        # permit is genuinely free afterward (not synthetically withheld).
        with t.background_slot(acquire_timeout=15.0) as ok:
            assert ok, "enrichment's first batch should acquire a free slot"
            enrichment_progress.append(1)
        # Park here — between batches, exactly where the real tick yields —
        # until the test has proven the other task got in. No sleep, no
        # guessed duration: this either unblocks the instant `other_acquired`
        # is observed, or the 15s safety bound fires (a real bug, not a slow
        # scheduler, at that point).
        resume_after_first_batch.wait(timeout=15.0)
        # The remaining batches, back to the normal per-batch acquire/release
        # shape — proves the tick as a whole still completes ALL its work.
        for _ in range(19):
            with t.background_slot(acquire_timeout=15.0) as ok:
                assert ok, "enrichment batch should eventually get a slot"
                enrichment_progress.append(1)
        enrichment_done.set()

    def other_background():
        # A different background task wanting a single slot.
        with t.background_slot(acquire_timeout=15.0) as ok:
            if ok:
                other_acquired.set()

    th_enrich = threading.Thread(target=long_enrichment)
    th_enrich.start()
    # Wait for the enrichment tick to have genuinely landed its first batch
    # (not a guessed sleep) before starting the other task.
    deadline = time.monotonic() + 15.0
    while not enrichment_progress and time.monotonic() < deadline:
        time.sleep(0.005)
    assert enrichment_progress, "enrichment tick did not start in time"

    th_other = threading.Thread(target=other_background)
    th_other.start()
    # The other background task acquires its slot while the enrichment tick
    # is deliberately parked between batches — proving the permit is yielded,
    # not held for the tick's whole duration. Bounded only by a generous
    # safety timeout, never by racing a fixed sleep window.
    assert other_acquired.wait(timeout=15.0), (
        "other background task was starved by the enrichment tick "
        "(permit not released between batches)"
    )
    assert not enrichment_done.is_set(), (
        "enrichment tick already finished — test did not exercise mid-tick interleaving"
    )
    resume_after_first_batch.set()  # let the enrichment tick continue

    th_enrich.join(timeout=20)
    th_other.join(timeout=20)
    assert not th_enrich.is_alive() and not th_other.is_alive()
    assert sum(enrichment_progress) == 20
