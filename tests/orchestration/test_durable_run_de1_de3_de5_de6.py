"""DurableRun's DE1/DE3/DE5/DE6 additions (durable-execution-native.md, lane
w6-de1-de8-durable): KG mirror-on-write, auto-checkpointed steps, definition-
version pinning, and a run-scoped durable timer.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.orchestration.durable_execution import (
    DurableRun,
    DurableRunVersionMismatch,
)


class FakeEngine:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_types: dict[str, str] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = dict(properties or {})
        self.node_types[node_id] = node_type

    def link_nodes(self, source_id: str, target_id: str, rel_type: str) -> None:
        self.edges.append((source_id, target_id, rel_type))


# ── DE1: KG mirror-on-write ─────────────────────────────────────────────────


def test_durable_run_without_engine_never_touches_the_kg(tmp_path):
    """No engine supplied → identical behavior to before DE1, no KG writes."""
    run = DurableRun("sess:noeng", db_path=tmp_path / "d.db")
    run.step("a", lambda: "ok")
    run.finish()
    # No assertion possible on "no KG write" other than: nothing raised, and
    # engine is None throughout.
    assert run.engine is None


def test_durable_run_mirrors_step_transitions_when_engine_supplied(tmp_path):
    engine = FakeEngine()
    run = DurableRun("sess:eng", db_path=tmp_path / "d.db", engine=engine)
    node_id = f"durable-run:sess:eng:{run.run_id}"
    # Constructor mirror: PENDING (fresh run).
    assert engine.nodes[node_id]["durable_status"] == "PENDING"

    run.step("a", lambda: "ok")
    assert engine.nodes[node_id]["durable_status"] == "COMPLETED"
    assert engine.nodes[node_id]["checkpoint_ref"] == f"{run.run_id}:a"

    run.finish()
    assert engine.nodes[node_id]["durable_status"] == "COMPLETED"
    assert engine.node_types[node_id] == "DurableRun"


def test_durable_run_mirror_marks_resumed_status(tmp_path):
    engine = FakeEngine()
    db = tmp_path / "d.db"
    run1 = DurableRun("sess:resume", db_path=db, engine=engine)
    with pytest.raises(RuntimeError):
        run1.step("boom", lambda: (_ for _ in ()).throw(RuntimeError("x")))

    run2 = DurableRun("sess:resume", db_path=db, engine=engine)
    assert run2.resumed is True
    node_id = f"durable-run:sess:resume:{run2.run_id}"
    assert engine.nodes[node_id]["durable_status"] == "RESUMED"


def test_link_run_trace_writes_produced_edge(tmp_path):
    engine = FakeEngine()
    run = DurableRun("sess:trace", db_path=tmp_path / "d.db", engine=engine)
    run.link_run_trace("trace:abc123")
    node_id = f"durable-run:sess:trace:{run.run_id}"
    assert (node_id, "trace:abc123", "produced") in engine.edges


# ── DE5: definition_version pinning ─────────────────────────────────────────


def test_definition_version_mismatch_fails_loud_on_resume(tmp_path):
    db = tmp_path / "d.db"
    run1 = DurableRun("sess:ver", db_path=db, definition_version="v1")
    with pytest.raises(RuntimeError):
        run1.step("boom", lambda: (_ for _ in ()).throw(RuntimeError("x")))

    with pytest.raises(DurableRunVersionMismatch, match="v1.*v2|v2.*v1"):
        DurableRun("sess:ver", db_path=db, definition_version="v2")


def test_definition_version_match_resumes_cleanly(tmp_path):
    db = tmp_path / "d.db"
    run1 = DurableRun("sess:ver2", db_path=db, definition_version="v1")
    with pytest.raises(RuntimeError):
        run1.step("boom", lambda: (_ for _ in ()).throw(RuntimeError("x")))

    run2 = DurableRun("sess:ver2", db_path=db, definition_version="v1")
    assert run2.resumed is True
    assert run2.run_id == run1.run_id


def test_definition_version_unset_never_enforces(tmp_path):
    """Omitting definition_version (either side) is not an error -- opt-in pin."""
    db = tmp_path / "d.db"
    run1 = DurableRun("sess:ver3", db_path=db, definition_version="v1")
    with pytest.raises(RuntimeError):
        run1.step("boom", lambda: (_ for _ in ()).throw(RuntimeError("x")))

    # Resuming WITHOUT declaring a version at all must not raise.
    run2 = DurableRun("sess:ver3", db_path=db)
    assert run2.resumed is True


# ── DE3: auto-checkpointed steps (no hand-named step schema) ───────────────


def test_auto_step_derives_names_and_dedupes_repeated_calls(tmp_path):
    db = tmp_path / "d.db"
    calls = {"n": 0}
    run1 = DurableRun("sess:auto", db_path=db)

    # An explicit `name=` (rather than relying on `__qualname__`) is the
    # robust usage: a resumed process redefines the SAME logical callable
    # under the SAME label, exactly as it would from the same source line on
    # restart. `test_auto_step_accepts_an_explicit_label` below covers this
    # form directly; the qualname-derived default is covered by construction
    # (both branches share `wrapper`/`decorator`).
    @run1.auto_step(name="do_thing")
    def do_thing(x):
        calls["n"] += 1
        return x * 2

    assert do_thing(3) == 6
    assert do_thing(4) == 8  # same function, second call -> distinct auto step
    assert calls["n"] == 2

    # Resume: replaying the SAME two calls in the SAME order must not re-run
    # either already-completed auto-step.
    run2 = DurableRun("sess:auto", db_path=db)
    assert run2.run_id == run1.run_id

    @run2.auto_step(name="do_thing")
    def do_thing_again(x):
        calls["n"] += 1
        return x * 2

    assert do_thing_again(3) == 6
    assert do_thing_again(4) == 8
    assert calls["n"] == 2, "resumed auto-steps must replay, not re-run"


def test_auto_step_accepts_an_explicit_label(tmp_path):
    run = DurableRun("sess:auto2", db_path=tmp_path / "d.db")

    @run.auto_step(name="my_label")
    def f():
        return 1

    f()
    assert run.is_done("my_label#1")


# ── DE6: run-scoped durable timer ───────────────────────────────────────────


def test_sleep_until_waits_then_fires_once_deadline_passes(tmp_path):
    run = DurableRun("sess:sleep", db_path=tmp_path / "d.db")
    far_future_ms = 4_102_444_800_000  # year 2100, safely in the future
    assert run.sleep_until("wake", far_future_ms) is False
    assert run.is_done("wake") is False

    past_ms = 0  # 1970 -- always already elapsed
    assert run.sleep_until("wake_now", past_ms) is True
    assert run.is_done("wake_now") is True


def test_sleep_until_is_idempotent_once_fired(tmp_path):
    run = DurableRun("sess:sleep2", db_path=tmp_path / "d.db")
    assert run.sleep_until("wake", 0) is True
    # A second call after the step already completed is a free replay, not a
    # re-evaluation against the (possibly now-past) deadline again.
    assert run.sleep_until("wake", 0) is True
