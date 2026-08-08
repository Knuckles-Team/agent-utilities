"""The unified durable-execution tool surface (DE2, CONCEPT:AU-ORCH.execution.durable-routing-trait).

Covers :mod:`agent_utilities.orchestration.durable_tool_surface`: the two
backed CallShapes (`run`, `sleep`, both real -- backed by `DurableRun`), the
KG status query, and that the three NOT-backed CallShapes fail loud with a
clear, named reason rather than a silent no-op or the wrong backend.
"""

from __future__ import annotations

import pytest

from agent_utilities.orchestration.durable_tool_surface import (
    DurableCallNotBacked,
    durable_call,
    durable_checkpoint,
    durable_run,
    durable_sleep,
    durable_state_get,
    durable_state_set,
    durable_status,
)


def test_durable_checkpoint_is_idempotent_across_stateless_calls(tmp_path, monkeypatch):
    """Each ``durable_checkpoint`` call reconstructs a fresh ``DurableRun``
    (the MCP/stateless-tool-call boundary), so ``auto_step``'s in-memory
    per-run sequence counter resets every call -- repeated calls with the SAME
    ``step_label`` on the SAME (resumed) run therefore resolve to the SAME
    auto-sequenced step and REPLAY its first-recorded result, exactly like a
    hand-named :meth:`DurableRun.step` idempotency key would. This is the
    correct, documented behavior at this boundary (see the function's own
    docstring) -- auto_step's "no hand-picked name needed" value is realized
    within ONE long-lived ``DurableRun`` driving a loop in-process, not across
    independently reconstructed stateless calls with a repeated label.
    """
    monkeypatch.setattr(
        "agent_utilities.orchestration.durable_execution._default_db_path",
        lambda: tmp_path / "durable_execution.db",
    )
    first = durable_checkpoint(None, "sess:ckpt", "event", {"n": 1})
    assert first["result"] == {"n": 1}
    second = durable_checkpoint(None, "sess:ckpt", "event", {"n": 2})
    assert second["result"] == {"n": 1}, (
        "replay, not re-record, across fresh DurableRun instances"
    )
    assert first["run_id"] == second["run_id"]

    # A DIFFERENT step_label on the same run is its own, independent step.
    third = durable_checkpoint(None, "sess:ckpt", "other_event", {"n": 3})
    assert third["result"] == {"n": 3}
    assert third["run_id"] == first["run_id"]


def test_durable_run_starts_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.orchestration.durable_execution._default_db_path",
        lambda: tmp_path / "durable_execution.db",
    )
    handle1 = durable_run(None, "sess:tool")
    assert handle1["resumed"] is False
    assert handle1["run_id"]

    handle2 = durable_run(None, "sess:tool")
    assert handle2["resumed"] is True
    assert handle2["run_id"] == handle1["run_id"]


def test_durable_sleep_waits_then_fires(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.orchestration.durable_execution._default_db_path",
        lambda: tmp_path / "durable_execution.db",
    )
    waiting = durable_sleep(None, "sess:sleep-tool", "wake", 4_102_444_800_000)
    assert waiting["fired"] is False

    fired = durable_sleep(None, "sess:sleep-tool2", "wake", 0)
    assert fired["fired"] is True


@pytest.mark.parametrize(
    "fn, args",
    [
        (durable_state_get, ("sess", "key")),
        (durable_state_set, ("sess", "key", "value")),
        (durable_call, ("sess", "name")),
    ],
)
def test_not_backed_call_shapes_fail_loud_not_silent(fn, args):
    with pytest.raises(DurableCallNotBacked, match="wire-protocol"):
        fn(None, *args)


def test_durable_status_delegates_to_the_kg_list_helper():
    class FakeEngine:
        def query_cypher(self, query, params=None):
            return []

    assert durable_status(FakeEngine()) == []
