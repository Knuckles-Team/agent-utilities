"""Real-engine tests for the reactive engine-subscription primitive (CONCEPT:KG-2.251).

USER DIRECTIVE (CONCEPT:KG-2.238): the reactive poll→push path is validated against
the ACTUAL ephemeral epistemic-graph engine the session fixture deploys — NOT
SQLite, NOT a mock for the engine. Each test requests the conftest ``engine_graph``
fixture (a FRESH per-test tenant on the one running engine) and asserts that a
WRITE to the engine is DELIVERED to the subscription handler via the engine's native
CDC/watch feed — i.e. the consumer reacts to the change-event, not a poll interval.

Covers:
* the ``EngineSubscription`` primitive (catch-up + watch deliver a real write);
* the world-model gate fires on a ``WorldModelTransition`` write (not a full re-scan);
* the autoscaler signal subscription fires on a control-plane ``:Task`` change.
"""

from __future__ import annotations

import pytest

from agent_utilities.graph.reactive import (
    EngineSubscription,
    resolve_streaming,
    subscribe,
)

pytestmark = pytest.mark.engine


# ── the primitive ───────────────────────────────────────────────────────


def test_resolve_streaming_from_compute(engine_graph) -> None:
    """``resolve_streaming`` finds the engine streaming client + graph name on a
    real GraphComputeEngine tenant."""
    resolved = resolve_streaming(engine_graph)
    assert resolved is not None
    streaming, graph_name = resolved
    assert graph_name == engine_graph.graph_name
    assert hasattr(streaming, "cdc_read")


def test_subscription_delivers_write_to_handler(engine_graph) -> None:
    """A write to the engine is PUSHED to the handler through the CDC/watch feed.

    This is the core poll→push assertion: no full graph re-scan — the subscription
    delivers exactly the changed node, identified by its CDC label."""
    seen: list[dict] = []
    sub = subscribe(engine_graph, "Widget", seen.append)
    assert sub.available

    # Nothing yet → a poll on an empty feed delivers nothing.
    assert sub.poll(block_ms=0) == 0
    assert seen == []

    # Write a labeled node; the engine commits a CDC change for it.
    engine_graph.add_node("w1", {"type": "Widget", "color": "blue"})

    delivered = sub.poll(block_ms=0)
    assert delivered == 1
    assert len(seen) == 1
    assert seen[0]["node_id"] == "w1"
    assert seen[0]["label"] == "Widget"

    # Cursor advanced — a second poll with no new writes delivers nothing.
    assert sub.poll(block_ms=0) == 0


def test_subscription_label_filter_isolates(engine_graph) -> None:
    """The subscription only delivers changes for its own label; an unrelated
    write does not wake it."""
    seen: list[dict] = []
    sub = subscribe(engine_graph, "Wanted", seen.append)

    engine_graph.add_node("other1", {"type": "Unwanted"})
    engine_graph.add_node("want1", {"type": "Wanted"})

    sub.poll(block_ms=0)
    assert [e["node_id"] for e in seen] == ["want1"]


def test_cold_start_catch_up_is_bounded(engine_graph) -> None:
    """A subscription created AFTER writes already exist catches up over the CDC
    tail on first poll (cold start), then advances — it does not miss the backlog."""
    engine_graph.add_node("pre1", {"type": "Backlog"})
    engine_graph.add_node("pre2", {"type": "Backlog"})

    seen: list[dict] = []
    sub = subscribe(engine_graph, "Backlog", seen.append)
    # First poll runs catch_up() then a watch — both pre-existing nodes delivered.
    sub.poll(block_ms=0)
    assert {e["node_id"] for e in seen} == {"pre1", "pre2"}


def test_unavailable_source_is_noop() -> None:
    """With no engine streaming surface, the subscription is a permanent no-op so a
    caller can wire it unconditionally and keep its periodic reconcile."""
    sub = EngineSubscription(object(), label="X", handler=lambda e: None)
    assert sub.available is False
    assert sub.poll(block_ms=0) == 0
    assert sub.catch_up() == 0


# ── world-model reactive gate ────────────────────────────────────────────


def test_world_model_subscription_fires_on_transition(engine_graph) -> None:
    """The world-model reactive gate (KG-2.251) registers ``pending`` on a real
    ``WorldModelTransition`` write — the daemon tick re-specializes on THIS change
    instead of re-querying the whole transition history."""
    from agent_utilities.harness.world_model_task import (
        WORLD_MODEL_TRANSITION_LABEL,
        world_model_subscription,
    )

    # The builder resolves the engine's content compute; feed it the per-test
    # tenant directly so the subscription watches exactly this graph.
    sub = world_model_subscription(engine_graph)
    assert sub.available
    assert sub.pending_state["pending"] == 0

    # No transitions yet → the gate stays closed (would skip the expensive cycle).
    sub.poll(block_ms=0)
    assert sub.pending_state["pending"] == 0

    # Persist a transition exactly as WorldModel.record_observation does.
    engine_graph.add_node(
        "wm_transition:abc123",
        {
            "type": WORLD_MODEL_TRANSITION_LABEL,
            "state": "s0",
            "action": "a0",
            "next_state": "s1",
        },
    )

    sub.poll(block_ms=0)
    assert sub.pending_state["pending"] == 1, "world model must react to the write"


# ── autoscaler reactive signal ───────────────────────────────────────────


def test_autoscale_subscription_fires_on_task_change(engine_graph) -> None:
    """The autoscaler reactive signal (KG-2.251) registers ``pending`` on a real
    control-plane ``:Task`` change — the autoscaler evaluates on the queue-depth
    change-event, not only on its poll interval."""
    from agent_utilities.orchestration.fleet_autoscaler import (
        TASK_LABEL,
        fleet_autoscale_subscription,
    )

    # ``fleet_autoscale_subscription`` prefers engine._control, then graph_compute.
    # The engine_graph compute has neither attribute, so it falls through to using
    # the compute object itself — watch ``:Task`` on this tenant.
    sub = fleet_autoscale_subscription(engine_graph)
    assert sub.available
    assert sub.pending_state["pending"] == 0

    # A queue-depth-moving change: a new :Task enqueued.
    engine_graph.add_node("task:1", {"type": TASK_LABEL, "status": "pending"})

    sub.poll(block_ms=0)
    assert sub.pending_state["pending"] == 1, "autoscaler must react to a :Task change"

    # A subsequent :Task mutation (claim) is another change-event.
    engine_graph.add_node("task:2", {"type": TASK_LABEL, "status": "pending"})
    sub.poll(block_ms=0)
    assert sub.pending_state["pending"] == 2
