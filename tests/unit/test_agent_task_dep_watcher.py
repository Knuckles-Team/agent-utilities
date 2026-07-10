"""``AgentTaskDepWatcher`` — CDC-first :AgentTask dependency firing, poll fallback (D13).

CONCEPT:AU-OS.state.cognitive-scheduler-preemption — Graph-Native Agent-OS Objects (C3/Phase 3b)

Covers: no engine change-feed reachable ⇒ every tick sweeps (Phase 3a
behavior, unchanged); a reachable change-feed with nothing new ⇒ zero Cypher
work; a reachable change-feed with a delivered change ⇒ the sweep runs once
and the dirty flag resets; a feed hiccup mid-poll ⇒ falls back to the sweep
rather than raising.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent_utilities.orchestration.fleet_reconciler import AgentTaskDepWatcher


class _FakeSubscription:
    """A minimal stand-in for ``EngineSubscription`` the watcher drives."""

    def __init__(self, *, available: bool, deliver_on_poll: bool = False):
        self.available = available
        self._deliver_on_poll = deliver_on_poll
        self.handler = None
        self.poll_calls = 0

    def poll(self, *, block_ms: int = 0) -> int:
        self.poll_calls += 1
        if self._deliver_on_poll and self.handler is not None:
            self.handler({"kind": "update", "label": "AgentTask", "node_id": "t1"})
            return 1
        return 0


def _watcher(subscription) -> AgentTaskDepWatcher:
    """Build a watcher bypassing the real ``subscribe()`` resolution.

    Mirrors ``_build_subscription``'s real wiring: the handler is bound to
    the subscription at "construction" time (never inside ``fire()``), so a
    subscription double must have its handler set here too.
    """
    watcher = AgentTaskDepWatcher.__new__(AgentTaskDepWatcher)
    watcher.engine = object()
    watcher._dirty = False
    watcher._subscription = subscription
    if subscription is not None:
        subscription.handler = watcher._on_change
    return watcher


def test_no_engine_changefeed_always_sweeps(monkeypatch):
    sweep_calls: list[object] = []
    monkeypatch.setattr(
        "agent_utilities.orchestration.fleet_reconciler.fire_ready_agent_tasks",
        lambda engine, **kw: sweep_calls.append(engine) or ["t1"],
    )
    watcher = _watcher(None)  # no subscription resolvable at all
    assert watcher.fire() == ["t1"]
    assert len(sweep_calls) == 1

    # An unavailable (but constructed) subscription behaves identically.
    watcher2 = _watcher(_FakeSubscription(available=False))
    assert watcher2.fire() == ["t1"]
    assert len(sweep_calls) == 2


def test_available_changefeed_with_nothing_new_skips_the_sweep(monkeypatch):
    sweep_calls: list[object] = []
    monkeypatch.setattr(
        "agent_utilities.orchestration.fleet_reconciler.fire_ready_agent_tasks",
        lambda engine, **kw: sweep_calls.append(engine) or ["should-not-see-this"],
    )
    sub = _FakeSubscription(available=True, deliver_on_poll=False)
    watcher = _watcher(sub)
    result = watcher.fire()
    assert result == []  # zero Cypher work — nothing changed
    assert sweep_calls == []
    assert sub.poll_calls == 1


def test_available_changefeed_with_a_delivered_change_sweeps_once(monkeypatch):
    sweep_calls: list[object] = []
    monkeypatch.setattr(
        "agent_utilities.orchestration.fleet_reconciler.fire_ready_agent_tasks",
        lambda engine, **kw: sweep_calls.append(engine) or ["t1"],
    )
    sub = _FakeSubscription(available=True, deliver_on_poll=True)
    watcher = _watcher(sub)
    result = watcher.fire()
    assert result == ["t1"]
    assert len(sweep_calls) == 1

    # The dirty flag resets — a subsequent tick with no new change is a no-op.
    sub._deliver_on_poll = False
    result2 = watcher.fire()
    assert result2 == []
    assert len(sweep_calls) == 1  # unchanged


def test_poll_failure_falls_back_to_the_sweep(monkeypatch):
    sweep_calls: list[object] = []
    monkeypatch.setattr(
        "agent_utilities.orchestration.fleet_reconciler.fire_ready_agent_tasks",
        lambda engine, **kw: sweep_calls.append(engine) or ["t1"],
    )

    class _BoomSubscription:
        available = True
        handler = None

        def poll(self, *, block_ms=0):
            raise RuntimeError("feed hiccup")

    watcher = _watcher(_BoomSubscription())
    result = watcher.fire()
    assert result == ["t1"]
    assert len(sweep_calls) == 1


def test_build_subscription_degrades_cleanly_when_engine_subscription_unimportable(
    monkeypatch,
):
    import agent_utilities.orchestration.fleet_reconciler as fr

    def _boom(*a, **k):
        raise ImportError("no such module")

    # Patch the module attribute the watcher imports lazily inside _build_subscription.
    monkeypatch.setattr(
        "agent_utilities.graph.reactive.engine_subscription.subscribe", _boom
    )
    watcher = fr.AgentTaskDepWatcher(SimpleNamespace())
    assert watcher._subscription is None
