"""Embedder circuit-breaker for the KG embedding-backfill loop (CONCEPT:EG-KG.storage.nonblocking-checkpoint).

When the embedding endpoint is down (e.g. the GPU host power-cycles → vLLM 502s),
the backfill tick must stop calling it so it doesn't retry-storm a dead endpoint
and peg the daemon (which makes the whole KG surface time out). These tests pin
the breaker state machine directly — no backend / embedder needed.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.engine_tasks import (
    _EMBED_CB_COOLDOWN,
    _EMBED_CB_THRESHOLD,
    TaskManagerMixin,
)


class _Obj:
    """Bare object the mixin's getattr/setattr-only helpers operate on."""


def test_breaker_opens_after_threshold_consecutive_failures():
    o = _Obj()
    now = 1000.0
    # Below threshold → stays CLOSED (probing still allowed).
    for _ in range(_EMBED_CB_THRESHOLD - 1):
        TaskManagerMixin._embed_circuit_record(o, False, now)
        assert TaskManagerMixin._embed_circuit_open(o, now) is False
    # The threshold-th consecutive failure OPENS the breaker for the cooldown.
    TaskManagerMixin._embed_circuit_record(o, False, now)
    assert TaskManagerMixin._embed_circuit_open(o, now) is True
    assert TaskManagerMixin._embed_circuit_open(o, now + _EMBED_CB_COOLDOWN - 1) is True


def test_breaker_allows_probe_after_cooldown():
    o = _Obj()
    now = 5000.0
    for _ in range(_EMBED_CB_THRESHOLD):
        TaskManagerMixin._embed_circuit_record(o, False, now)
    assert TaskManagerMixin._embed_circuit_open(o, now) is True
    # Once the cooldown elapses the breaker no longer skips → one probe tick runs.
    assert (
        TaskManagerMixin._embed_circuit_open(o, now + _EMBED_CB_COOLDOWN + 1) is False
    )


def test_success_resets_failures_and_closes():
    o = _Obj()
    now = 9000.0
    for _ in range(_EMBED_CB_THRESHOLD):
        TaskManagerMixin._embed_circuit_record(o, False, now)
    assert TaskManagerMixin._embed_circuit_open(o, now) is True
    # A healthy embed closes the breaker and clears the failure count.
    TaskManagerMixin._embed_circuit_record(o, True, now)
    assert o._embed_cb_failures == 0
    assert TaskManagerMixin._embed_circuit_open(o, now) is False


def test_closed_by_default():
    assert TaskManagerMixin._embed_circuit_open(_Obj(), 0.0) is False
