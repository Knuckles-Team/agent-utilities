"""Tests for cross-agent trace correlation (CONCEPT:OS-5.11)."""

from __future__ import annotations

import contextvars

from agent_utilities.observability import correlation as corr


def _fresh():
    """Run each assertion in an isolated contextvars context."""
    return contextvars.copy_context()


def test_ensure_correlation_id_is_stable():
    ctx = _fresh()

    def body():
        a = corr.ensure_correlation_id()
        b = corr.ensure_correlation_id()
        assert a == b
        assert corr.get_correlation_id() == a
        return a

    assert ctx.run(body)


def test_carrier_round_trips_correlation_id():
    parent_ctx = _fresh()

    def parent():
        cid = corr.ensure_correlation_id()
        carrier = corr.current_carrier()
        assert carrier[corr.CORRELATION_HEADER] == cid
        # A "spawned child" in a fresh context binds the carrier and inherits it.
        child_ctx = _fresh()

        def child():
            with corr.bind_carrier(carrier) as effective:
                assert effective == cid
                assert corr.get_correlation_id() == cid

        child_ctx.run(child)
        return cid

    assert parent_ctx.run(parent)


def test_inject_and_extract_headers():
    ctx = _fresh()

    def body():
        cid = corr.ensure_correlation_id()
        headers = corr.inject({"content-type": "application/json"})
        assert headers[corr.CORRELATION_HEADER] == cid
        assert headers["content-type"] == "application/json"
        # extract is the inverse (case-insensitive).
        carrier = corr.extract({"X-Correlation-Id": cid})
        assert carrier[corr.CORRELATION_HEADER] == cid

    ctx.run(body)


def test_traceparent_is_w3c_shaped():
    ctx = _fresh()

    def body():
        from agent_utilities.harness import tracing

        tracing._current_trace_id.set("trace-abc")
        tracing._current_span_id.set("span-xyz")
        headers = corr.inject()
        tp = headers.get(corr.TRACEPARENT_HEADER)
        assert tp is not None
        parts = tp.split("-")
        assert parts[0] == "00"
        assert len(parts[1]) == 32  # trace-id
        assert len(parts[2]) == 16  # parent-id
        assert parts[3] == "01"

    ctx.run(body)


# --- identity propagation (CONCEPT:OS-5.14 + OS-5.11) -----------------------


def test_inject_carries_tenant_and_actor():
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.observability.correlation import (
        ACTOR_HEADER,
        TENANT_HEADER,
        ensure_correlation_id,
        inject,
    )
    from agent_utilities.security.brain_context import ActorContext, use_actor

    actor = ActorContext(
        actor_id="alice",
        actor_type=ActorType.HUMAN,
        tenant_id="acme",
        authenticated=True,
    )
    with use_actor(actor):
        ensure_correlation_id()
        headers = inject({})
    assert headers[TENANT_HEADER] == "acme"
    assert headers[ACTOR_HEADER] == "alice"


def test_inject_omits_system_actor_identity():
    from agent_utilities.observability.correlation import (
        ACTOR_HEADER,
        TENANT_HEADER,
        inject,
    )

    # No actor in scope → ambient SYSTEM_ACTOR (tenant="", actor="system");
    # identity headers must be absent (nothing to attribute).
    headers = inject({})
    assert TENANT_HEADER not in headers
    assert ACTOR_HEADER not in headers
