"""The router RECEIVES capability designation from EG (EG-DECISION-ENGINE-R172).

agent-utilities keeps no capability index and does no scoring: the enricher
forwards the router's question and outcomes through the designation port bound
on the engine, and returns ``None`` (keyword-scan fallback) without one.
"""

from __future__ import annotations

import types

from agent_utilities.graph.routing.enrichers.capability_designation import (
    designate_specialists,
    record_capability_outcome,
)
from tests.capability_port_fake import FakeCapabilityDesignationPort


def _engine(port=None):
    engine = types.SimpleNamespace()
    if port is not None:
        engine.capability_designator = port
    return engine


def _port():
    port = FakeCapabilityDesignationPort()
    port.add("tool:search", [1.0, 0.0, 0.0], ["web"])
    port.add("tool:calc", [0.0, 1.0, 0.0], ["math"])
    return port


def test_designate_forwards_the_question_and_returns_eg_order():
    port = _port()
    out = designate_specialists(
        _engine(port),
        "find it",
        k=2,
        required_caps=["web"],
        tenant="t1",
        policy_tags=["public"],
        embed_fn=lambda _q: [1.0, 0.0, 0.0],
    )
    assert out == ["tool:search"]
    request = port.requests[-1]
    assert request.required_capabilities == ("web",)
    assert request.required_policy_tags == ("public",)
    assert request.tenant == "t1"
    assert request.k == 2


def test_designate_returns_none_without_a_port():
    out = designate_specialists(_engine(), "q", embed_fn=lambda _q: [1.0, 0.0, 0.0])
    assert out is None


def test_designate_returns_none_when_embedding_unavailable():
    assert (
        designate_specialists(_engine(_port()), "q", embed_fn=lambda _q: None) is None
    )


def test_designate_returns_none_when_the_port_fails():
    class _Broken(FakeCapabilityDesignationPort):
        def designate(self, request):
            raise RuntimeError("engine down")

    out = designate_specialists(
        _engine(_Broken()), "q", embed_fn=lambda _q: [1.0, 0.0, 0.0]
    )
    assert out is None


def test_record_outcome_forwards_to_eg():
    port = _port()
    updated = record_capability_outcome(_engine(port), "tool:search", success=True)
    assert port.outcomes == [("tool:search", 1.0)]
    assert updated > 0.5


def test_record_outcome_never_raises_without_a_backend():
    assert record_capability_outcome(_engine(), "tool:search", success=True) == 0.5
