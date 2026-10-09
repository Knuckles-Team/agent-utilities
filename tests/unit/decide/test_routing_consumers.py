"""Routing decision points decide through EG, or keep their deterministic rule."""

from __future__ import annotations

from types import SimpleNamespace

from agent_utilities import decide
from agent_utilities.decide.consumers.routing import route_by_cost, route_model
from agent_utilities.orchestration.outcome_router import OutcomeRouter
from tests.unit.decide.fakes import FakeTransport, abstained, acted


def _keys(transport: FakeTransport) -> list[list[str]]:
    options = transport.requests[-1]["candidates"]["options"]
    return [[n["key"] for n in o["numbers"]] for o in options]


def test_outcome_router_takes_eg_s_decision(eg: FakeTransport) -> None:
    eg.answer = acted("deep")
    chosen = OutcomeRouter("shape").select("qa", "fast", ("fast", "deep"))
    assert chosen == "deep"
    assert eg.requests[0]["question"]["question_id"] == "au.route.choice"
    assert _keys(eg) == [["prior", "reward"], ["prior", "reward"]]


def test_outcome_router_keeps_its_prior_when_eg_abstains(eg: FakeTransport) -> None:
    eg.answer = abstained()
    router = OutcomeRouter("shape")
    assert router.select("qa", "fast", ("fast", "deep")) == "fast"
    decide.install_runner(None)
    token = decide.use_runner(None)
    try:
        assert router.select("qa", "fast", ("fast", "deep")) == "fast", "same rule"
    finally:
        decide.reset_runner(token)
    assert eg.op_names()[0] == "commit", "the abstention is recorded"


def _model(model_id: str, tier: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=model_id, tier=tier, cost=SimpleNamespace(input=1.0, output=2.0)
    )


def test_model_routing_decides_among_eligible_models(eg: FakeTransport) -> None:
    light, heavy = _model("m-light", "light"), _model("m-heavy", "heavy")
    eg.answer = acted("m-light")
    assert route_model("coder", [light, heavy], heavy) is light
    eg.answer = abstained()
    assert route_model("coder", [light, heavy], heavy) is heavy


class _Observed:
    def observed_cost(self, model_id: str) -> float | None:
        return {"m-light": 0.5}.get(model_id)


def test_cost_routing_declares_observed_cost_only_when_l5_holds_it(
    eg: FakeTransport,
) -> None:
    models = [_model("m-heavy", "heavy"), _model("m-light", "light")]
    eg.answer = abstained("unknown_fact")
    choice = route_by_cost(models, models[0])
    assert choice.option_id == "m-heavy" and not choice.decided
    assert _keys(eg) == [["declared_cost"], ["declared_cost"]]
    route_by_cost(models, models[0], _Observed())
    assert _keys(eg) == [["declared_cost"], ["declared_cost", "l5.observed_cost"]]
