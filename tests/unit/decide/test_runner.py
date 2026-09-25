"""The shared consumer contract: decide, fall back, record, resolve."""

from __future__ import annotations

import asyncio

import pytest

from agent_utilities import decide
from agent_utilities.decide import Escalated, Option
from agent_utilities.decide.transport import GeneratedTransport
from agent_utilities.layers.clients import LayerUnavailable
from tests.unit.decide.fakes import FakeTransport, abstained, acted, runner

OPTIONS = [Option("plan-b", {"threshold": 0.28}), Option("plan-a", {"threshold": 0.38})]


def test_an_unbound_point_never_calls_eg_and_falls_back() -> None:
    transport = FakeTransport(answer=acted("plan-a"))
    choice = runner(transport, bound=False).choose(
        "au.retrieval.plan", OPTIONS, lambda: "plan-b"
    )
    assert (choice.option_id, choice.decided, choice.reason) == (
        "plan-b",
        False,
        "unbound",
    )
    assert transport.requests == []


def test_an_executed_offered_option_is_the_answer_and_is_logged() -> None:
    transport = FakeTransport(answer=acted("plan-a"))
    choice = runner(transport).choose("au.retrieval.plan", OPTIONS, lambda: "plan-b")
    assert (choice.option_id, choice.decided) == ("plan-a", True)
    assert transport.op_names() == ["commit"]
    request = transport.requests[0]
    assert request["question"]["kind"] == "retrieval_plan"
    ids = [o["option_id"] for o in request["candidates"]["options"]]
    assert ids == ["plan-a", "plan-b"], "declared options are sorted"
    assert request["candidates"]["options"][0]["numbers"][0]["q32"] == round(
        0.38 * (1 << 32)
    )


def test_an_abstention_falls_back_and_is_still_recorded() -> None:
    transport = FakeTransport(answer=abstained())
    choice = runner(transport).choose("au.retrieval.plan", OPTIONS, lambda: "plan-b")
    assert (choice.option_id, choice.decided) == ("plan-b", False)
    assert choice.reason == "abstained: insufficient_confidence"
    assert choice.record_id is not None and choice.logged
    assert transport.op_names() == ["commit"]


def test_an_option_eg_was_not_offered_is_never_obeyed() -> None:
    transport = FakeTransport(answer=acted("plan-invented"))
    choice = runner(transport).choose("au.retrieval.plan", OPTIONS, lambda: "plan-b")
    assert (choice.option_id, choice.decided) == ("plan-b", False)
    assert choice.reason == "foreign_option: plan-invented"


def test_an_unavailable_engine_costs_only_the_fallback() -> None:
    transport = FakeTransport(fail=ConnectionError("engine down"))
    choice = runner(transport).choose("au.retrieval.plan", OPTIONS, lambda: "plan-b")
    assert choice.option_id == "plan-b"
    assert choice.reason.startswith("unavailable: ConnectionError: engine down")


def test_an_escalated_answer_re_enters_as_a_resolution() -> None:
    transport = FakeTransport(answer=abstained())
    escalated = Escalated(
        "plan-a", {"resolver": "model", "producer": "au-escalation"}, "r-1"
    )
    choice = runner(transport).choose("au.entity.same_as", OPTIONS, lambda: escalated)
    assert (choice.option_id, choice.decided) == ("plan-a", False)
    assert transport.op_names() == ["commit", "resolve"]
    resolution = transport.ops[1]["resolution"]
    assert resolution["record_id"] == choice.record_id
    assert resolution["resolver"]["resolver"] == "model"


def test_a_sampled_point_logs_only_its_reproducible_sample() -> None:
    kept = FakeTransport(answer=abstained(digest="sha256:" + "0" * 64))
    runner(kept).choose("au.route.choice", OPTIONS, lambda: "plan-b")
    skipped = FakeTransport(answer=abstained(digest="sha256:" + "0" * 63 + "1"))
    runner(skipped).choose("au.route.choice", OPTIONS, lambda: "plan-b")
    assert kept.op_names() == ["commit"]
    assert skipped.op_names() == []


async def test_the_async_path_shares_the_contract() -> None:
    transport = FakeTransport(answer=acted("plan-a"))
    choice = await runner(transport).achoose(
        "au.retrieval.plan", OPTIONS, lambda: "plan-b"
    )
    assert (choice.option_id, choice.decided) == ("plan-a", True)


def test_without_a_runner_the_call_site_keeps_its_rule() -> None:
    decide.install_runner(None)
    choice = decide.choose("au.retrieval.plan", OPTIONS, lambda: "plan-b")
    assert (choice.option_id, choice.reason) == ("plan-b", "no_runner")


def test_a_sync_transport_without_an_engine_loop_refuses_instead_of_blocking() -> None:
    transport = GeneratedTransport(client=object())

    async def never() -> None:
        raise AssertionError("must not run")

    with pytest.raises(LayerUnavailable):
        transport.run(never())
    loop = asyncio.new_event_loop()
    loop.close()
    with pytest.raises(LayerUnavailable):
        GeneratedTransport(client=object(), loop=loop).run(never())


def test_a_bound_evaluator_is_named_at_commit() -> None:
    """EH-395: the binding's evaluator rides the commit as an expiring grant."""
    from agent_utilities.decide import Binding
    from agent_utilities.decide.outcome import commit_op

    binding = Binding(
        feature_schema={},
        policy={"policy": "default"},
        evaluator="principal:sha256:ab",
        evaluator_ttl_s=60,
    )
    op = commit_op({"record_id": "r"}, binding, 1_000)
    assert op["evaluator"] == {
        "principal": "principal:sha256:ab",
        "role": None,
        "expires_at_ms": 61_000,
    }
    assert commit_op({"record_id": "r"}, None, 1_000)["evaluator"] is None
    by_role = commit_op({"record_id": "r"}, None, 1_000, "decide-evaluator")
    assert by_role["evaluator"] == {
        "principal": None,
        "role": "decide-evaluator",
        "expires_at_ms": 3_601_000,
    }
    overridden = commit_op({"record_id": "r"}, binding, 1_000, "decide-evaluator")
    assert overridden["evaluator"]["role"] is None


def test_the_retrieval_plan_commit_names_its_policy_evaluator_role() -> None:
    """EH-395 (policy role): the LIVE commit path -- ``DecisionRunner.choose``
    on the retrieval-plan point -- names the point's declared evaluator role."""
    from agent_utilities.decide.points import DECIDE_EVALUATOR_ROLE, POINTS

    assert POINTS["au.retrieval.plan"].evaluator_role == DECIDE_EVALUATOR_ROLE
    transport = FakeTransport(answer=acted("plan-a"))
    runner(transport).choose("au.retrieval.plan", OPTIONS, lambda: "plan-b")
    (commit,) = transport.ops
    assert commit["evaluator"]["role"] == DECIDE_EVALUATOR_ROLE
    assert commit["evaluator"]["principal"] is None
    other = FakeTransport(answer=acted("plan-a"))
    runner(other).choose("au.route.cost", OPTIONS, lambda: "plan-b")
    assert other.ops[0]["evaluator"] is None, "only a declaring point names one"
