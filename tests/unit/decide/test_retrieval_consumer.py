"""EH-029: the retrieval plan template is a decision; the requested mode is the fallback."""

from __future__ import annotations

from agent_utilities.decide.consumers.retrieval import choose_retrieval_plan
from tests.unit.decide.fakes import FakeTransport, abstained, acted


def test_eg_picks_the_plan_template(eg: FakeTransport) -> None:
    eg.answer = acted("deep")
    assert choose_retrieval_plan("who owns billing?", "hyde") == "deep"
    request = eg.requests[0]
    assert [o["option_id"] for o in request["candidates"]["options"]] == [
        "deep",
        "hyde",
        "standard",
    ]
    assert request["params"] == [
        {"name": "query", "value": {"type": "text", "value": "who owns billing?"}}
    ]


def test_an_abstention_keeps_the_requested_mode(eg: FakeTransport) -> None:
    eg.answer = abstained()
    assert choose_retrieval_plan("who owns billing?", "hyde") == "hyde"
    assert eg.op_names() == ["commit"]


def test_an_unknown_mode_is_passed_through_untouched(eg: FakeTransport) -> None:
    assert choose_retrieval_plan("q", "graph-walk") == "graph-walk"
    assert eg.requests == []
