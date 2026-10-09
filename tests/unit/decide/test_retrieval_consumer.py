"""The retrieval plan template is a decision; the requested mode is the fallback."""

from __future__ import annotations

import pytest

from agent_utilities.decide.consumers.retrieval import (
    choose_retrieval,
    choose_retrieval_plan,
)
from agent_utilities.decide.options import q32
from tests.unit.decide.fakes import FakeTransport, abstained, acted


@pytest.mark.spec("AU-CONTEXT-R001")
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


@pytest.mark.spec("AU-CONTEXT-R001")
def test_a_proven_path_reaches_eg_as_a_candidate_with_no_local_promotion(
    eg: FakeTransport,
) -> None:
    """A proven RetrievalPath template is submitted to EG's decision engine as
    just another candidate option, carrying its judged record as facts. AU
    never ranks or activates it itself: the chosen option is whatever EG's
    answer names, and the only operation AU performs locally is the decision
    request itself (plus its commit) -- no separate write promotes the path."""
    path = {"template_digest": "sha256:deadbeef", "successes": 40, "failures": 2}
    eg.answer = acted("path:sha256:deadbeef")

    choice = choose_retrieval("who owns billing?", "hyde", paths=[path])

    assert choice.option_id == "path:sha256:deadbeef"
    request = eg.requests[0]
    options = {o["option_id"]: o for o in request["candidates"]["options"]}
    assert "path:sha256:deadbeef" in options
    numbers = {n["key"]: n["q32"] for n in options["path:sha256:deadbeef"]["numbers"]}
    assert numbers["successes"] == q32(40.0) and numbers["failures"] == q32(2.0)
    assert numbers["requested"] == q32(0.0), "a proven path is never the fallback"
    # The decision + its commit are the only operations AU performs -- no
    # local promotion/write of the path into any ranking table happens here.
    assert eg.op_names() == ["commit"]


def test_an_abstention_keeps_the_requested_mode(eg: FakeTransport) -> None:
    eg.answer = abstained()
    assert choose_retrieval_plan("who owns billing?", "hyde") == "hyde"
    assert eg.op_names() == ["commit"]


def test_an_unknown_mode_is_passed_through_untouched(eg: FakeTransport) -> None:
    assert choose_retrieval_plan("q", "graph-walk") == "graph-walk"
    assert eg.requests == []
