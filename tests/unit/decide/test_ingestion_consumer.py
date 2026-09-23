"""EH-030: the ingestion lane is a decision; the structure classifier is the fallback."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from agent_utilities import decide
from agent_utilities.decide.consumers.ingestion import choose_ingestion_lane
from agent_utilities.knowledge_graph.ingestion.engine import IngestionEngine
from tests.unit.decide.fakes import FakeTransport, abstained, acted, runner


@pytest.fixture
def eg() -> Iterator[FakeTransport]:
    transport = FakeTransport(answer=abstained(digest="sha256:" + "0" * 64))
    token = decide.use_runner(runner(transport))
    yield transport
    decide._RUNNER.reset(token)


def test_eg_routes_a_window_to_a_lane(eg: FakeTransport) -> None:
    eg.answer = acted("structured")
    assert choose_ingestion_lane("invoice", lambda: "prose") == "structured"
    options = eg.requests[0]["candidates"]["options"]
    votes = {o["option_id"]: o["numbers"][0]["q32"] for o in options}
    assert votes == {"mixed": 0, "prose": 1 << 32, "structured": 0}


def test_the_engine_keeps_the_classifier_s_lane_when_eg_abstains(
    eg: FakeTransport,
) -> None:
    text = '{"invoice": 12, "total": "9.50"}'
    assert IngestionEngine._classify_structure(text, "invoice") == "structured"
    assert eg.requests[0]["question"]["question_id"] == "au.ingestion.lane"
