"""EH-031: enrichment scheduling is a decision; running is the fallback."""

from __future__ import annotations

from agent_utilities.decide.consumers.enrichment import schedule_enricher
from tests.unit.decide.fakes import FakeTransport, abstained, acted


def test_eg_may_defer_an_enricher(eg: FakeTransport) -> None:
    eg.answer = acted("defer")
    assert schedule_enricher("code_cards", 64.0, 0.25) is False
    keys = [
        [n["key"] for n in o["numbers"]]
        for o in eg.requests[0]["candidates"]["options"]
    ]
    assert keys == [["declared_cost", "expected_yield"]] * 2


def test_an_unknown_yield_is_absent_and_the_enricher_runs(eg: FakeTransport) -> None:
    eg.answer = abstained("unknown_fact")
    assert schedule_enricher("code_cards", 64.0, None) is True
    keys = [
        [n["key"] for n in o["numbers"]]
        for o in eg.requests[0]["candidates"]["options"]
    ]
    assert keys == [["declared_cost"]] * 2
