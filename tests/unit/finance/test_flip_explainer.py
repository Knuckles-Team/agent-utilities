"""EH-419: the flip explainer states the math, then only source-backed claims."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_utilities.api.finance import Evidence, explain_flip
from agent_utilities.domains.finance.flip_explainer import (
    SourcedClaim,
    flip_math,
    keep_sourced,
)

ALERT = {
    "listing_id": "binance:BTC/USDT:spot",
    "asset_class": "crypto",
    "timeframe": "1D",
    "informational_only": True,
    "record": {
        "record_id": "r1",
        "status": "emitted",
        "revises": None,
        "recorded_at": 5,
        "flip": {
            "event_id": "e1",
            "key_digest": "k",
            "from": "bearish",
            "to": "bullish",
            "bar_open": 1,
            "effective_at": 2,
            "observed_at": 2,
            "price": 6_500_000,
            "line": 6_400_000,
            "bar_revision": 0,
        },
    },
}
REAL = Evidence(url="https://news.example/etf-inflows", title="ETF inflows jump")


def _model(claims: list[dict[str, Any]], seen: list[str]) -> FunctionModel:
    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(str(messages[-1]))
        tool = info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(tool.name, json.dumps({"response": claims}))]
        )

    return FunctionModel(answer)


def test_the_math_is_read_off_the_record_without_a_model() -> None:
    math = flip_math(ALERT)
    assert (math.direction_from, math.direction_to) == ("bearish", "bullish")
    assert (math.close_ticks, math.line_ticks) == (6_500_000, 6_400_000)
    assert "closed above the trailing trend line" in math.rule


async def test_claims_citing_anything_but_the_evidence_are_dropped() -> None:
    seen: list[str] = []
    model = _model(
        [
            {"kind": "news", "text": "Spot ETF inflows rose.", "sources": [REAL.url]},
            {
                "kind": "news",
                "text": "A whale bought.",
                "sources": ["https://made.up/x"],
            },
        ],
        seen,
    )
    explanation = await explain_flip(ALERT, [REAL], model=model)
    assert [claim.text for claim in explanation.claims] == ["Spot ETF inflows rose."]
    assert explanation.dropped_claims == 1 and explanation.evidence_count == 1
    assert explanation.informational_only is True
    assert "not investment advice" in explanation.notice
    assert REAL.url in seen[0] and "Math (fact)" in seen[0]


async def test_without_evidence_the_math_stands_alone_and_no_model_runs() -> None:
    def refuse(*_args: Any) -> ModelResponse:
        raise AssertionError("no evidence means no model call")

    explanation = await explain_flip(ALERT, [], model=FunctionModel(refuse))
    assert explanation.claims == [] and explanation.math.status == "emitted"


def test_a_claim_needs_a_source_and_evidence_must_be_https() -> None:
    with pytest.raises(ValueError):
        SourcedClaim(text="no source", sources=[])
    with pytest.raises(ValueError):
        Evidence(url="http://insecure.example")
    kept, dropped = keep_sourced(
        [SourcedClaim(text="half", sources=[REAL.url, "https://other"])], [REAL]
    )
    assert (kept, dropped) == ([], 1)
