"""EH-040: a head is promoted only by publishing it with a PASSED eval receipt."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from agent_utilities.decide.promotion import HeadPromoter, PromotionPlan

SCHEMA = {"component_id": "s", "kind": "feature_schema", "definition_digest": "d"}


def _job(output: dict[str, Any]) -> dict[str, Any]:
    return {"state": {"state": "succeeded", "output": output}}


FIT = _job(
    {
        "output": "fit",
        "draft_sha256": "abc",
        "draft_length": 10,
        "draft": {"kind": "listwise_logistic"},
    }
)


@pytest.fixture
def jobs(monkeypatch) -> dict[str, Any]:
    state: dict[str, Any] = {"sent": [], "eval": None}

    async def fit(client, params, graph=None, *, idempotency_key=None):
        state["sent"].append(("fit", params))
        return FIT

    async def eval_(client, params, graph=None, *, idempotency_key=None):
        state["sent"].append(("eval", params))
        return state["eval"]

    module = types.ModuleType("epistemic_graph.generated.coordination")
    setattr(module, "send_decision_fit", fit)
    setattr(module, "send_decision_eval", eval_)
    monkeypatch.setitem(sys.modules, "epistemic_graph.generated.coordination", module)
    return state


def _promoter(published: list[Any]) -> HeadPromoter:
    async def publish(draft: Any, receipt: str) -> dict[str, Any]:
        published.append((draft, receipt))
        return {"component_id": "head", "kind": "decision_head"}

    return HeadPromoter(client=object(), tenant="t", publish=publish)


async def test_a_passed_receipt_publishes_the_exact_draft(jobs) -> None:
    jobs["eval"] = _job(
        {"output": "eval", "receipt": {"passed": True, "receipt_digest": "r1"}}
    )
    published: list[Any] = []
    result = await _promoter(published).promote(
        PromotionPlan("au.route.choice", SCHEMA), "k"
    )
    assert result.promoted and published == [({"kind": "listwise_logistic"}, "r1")]
    eval_request = jobs["sent"][1][1]["op"]["request"]
    assert eval_request["candidate"] == {
        "candidate": "draft_artifact",
        "sha256": "abc",
        "length": 10,
    }
    assert jobs["sent"][0][1]["op"]["request"]["source"] == {
        "source": "logged",
        "question_id": "au.route.choice",
    }


async def test_a_failed_receipt_names_its_gates_and_publishes_nothing(jobs) -> None:
    jobs["eval"] = _job(
        {
            "output": "eval",
            "receipt": {"passed": False, "failed_gates": ["min_ess"]},
        }
    )
    published: list[Any] = []
    result = await _promoter(published).promote(
        PromotionPlan("au.route.choice", SCHEMA), "k"
    )
    assert not result.promoted and result.failed_gates == ("min_ess",)
    assert published == []
