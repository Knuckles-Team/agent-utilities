"""EH-396 / EH-397: adapters and embedding generations move only with EG's
receipts; the retrieval path resolves the active generation."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from agent_utilities.decide.learning import generation
from agent_utilities.decide.learning.adapter import AdapterPlan, AdapterPromoter
from agent_utilities.decide.learning.generation import (
    GenerationPlan,
    GenerationSwap,
    active_generation,
    resolve_generation,
)
from agent_utilities.decide.learning.session import LearningSession
from tests.unit.decide.fakes import FakeTransport


def _pointer(target: str | None) -> dict[str, Any]:
    active = None if target is None else {"transition": "activated", "target": target}
    return {
        "result": "pointer",
        "key": "k",
        "active": active,
        "stack": [],
        "history": [],
    }


def _scripted(answers: Mapping[str, Any]) -> FakeTransport:
    def answer(op: Mapping[str, Any]) -> Any:
        retrieval = op.get("retrieval")
        key = retrieval["action"] if retrieval else op["op"]
        value = answers[key]
        return value(op) if callable(value) else value

    return FakeTransport(log_answer=answer)


def _session(transport: FakeTransport) -> LearningSession:
    return LearningSession(transport, "tenant-t")


FITTED = {
    "result": "fitted",
    "adapter_digest": "sha256:a",
    "receipt_digest": "sha256:r",
    "body": {},
    "receipt": {"passed": True, "wins": 40, "losses": 2},
}


def test_an_adapter_is_activated_only_with_its_passing_receipt() -> None:
    eg = _scripted({"fit_adapter": FITTED, "activate_adapter": _pointer("sha256:a")})
    promoter = AdapterPromoter(_session(eg))
    done = asyncio.run(promoter.promote(AdapterPlan("kg", "sha256:space")))
    assert done.promoted and done.pointer is not None
    fit, activate = (op["retrieval"] for op in eg.ops)
    assert fit["request"]["graph"] == "kg" and fit["request"]["max_gain_q16"] <= 1 << 15
    assert (activate["adapter_digest"], activate["receipt_digest"]) == (
        "sha256:a",
        "sha256:r",
    )


def test_a_failed_receipt_activates_nothing() -> None:
    failed = {**FITTED, "receipt": {"passed": False}}
    eg = _scripted({"fit_adapter": failed})
    done = asyncio.run(AdapterPromoter(_session(eg)).promote(AdapterPlan("kg", "s")))
    assert (done.promoted, done.stage) == (False, "eval")
    assert [op["retrieval"]["action"] for op in eg.ops] == ["fit_adapter"]


class _Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def get_text_embedding(self, text: str) -> list[float]:
        return [1.0, 0.0] if self.model_name == "base" else [0.0, 1.0]


def _swap(eg: FakeTransport, trained: str | None, leases: list[str]) -> GenerationSwap:
    @contextlib.contextmanager
    def lease() -> Iterator[None]:
        leases.append("held")
        yield
        leases.append("released")

    async def reembed(active: str, shadow: str, model: str) -> int:
        leases.append(f"reembed {active}->{shadow} with {model}")
        return 10

    return GenerationSwap(_session(eg), lambda base: trained, reembed, lease, _Embedder)


PLAN = GenerationPlan("kg", "kg", "sha256:s1", "kg-2", "sha256:s2", min_eval_items=1)
VIEW = {"columns": ["record_id"], "rows": [[{"cell": "text", "value": "rec-1"}]]}
ENTRY = {
    "record": {
        "inputs": {
            "params": [{"name": "query", "value": {"type": "text", "value": "q"}}]
        }
    }
}


def test_a_generation_swaps_only_with_a_passing_receipt_and_rolls_back() -> None:
    evaluated = {
        "result": "generation",
        "receipt_digest": "sha256:g",
        "receipt": {"passed": True},
    }
    eg = _scripted(
        {
            "query": VIEW,
            "get": ENTRY,
            "evaluate_generation": evaluated,
            "activate_generation": _pointer("kg-2"),
            "rollback_generation": _pointer(None),
        }
    )
    leases: list[str] = []
    done = asyncio.run(_swap(eg, "tuned", leases).run(PLAN, "base"))
    assert done.activated, done
    assert leases == ["held", "reembed kg->kg-2 with tuned", "released"]
    request = next(
        op["retrieval"]["request"]
        for op in eg.ops
        if (op.get("retrieval") or {}).get("action") == "evaluate_generation"
    )
    assert request["items"] == [
        {"record_id": "rec-1", "active_q16": [65536, 0], "shadow_q16": [0, 65536]}
    ]
    assert request["active_space"] != request["shadow_space"]
    rolled = asyncio.run(_swap(eg, "tuned", []).rollback("kg"))
    assert rolled["active"] is None


def test_no_trained_model_or_a_failed_receipt_stops_the_swap() -> None:
    eg = _scripted({})
    assert asyncio.run(_swap(eg, None, []).run(PLAN, "base")).stage == "train"
    failed = {
        "result": "generation",
        "receipt_digest": "d",
        "receipt": {"passed": False},
    }
    eg = _scripted({"query": VIEW, "get": ENTRY, "evaluate_generation": failed})
    done = asyncio.run(_swap(eg, "tuned", []).run(PLAN, "base"))
    assert (done.activated, done.stage) == (False, "eval")


class _GraphView:
    def __init__(self, name: str) -> None:
        self.graph_name = name

    def for_graph(self, name: str) -> _GraphView:
        return _GraphView(name)


@pytest.fixture(autouse=True)
def _fresh_pointers() -> Iterator[None]:
    generation._RESOLVED.clear()
    yield
    generation._RESOLVED.clear()


def test_the_retrieval_path_resolves_the_active_generation(eg: FakeTransport) -> None:
    eg.log_answer = lambda op: _pointer("kg-2")
    view = active_generation(_GraphView("kg"))
    assert view.graph_name == "kg-2"
    assert resolve_generation("kg") == "kg-2"
    assert len(eg.ops) == 1, "the pointer is read once per TTL"
    assert resolve_generation("kg", now=lambda: 1e12) == "kg-2"
    assert len(eg.ops) == 2


def test_no_session_or_no_pointer_keeps_the_logical_graph(eg: FakeTransport) -> None:
    eg.log_answer = lambda op: _pointer(None)
    assert active_generation(_GraphView("kg")).graph_name == "kg"
    assert active_generation(None) is None
