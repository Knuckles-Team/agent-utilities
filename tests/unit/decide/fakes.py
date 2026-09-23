"""A scripted in-memory EG ``Decide``/``DecisionLog`` transport for consumer tests."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.decide import Binding, DecisionRunner, StaticBindings
from agent_utilities.decide.points import POINTS

SCHEMA = {
    "component_id": "decide.schema.x",
    "kind": "feature_schema",
    "definition_digest": "sha256:" + "0" * 64,
}


def record(outcome: dict[str, Any], *, digest: str = "sha256:" + "0" * 64) -> dict:
    return {
        "record_id": "decision:" + digest[-8:],
        "record_digest": digest,
        "outcome": outcome,
    }


def acted(option_id: str, **kw: Any) -> dict[str, Any]:
    return {"records": [record({"outcome": "acted", "option_id": option_id}, **kw)]}


def abstained(reason: str = "insufficient_confidence", **kw: Any) -> dict[str, Any]:
    outcome = {"outcome": "abstained", "reasons": [{"reason": reason}]}
    return {"records": [record(outcome, **kw)]}


@dataclass
class FakeTransport:
    """Answers every ``Decide`` with ``answer``; records every request and op."""

    answer: Any = None
    fail: BaseException | None = None
    requests: list[Mapping[str, Any]] = field(default_factory=list)
    ops: list[Mapping[str, Any]] = field(default_factory=list)

    async def decide(self, request: Mapping[str, Any]) -> Any:
        self.requests.append(request)
        if self.fail is not None:
            raise self.fail
        return self.answer

    async def log(self, op: Mapping[str, Any]) -> Any:
        self.ops.append(op)
        return {"record_id": "logged"}

    def run(self, call: Any) -> Any:
        return asyncio.run(call)

    def op_names(self) -> list[str]:
        return [str(op["op"]) for op in self.ops]


def runner(transport: FakeTransport, *, bound: bool = True) -> DecisionRunner:
    bindings = {
        qid: Binding(feature_schema=SCHEMA, policy={"policy": "default"})
        for qid in POINTS
    }
    return DecisionRunner(
        transport=transport,
        bindings=StaticBindings(bindings if bound else {}),
        tenant="tenant-t",
    )
