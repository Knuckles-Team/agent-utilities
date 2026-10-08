"""A scripted in-memory EG ``Decide``/``DecisionLog`` transport for consumer tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
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
    #: Answers a ``DecisionLog`` op (default: a bare logged acknowledgement).
    log_answer: Callable[[Mapping[str, Any]], Any] = lambda op: {"record_id": "logged"}
    #: Answers a SQL read of the decision views (default: no rows).
    sql_answer: Callable[[str], Any] = lambda query: {"columns": [], "rows": []}
    queries: list[str] = field(default_factory=list)

    async def decide(self, request: Mapping[str, Any]) -> Any:
        self.requests.append(request)
        if self.fail is not None:
            raise self.fail
        return self.answer

    async def log(self, op: Mapping[str, Any]) -> Any:
        self.ops.append(op)
        return self.log_answer(op)

    async def sql(self, query: str) -> Any:
        self.queries.append(query)
        return self.sql_answer(query)

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


#: What a committed ``DecisionCommitResult`` looks like to AU.
COMMITTED = {
    "record_id": "decision:abc",
    "component": {
        "component": {"kind": "decision_record", "definition_digest": "sha256:rec"}
    },
}


class FakeGraphs:
    """An L3 agent-graph client answering every assembly with ``result``."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.requests: list[Any] = []
        self.commits: list[Any] = []
        self.published: list[Any] = []

    async def assemble(self, request: Any) -> Any:
        self.requests.append(request)
        return self.result

    async def commit_decision(self, request: Any) -> Any:
        self.commits.append(request)
        return COMMITTED

    async def publish_graph(
        self,
        draft: Any,
        context: Any,
        *,
        evidence: Any = None,
        idempotency_key: Any = None,
    ) -> Any:
        self.published.append((draft, context, evidence))
        return {"graph_id": draft["graph_id"]}
