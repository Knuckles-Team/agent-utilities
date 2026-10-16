"""Test double for the EG capability-designation port (EG-DECISION-ENGINE-R172).

Stands in for the engine in AU unit tests: AU code under test only forwards
requests/outcomes through the port, so the double needs just enough behaviour
(capability filter, cosine + reward order, EMA outcome) to observe the wiring.
It is NOT an AU designation implementation.
"""

from __future__ import annotations

import math
from typing import Any

from agent_utilities.knowledge_graph.retrieval.capability_designation_port import (
    CapabilityDesignation,
    CapabilityDesignationRequest,
)


class FakeCapabilityDesignationPort:
    def __init__(self) -> None:
        self.records: dict[str, tuple[list[float], frozenset[str]]] = {}
        self.rewards: dict[str, float] = {}
        self.requests: list[CapabilityDesignationRequest] = []
        self.outcomes: list[tuple[str, float]] = []

    def add(self, entity_id: str, vector: list[float], capabilities: Any = ()) -> None:
        self.records[entity_id] = (list(vector), frozenset(capabilities))

    def designate(
        self, request: CapabilityDesignationRequest
    ) -> list[CapabilityDesignation]:
        self.requests.append(request)
        q = list(request.query_vector)
        qn = math.sqrt(sum(v * v for v in q)) or 1.0
        out = []
        for eid, (vec, caps) in self.records.items():
            if not set(request.required_capabilities) <= caps:
                continue
            vn = math.sqrt(sum(v * v for v in vec)) or 1.0
            cos = sum(a * b for a, b in zip(q, vec, strict=False)) / (qn * vn)
            reward = self.rewards.get(eid, 0.5)
            out.append(
                CapabilityDesignation(
                    id=eid,
                    score=cos + 0.15 * (reward - 0.5),
                    capabilities=caps,
                    provenance={"reward": reward},
                )
            )
        out.sort(key=lambda d: (-d.score, d.id))
        return out[: request.k]

    def record_outcome(
        self,
        entity_id: str,
        *,
        success: bool | None = None,
        reward: float | None = None,
        **_: Any,
    ) -> float:
        r = float(reward) if reward is not None else (1.0 if success else 0.0)
        new = 0.7 * self.rewards.get(entity_id, 0.5) + 0.3 * r
        self.rewards[entity_id] = new
        self.outcomes.append((entity_id, r))
        return new

    def reward_of(self, entity_id: str) -> float:
        return self.rewards.get(entity_id, 0.5)
