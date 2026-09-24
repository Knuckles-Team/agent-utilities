"""EH-464 / ST-12: the statistical rung's AU side -- features, independent slate
crediting and the synthetic full-label gold set."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from agent_utilities.decide.consumers.topology_learning import (
    TopologyOutcome,
    credit_topology_outcome,
    gold_dataset,
    gold_items,
    plan_features,
)
from agent_utilities.decide.schemas import feature_schema_body
from agent_utilities.decide.topology import REFERENCE_TEMPLATES
from tests.unit.decide.fakes import FakeTransport

PLAN = {
    "slots": [{"width": 1, "rounds": 1}, {"width": 4, "rounds": 2}],
    "lease": {"per_cell": [{"amount": 5}]},
    "makespan_ms": 700,
}


def _digest(value: Any) -> str:
    data = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def test_features_match_the_published_schema_keys() -> None:
    features = plan_features(PLAN, subtasks=4, headroom=10)
    assert features["width"] == 5.0 and features["rounds"] == 2.0
    assert features["headroom_ratio"] == 0.5
    names = {f["name"] for f in feature_schema_body("au.swarm.topology")["features"]}
    assert set(features) <= names
    pooled = plan_features(PLAN, subtasks=4, headroom=10, pooled={"class": 0.7})
    assert pooled["pooled.class"] == 0.7


def test_an_independent_evaluation_is_credited_and_censoring_drops_the_label() -> None:
    transport = FakeTransport()
    outcome = TopologyOutcome(
        "decision:1", "eval:1", "agent-lead", "graph-os", "tool_calls", True
    )
    asyncio.run(credit_topology_outcome(transport, "tenant-t", outcome))
    censored = TopologyOutcome(
        "decision:1", "eval:2", "agent-lead", "graph-os", "cancelled", True
    )
    asyncio.run(credit_topology_outcome(transport, "tenant-t", censored))
    first, second = transport.ops
    assert first["op"] == "evaluate" and first["evaluation"]["success"] is True
    assert first["evaluation"]["selected_agent"] == "agent-lead"
    assert second["evaluation"]["success"] is None


def test_the_gold_set_lists_acceptable_topologies_by_construction() -> None:
    items = {item.item_id: item for item in gold_items()}
    survey = items["gold:independent-6"]
    assert set(survey.acceptable) == {"swarm:fan-out-join", "swarm:supervisor-workers"}
    assert len(survey.candidates) == len(REFERENCE_TEMPLATES)
    assert items["gold:independent-check"].acceptable == ("swarm:critique-loop",)
    dataset, digest = gold_dataset(list(items.values()), digest_of=_digest)
    assert dataset["synthetic"] is True
    names = dataset["feature_names"]
    for row in dataset["items"]:
        assert len(row["features"]) == len(row["candidate_ids"]) * len(names)
        assert row["label"]["source"] == "synthetic_construction"
        assert set(row["label"]["acceptable"]) <= set(row["candidate_ids"])
    assert digest == _digest(dataset)
