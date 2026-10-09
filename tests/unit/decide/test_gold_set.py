"""The synthetic full-label gold set over the reference swarm topologies."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agent_utilities.decide.topology import REFERENCE_TEMPLATES
from agent_utilities.decide.topology.gold_set import gold_dataset, gold_items


def _digest(value: Any) -> str:
    data = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


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
