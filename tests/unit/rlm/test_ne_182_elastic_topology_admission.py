"""NE-182 bounded elastic topology/RLM admission contracts.

These are deterministic contract fixtures only.  They do not start a model,
container, forkserver, microVM, or Wasmtime runtime; runtime lanes own those
expensive validations.
"""

from __future__ import annotations

import dataclasses

import pytest

from agent_utilities.graph.topology_engine import (
    ElasticTopologyAdmission,
    SandboxResourceLimits,
    TopologyAdmissionError,
    TopologyEngine,
)
from agent_utilities.models.knowledge_graph import TeamComposition
from agent_utilities.rlm.config import RLMConfig
from agent_utilities.rlm.repl import RLMEnvironment


def _admission(**overrides) -> ElasticTopologyAdmission:
    values = {
        "tenant": "tenant:test",
        "delegation_id": "delegation:test",
        "capabilities": ("rlm.execute", "topology.materialize"),
        "issued_at": 2_000_000_000.0,
    }
    values.update(overrides)
    return ElasticTopologyAdmission(**values)


def _team(count: int = 2, *, mode: str = "parallel") -> TeamComposition:
    return TeamComposition(
        team_id="team:test",
        adaptive_agent_router=[
            {"role": f"worker-{idx}", "agent_id": f"agent-{idx}"}
            for idx in range(count)
        ],
        execution_mode=mode,
    )


def test_admission_is_immutable_and_content_addressed():
    admission = _admission()
    assert admission.digest == _admission().digest
    metadata = admission.work_item_metadata()
    admission.require_work_item(
        {
            "tenant": admission.tenant,
            "deadline_unix": admission.deadline_unix,
            "metadata": metadata,
        }
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        admission.max_nodes = 1  # type: ignore[misc]


def test_work_item_admission_digest_is_exact_and_non_replayable():
    admission = _admission()
    item = {
        "tenant": admission.tenant,
        "deadline_unix": admission.deadline_unix,
        "metadata": admission.work_item_metadata(),
    }
    item["metadata"]["metadata"]["admission_digest"] = "stale"
    with pytest.raises(TopologyAdmissionError, match="digest"):
        admission.require_work_item(item)


def test_resource_page_limit_must_fit_actual_memory_limit():
    with pytest.raises(TopologyAdmissionError, match="page limit"):
        SandboxResourceLimits(memory_bytes=64 * 1024 * 1024, max_wasm_pages=2_048)


def test_materialization_rejects_fanout_before_engine_write():
    admission = _admission(max_fan_out=1)
    with pytest.raises(TopologyAdmissionError, match="fan-out"):
        TopologyEngine(admission=admission).materialize(_team())


def test_materialization_rejects_depth_and_node_bounds():
    admission = _admission(max_nodes=1, max_depth=1)
    with pytest.raises(TopologyAdmissionError, match="node count"):
        TopologyEngine(admission=admission).materialize(
            _team(count=2, mode="sequential")
        )


@pytest.mark.asyncio
async def test_rlm_fanout_and_payload_are_admission_bounded():
    admission = _admission(max_fan_out=2, max_parallelism=1, max_payload_bytes=256)
    env = RLMEnvironment(
        context="small",
        config=RLMConfig(max_depth=1),
        admission=admission,
    )
    with pytest.raises(TopologyAdmissionError, match="fan-out"):
        await env.run_parallel_sub_calls(
            [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
        )
    with pytest.raises(TopologyAdmissionError, match="payload"):
        await env.execute("x = '" + ("x" * 300) + "'")


def test_engine_less_retirement_is_not_a_second_lifecycle_authority():
    assert TopologyEngine().retire("workitem:missing") is False
