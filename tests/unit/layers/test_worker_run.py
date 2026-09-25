"""A signed WorkItem claim becomes one bounded harness run and L5 commit."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
from typing import Any

import msgpack
import pytest

from agent_utilities.layers.contracts import McpEndpoint
from agent_utilities.layers.l5_writer import L5CommitRejected, RunOutcomeWriter
from agent_utilities.layers.worker_run import build_worker_run, run_worker_harness

DIGEST = hashlib.sha256(b"pinned").hexdigest()
ENDPOINT = McpEndpoint(name="graphos", url="https://graphos.example/mcp")


def _bound() -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    row = {
        "id": "wi-one",
        "tenant": "tenant-one",
        "description": "Inspect the package",
        "policy_digest": DIGEST,
        "catalog_digest": DIGEST,
        "model_digest": DIGEST,
        "metadata": {
            "au:description": "Inspect the package",
            "delegation_id": "job-one",
            "run_id": "job-one",
            "agent_id": "component-one",
            "agent_name": "package-agent",
            "delegator_id": "principal-one",
            "capability_digest": DIGEST,
        },
    }
    claim = {
        "work_item_id": "wi-one",
        "lease_owner": "worker-one",
        "lease_epoch": 3,
        "fencing_token": 8,
    }
    envelope = SimpleNamespace(
        job_id="job-one",
        agent_name="package-agent",
        session_id="session-one",
        allowed_tools=("find", "ask"),
    )
    carrier = SimpleNamespace(model_dump_json=lambda: '{"verified":"carrier"}')
    return row, claim, envelope, carrier


def test_worker_run_binds_pinned_admission_claim_and_context() -> None:
    row, claim, envelope, carrier = _bound()
    bound = build_worker_run(row, claim, envelope, ENDPOINT, carrier)
    assert bound.binding.selected_agent_id == "component-one"
    assert bound.binding.capability_digest == DIGEST
    assert bound.spec.agent_ref == "package-agent"
    assert bound.spec.toolset.context_endpoint == ENDPOINT
    assert bound.spec.toolset.allowed_tools == ("find", "ask")
    assert bound.spec.side_effects == "irreversible"
    assert bound.claim.fencing_token == 8


def test_worker_run_refuses_missing_or_mismatched_bindings() -> None:
    row, claim, envelope, carrier = _bound()
    row["metadata"]["agent_name"] = "other-agent"
    with pytest.raises(ValueError, match="disagree"):
        build_worker_run(row, claim, envelope, ENDPOINT, carrier)
    row["metadata"]["agent_name"] = envelope.agent_name
    row["metadata"].pop("capability_digest")
    with pytest.raises(ValueError, match="capability_digest"):
        build_worker_run(row, claim, envelope, ENDPOINT, carrier)


def test_worker_harness_commits_once_with_real_trace(monkeypatch: Any) -> None:
    row, claim, envelope, carrier = _bound()
    bound = build_worker_run(row, claim, envelope, ENDPOINT, carrier)
    seen: dict[str, Any] = {}

    class Runner:
        async def execute_agent(
            self, agent_name: str, task: str, **options: Any
        ) -> str:
            seen.update(agent_name=agent_name, task=task, options=options)
            return "inspected"

    class WorkItems:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def receipt_properties(self, properties: dict[str, Any]) -> bytes:
            return msgpack.packb(properties, use_bin_type=True)

        async def commit_result(self, **kwargs: Any) -> dict[str, str]:
            self.calls.append(kwargs)
            return {"status": "committed"}

    work_items = WorkItems()
    outcome, receipt = asyncio.run(
        run_worker_harness(bound, Runner(), RunOutcomeWriter(work_items))
    )
    assert receipt.status == "committed"
    assert outcome.result.status == "succeeded"
    assert len(work_items.calls) == 1
    assert (
        work_items.calls[0]["outcome_extension"]["outcome_bundle"]["run_id"]
        == "job-one"
    )
    assert seen["options"]["context_endpoint"] == ENDPOINT
    assert seen["options"]["execution_mode"] == "pydantic_graph"


def test_worker_harness_refuses_fenced_terminal_commit() -> None:
    row, claim, envelope, carrier = _bound()
    bound = build_worker_run(row, claim, envelope, ENDPOINT, carrier)

    class Runner:
        async def execute_agent(self, *_args: Any, **_kwargs: Any) -> str:
            return "inspected"

    class WorkItems:
        def receipt_properties(self, properties: dict[str, Any]) -> bytes:
            return msgpack.packb(properties, use_bin_type=True)

        async def commit_result(self, **_kwargs: Any) -> dict[str, str]:
            return {"status": "fenced"}

    with pytest.raises(L5CommitRejected, match="fenced"):
        asyncio.run(run_worker_harness(bound, Runner(), RunOutcomeWriter(WorkItems())))


def test_hosted_worker_uses_verified_endpoint_and_one_l5_writer(
    monkeypatch: Any,
) -> None:
    from agent_utilities.knowledge_graph.core import work_durability
    from agent_utilities.layers import l5_writer, worker_run
    from agent_utilities.orchestration import agent_dispatch_worker, manager
    from agent_utilities.orchestration.agent_dispatch import (
        KIND_WORK_ITEM_TURN,
        AgentTurnEnvelope,
    )

    row, claim, _envelope, _carrier = _bound()
    row["id"] = "wi-one"
    monkeypatch.setenv("AGENT_UTILITIES_TOKEN_SECRET", "worker-run-test-secret")
    envelope = AgentTurnEnvelope(
        job_id="job-one",
        session_id="session-one",
        kind=KIND_WORK_ITEM_TURN,
        payload_ref="wi-one",
        agent_name="package-agent",
        tenant="tenant-one",
        allowed_tools=("find", "ask"),
    )
    envelope.ensure_authenticated_carrier()
    monkeypatch.setattr(work_durability, "get_work_item", lambda *_args: row)
    monkeypatch.setattr(manager, "Orchestrator", lambda _engine: object())
    writer = object()
    monkeypatch.setattr(
        l5_writer.RunOutcomeWriter, "from_engine", lambda _engine: writer
    )
    seen: dict[str, Any] = {}

    async def execute(bound: Any, runner: Any, actual_writer: Any) -> Any:
        seen.update(spec=bound.spec, writer=actual_writer)
        return SimpleNamespace(result=SimpleNamespace(status="succeeded")), object()

    monkeypatch.setattr(worker_run, "run_worker_harness", execute)

    async def context(_envelope: Any) -> McpEndpoint:
        return ENDPOINT

    lease = SimpleNamespace(require_current=lambda: None)
    result = agent_dispatch_worker._execute_work_item_turn(
        envelope, object(), lease, claim, context
    )
    assert result == "completed"
    assert seen["spec"].task == "Inspect the package"
    assert seen["spec"].toolset.context_endpoint == ENDPOINT
    assert seen["writer"] is writer
