"""Bind one verified WorkItem claim to an L4 run and its L5 receipt."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from agent_utilities.layers.adapters.pydantic_ai import PydanticAiHarness
from agent_utilities.layers.contracts import McpEndpoint, RunBudget, RunSpec, RunToolset
from agent_utilities.layers.execution import RunOutcome, run_to_completion
from agent_utilities.layers.l5_writer import (
    DelegationBinding,
    L5Receipt,
    LeaseClaim,
    RunOutcomeWriter,
)
from agent_utilities.layers.negotiation import HarnessPolicy
from agent_utilities.layers.trace_ownership import l5_terminal_trace_owner


@dataclass(frozen=True, slots=True)
class WorkerRun:
    """Trusted inputs read back from EG and the verified dispatch carrier."""

    spec: RunSpec
    binding: DelegationBinding
    claim: LeaseClaim


def _required(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"WorkItem has no {name} binding")
    return value


def _deadline_budget(deadline: Any) -> RunBudget:
    if deadline is None:
        return RunBudget()
    remaining = float(deadline) - time.time()
    if remaining <= 0:
        raise ValueError("WorkItem deadline has expired")
    return RunBudget(max_wall_s=min(remaining, 86_400.0))


def build_worker_run(
    row: dict[str, Any],
    claim: dict[str, Any],
    envelope: Any,
    context_endpoint: McpEndpoint,
    verified_carrier: Any,
) -> WorkerRun:
    """Refuse any missing or mismatched admission and dispatch binding."""

    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("WorkItem has no bound metadata")
    run_id = _required(metadata.get("run_id"), "run_id")
    agent_name = _required(metadata.get("agent_name"), "agent_name")
    if run_id != envelope.job_id or agent_name != envelope.agent_name:
        raise ValueError("WorkItem and signed dispatch disagree")
    item_id = _required(row.get("id"), "id")
    if claim.get("work_item_id") != item_id:
        raise ValueError("WorkItem claim does not match the admitted item")
    carrier_bytes = verified_carrier.model_dump_json().encode("utf-8")
    carrier_digest = hashlib.sha256(carrier_bytes).hexdigest()
    binding = DelegationBinding(
        delegation_id=_required(metadata.get("delegation_id"), "delegation_id"),
        delegator_id=_required(metadata.get("delegator_id"), "delegator_id"),
        selected_agent_id=_required(metadata.get("agent_id"), "agent_id"),
        capability_digest=_required(
            metadata.get("capability_digest"), "capability_digest"
        ),
        catalog_digest=_required(row.get("catalog_digest"), "catalog_digest"),
        policy_digest=_required(row.get("policy_digest"), "policy_digest"),
        model_digest=_required(row.get("model_digest"), "model_digest"),
        carrier_digest=carrier_digest,
    )
    if binding.delegation_id != run_id:
        raise ValueError("WorkItem delegation and run IDs disagree")
    lease = LeaseClaim(
        tenant=_required(row.get("tenant"), "tenant"),
        work_item_id=item_id,
        worker_id=_required(claim.get("lease_owner"), "lease_owner"),
        lease_epoch=int(claim["lease_epoch"]),
        fencing_token=int(claim["fencing_token"]),
    )
    spec = RunSpec(
        run_id=run_id,
        task=_required(row.get("description"), "description"),
        agent_ref=agent_name,
        toolset=RunToolset(
            context_endpoint=context_endpoint,
            allowed_tools=envelope.allowed_tools,
        ),
        budget=_deadline_budget(row.get("deadline_unix")),
        policy_ref=f"policy:{binding.policy_digest}",
        authorization_ref=f"carrier:{carrier_digest}",
        side_effects="irreversible",
        runtime_options={"session_id": envelope.session_id},
    )
    return WorkerRun(spec=spec, binding=binding, claim=lease)


async def run_worker_harness(
    worker_run: WorkerRun, runner: Any, writer: RunOutcomeWriter
) -> tuple[RunOutcome, L5Receipt]:
    """Execute under one trace owner and commit the terminal EG extension."""

    harness = PydanticAiHarness(runner)
    with l5_terminal_trace_owner():
        outcome = await run_to_completion(
            harness,
            worker_run.spec,
            policy=HarnessPolicy(),
            lease=None,
        )
    receipt = await writer.commit(
        worker_run.claim,
        worker_run.binding,
        outcome.result,
        outcome.trace,
    )
    return outcome, receipt
