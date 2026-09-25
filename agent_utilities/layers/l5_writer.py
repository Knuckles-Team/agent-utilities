"""The single L5 reconciliation writer (RF-ADR-010 §8).

Exactly one write authority for run outcomes and traces: a harness run's
terminal result, its RunTrace/ToolCall/OutcomeEvaluation receipts and its one
RunEvent are committed **atomically with the WorkItem terminal transition**,
under the claiming worker's lease fence, through EG's
``work_items.commit_result(outcome_extension=...)`` (GOC-20
``TerminalOutcomeExtension``). There is no second trace writer and no
fallback: when the connected EG cannot take the extension the writer fails
closed with :class:`L5WriterUnavailable`.

Rules carried from the ADR:

* an ``outcome_uncertain`` run is committed as a **non-retryable failure**
  marked ``degraded`` with ``reconciliation_receipt`` missing, so nothing
  retries or fails over automatically;
* an incomplete trace is never committed as ``complete``;
* the engine fills every ``outbox_id`` and receipt ``payload_digest``
  (EG-4); AU never computes EG digests;
* a transport failure during the commit is reconciled by reading the
  committed outcome back (:meth:`RunOutcomeWriter.reconcile`) before anything
  is retried.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_utilities.layers.contracts import RunEvent, RunResult, RunStatus

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

#: Harness status -> EG terminal outcome.
_TERMINAL_OUTCOME: dict[RunStatus, str] = {
    "succeeded": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
    "outcome_uncertain": "failed",
}
#: Longest redacted output kept in the OutcomeEvaluation receipt.
MAX_OUTCOME_OUTPUT_CHARS = 64_000
#: Longest tool-call detail kept per ToolCall receipt.
MAX_TOOL_DETAIL_CHARS = 2_000

L5Status = Literal[
    "committed", "replayed", "fenced", "missing", "conflict", "uncertain"
]


class L5WriterUnavailable(RuntimeError):
    """The connected EG does not serve the terminal outcome extension."""


class L5CommitRejected(RuntimeError):
    """EG rejected the terminal commit (fence, binding or validation)."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DelegationBinding(_Frozen):
    """Identities the WorkItem was admitted under; EG checks every one."""

    delegation_id: str = Field(min_length=1, max_length=512)
    #: The authenticated delegator (the admission ``context.agent_id``).
    delegator_id: str = Field(min_length=1, max_length=512)
    #: The selected agent (``metadata.agent_id`` at admission).
    selected_agent_id: str = Field(min_length=1, max_length=512)
    capability_digest: Sha256Hex
    catalog_digest: Sha256Hex
    policy_digest: Sha256Hex
    model_digest: Sha256Hex
    #: Digest of the verified carrier/session the run executed under.
    carrier_digest: Sha256Hex


class LeaseClaim(_Frozen):
    """The claiming worker's fenced lease on the WorkItem."""

    tenant: str = Field(min_length=1, max_length=512)
    work_item_id: str = Field(min_length=1, max_length=512)
    worker_id: str = Field(min_length=1, max_length=512)
    lease_epoch: int = Field(ge=0)
    fencing_token: int = Field(ge=1)


@dataclass(frozen=True, slots=True)
class L5Receipt:
    """What the single writer established about one terminal commit."""

    status: L5Status
    outcome: str
    completeness: str
    trace_ref: str
    outcome_ref: str
    tool_call_refs: tuple[str, ...]
    reconciled: bool = False


def delegation_metadata(binding: DelegationBinding, run_id: str) -> dict[str, str]:
    """WorkItem admission metadata the terminal extension is later bound to."""
    return {
        "delegation_id": binding.delegation_id,
        "run_id": run_id,
        "agent_id": binding.selected_agent_id,
        "capability_digest": binding.capability_digest,
    }


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _node_ref(kind: str, run_id: str, suffix: str = "") -> str:
    return f"au-{kind}-{_sha(run_id)[:40]}{suffix}"


@dataclass(frozen=True, slots=True)
class _Plan:
    """Everything that is identical across the bundle, receipts and event."""

    claim: LeaseClaim
    binding: DelegationBinding
    result: RunResult
    outcome: str
    completeness: str
    missing_refs: tuple[str, ...]
    result_ref: str | None
    result_digest: str | None
    trace_ref: str
    outcome_ref: str
    tool_calls: tuple[RunEvent, ...]
    event_sequence: int

    def tool_refs(self) -> tuple[str, ...]:
        return tuple(
            _node_ref("toolcall", self.result.run_id, f"-{event.seq}")
            for event in self.tool_calls
        )


def _redacted(text: str) -> str:
    from agent_utilities.orchestration.task_guard import redact_agent_task

    return redact_agent_task(text)[:MAX_OUTCOME_OUTPUT_CHARS] if text else ""


def _missing_refs(result: RunResult) -> tuple[str, ...]:
    missing: list[str] = []
    if result.status != "succeeded":
        missing.append("result")
    if result.trace != "complete":
        missing.append(f"trace_after_seq:{result.high_watermark}")
    if result.status == "outcome_uncertain":
        missing.append("reconciliation_receipt")
    return tuple(missing)


def plan_commit(
    claim: LeaseClaim,
    binding: DelegationBinding,
    *,
    result: RunResult,
    trace: tuple[RunEvent, ...],
) -> _Plan:
    """Derive the terminal identities for one run (pure)."""
    from agent_utilities.observability.trace_ontology import next_event_sequence

    missing = _missing_refs(result)
    succeeded = result.status == "succeeded"
    output = _redacted(result.output) if succeeded else ""
    return _Plan(
        claim=claim,
        binding=binding,
        result=result,
        outcome=_TERMINAL_OUTCOME[result.status],
        completeness="degraded" if missing else "complete",
        missing_refs=missing,
        result_ref=_node_ref("result", result.run_id) if succeeded else None,
        result_digest=_sha(output) if succeeded else None,
        trace_ref=_node_ref("trace", result.run_id),
        outcome_ref=_node_ref("outcome", result.run_id),
        tool_calls=tuple(event for event in trace if event.kind == "tool_call"),
        event_sequence=next_event_sequence(),
    )


def _identity(plan: _Plan) -> dict[str, Any]:
    binding = plan.binding
    return {
        "delegation_id": binding.delegation_id,
        "delegator_id": binding.delegator_id,
        "selected_agent_id": binding.selected_agent_id,
        "executor_lease_actor": plan.claim.worker_id,
        "outcome": plan.outcome,
        "work_item_id": plan.claim.work_item_id,
        "run_id": plan.result.run_id,
        "outbox_id": "",
        "capability_digest": binding.capability_digest,
        "catalog_digest": binding.catalog_digest,
        "policy_digest": binding.policy_digest,
        "model_digest": binding.model_digest,
    }


def _receipt_properties(plan: _Plan, node_id: str, kind: str) -> dict[str, Any]:
    return {
        **_identity(plan),
        "node_id": node_id,
        "kind": kind,
        "payload_ref": f"au-payload-{node_id}",
        "fence_token": plan.claim.fencing_token,
        "result_ref": plan.result_ref,
        "result_digest": plan.result_digest,
        "event_sequence": plan.event_sequence,
        "completeness": plan.completeness,
        "missing_refs": list(plan.missing_refs),
        "harness": plan.result.harness,
        "spec_digest": plan.result.spec_digest,
        "fidelity": plan.result.fidelity,
        "environment": plan.result.environment,
    }


def _trace_body(plan: _Plan) -> dict[str, Any]:
    result = plan.result
    return {
        "trace": result.trace,
        "high_watermark": result.high_watermark,
        "gap_reason": result.gap_reason,
        "provider_session": result.provider_session,
        "tool_call_count": len(plan.tool_calls),
    }


def _outcome_body(plan: _Plan) -> dict[str, Any]:
    result = plan.result
    output = _redacted(result.output) if result.status == "succeeded" else ""
    return {
        "status": result.status,
        "output": output,
        "error_kind": result.error_kind,
        "error": result.error[:MAX_TOOL_DETAIL_CHARS],
        "usage": result.usage.model_dump(mode="json"),
        "evidence": "claim",
    }


def _tool_body(event: RunEvent) -> dict[str, Any]:
    return {
        "tool_name": event.name,
        "seq": event.seq,
        "evidence": event.evidence,
        "detail": event.detail[:MAX_TOOL_DETAIL_CHARS],
    }


def _receipt(
    plan: _Plan, encoder: Any, node_id: str, *, kind: str, body: dict[str, Any]
) -> dict[str, Any]:
    properties = {**_receipt_properties(plan, node_id, kind), "body": body}
    return {
        "node_id": node_id,
        "kind": kind,
        "delegation_id": plan.binding.delegation_id,
        "work_item_id": plan.claim.work_item_id,
        "run_id": plan.result.run_id,
        "fence_token": plan.claim.fencing_token,
        "result_ref": plan.result_ref,
        "outbox_id": "",
        "payload_ref": f"au-payload-{node_id}",
        "payload_digest": "",
        "properties_msgpack": encoder(properties),
    }


def _run_event(plan: _Plan, outcome_body: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(outcome_body, sort_keys=True, separators=(",", ":"))
    return {
        **_identity(plan),
        "schema_version": 1,
        "fence_token": plan.claim.fencing_token,
        "result_ref": plan.result_ref,
        "event_sequence": plan.event_sequence,
        "completeness": plan.completeness,
        "missing_refs": list(plan.missing_refs),
        "kind": "outcome" if plan.completeness == "complete" else "degraded",
        "outcome_ref": plan.outcome_ref,
        "payload_digest": _sha(payload),
        "timestamp_ms": int(time.time() * 1000),
        "cursor_token": f"au-cursor-{_sha(plan.result.run_id)[:32]}-"
        f"{plan.event_sequence}",
        "carrier_digest": plan.binding.carrier_digest,
    }


def build_outcome_extension(plan: _Plan, encoder: Any) -> dict[str, Any]:
    """The ``TerminalOutcomeExtension`` for one planned commit."""
    tool_refs = plan.tool_refs()
    outcome_body = _outcome_body(plan)
    receipts = [
        _receipt(
            plan, encoder, plan.trace_ref, kind="run_trace", body=_trace_body(plan)
        ),
        *(
            _receipt(plan, encoder, ref, kind="tool_call", body=_tool_body(event))
            for ref, event in zip(tool_refs, plan.tool_calls, strict=True)
        ),
        _receipt(
            plan,
            encoder,
            plan.outcome_ref,
            kind="outcome_evaluation",
            body=outcome_body,
        ),
    ]
    bundle = {
        **_identity(plan),
        "schema_version": 1,
        "fence_token": plan.claim.fencing_token,
        "result_ref": plan.result_ref,
        "result_digest": plan.result_digest,
        "artifacts": [],
        "trace_ref": plan.trace_ref,
        "tool_call_refs": list(tool_refs),
        "outcome_ref": plan.outcome_ref,
        "event_sequence": plan.event_sequence,
        "completeness": plan.completeness,
        "missing_refs": list(plan.missing_refs),
        "langfuse_observation_refs": [],
    }
    return {
        "outcome_bundle": bundle,
        "receipt_nodes": receipts,
        "run_event": _run_event(plan, outcome_body),
    }


#: EG commit status -> writer status.
_COMMIT_STATUS: dict[str, L5Status] = {
    "committed": "committed",
    "succeeded": "committed",
    "failed": "committed",
    "cancelled": "committed",
    "noop": "replayed",
    "already_committed": "replayed",
    "fenced": "fenced",
    "missing": "missing",
    "conflict": "conflict",
}


class RunOutcomeWriter:
    """The one L5 writer over EG's WorkItem namespace (``client.work_items``)."""

    def __init__(self, work_items: Any) -> None:
        if work_items is None:
            raise L5WriterUnavailable("an EG work_items namespace is required")
        self._work_items = work_items

    @classmethod
    def from_engine(cls, engine: Any) -> RunOutcomeWriter:
        """Use one process EG transport with sync receipt bytes and async RPCs."""

        sync_client = getattr(engine, "client", None)
        async_client = getattr(engine, "async_client", None)
        sync_items = getattr(sync_client, "work_items", None)
        async_items = getattr(async_client, "work_items", None)
        if sync_items is None or async_items is None:
            raise L5WriterUnavailable("engine lacks a paired WorkItem client")
        return cls(_EngineWorkItems(sync_items, async_items))

    def _encoder(self) -> Any:
        encoder = getattr(self._work_items, "receipt_properties", None)
        if not callable(encoder):
            raise L5WriterUnavailable(
                "the connected EG client cannot encode terminal receipts"
            )
        return encoder

    async def commit(
        self,
        claim: LeaseClaim,
        binding: DelegationBinding,
        result: RunResult,
        trace: tuple[RunEvent, ...],
    ) -> L5Receipt:
        """Commit the terminal outcome and its provenance atomically."""
        plan = plan_commit(claim, binding, result=result, trace=trace)
        extension = build_outcome_extension(plan, self._encoder())
        try:
            answer = await self._work_items.commit_result(
                tenant=claim.tenant,
                work_item_id=claim.work_item_id,
                worker_id=claim.worker_id,
                lease_epoch=claim.lease_epoch,
                fencing_token=claim.fencing_token,
                idempotency_key=f"au-l5:{claim.work_item_id}:{claim.lease_epoch}",
                outcome=plan.outcome,
                now_ms=int(time.time() * 1000),
                result_ref=plan.result_ref,
                error_ref=None
                if plan.result_ref
                else _node_ref("error", result.run_id),
                retryable=False,
                outcome_extension=extension,
            )
        except (OSError, TimeoutError) as exc:
            return await self.reconcile(claim, plan, cause=exc)
        except RuntimeError as exc:
            raise L5CommitRejected(str(exc)) from exc
        return _receipt_for(plan, _status_of(answer))

    async def reconcile(
        self, claim: LeaseClaim, plan: _Plan, *, cause: BaseException
    ) -> L5Receipt:
        """Establish whether an interrupted commit landed, by reading it back."""
        reader = getattr(self._work_items, "get_outcome", None)
        if not callable(reader):
            raise L5WriterUnavailable("EG does not serve GetWorkItemOutcome") from cause
        landed = await reader(tenant=claim.tenant, work_item_id=claim.work_item_id)
        if landed is not None and landed.get("outcome_ref") == plan.outcome_ref:
            return _receipt_for(plan, "committed", reconciled=True)
        return _receipt_for(plan, "uncertain", reconciled=True)


class _EngineWorkItems:
    """Keep EG's receipt encoder synchronous without a second connection."""

    def __init__(self, sync_items: Any, async_items: Any) -> None:
        self._sync = sync_items
        self._async = async_items

    def receipt_properties(self, properties: dict[str, Any]) -> bytes:
        encoder = getattr(self._sync, "receipt_properties", None)
        if not callable(encoder):
            raise L5WriterUnavailable("EG receipt encoder is unavailable")
        return encoder(properties)

    async def commit_result(self, **kwargs: Any) -> Any:
        return await self._async.commit_result(**kwargs)

    async def get_outcome(self, **kwargs: Any) -> Any:
        return await self._async.get_outcome(**kwargs)


def _status_of(answer: Any) -> L5Status:
    fields = answer if isinstance(answer, dict) else {}
    raw = str(fields.get("status") or fields.get("result") or "").lower()
    status = _COMMIT_STATUS.get(raw)
    if status is None:
        raise L5CommitRejected(f"EG returned an unknown commit status {raw!r}")
    return status


def _receipt_for(
    plan: _Plan, status: L5Status, *, reconciled: bool = False
) -> L5Receipt:
    return L5Receipt(
        status=status,
        outcome=plan.outcome,
        completeness=plan.completeness,
        trace_ref=plan.trace_ref,
        outcome_ref=plan.outcome_ref,
        tool_call_refs=plan.tool_refs(),
        reconciled=reconciled,
    )


__all__ = [
    "MAX_OUTCOME_OUTPUT_CHARS",
    "DelegationBinding",
    "L5CommitRejected",
    "L5Receipt",
    "L5Status",
    "L5WriterUnavailable",
    "LeaseClaim",
    "RunOutcomeWriter",
    "build_outcome_extension",
    "delegation_metadata",
    "plan_commit",
]
