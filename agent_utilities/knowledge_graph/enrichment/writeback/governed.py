"""Execute every write-back sink through the SDK governed write-back contract.

AU-BOUNDARY-R049: ``run_writeback`` no longer calls a sink directly. Each
invocation becomes one canonical :class:`SourceChangeSet` and runs through
:class:`~agent_connector_sdk.writeback.connector.DurableWritableConnector`.
The SDK then validates the digest, expiry, field scope, base version and
authorization. It reserves an audit record before the source call and
appends the attempt to a :class:`WriteBackLedger`.

The change set carries the sink's JSON-safe operations as one ``ops`` field.
An AU sink applies a batch, not one entity field, and it has no source
read-back. The source version therefore comes from the caller's
``expected_source_version``/``current_source_version`` signals, or it is
``unversioned``. A sink failure during apply is recorded as an uncertain
outcome, and the original error is raised to the caller unchanged.

A dry run registers its change set in a throwaway ledger because it records
no effect. A live apply uses a durable file ledger under the AU runtime
directory, or an injected ledger root.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_connector_sdk.writeback.connector import DurableWritableConnector
from agent_connector_sdk.writeback.durable_ledger import FileWriteBackLedger
from agent_connector_sdk.writeback.errors import OutcomeUncertainError
from agent_connector_sdk.writeback.models import DryRunObservation, SourceSnapshot
from epistemic_graph.generated.write_back import (
    ReconciliationObservation,
    SourceChangeSet,
    WriteBackAttempt,
    WriteBackAttemptKind,
    WriteBackAuthorizationDecision,
    WriteBackAuthorizationMode,
    WriteBackEffectStatus,
    WriteBackOutcome,
)

from agent_utilities.core.event_loop import run_sync_isolated
from agent_utilities.core.paths import runtime_dir

from .core import PROVENANCE_TAG, WritebackContext, WritebackResult, WritebackSink

__all__ = [
    "GovernedOutcome",
    "SinkGrant",
    "SinkWriteBackTransport",
    "build_change_set",
    "execute_governed",
    "ledger_root",
]

#: SDK connector id prefix for AU write-back sinks.
CONNECTOR_PREFIX = "agent-utilities.writeback"
#: the single change-set field that carries a sink's operations.
PATCH_FIELD = "ops"
#: source version used when the caller supplies no version signal.
UNVERSIONED = "unversioned"
_TENANT = "agent-utilities"
_ACTOR = "agent-utilities:graph_writeback"
_TTL_MS = 300_000
_SCHEMA_VERSION = 1


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _json_ops(ops: dict[str, Any]) -> dict[str, Any]:
    """Project ``ops`` to canonical JSON, dropping private keys and clients."""
    public = {
        key: value
        for key, value in ops.items()
        if not key.startswith("_") and key != "client"
    }
    return json.loads(json.dumps(public, sort_keys=True, default=str))


@dataclass(frozen=True)
class SinkGrant:
    """The fail-closed write grant that ``run_writeback`` already resolved."""

    target: str
    enable_flag: str
    write_enabled: bool
    risk_tier: str = "standard"
    approved: bool = False

    def mode(self) -> WriteBackAuthorizationMode:
        """Approved proposals replay under approval; others under standing policy."""
        if self.approved:
            return WriteBackAuthorizationMode.PROPOSAL_APPROVAL
        return WriteBackAuthorizationMode.STANDING_POLICY

    def policy_digest(self) -> str:
        """Digest the policy inputs the grant was decided from."""
        return _digest(
            {
                "enable_flag": self.enable_flag,
                "risk_tier": self.risk_tier,
                "write_enabled": self.write_enabled,
                "approved": self.approved,
            }
        )

    def decision(self, input_digest: str) -> WriteBackAuthorizationDecision:
        """Return the exact authorization decision for one input digest."""
        output_digest = _digest({"target": self.target, "input": input_digest})
        fields = {
            "mode": self.mode().value,
            "authorization_ref": self.enable_flag,
            "authorized": self.write_enabled,
            "input_digest": input_digest,
            "output_digest": output_digest,
            "policy_digest": self.policy_digest(),
        }
        return WriteBackAuthorizationDecision(
            mode=self.mode(),
            authorization_ref=self.enable_flag,
            authorized=self.write_enabled,
            decision_digest=_digest(fields),
            input_digest=input_digest,
            output_digest=output_digest,
        )


def _base_version(ops: dict[str, Any]) -> str:
    return str(ops.get("expected_source_version") or UNVERSIONED)


def build_change_set(
    grant: SinkGrant, ops: dict[str, Any], *, now_ms: int | None = None
) -> SourceChangeSet:
    """Build the canonical SDK change set for one sink invocation."""
    patch = {PATCH_FIELD: _json_ops(ops)}
    input_digest = _digest(patch)
    change_set_id = uuid.uuid4().hex
    issued_ms = time.time_ns() // 1_000_000 if now_ms is None else now_ms
    draft = SourceChangeSet(
        actor=_ACTOR,
        authorization=grant.decision(input_digest),
        base_source_version=_base_version(ops),
        change_set_digest="0" * 64,
        change_set_id=change_set_id,
        connector_id=f"{CONNECTOR_PREFIX}.{grant.target}",
        desired_patch=patch,
        entity_id=f"{grant.target}:batch:{input_digest[:16]}",
        expires_at_ms=issued_ms + _TTL_MS,
        field_provenance={PATCH_FIELD: PROVENANCE_TAG},
        field_scope=[PATCH_FIELD],
        idempotency_key=f"{grant.target}:{change_set_id}",
        policy_digest=grant.policy_digest(),
        purpose="graph_writeback",
        reconciliation_procedure="operator-reconciles: sink has no read-back",
        required_capability=f"writeback:{grant.target}",
        schema_version=_SCHEMA_VERSION,
        source_instance_id=grant.target,
        source_of_truth_rule="system-of-record-wins",
        tenant_id=_TENANT,
    )
    return draft.model_copy(update={"change_set_digest": draft.canonical_digest()})


class SinkWriteBackTransport:
    """Adapt one AU :class:`WritebackSink` call to the SDK transport port.

    The sink runs on a worker thread, so a sink that bridges to async code
    never re-enters the SDK's event loop.
    """

    def __init__(
        self, sink: WritebackSink, ctx: WritebackContext, ops: dict[str, Any]
    ) -> None:
        self._sink = sink
        self._ctx = ctx
        self._ops = ops
        self._effect: WriteBackAttempt | None = None
        self.result: WritebackResult | None = None
        self.failure: BaseException | None = None

    async def read_current(self, change_set: SourceChangeSet) -> SourceSnapshot:
        """Report the caller's current version signal; sinks have no read-back."""
        version = self._ops.get("current_source_version")
        return SourceSnapshot(
            source_version=str(version or change_set.base_source_version),
            fields={PATCH_FIELD: None},
        )

    async def preview(
        self, change_set: SourceChangeSet, current: SourceSnapshot
    ) -> DryRunObservation:
        """Run the sink in dry-run mode and bind its proposals to the change set."""
        self.result = await self._run(dry_run=True)
        changed = (PATCH_FIELD,) if self.result.proposals else ()
        return DryRunObservation(
            change_set_digest=change_set.change_set_digest,
            source_version=current.source_version,
            desired_patch_digest=change_set.patch_digest(),
            changed_fields=changed,
            before={PATCH_FIELD: None},
            after=dict(change_set.desired_patch),
        )

    async def prior_effect(
        self, change_set: SourceChangeSet
    ) -> WriteBackAttempt | None:
        """Return this invocation's effect; each invocation is a new change set."""
        return self._effect

    async def apply(
        self, change_set: SourceChangeSet, expected_version: str
    ) -> WriteBackAttempt:
        """Run the sink live; any sink error is an uncertain outcome."""
        try:
            self.result = await self._run(dry_run=False)
        except Exception as exc:  # a partial batch may have applied
            self.failure = exc
            raise OutcomeUncertainError(str(exc)) from exc
        self._effect = self._attempt(change_set, expected_version)
        return self._effect

    async def reconcile(self, change_set: SourceChangeSet) -> ReconciliationObservation:
        """Sinks cannot observe the source, so the outcome stays uncertain."""
        return ReconciliationObservation(
            tenant_id=change_set.tenant_id,
            change_set_id=change_set.change_set_id,
            change_set_digest=change_set.change_set_digest,
            idempotency_key=change_set.idempotency_key,
            observed_source_version=change_set.base_source_version,
            effect_status=WriteBackEffectStatus.OUTCOME_UNCERTAIN,
            retry_allowed=False,
            evidence_digest=_digest({"reason": "sink has no read-back"}),
            connector_observation_digest=_digest({"target": self._sink.domain}),
            provenance_digest=_digest(change_set.field_provenance),
        )

    async def _run(self, *, dry_run: bool) -> WritebackResult:
        return await asyncio.to_thread(
            self._sink.run, self._ctx, self._ops, dry_run=dry_run
        )

    def _attempt(
        self, change_set: SourceChangeSet, expected_version: str
    ) -> WriteBackAttempt:
        authorization = change_set.authorization
        counts = self.result.as_dict() if self.result is not None else {}
        return WriteBackAttempt(
            tenant_id=change_set.tenant_id,
            change_set_id=change_set.change_set_id,
            change_set_digest=change_set.change_set_digest,
            idempotency_key=change_set.idempotency_key,
            kind=WriteBackAttemptKind.APPLY,
            input_digest=authorization.input_digest,
            output_digest=authorization.output_digest,
            pre_source_version=expected_version,
            post_source_version=expected_version,
            applied_field_digest=change_set.patch_digest(),
            outcome=WriteBackOutcome.APPLIED,
            effect_status=WriteBackEffectStatus.APPLIED,
            connector_observation_digest=_digest(counts),
            provenance_digest=_digest(change_set.field_provenance),
        )


class FileAuditReservation:
    """Persist one audit reservation beside the change set's durable ledger."""

    def __init__(self, directory: Path) -> None:
        self._path = directory / "audit_reservation.json"

    async def reserve(self, change_set: SourceChangeSet) -> str:
        """Return the existing reservation, or write a new one atomically."""
        return await asyncio.to_thread(self._reserve, change_set)

    def _reserve(self, change_set: SourceChangeSet) -> str:
        if self._path.exists():
            return str(json.loads(self._path.read_text())["reservation_id"])
        reservation_id = (
            f"audit:{change_set.idempotency_key}:{change_set.change_set_digest}"
        )
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"reservation_id": reservation_id}))
        os.replace(tmp, self._path)
        return reservation_id


@dataclass(frozen=True)
class GovernedOutcome:
    """The sink result plus the SDK change-set evidence for one invocation."""

    result: WritebackResult
    change_set: SourceChangeSet
    attempt: WriteBackAttempt | None = None

    def evidence(self) -> dict[str, Any]:
        """Return the JSON evidence a write-back manifest carries."""
        effect = self.attempt.effect_status.value if self.attempt else None
        return {
            "connector_id": self.change_set.connector_id,
            "change_set_id": self.change_set.change_set_id,
            "change_set_digest": self.change_set.change_set_digest,
            "authorization_mode": self.change_set.authorization.mode.value,
            "effect_status": effect,
        }


def ledger_root() -> Path:
    """Return the durable ledger root for live write-back change sets."""
    return runtime_dir() / "writeback-ledger"


def _connector(
    transport: SinkWriteBackTransport, change_set: SourceChangeSet, directory: Path
) -> DurableWritableConnector:
    return DurableWritableConnector(
        change_set.connector_id,
        transport,
        FileWriteBackLedger(directory),
        audit=FileAuditReservation(directory),
    )


async def _dry_run(
    transport: SinkWriteBackTransport, change_set: SourceChangeSet, directory: Path
) -> None:
    connector = _connector(transport, change_set, directory)
    await connector.register(change_set)
    await connector.dry_run(change_set)


async def _apply(
    transport: SinkWriteBackTransport, change_set: SourceChangeSet, directory: Path
) -> WriteBackAttempt:
    connector = _connector(transport, change_set, directory)
    await connector.register(change_set)
    return await connector.apply(change_set)


def _preview(transport: SinkWriteBackTransport, change_set: SourceChangeSet) -> None:
    with tempfile.TemporaryDirectory(prefix="au-writeback-") as tmp:
        run_sync_isolated(
            lambda: asyncio.run(_dry_run(transport, change_set, Path(tmp)))
        )


def execute_governed(
    sink: WritebackSink,
    ctx: WritebackContext,
    ops: dict[str, Any],
    *,
    grant: SinkGrant,
    dry_run: bool,
    root: Path | None = None,
) -> GovernedOutcome:
    """Run one sink invocation through the SDK governed write-back contract.

    Raises the sink's own error after the SDK records an uncertain outcome.
    SDK refusals (expiry, scope, version, authorization, audit) raise
    :class:`~agent_connector_sdk.writeback.errors.WriteBackError` before any
    live sink call.
    """
    change_set = build_change_set(grant, ops)
    transport = SinkWriteBackTransport(sink, ctx, ops)
    if dry_run:
        _preview(transport, change_set)
        return GovernedOutcome(_result(transport, grant), change_set)
    directory = (root or ledger_root()) / change_set.change_set_id
    attempt = run_sync_isolated(
        lambda: asyncio.run(_apply(transport, change_set, directory))
    )
    if transport.failure is not None:
        raise transport.failure
    return GovernedOutcome(_result(transport, grant), change_set, attempt)


def _result(transport: SinkWriteBackTransport, grant: SinkGrant) -> WritebackResult:
    if transport.result is None:
        return WritebackResult(target=grant.target)
    return transport.result
