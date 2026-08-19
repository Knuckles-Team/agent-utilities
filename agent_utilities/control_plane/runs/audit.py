"""Bounded hash-chained audit records for policy/run admission."""

from __future__ import annotations

import threading
from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from agent_utilities.control_plane.policy.models import canonical_digest

__all__ = ["AuditChain", "AuditDiscontinuityError", "AuditEvent"]


Digest: TypeAlias = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
StableId: TypeAlias = Annotated[
    str, Field(pattern=r"^[A-Za-z][A-Za-z0-9:_./-]{0,127}$", min_length=1)
]
OpaqueRef: TypeAlias = Annotated[
    str, Field(pattern=r"^[A-Za-z][A-Za-z0-9:_./-]{0,255}$", min_length=1)
]

GENESIS_HASH = canonical_digest("control-plane-audit-genesis-v1")


class AuditDiscontinuityError(ValueError):
    """The chain is tampered, truncated without an anchor, or over capacity."""


class AuditEvent(BaseModel):
    """One material-free chain entry."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sequence: int = Field(ge=0)
    event_id: StableId
    kind: StableId
    subject_ref: OpaqueRef
    payload_digest: Digest
    previous_hash: Digest
    chain_hash: Digest
    observed_at: int = Field(ge=0)


class AuditChain:
    """Append-only bounded chain that refuses discontinuity and overflow."""

    def __init__(self, *, max_events: int = 256) -> None:
        if max_events < 1 or max_events > 4096:
            raise ValueError("audit_capacity_invalid")
        self.max_events = max_events
        self._lock = threading.RLock()
        self._events: list[AuditEvent] = []

    def can_append(self, count: int = 1) -> bool:
        with self._lock:
            return count >= 0 and len(self._events) + count <= self.max_events

    def append(
        self,
        *,
        kind: str,
        subject_ref: str,
        payload_digest: str,
        observed_at: int,
    ) -> AuditEvent:
        with self._lock:
            if not self.verify():
                raise AuditDiscontinuityError("audit_chain_discontinuous")
            if not self.can_append():
                raise AuditDiscontinuityError("audit_capacity_exceeded")
            sequence = len(self._events)
            previous_hash = self._events[-1].chain_hash if self._events else GENESIS_HASH
            chain_hash = canonical_digest(
                {
                    "sequence": sequence,
                    "kind": kind,
                    "subject_ref": subject_ref,
                    "payload_digest": payload_digest,
                    "previous_hash": previous_hash,
                    "observed_at": observed_at,
                }
            )
            event = AuditEvent(
                sequence=sequence,
                event_id=f"audit:{chain_hash.removeprefix('sha256:')}",
                kind=kind,
                subject_ref=subject_ref,
                payload_digest=payload_digest,
                previous_hash=previous_hash,
                chain_hash=chain_hash,
                observed_at=observed_at,
            )
            self._events.append(event)
            return event

    def snapshot(self) -> tuple[AuditEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def verify(self) -> bool:
        with self._lock:
            previous = GENESIS_HASH
            for expected_sequence, event in enumerate(self._events):
                if event.sequence != expected_sequence or event.previous_hash != previous:
                    return False
                expected_hash = canonical_digest(
                    {
                        "sequence": event.sequence,
                        "kind": event.kind,
                        "subject_ref": event.subject_ref,
                        "payload_digest": event.payload_digest,
                        "previous_hash": event.previous_hash,
                        "observed_at": event.observed_at,
                    }
                )
                if event.chain_hash != expected_hash:
                    return False
                if event.event_id != f"audit:{event.chain_hash.removeprefix('sha256:')}":
                    return False
                previous = event.chain_hash
            return True

    def require_valid(self) -> None:
        if not self.verify():
            raise AuditDiscontinuityError("audit_chain_discontinuous")
