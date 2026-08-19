"""One-run/one-native-WorkItem admission protocol."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from agent_utilities.control_plane.policy.models import canonical_digest

from .audit import AuditChain, AuditDiscontinuityError
from .models import (
    AdmissionReceipt,
    NativeAdmissionRequest,
    RunRecord,
)

__all__ = [
    "InMemoryNativeAdmission",
    "NativeAdmissionError",
    "NativeWorkItemAdmissionProtocol",
    "ResolvedAuthorizationVerifier",
    "ReplayDriftError",
]


class NativeAdmissionError(ValueError):
    """The native WorkItem admission precondition failed."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(code if not detail else f"{code}: {detail}")


class ReplayDriftError(NativeAdmissionError):
    """A duplicate identity arrived with a different body or resolution."""


@runtime_checkable
class NativeWorkItemAdmissionProtocol(Protocol):
    """The only execution-plane seam exposed by this domain core."""

    def admit_once(self, request: NativeAdmissionRequest) -> AdmissionReceipt:
        """Atomically admit one resolved run and one native WorkItem."""


class ResolvedAuthorizationVerifier(Protocol):
    def verify_authorization(
        self, authorization: object, *, now: int | None = None
    ) -> None:
        """Verify the exact persisted policy authorization."""


class InMemoryNativeAdmission:
    """Reference implementation with atomic duplicate/concurrency behavior.

    It stores the exact resolved version and never re-resolves a policy during
    duplicate delivery or recovery.  It deliberately exposes no claim, lease,
    fencing, completion, or result methods; the native engine owns those.
    """

    def __init__(
        self,
        *,
        audit: AuditChain | None = None,
        clock: Callable[[], int] | None = None,
        authorization_verifier: ResolvedAuthorizationVerifier | None = None,
    ) -> None:
        self.audit = audit
        self.clock = clock or (lambda: 0)
        self.authorization_verifier = authorization_verifier
        self._lock = threading.RLock()
        self._records: dict[str, RunRecord] = {}
        self._work_items: dict[str, str] = {}

    def admit_once(self, request: NativeAdmissionRequest) -> AdmissionReceipt:
        with self._lock:
            run_id = request.resolution.run_id
            work_item_id = request.work_item.work_item_id
            existing = self._records.get(run_id)
            if existing is not None:
                if (
                    existing.resolution != request.resolution
                    or existing.work_item != request.work_item
                ):
                    raise ReplayDriftError("replay_or_body_drift")
                return self._receipt(existing, created=False)

            owner = self._work_items.get(work_item_id)
            if owner is not None:
                raise NativeAdmissionError("work_item_identity_conflict")
            if self.authorization_verifier is not None:
                try:
                    self.authorization_verifier.verify_authorization(
                        request.resolution.authorization,
                        now=self.clock(),
                    )
                except Exception as exc:  # fail closed at the admission boundary
                    raise NativeAdmissionError("authorization_stale_or_invalid") from exc
            if self.audit is not None:
                try:
                    self.audit.require_valid()
                except AuditDiscontinuityError as exc:
                    raise NativeAdmissionError("audit_discontinuity") from exc
                if not self.audit.can_append(2):
                    raise NativeAdmissionError("audit_capacity_exceeded")

            audit_ids: list[str] = []
            now = self.clock()
            if self.audit is not None:
                run_event = self.audit.append(
                    kind="run.admitted",
                    subject_ref=run_id,
                    payload_digest=request.resolution.resolution_digest,
                    observed_at=now,
                )
                item_event = self.audit.append(
                    kind="work_item.admitted",
                    subject_ref=work_item_id,
                    payload_digest=canonical_digest(request.work_item),
                    observed_at=now,
                )
                audit_ids.extend((run_event.event_id, item_event.event_id))

            record = RunRecord(
                resolution=request.resolution,
                work_item=request.work_item,
                audit_event_ids=tuple(audit_ids),
            )
            self._records[run_id] = record
            self._work_items[work_item_id] = run_id
            return self._receipt(record, created=True)

    def read(self, run_id: str) -> RunRecord | None:
        """Return the exact persisted resolution for crash/reclaim recovery."""

        with self._lock:
            return self._records.get(run_id)

    def _receipt(self, record: RunRecord, *, created: bool) -> AdmissionReceipt:
        return AdmissionReceipt(
            run_id=record.resolution.run_id,
            work_item_id=record.work_item.work_item_id,
            resolution_digest=record.resolution.resolution_digest,
            work_item_digest=canonical_digest(record.work_item),
            created=created,
        )
