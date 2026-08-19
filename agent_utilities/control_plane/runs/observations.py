"""Monotonic, non-authoritative relational observations."""

from __future__ import annotations

import threading
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from agent_utilities.control_plane.policy.models import canonical_digest

__all__ = [
    "InMemoryObservationLedger",
    "ObservationDiscontinuityError",
    "RelationalObservation",
]


Digest: TypeAlias = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
StableId: TypeAlias = Annotated[
    str, Field(pattern=r"^[A-Za-z][A-Za-z0-9:_./-]{0,127}$", min_length=1)
]
OpaqueRef: TypeAlias = Annotated[
    str, Field(pattern=r"^[A-Za-z][A-Za-z0-9:_./-]{0,255}$", min_length=1)
]
ObservedState = Literal["unknown", "discovered", "pending", "blocked", "drifted"]


class ObservationDiscontinuityError(ValueError):
    """A relational observation regressed or changed at an existing sequence."""


class RelationalObservation(BaseModel):
    """Monotonic projection evidence, never an execution authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    observation_id: StableId
    run_id: StableId
    work_item_id: StableId
    sequence: int = Field(ge=0)
    observed_state: ObservedState
    source_ref: OpaqueRef
    evidence_digest: Digest
    observed_at: int = Field(ge=0)

    @property
    def observation_digest(self) -> str:
        return canonical_digest(self)


class InMemoryObservationLedger:
    """Append-only observations keyed by native WorkItem identity."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._rows: dict[str, list[RelationalObservation]] = {}
        self._observations: dict[str, RelationalObservation] = {}

    def append(self, observation: RelationalObservation) -> bool:
        with self._lock:
            prior_identity = self._observations.get(observation.observation_id)
            if prior_identity is not None and prior_identity != observation:
                raise ObservationDiscontinuityError("observation_identity_drift")
            rows = self._rows.setdefault(observation.work_item_id, [])
            if rows:
                prior = rows[-1]
                if observation.run_id != prior.run_id:
                    raise ObservationDiscontinuityError("observation_run_identity_drift")
                if observation.sequence < prior.sequence:
                    raise ObservationDiscontinuityError("observation_sequence_regressed")
                if observation.observed_at < prior.observed_at:
                    raise ObservationDiscontinuityError("observation_time_regressed")
                if observation.sequence == prior.sequence:
                    if observation == prior:
                        return False
                    raise ObservationDiscontinuityError("observation_sequence_drift")
            rows.append(observation)
            self._observations[observation.observation_id] = observation
            return True

    def read(self, work_item_id: str) -> tuple[RelationalObservation, ...]:
        with self._lock:
            return tuple(self._rows.get(work_item_id, ()))
