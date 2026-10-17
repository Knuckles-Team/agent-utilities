"""Provenance sink port for the developer workspace.

The workspace emits every action and observation of one run to a provenance
sink. The harness does not store those records: the host supplies the sink
(graph-os binds one that writes trace records into epistemic-graph). Without a
host sink the null mirror drops the events.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .events import Action, Observation


@runtime_checkable
class ProvenanceSink(Protocol):
    """Receives each workspace action and its observation, in step order."""

    def mirror_action(self, action: Action) -> None: ...

    def mirror_observation(
        self, action: Action, observation: Observation, root: Path | str
    ) -> None: ...


class ProvenanceMirror:
    """Null provenance sink: accepts every event and records nothing."""

    def mirror_action(self, action: Action) -> None:
        return None

    def mirror_observation(
        self, action: Action, observation: Observation, root: Path | str
    ) -> None:
        return None


__all__ = ["ProvenanceMirror", "ProvenanceSink"]
