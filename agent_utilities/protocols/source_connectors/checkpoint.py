from __future__ import annotations

"""Checkpointed incremental poll — resumable connector watermarks.

CONCEPT:AU-ECO.connector.incremental-poll-watermark — Checkpointed Incremental Poll

A :class:`ConnectorCheckpoint` is the opaque, serializable cursor a
:class:`~agent_utilities.protocols.source_connectors.base.PollConnector` returns
from one ``poll`` so the next ``poll`` resumes exactly where it left off — Onyx's
generic checkpoint typing, made concrete.

It round-trips through the existing ``DeltaManifest`` (CONCEPT:EG-KG.storage.nonblocking-checkpoint) **without a
schema change**: the JSON form (:meth:`to_json`) is stored in the manifest's
``content_hash`` column under a dedicated ``connector_checkpoint`` category, keyed
by the connector's source URI. The next ingestion run reads it back with
:meth:`from_json` and hands it to ``poll`` — so an unchanged source costs a delta,
not a full re-scan.
"""

import json
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["ConnectorCheckpoint", "CheckpointedBatch"]


class ConnectorCheckpoint(BaseModel):
    """Resumable cursor for an incremental poll.

    CONCEPT:AU-ECO.connector.incremental-poll-watermark.

    Attributes:
        has_more: True while the source still has pages/batches to drain; the
            drain loop (``PollConnector.poll_all``) stops when this is false.
        cursor: Opaque next-page token (API pagination cursor / continuation).
        watermark: High-water mark for incrementality — an ISO timestamp, a
            monotonic mtime, or any opaque marker the connector compares against
            ``SourceDocument.updated_at`` to emit only changed documents.
        seen_ids: Document ids already emitted in this poll session (dedup across
            overlapping pages). Kept bounded by connectors that page by time.
        state: Free-form connector-private state carried across polls.
    """

    has_more: bool = False
    cursor: str | None = None
    watermark: str | None = None
    seen_ids: list[str] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to the compact JSON stored in ``DeltaManifest`` (KG-2.8)."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str | None) -> ConnectorCheckpoint | None:
        """Rehydrate from the manifest JSON; ``None``/garbage → ``None``."""
        if not raw:
            return None
        try:
            return cls.model_validate(json.loads(raw))
        except (ValueError, TypeError):
            return None


class CheckpointedBatch(BaseModel):
    """One incremental batch: the documents plus the checkpoint to resume from.

    CONCEPT:AU-ECO.connector.incremental-poll-watermark — the return type of ``PollConnector.poll``.
    """

    documents: list[Any] = Field(default_factory=list)
    checkpoint: ConnectorCheckpoint = Field(default_factory=ConnectorCheckpoint)
