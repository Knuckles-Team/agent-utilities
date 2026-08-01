#!/usr/bin/python
from __future__ import annotations

"""Fact retraction/supersession preserving history (CONCEPT:AU-KG.ingest.fact-supersession).

The universal-ingestion program's Track B "retraction/supersession must
preserve history" requirement: a superseded fact stays inspectable, with the
evidence that retired it, rather than being deleted.

Assembled from pieces that already exist, not rebuilt:

* **Tombstoning** — a ``ChangeEnvelope(operation="delete")`` through
  :func:`~.envelope_ingest.ingest_envelope` already archives a node
  (``archived=True``, ``archivedReason``) and closes its bitemporal validity
  interval (``_stamp_ambient_valid_until``) WITHOUT deleting it — the node
  stays gettable by id, still carrying every property it ever had. This
  module reuses that exact path rather than a second delete implementation.
* **The supersession edge** — :data:`RegistryEdgeType.SUPERSEDES`, the same
  edge type :func:`~..assimilation.dedup.dedup_features` already writes
  (survivor -> duplicate) to keep a dedup'd node inspectable. This module
  writes the identical edge shape (``_rel`` mirrored into properties for
  backend-portable traversal) for the retraction case.

So "retire this fact" is: tombstone via the existing fail-closed envelope
path (never a raw property mutation or a direct delete), then link
``superseded_by -> old`` with the reason and, when known, the claim whose
retraction caused it — the durable evidence a reviewer follows from the
retired fact back to why it was retired.
"""

import logging
from typing import Any

from ...models.knowledge_graph import RegistryEdgeType
from .change_envelope import ChangeEnvelope

logger = logging.getLogger(__name__)

__all__ = ["retire_fact"]


def retire_fact(
    engine: Any,
    *,
    entity_id: str,
    connector: str,
    reason: str,
    superseded_by_id: str | None = None,
    retracted_by_claim: str | None = None,
    tenant: str = "",
) -> dict[str, Any]:
    """Retire ``entity_id`` WITHOUT deleting it: tombstone through the
    existing fail-closed envelope path, then link the evidence that retired
    it as a ``SUPERSEDES`` edge — so the retired fact remains inspectable.

    Returns the ``ingest_envelope`` tombstone result plus whether the
    evidence edge was written. Never raises: a failed tombstone is reported
    (``status`` != ``"committed"``-shaped success), never silently skipped —
    the caller (e.g. ``promotion.retract_and_supersede``) is responsible for
    surfacing a failure rather than treating it as success.
    """
    from .envelope_ingest import ingest_envelope

    tombstone = ChangeEnvelope(
        connector=connector,
        operation="delete",
        tenant=tenant,
        source_object_id=entity_id,
        provenance={
            "retirement_reason": reason,
            **(
                {"retracted_by_claim": retracted_by_claim} if retracted_by_claim else {}
            ),
        },
    )
    result = ingest_envelope(engine, tombstone)

    linked = False
    evidence_id = superseded_by_id or retracted_by_claim
    if evidence_id:
        try:
            engine.link_nodes(
                evidence_id,
                entity_id,
                RegistryEdgeType.SUPERSEDES,
                properties={
                    "_rel": "SUPERSEDES",
                    "reason": reason,
                    "concept": "AU-KG.ingest.fact-supersession",
                },
            )
            linked = True
        except Exception as exc:  # noqa: BLE001 — the evidence edge is best-effort provenance
            logger.debug(
                "supersession: could not link %s -> %s: %s",
                evidence_id,
                entity_id,
                exc,
                exc_info=True,
            )

    return {
        "entity_id": entity_id,
        "tombstone": result,
        "evidence_linked": linked,
    }
