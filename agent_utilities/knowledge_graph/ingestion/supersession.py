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
* **The supersession edge** — ``SUPERSEDES``, the same
  edge type :func:`~..assimilation.dedup.dedup_features` already writes
  (survivor -> duplicate) to keep a dedup'd node inspectable. EG derives the
  identical edge shape (``_rel`` mirrored into properties for backend-portable
  traversal) for the retraction case.

So "retire this fact" is one native ApplyChangeEnvelope transaction carrying
the tombstone and an optional ``superseded_by -> old`` evidence edge. A failed
edge or tombstone leaves neither committed.
"""

from typing import Any

from epistemic_graph.ingestion.supersession_derivation import supersession_material

from .change_envelope import ChangeEnvelope

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
    lifecycle_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retire ``entity_id`` through the native tombstone and edge transaction.

    Returns the native result and whether the optional edge was committed.
    A failed transaction is reported under ``tombstone.status``.
    """
    from .envelope_ingest import ingest_envelope

    evidence_id = superseded_by_id or retracted_by_claim
    source_version, sidecar = supersession_material(
        entity_id, evidence_id, reason, lifecycle_event=lifecycle_event
    )
    tombstone = ChangeEnvelope(
        connector=connector,
        operation="delete",
        tenant=tenant,
        source_object_id=entity_id,
        source_version=source_version,
        typed_payload=sidecar,
        provenance={
            "retirement_reason": reason,
            **(
                {"retracted_by_claim": retracted_by_claim} if retracted_by_claim else {}
            ),
        },
    )
    result = ingest_envelope(engine, tombstone)

    return {
        "entity_id": entity_id,
        "tombstone": result,
        "evidence_linked": bool(evidence_id)
        and result.get("status") in {"success", "skipped"},
        "lifecycle_event_committed": lifecycle_event is not None
        and result.get("status") in {"success", "skipped"},
    }
