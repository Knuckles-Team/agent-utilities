#!/usr/bin/python
from __future__ import annotations

"""Source-to-claim lineage (CONCEPT:AU-KG.retrieval.source-to-claim-lineage).

Track 9 of the universal-ingestion program (evaluation + observability): given
a fact/claim's cited fragment, walk backward through the evidence spine to its
full provenance chain —

    claim  ->  Fragment  ->  Artifact  ->  ChangeEnvelope fields  ->  pack version

— so an operator (or an eval gate) can answer "what backs this, and is it still
true" for any retrieval result, not just trust the citation blindly. This is
the deliverable the charter names directly: "given a fact, show the claim, the
fragment, the artifact, the envelope, and the pack version that produced it."

**Nothing here re-derives evidence.** It reuses the evidence spine's own
readers verbatim:

* :func:`~..ingestion.evidence_spine.load_fragments` — the fragment reader.
* :func:`~..ingestion.evidence_spine.citation_status` — the four-outcome
  citation-drift check (current/moved/stale/lost).
* The stored :class:`~..ingestion.evidence_spine.Artifact` node's OWN
  properties — ``envelope_id``, ``source_version``, ``schema_version``,
  ``ontology_mapping_version`` (the domain-pack revision) — never re-derived
  from the payload, mirroring ``Artifact.from_envelope``'s own rule that
  governance/revision come off the envelope, not the content.

A citation that cannot resolve at all (``citation_status`` reports ``lost``)
raises :class:`LineageNotFoundError` — mandatory evidence citation
(CONCEPT:AU-KG.retrieval.mandatory-evidence-citation) means an unresolvable
citation is a defect to surface, never a silently degraded/empty answer.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["LineageNotFoundError", "LineageRecord", "resolve_lineage"]


class LineageNotFoundError(LookupError):
    """No resolvable evidence trail exists for the given fragment citation.

    Raised when :func:`~..ingestion.evidence_spine.citation_status` reports
    ``lost`` (or the fragment id resolves to no artifact/fragment set at all)
    — never silently collapsed to an empty/default lineage record, so a
    caller can distinguish "there genuinely is no trail" from a legitimate
    one that happens to be thin.
    """


#: The full lineage record shape returned by :func:`resolve_lineage`.
LineageRecord = dict[str, Any]


def _query_cypher_rows(engine: Any, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Best-effort ``engine.query_cypher`` read; ``[]`` on any unavailability.

    Mirrors :func:`~..ingestion.evidence_spine.load_fragments`'s own tolerance:
    an unavailable/uningested graph reads as "nothing found", never a hard
    failure of an otherwise-working lineage walk.
    """
    if engine is None:
        return []
    try:
        run = getattr(engine, "query_cypher", None)
        if not callable(run):
            return []
        return list(run(cypher, params) or [])
    except Exception as exc:  # noqa: BLE001 — advisory read, see docstring
        logger.debug("lineage graph read failed: %s", exc)
        return []


def _artifact_id_from_fragment(engine: Any, fragment_id: str) -> str:
    """Resolve a fragment's owning ``artifact_id`` from its stored node.

    The fragment id is a content-independent hash of ``(artifact_id, path)``
    (:func:`~..ingestion.evidence_spine.fragment_id_for`) — it cannot be
    reverse-derived, so this is a direct lookup of the stored property.
    """
    from ..ingestion.evidence_spine import FRAGMENT_NODE_TYPE

    rows = _query_cypher_rows(
        engine,
        f"MATCH (f:{FRAGMENT_NODE_TYPE}) WHERE f.id = $key RETURN f.artifact_id AS artifact_id",
        {"key": fragment_id},
    )
    for row in rows:
        if isinstance(row, dict) and row.get("artifact_id"):
            return str(row["artifact_id"])
    return ""


def _load_artifact_row(engine: Any, artifact_id: str) -> dict[str, Any]:
    """Best-effort read of one stored :class:`Artifact` node's properties."""
    from ..ingestion.evidence_spine import ARTIFACT_NODE_TYPE

    if not artifact_id:
        return {}
    rows = _query_cypher_rows(
        engine,
        (
            f"MATCH (a:{ARTIFACT_NODE_TYPE}) WHERE a.id = $key RETURN "
            "a.connector AS connector, a.source_instance AS source_instance, "
            "a.source_object_id AS source_object_id, a.content_hash AS content_hash, "
            "a.envelope_id AS envelope_id, a.source_version AS source_version, "
            "a.schema_version AS schema_version, "
            "a.ontology_mapping_version AS ontology_mapping_version, "
            "a.classification AS classification, a.title AS title"
        ),
        {"key": artifact_id},
    )
    for row in rows:
        if isinstance(row, dict):
            return row
    return {}


def _record_citation_metric(status: str) -> None:
    try:
        from agent_utilities.observability.gateway_metrics import (
            RETRIEVAL_CITATION_RESOLUTION,
        )

        RETRIEVAL_CITATION_RESOLUTION.labels(status=status).inc()
    except Exception as exc:  # noqa: BLE001 — metrics must never break the lineage walk
        logger.debug("citation-resolution metric emission failed: %s", exc)


def resolve_lineage(
    engine: Any,
    *,
    fragment_id: str,
    claim_text: str = "",
    content_hash: str = "",
) -> LineageRecord:
    """Walk ``claim -> Fragment -> Artifact -> envelope/pack-version`` for one citation.

    Args:
        engine: The live engine (anything exposing ``query_cypher``); ``None``
            or an unreachable graph degrades every read to "nothing found"
            (never raises an infrastructure error) — this mirrors
            ``load_fragments``'s own contract.
        fragment_id: The cited fragment's stable address — the mandatory
            evidence citation every retrieval result must carry.
        claim_text: The claim/fact text this lineage is FOR. Carried through
            verbatim, never re-derived — this module resolves evidence for a
            claim, it does not extract or validate the claim itself (that is
            track 4, candidate-claim extraction).
        content_hash: The content hash the citation was originally made
            against, if known — sharpens ``citation_status``'s moved/stale
            distinction (a fragment at the same address with DIFFERENT
            content is ``stale``, not silently reported as ``current``).

    Returns:
        A lineage record with ``claim``, ``fragment`` (address/content_hash/
        text/citation_status), ``artifact`` (id/connector/source_instance/
        content_hash/title), ``envelope`` (envelope_id/source_version/
        schema_version), and ``pack_version`` (``ontology_mapping_version``).

    Raises:
        ValueError: ``fragment_id`` is empty — there is nothing to resolve.
        LineageNotFoundError: the citation resolves to NO evidence at all
            (``citation_status`` reports ``lost``).
    """
    if not fragment_id:
        raise ValueError("resolve_lineage requires a fragment_id")

    from ..ingestion.evidence_spine import citation_status, load_fragments

    artifact_id = _artifact_id_from_fragment(engine, fragment_id)
    fragments = load_fragments(engine, artifact_id=artifact_id) if artifact_id else ()
    status = citation_status(
        fragments, fragment_id=fragment_id, content_hash=content_hash
    )
    _record_citation_metric(status["status"])
    if status["status"] == "lost":
        raise LineageNotFoundError(
            f"fragment {fragment_id!r} resolves to no evidence — the citation is lost"
        )

    artifact_row = _load_artifact_row(engine, artifact_id)
    return {
        "claim": claim_text,
        "fragment": {
            "fragment_id": status["fragment_id"],
            "address": status["address"],
            "content_hash": status["content_hash"],
            "text": status["text"],
            "citation_status": status["status"],
        },
        "artifact": {
            "artifact_id": artifact_id,
            "connector": artifact_row.get("connector", ""),
            "source_instance": artifact_row.get("source_instance", ""),
            "source_object_id": artifact_row.get("source_object_id", ""),
            "content_hash": artifact_row.get("content_hash", ""),
            "title": artifact_row.get("title", ""),
        },
        "envelope": {
            "envelope_id": artifact_row.get("envelope_id", ""),
            "source_version": artifact_row.get("source_version", ""),
            "schema_version": artifact_row.get("schema_version", ""),
        },
        "pack_version": artifact_row.get("ontology_mapping_version", ""),
    }
