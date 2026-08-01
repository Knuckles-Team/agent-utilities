"""Bridge from proposed facts to the existing governed promotion path
(CONCEPT:AU-KG.ingest.domain-pack-framework).

This is the ONLY module in the domain-pack framework that may reach the graph
— and even here, it never writes directly: it always goes through
``ingestion.envelope_ingest.ingest_graph_slice``/``ingest_envelope``, the
SAME atomic, SHACL-gated, identity-resolved commit path every other connector
already uses. A domain pack cannot grant itself write authority; the most
this module does is hand the existing gate a batch of proposed facts stamped
with the pack's own ``ontology_mapping_version`` for traceability.

**Dry-run vs write is a single boolean, and dry-run never gets near the
gate**: :func:`run_pack` with ``dry_run=True`` (the default) returns the exact
entity/relationship dicts a write would submit — computed by the same pure
:func:`~.dsl.evaluate_mapping` a preview or a write both start from — without
ever importing or calling ``ingest_graph_slice``. Only ``dry_run=False`` calls
it, and only then does anything reach the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ingestion.evidence_spine import Artifact, Fragment
from .domain_pack import DomainPackManifest
from .dsl import MappingResult, evaluate_mapping

__all__ = ["PackRunPreview", "preview_pack", "run_pack"]


@dataclass(frozen=True)
class PackRunPreview:
    """What :func:`run_pack(dry_run=True)` returns — inspectable, unwritten."""

    pack: str
    version: str
    mapping_version: str
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    unmatched_fragment_ids: list[str]


def preview_pack(
    manifest: DomainPackManifest,
    artifact: Artifact,
    fragments: list[Fragment],
) -> PackRunPreview:
    """Evaluate a pack's mappings and return the facts it WOULD produce.

    Pure — no import of, or call into, the ingest path. This is what a
    ``--dry-run`` CLI/tool action calls.
    """
    result: MappingResult = evaluate_mapping(manifest, artifact, fragments)
    return PackRunPreview(
        pack=manifest.pack,
        version=manifest.version,
        mapping_version=manifest.mapping_version,
        entities=result.entity_dicts(),
        relationships=result.relationship_dicts(),
        unmatched_fragment_ids=result.unmatched_fragment_ids,
    )


def run_pack(
    engine: Any,
    manifest: DomainPackManifest,
    artifact: Artifact,
    fragments: list[Fragment],
    *,
    dry_run: bool = True,
) -> PackRunPreview | dict[str, Any]:
    """Evaluate a pack's mappings; write them ONLY when ``dry_run=False``.

    ``dry_run=True`` (default): identical to :func:`preview_pack` — no ingest
    module is even imported, so a preview cannot accidentally reach the graph.

    ``dry_run=False``: submits the same facts through
    ``ingestion.envelope_ingest.ingest_graph_slice`` — the existing governed,
    atomic, SHACL-validated commit path — tagged with this pack's
    ``ontology_mapping_version`` so every resulting fact traces back to the
    exact pack revision that produced it.
    """
    preview = preview_pack(manifest, artifact, fragments)
    if dry_run:
        return preview

    from ..ingestion.envelope_ingest import ingest_graph_slice

    if not preview.entities and not preview.relationships:
        return {"status": "skipped", "reason": "no facts produced"}

    return ingest_graph_slice(
        engine,
        manifest.connector_id,
        preview.entities,
        preview.relationships,
        source_instance=artifact.artifact_id,
        ontology_mapping_version=manifest.mapping_version,
        classification=manifest.default_classification,
    )
