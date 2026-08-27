"""Ontology-pack pipeline (CONCEPT:AU-KG.ontology.pack-pipeline) load->compile->gate->apply->publish->verify orchestration — CA-23.

Composes the pipeline stages ``DEC-CA-06`` names — load -> compile -> SHACL gate ->
apply -> publish -> verify -- out of components that already exist on ``main``:
:func:`manifest_compiler.compile_manifest`/``export_manifest_ttl``/``apply_manifest``,
the advisory :func:`pipeline.phases.shacl_gate.validate_graph`, and
:meth:`backends.sparql.jena_fuseki_backend.JenaFusekiBackend.upload_graph`. Nothing here
is a new write authority — see ``run_pack``'s docstring for exactly what "apply" and
"verify" mean today, corrected against two premises CA-23's lane brief got wrong (both
confirmed by grep against this checkout, both recorded here rather than silently
special-cased away — the lane's own refusal rule).

**W01 premise correction #1 — "apply" is not ``ApplyChangeEnvelope``.**
The brief's Design assumed ``apply_manifest`` "Apply[ies] via ``ApplyChangeEnvelope``
(per ``DEC-CA-01``'s mutation boundary)". MEASURED: it does not.
``manifest_compiler.apply_manifest`` (and the ``leanix_metamodel.apply_leanix_metamodel``
it generalizes) does exactly two things after its fail-closed hash/signature check: write
the compiled Turtle to ``ttl_path`` on disk, and call
``owl_bridge.register_promotable_node_types`` (an in-process set update). A repo-wide grep
for ``ApplyChangeEnvelope`` (checked 2026-08-26) shows every call site is
``envelope_ingest``/``source_sync``/``materialize``/``native_ingest`` — the LPG
record-data write path — with zero references from ``manifest_compiler.py``,
``connector_manifest.py``, or ``shacl_gate.py``. There is no code path in this checkout
today that applies an OWL ontology pack's compiled Turtle to eg's live graph state via
``ApplyChangeEnvelope`` — an ontology pack becomes live only when eg's own OWL loader
picks up the regenerated ``ontology_<source>.ttl`` file (an ``owl:imports`` target of the
canonical ``ontology.ttl``) at its own load time, which is outside this lane's/this
module's control. ``run_pack``'s "apply" stage is therefore exactly what
``apply_manifest`` really is: the fail-closed local artifact emission, not a live eg
write — and this module's "verify" stage compares Fuseki's published triple count
against that LOCAL artifact's own recomputed count (``apply_manifest``'s own
``triple_count``/``canonical_hash`` return fields), not a live eg query, because there is
nothing live to query for a freshly-applied pack today.

**W01 premise correction #2 — "eg's native ShaclValidate inside ApplyChangeEnvelope" is
not this pipeline's fail-closed SHACL boundary.** Following directly from correction #1:
since no ``ApplyChangeEnvelope`` call exists in this pack's apply path, eg's native
``ShaclValidate`` (which IS the authoritative gate for LPG ``ChangeEnvelope`` commits,
per ``shacl_gate.py``'s own docstring) is simply never invoked for an OWL pack. The
*only* SHACL enforcement available to an ontology pack in this codebase today is the
Python-side :func:`pipeline.phases.shacl_gate.validate_graph` -- which that module's own
docstring calls advisory, not a second security authority, precisely because it assumes
a native gate sits downstream of it for LPG data. For OWL packs, no such downstream gate
exists yet (an eg-side gap, out of this lane's repo). ``run_pack`` therefore makes
:func:`~pipeline.phases.shacl_gate.validate_graph`'s verdict fail-closed FOR THIS
PIPELINE SPECIFICALLY (a pack whose supplied ``instance_sample`` fails SHACL is rejected
before publish, full stop) -- an honest, lane-local strengthening, not a claim that eg's
native gate is what is actually enforcing it. Closing that gap for real (an eg-side OWL
pack instance validated by eg's native SHACL engine before being considered "applied")
is program-level follow-up beyond CA-23's scope, not something this module can silently
satisfy by asserting it.

The fail-closed guarantee this pipeline DOES deliver end-to-end, with no gap: a pack
whose ``provenance.integrity.hash``/``signature`` fails re-verification, or whose
ontology IRI is not anti-sprawl-wired, is rejected by ``apply_manifest`` before anything
is written or published -- unchanged, unweakened, exercised by this module's own tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import ontology_integrity
from .connector_manifest import ConnectorManifest
from .manifest_compiler import (
    AntiSprawlError,
    SignatureVerificationError,
    apply_manifest,
    compile_manifest,
    export_manifest_ttl,
)
from .r2rml_generator import generate_r2rml, unmapped_relations

if TYPE_CHECKING:
    from ..backends.sparql.jena_fuseki_backend import JenaFusekiBackend

logger = logging.getLogger(__name__)

__all__ = [
    "PackRejected",
    "PackResult",
    "run_pack",
    "count_graph_triples",
]


class PackRejected(RuntimeError):
    """Raised (or returned as a rejected :class:`PackResult`, caller's choice via
    ``raise_on_reject``) when a pack fails a fail-closed gate before publish: the
    ``apply_manifest`` hash/signature/anti-sprawl check, or (when ``instance_sample`` is
    supplied) the pipeline's own SHACL verdict. Never partially applied, never
    published -- P5's "rejected outright" contract."""


@dataclass
class PackResult:
    """One ``run_pack`` outcome. ``rejected`` is the single source of truth for whether
    anything was written/published; every other field is diagnostic."""

    connector: str
    rejected: bool
    rejection_stage: str | None = None  # "shacl" | "apply" | None
    rejection_reason: str | None = None
    shacl_violated_shapes: tuple[str, ...] = field(default_factory=tuple)
    classes: int = 0
    object_properties: int = 0
    datatype_properties: int = 0
    ttl_path: str | None = None
    local_triple_count: int = 0
    canonical_hash: str | None = None
    dry_run: bool = False
    published: bool = False
    fuseki_graph_uri: str | None = None
    fuseki_triple_count: int | None = None
    counts_equal: bool | None = None
    r2rml_turtle: str | None = None
    r2rml_triples_map_count: int = 0
    unmapped_relations: tuple[str, ...] = field(default_factory=tuple)


def count_graph_triples(fuseki: JenaFusekiBackend, graph_uri: str) -> int:
    """``SELECT (COUNT(*) AS ?c) WHERE { GRAPH <graph_uri> { ?s ?p ?o } }`` against
    ``fuseki`` -- the read half of P5's ``COUNT(*)`` equality check. Returns ``0`` for a
    graph that does not exist (Fuseki answers a normal empty-result SELECT, not an
    error, for an absent named graph -- this is the negative-case behaviour P5 requires:
    a rejected pack's graph reads back as count ``0``, not an error)."""
    query = f"SELECT (COUNT(*) AS ?c) WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }} }}"
    rows = fuseki.execute_sparql_query(query)
    if not rows:
        return 0
    row = rows[0]
    if "error" in row:
        raise RuntimeError(f"Fuseki COUNT(*) query failed: {row['error']}")
    value = row.get("c", "0")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _default_graph_uri(manifest: ConnectorManifest) -> str:
    return f"http://knuckles.team/kg/{manifest.resolved_ontology_source}"


def run_pack(
    manifest: ConnectorManifest,
    *,
    ttl_path: str | Path,
    fuseki: JenaFusekiBackend | None = None,
    graph_uri: str | None = None,
    instance_sample: Any | None = None,
    shacl_shapes_path: str | None = None,
    dry_run: bool = False,
    trusted_signers: tuple[str, ...] = ontology_integrity.DEFAULT_TRUSTED_SIGNERS,
    trusted_public_keys: tuple[str, ...] | None = None,
    raise_on_reject: bool = False,
) -> PackResult:
    """Run the ontology-pack pipeline: compile -> (optional) SHACL gate -> apply ->
    publish -> verify. See the module docstring for exactly what "apply" and "verify"
    mean in this checkout today.

    ``instance_sample`` is an optional LPG-shaped object (e.g. a
    :class:`~core.graph_compute.GraphComputeEngine`, or anything exposing
    ``.nodes(data=True)``) representing sample instance data for this pack's classes --
    when supplied, it is run through
    :func:`~pipeline.phases.shacl_gate.validate_graph` and a non-conforming sample
    rejects the pack (fail-closed, before ``apply_manifest`` runs) naming the violated
    shapes/messages. When omitted (the common case for a pure TBox/ontology-only pack —
    e.g. this lane's own golden-file manifests, which declare classes/relations but no
    instance data), the SHACL stage is skipped and only ``apply_manifest``'s
    hash/signature/anti-sprawl fail-closed check gates the pack — the SHACL gate cannot
    reject what it was never given anything to validate.

    ``fuseki`` omitted (or ``dry_run=True``) skips the publish/verify stages entirely --
    ``published`` stays ``False`` and ``fuseki_triple_count``/``counts_equal`` stay
    ``None``, never fabricated as a false negative/positive.

    Raises :class:`PackRejected` when ``raise_on_reject=True`` and the pack is rejected;
    otherwise returns a :class:`PackResult` with ``rejected=True`` and the reason.
    """
    connector = manifest.connector
    spec = compile_manifest(manifest)
    ttl = export_manifest_ttl(spec, source=manifest.resolved_ontology_source)

    violated_shapes: tuple[str, ...] = ()
    if instance_sample is not None:
        from ..pipeline.phases.shacl_gate import _DEFAULT_SHAPES, validate_graph

        shapes_path = shacl_shapes_path or _DEFAULT_SHAPES
        conforms, violations, _report_text = validate_graph(
            instance_sample, shapes_path
        )
        if not conforms:
            violated_shapes = tuple(sorted(violations))
            reason = (
                f"{connector}: SHACL gate rejected {len(violations)} instance "
                f"sample node(s): {'; '.join(f'{k}: {v}' for k, v in sorted(violations.items()))}"
            )
            logger.warning(
                "pack_pipeline: SHACL rejection for %s (%d node(s))",
                connector,
                len(violations),
            )
            result = PackResult(
                connector=connector,
                rejected=True,
                rejection_stage="shacl",
                rejection_reason=reason,
                shacl_violated_shapes=violated_shapes,
                classes=len(spec.classes),
                object_properties=len(spec.object_properties),
                datatype_properties=len(spec.datatype_properties),
            )
            if raise_on_reject:
                raise PackRejected(reason)
            return result

    try:
        applied = apply_manifest(
            manifest,
            spec,
            ttl_path=ttl_path,
            dry_run=dry_run,
            trusted_signers=trusted_signers,
            trusted_public_keys=trusted_public_keys,
        )
    except (SignatureVerificationError, AntiSprawlError) as exc:
        reason = str(exc)
        logger.warning("pack_pipeline: apply rejected %s: %s", connector, reason)
        result = PackResult(
            connector=connector,
            rejected=True,
            rejection_stage="apply",
            rejection_reason=reason,
            classes=len(spec.classes),
            object_properties=len(spec.object_properties),
            datatype_properties=len(spec.datatype_properties),
        )
        if raise_on_reject:
            raise PackRejected(reason) from exc
        return result

    result = PackResult(
        connector=connector,
        rejected=False,
        classes=applied["classes"],
        object_properties=applied["object_properties"],
        datatype_properties=applied["datatype_properties"],
        ttl_path=applied["ttl_path"],
        local_triple_count=applied["triple_count"],
        canonical_hash=applied["canonical_hash"],
        dry_run=applied["dry_run"],
    )

    # "Generate mapping" (DEC-CA-06's pipeline-stages table) -- independent of Fuseki,
    # always computed once the pack is applied so a caller/log has the R2RML output and
    # any skipped (never-guessed) relations even in dry-run mode. One TriplesMap per
    # resource (see r2rml_generator.generate_r2rml_report, which this wraps).
    result.r2rml_turtle = generate_r2rml(manifest)
    result.r2rml_triples_map_count = len(manifest.resources)
    result.unmapped_relations = tuple(unmapped_relations(manifest))

    if dry_run or fuseki is None:
        return result

    resolved_graph_uri = graph_uri or _default_graph_uri(manifest)
    # Idempotent by construction: re-uploading identical Turtle content to the same
    # named graph overwrites with the same triples (Fuseki graph-store PUT semantics
    # via JenaFusekiBackend.upload_graph) -- a re-publish of an already-applied pack
    # (same canonical hash) is a safe no-op at this step, per the lane's Idempotency
    # invariant.
    fuseki.upload_graph(ttl, graph_uri=resolved_graph_uri)
    fuseki_count = count_graph_triples(fuseki, resolved_graph_uri)

    result.published = True
    result.fuseki_graph_uri = resolved_graph_uri
    result.fuseki_triple_count = fuseki_count
    result.counts_equal = fuseki_count == result.local_triple_count
    return result
