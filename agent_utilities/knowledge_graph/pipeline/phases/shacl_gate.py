#!/usr/bin/python
"""Advisory SHACL quarantine phase (CONCEPT:EG-KG.storage.nonblocking-checkpoint).

This in-memory pipeline phase classifies invalid candidate nodes before ``sync``
so operators can inspect a quarantine projection. The authoritative ingestion
boundary independently invokes Epistemic Graph's native ``ShaclValidate`` method
and fails closed before applying any ChangeEnvelope; this phase is not a second
security authority. Nodes whose focus-node
shape constraints are violated are **quarantined** rather than committed
as their declared type:

  * the node is marked with a ``shacl_valid = False`` flag,
  * an ``_invalid`` rdf-type marker (``:Invalid`` by default) is recorded
    in ``shacl_quarantine_type`` so downstream ``sync`` persists it under
    the quarantine label instead of its real class, and
  * the human-readable violation report is attached to the node under
    ``shacl_report`` for audit/triage.

Valid nodes pass through untouched.  This makes ingestion a *gated*
operation: a Tool missing its required ``name``/``capabilityCategory``,
or an Agent missing its ``name``, never silently lands in the graph as a
first-class citizen.

The phase sends epistemic-graph either nothing (an engine-backed graph: EG
validates its own live RDF) or the candidate nodes as typed triples in the
``http://knuckles.team/kg#`` namespace, always with the ``shapes`` request
field omitted, so validation uses the committed, digest-bound GraphSchema
snapshot. Agent Utilities renders, parses and validates no RDF itself.
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.typed_triples import KG_NS, TypedTriple, kg, literal, triple, typed
from ..types import PhaseResult, PipelineContext, PipelinePhase

logger = logging.getLogger(__name__)


# Properties that should never be promoted into the RDF data graph
# (large float arrays, internal bookkeeping).
_SKIP_PROPS = {"embedding", "ewc_fisher_diag"}


def _class_iri(node_type: str) -> str:
    """Map an LPG ``type`` string to its kg# class local name.

    ``"tool"`` -> ``Tool``, ``"service_capability"`` -> ``ServiceCapability``,
    ``"Agent"`` -> ``Agent``.  Mirrors the casing used in the ontology so
    ``sh:targetClass`` matches.
    """
    cleaned = str(node_type).strip()
    if not cleaned:
        return "Thing"
    if any(ch in cleaned for ch in (" ", "_", "-")):
        parts = cleaned.replace("-", " ").replace("_", " ").split()
        return "".join(p[:1].upper() + p[1:] for p in parts)
    # Already a single token; preserve given casing but ensure leading cap.
    return cleaned[:1].upper() + cleaned[1:]


def build_data_triples(graph: Any) -> list[TypedTriple] | None:
    """The data EG validates for ``graph``, as typed triples (EH-472).

    An engine-backed graph (one exposing ``get_rdf``) returns ``None``: EG then
    validates its OWN live RDF projection of the request graph -- the exact
    triples ``GetRdf`` serializes -- without a round-trip through Agent Utilities.
    Any other graph is materialized node by node in the ``kg#`` namespace: each
    node becomes ``kg:<id> rdf:type kg:<Class>`` plus its string/numeric
    properties, so SHACL shapes targeting ``:Tool``/``:Agent``/... can validate
    them. Agent Utilities never renders or parses RDF text.
    """
    if getattr(graph, "get_rdf", None) is not None:
        return None
    triples: list[TypedTriple] = []
    for node_id, data in graph.nodes(data=True):
        triples.extend(_node_triples(str(node_id), data))
    return triples


def _node_triples(node_id: str, data: dict[str, Any]) -> list[TypedTriple]:
    subject = kg(node_id.replace(" ", "_"))
    triples = [typed(subject, kg(_class_iri(data.get("node_type", "Thing"))))]
    for key, value in data.items():
        if key in _SKIP_PROPS or key == "node_type" or isinstance(value, bool):
            continue
        if (isinstance(value, str) and value) or isinstance(value, int | float):
            triples.append(triple(subject, kg(key), literal(value)))
    return triples


def _committed_kwargs(graph: Any) -> dict[str, Any]:
    triples = build_data_triples(graph)
    return {} if triples is None else {"data_triples": triples}


def _result_node_id(focus_node: str) -> str:
    """Convert the engine's N-Triples lexical focus node into an AU node id."""

    value = focus_node
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    return value[len(KG_NS) :] if value.startswith(KG_NS) else value


def _violation_map(report: Any) -> dict[str, list[str]]:
    violations: dict[str, list[str]] = {}
    for result in report.results:
        if getattr(result.severity, "value", result.severity) != "Violation":
            continue
        violations.setdefault(_result_node_id(result.focus_node), []).append(
            result.message or "SHACL constraint violated"
        )
    return violations


def validate_graph(graph: Any) -> tuple[bool, dict[str, list[str]], str]:
    """Validate through EG's committed, digest-bound GraphSchema authority."""

    report = graph.shacl_validate_committed(**_committed_kwargs(graph))
    return bool(report.conforms), _violation_map(report), repr(report.results)


def _resolve_node_id(graph: Any, raw_id: str) -> str | None:
    """Map an RDF local-name back to the original LPG node id."""
    if graph.has_node(raw_id):
        return raw_id
    spaced = raw_id.replace("_", " ")
    if graph.has_node(spaced):
        return spaced
    return None


async def execute_shacl_gate(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Gate phase: validate nodes against SHACL shapes before commit.

    Violating nodes are quarantined (marked + report attached) instead of
    being committed cleanly by the downstream ``sync`` phase.
    """
    if not ctx.config.enable_shacl_gate:
        return {"status": "skipped", "reason": "SHACL gate disabled"}

    try:
        report = await ctx.graph.shacl_validate_committed_async(
            **_committed_kwargs(ctx.graph)
        )
        conforms = bool(report.conforms)
        violations = _violation_map(report)
        report_text = repr(report.results)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "Advisory SHACL validation failed (exception_type=%s)",
            type(exc).__name__,
        )
        return {"status": "error", "reason": "SHACL validation failed"}

    quarantine_type = ctx.config.shacl_quarantine_marker
    quarantined: list[str] = []

    for raw_id, messages in violations.items():
        node_id = _resolve_node_id(ctx.graph, raw_id)
        if node_id is None:
            continue
        props = dict(ctx.graph.nodes.get(node_id, {}) or {})
        original_type = props.get("node_type")
        props["shacl_valid"] = False
        props["shacl_quarantine_type"] = quarantine_type
        props["shacl_original_type"] = original_type
        props["shacl_report"] = "\n".join(messages)
        # Re-route the node's effective type so the commit phase persists it
        # under the quarantine label instead of as a first-class citizen.
        props["node_type"] = quarantine_type
        ctx.graph.add_node(node_id, properties=props)
        quarantined.append(node_id)

    if quarantined:
        logger.warning("SHACL gate quarantined %d node(s)", len(quarantined))

    return {
        "status": "completed",
        "conforms": bool(conforms) and not quarantined,
        "quarantined_count": len(quarantined),
        "report_available": bool(report_text) and bool(quarantined),
    }


shacl_gate_phase = PipelinePhase(
    name="shacl_gate",
    # Runs after nodes are resolved/built, before the sync (commit) phase.
    deps=["resolve"],
    execute_fn=execute_shacl_gate,
)
