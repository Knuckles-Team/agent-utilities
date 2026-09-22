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

The phase builds only the candidate RDF data document in the
``http://knuckles.team/kg#`` namespace. It sends that document to
epistemic-graph with the ``shapes`` request field omitted, so validation uses
the committed, digest-bound GraphSchema snapshot rather than an AU-local shape
file or Python validator.
"""

from __future__ import annotations

import logging
from typing import Any

from ..types import PhaseResult, PipelineContext, PipelinePhase

logger = logging.getLogger(__name__)

# kg# namespace — the ``:`` prefix used by every ontology + shapes file.
KG_NS = "http://knuckles.team/kg#"

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


def _graph_has_nodes(graph: Any) -> bool:
    """Best-effort "does this graph hold any node at all" probe (BUG-281).

    Used ONLY to tell an honest empty RDF projection (empty graph) apart from a
    degraded one (populated graph, no triples). Deliberately conservative: if the
    node count cannot be established, return ``False`` so an indeterminate probe
    never manufactures a :class:`EngineRdfProjectionError` on its own -- the
    escalation has to be evidence-backed, not a guess.
    """
    try:
        nodes = graph.nodes()
    except Exception:  # noqa: BLE001 -- an unprobeable graph must not itself escalate
        return False
    try:
        return any(True for _ in nodes)
    except Exception:  # noqa: BLE001 -- same rationale as above
        return False


class EngineRdfProjectionError(RuntimeError):
    """The engine's RDF projection was REACHABLE but did not yield a usable data
    graph (BUG-281).

    Distinct from "no engine attached" (which :func:`_data_graph_from_engine_rdf`
    still signals with ``None``, the legitimate fall-back-to-LPG case). The two used
    to be indistinguishable: a raised ``GetRdf`` and an empty projection were both
    swallowed into the SAME ``None`` a healthy no-engine deployment returns, so a
    DEGRADED read silently became "validate the LPG instead" with no signal that the
    authoritative source had failed -- the fail-open shape this program's standards
    call out (a reader that swallows an exception and returns a falsy value is
    indistinguishable from a healthy negative at the call site).
    """


def _data_graph_from_engine_rdf(graph: Any) -> Any | None:
    """Build the SHACL data graph from the ENGINE's RDF projection (CONCEPT:AU-KG.compute.native-sparql-owl-shacl).

    Routes the SHACL *graph source* to the engine: pulls the canonical N-Triples
    document from ``GetRdf`` (one round-trip over the live graph) and parses it
    without datatype/language loss.

    Returns ``None`` for EXACTLY ONE condition -- no engine RDF surface is attached
    at all -- so the caller's fall back to per-node LPG iteration stays the
    deliberate, correct behaviour for an offline/dev deployment. Every OTHER failure
    (``GetRdf`` raising, or an empty projection over a non-empty graph) raises
    :class:`EngineRdfProjectionError` WITH its cause attached, because those mean the
    authoritative source is degraded and the caller must decide that explicitly
    rather than inherit a silent downgrade (BUG-281). PySHACL is used only for this
    advisory quarantine view; the native engine validator remains authoritative at
    materialization time.
    """
    get_rdf = getattr(graph, "get_rdf", None)
    if get_rdf is None:
        # The one legitimate None: nothing to route to.
        return None
    try:
        ntriples = get_rdf()
    except Exception as exc:
        raise EngineRdfProjectionError(
            "engine GetRdf failed; the RDF projection is degraded"
        ) from exc
    if not ntriples:
        # An empty projection is only honest when the graph itself is empty.
        # Empty triples over a POPULATED graph is a degraded read, not a clean
        # negative -- exactly the case that used to vanish into `None`.
        if _graph_has_nodes(graph):
            raise EngineRdfProjectionError(
                "engine GetRdf returned an empty RDF projection for a non-empty "
                "graph; refusing to treat it as a clean empty result"
            )
        return None

    import rdflib

    g = rdflib.Graph()
    kg = rdflib.Namespace(KG_NS)
    g.bind("", kg)
    g.bind("rdf", rdflib.RDF)
    g.bind("rdfs", rdflib.RDFS)

    g.parse(data=ntriples, format="nt")
    return g


def build_data_graph(graph: Any) -> Any:
    """Materialize the LPG into an rdflib Graph in the kg# namespace.

    The data SHACL validates is sourced from the ENGINE's RDF projection first
    (``get_rdf`` -- one N-Triples round-trip over the live graph, CONCEPT:AU-KG.compute.native-sparql-owl-shacl); when no
    engine is reachable this falls back to per-node iteration of the LPG.

    Each node becomes ``kg:<id> rdf:type kg:<Class>`` plus its string/numeric
    properties as datatype-property assertions, so SHACL shapes targeting
    ``:Tool``/``:Agent``/``:ServiceCapability`` etc. can validate them.

    BUG-281: a DEGRADED engine projection (as opposed to an absent one) now surfaces
    as :class:`EngineRdfProjectionError` from the helper. This function still falls
    back to the LPG -- an advisory quarantine view is better than none -- but the
    downgrade is now LOGGED with its cause instead of being invisible, so a degraded
    authoritative source is observable rather than silently tolerated.
    """
    try:
        engine_graph = _data_graph_from_engine_rdf(graph)
    except EngineRdfProjectionError:
        logger.warning(
            "SHACL data graph: engine RDF projection degraded; falling back to "
            "per-node LPG iteration (advisory view only)",
            exc_info=True,
        )
        engine_graph = None
    if engine_graph is not None:
        return engine_graph

    import rdflib

    g = rdflib.Graph()
    kg = rdflib.Namespace(KG_NS)
    g.bind("", kg)
    g.bind("rdf", rdflib.RDF)
    g.bind("rdfs", rdflib.RDFS)

    for node_id, data in graph.nodes(data=True):
        node_uri = kg[str(node_id).replace(" ", "_")]
        node_type = data.get("node_type", "Thing")
        g.add((node_uri, rdflib.RDF.type, kg[_class_iri(node_type)]))
        for key, value in data.items():
            if key in _SKIP_PROPS or key == "node_type":
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, str) and value:
                g.add((node_uri, kg[key], rdflib.Literal(value)))
            elif isinstance(value, int | float):
                g.add((node_uri, kg[key], rdflib.Literal(value)))
    return g


def _data_turtle(graph: Any) -> str:
    rendered = build_data_graph(graph).serialize(format="turtle")
    return rendered.decode() if isinstance(rendered, bytes) else str(rendered)


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

    report = graph.shacl_validate_committed(_data_turtle(graph))
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
        report = await ctx.graph.shacl_validate_committed_async(_data_turtle(ctx.graph))
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
