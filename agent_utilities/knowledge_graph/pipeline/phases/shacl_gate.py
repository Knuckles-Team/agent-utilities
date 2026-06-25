#!/usr/bin/python
"""SHACL Ingestion Gate Pipeline Phase (CONCEPT:KG-2.8).

Runs ``pyshacl.validate`` against the bundled governance SHACL shapes
*before* the commit/persist phase (``sync``).  Nodes whose focus-node
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

The phase is deliberately self-contained — it builds its own RDF data
graph in the ``http://knuckles.team/kg#`` namespace (matching the
``sh:targetClass`` IRIs in ``governance.shapes.ttl``) so it can run before
any RDF materialization performed by the OWL reasoning phase.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..types import PhaseResult, PipelineContext, PipelinePhase

logger = logging.getLogger(__name__)

# kg# namespace — the ``:`` prefix used by every ontology + shapes file.
KG_NS = "http://knuckles.team/kg#"

# Default shapes: bundled alongside the knowledge_graph package.
_DEFAULT_SHAPES = str(
    Path(__file__).parent.parent.parent / "shapes" / "governance.shapes.ttl"
)

# Properties that should never be promoted into the RDF data graph
# (large float arrays, internal bookkeeping).
_SKIP_PROPS = {"embedding", "ewc_fisher_diag"}

try:
    import pyshacl  # noqa: F401
    import rdflib  # noqa: F401

    SHACL_SUPPORT = True
except ImportError:  # pragma: no cover - exercised only when deps missing
    SHACL_SUPPORT = False


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


def _data_graph_from_engine_triples(graph: Any) -> Any | None:
    """Build the SHACL data graph from the ENGINE's RDF projection (CONCEPT:KG-2.242).

    Routes the SHACL *graph source* to the engine: pulls ``[s, p, o]`` triples from
    the engine's native ``get_triples`` op (one round-trip over the live graph) and
    maps them into the kg# namespace rdflib data graph SHACL validates. Returns
    ``None`` when the engine/op is unavailable so the caller falls back to per-node
    iteration of the LPG. pyshacl remains the validator (the engine has no native
    SHACL method); only the data the validator sees now comes from the engine.
    """
    get_triples = getattr(graph, "get_triples", None)
    if get_triples is None:
        return None
    try:
        triples = get_triples()
    except Exception:  # noqa: BLE001 -- engine/op unavailable -> caller falls back
        return None
    if not triples:
        return None

    import rdflib

    g = rdflib.Graph()
    kg = rdflib.Namespace(KG_NS)
    g.bind("", kg)
    g.bind("rdf", rdflib.RDF)
    g.bind("rdfs", rdflib.RDFS)

    subjects = {str(t[0]) for t in triples if len(t) == 3}
    for t in triples:
        if len(t) != 3:
            continue
        s, pred, o = str(t[0]), str(t[1]), t[2]
        subj = kg[s.replace(" ", "_")]
        if pred == "rdf:type":
            g.add((subj, rdflib.RDF.type, kg[_class_iri(str(o))]))
        elif str(o) in subjects:
            g.add((subj, kg[pred], kg[str(o).replace(" ", "_")]))
        else:
            if isinstance(o, bool):
                continue
            g.add((subj, kg[pred], rdflib.Literal(o)))
    return g


def build_data_graph(graph: Any) -> Any:
    """Materialize the LPG into an rdflib Graph in the kg# namespace.

    The data SHACL validates is sourced from the ENGINE's RDF projection first
    (``get_triples`` -- one round-trip over the live graph, CONCEPT:KG-2.242); when no
    engine is reachable this falls back to per-node iteration of the LPG.

    Each node becomes ``kg:<id> rdf:type kg:<Class>`` plus its string/numeric
    properties as datatype-property assertions, so SHACL shapes targeting
    ``:Tool``/``:Agent``/``:ServiceCapability`` etc. can validate them.
    """
    engine_graph = _data_graph_from_engine_triples(graph)
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
        node_type = data.get("type", "Thing")
        g.add((node_uri, rdflib.RDF.type, kg[_class_iri(node_type)]))
        for key, value in data.items():
            if key in _SKIP_PROPS or key == "type":
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, str) and value:
                g.add((node_uri, kg[key], rdflib.Literal(value)))
            elif isinstance(value, int | float):
                g.add((node_uri, kg[key], rdflib.Literal(value)))
    return g


def validate_graph(
    graph: Any, shapes_path: str
) -> tuple[bool, dict[str, list[str]], str]:
    """Run pyshacl against *graph* and return per-focus-node violations.

    Returns ``(conforms, {node_id: [messages]}, report_text)``.  Only
    ``sh:Violation`` severity results trigger quarantine; warnings/info are
    surfaced in the report text but do not gate ingestion.
    """
    import pyshacl
    import rdflib

    data_graph = build_data_graph(graph)
    shapes_graph = rdflib.Graph()
    shapes_graph.parse(shapes_path, format="turtle")

    # CONCEPT:KG-2.39 — also load value-type-generated SHACL shapes so value-type
    # constraints (EmailAddress, Percentage, …) are enforced alongside the bundled
    # governance shapes. Best-effort: a malformed/absent fragment must not break
    # the gate (the parse is isolated in try/except).
    try:
        from agent_utilities.knowledge_graph.ontology.value_types import VALUE_TYPES

        frags = [vt.to_shacl() for vt in VALUE_TYPES.values()]
        if frags:
            header = (
                "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
                "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
                "@prefix qudt: <http://qudt.org/schema/qudt/> .\n"
                f"@prefix : <{KG_NS}> .\n\n"
            )
            shapes_graph.parse(data=header + "\n".join(frags), format="turtle")
    except Exception as _vt_e:  # noqa: BLE001 — best-effort coupling
        logger.debug("value-type SHACL shapes not merged: %s", _vt_e)

    conforms, results_graph, results_text = pyshacl.validate(
        data_graph,
        shacl_graph=shapes_graph,
        inference="none",
        abort_on_first=False,
        meta_shacl=False,
        advanced=True,
    )

    SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")
    kg = rdflib.Namespace(KG_NS)
    violations: dict[str, list[str]] = {}

    for result in results_graph.subjects(rdflib.RDF.type, SH.ValidationResult):
        severity = results_graph.value(result, SH.resultSeverity)
        if severity is not None and severity != SH.Violation:
            continue  # warnings / info do not gate ingestion
        focus = results_graph.value(result, SH.focusNode)
        msg = results_graph.value(result, SH.resultMessage)
        if focus is None:
            continue
        focus_str = str(focus)
        if focus_str.startswith(str(kg)):
            focus_str[len(str(kg)) :].replace("_", " ")
            # Prefer the exact id form first (ids without spaces stay intact)
            raw_id = focus_str[len(str(kg)) :]
        else:
            raw_id = focus_str
        key = raw_id
        violations.setdefault(key, []).append(
            str(msg) if msg is not None else "SHACL constraint violated"
        )

    return conforms, violations, str(results_text)


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

    if not SHACL_SUPPORT:
        return {
            "status": "skipped",
            "reason": "pyshacl/rdflib not installed (pip install pyshacl rdflib)",
        }

    shapes_path = ctx.config.shacl_shapes_path or _DEFAULT_SHAPES
    if not Path(shapes_path).exists():
        return {"status": "error", "reason": f"SHACL shapes not found: {shapes_path}"}

    try:
        conforms, violations, report_text = validate_graph(ctx.graph, shapes_path)
    except Exception as e:  # pragma: no cover - defensive
        logger.error("SHACL validation failed: %s", e)
        return {"status": "error", "reason": str(e)}

    quarantine_type = ctx.config.shacl_quarantine_marker
    quarantined: list[str] = []

    for raw_id, messages in violations.items():
        node_id = _resolve_node_id(ctx.graph, raw_id)
        if node_id is None:
            continue
        props = dict(ctx.graph.nodes.get(node_id, {}) or {})
        original_type = props.get("type")
        props["shacl_valid"] = False
        props["shacl_quarantine_type"] = quarantine_type
        props["shacl_original_type"] = original_type
        props["shacl_report"] = "\n".join(messages)
        # Re-route the node's effective type so the commit phase persists it
        # under the quarantine label instead of as a first-class citizen.
        props["type"] = quarantine_type
        ctx.graph.add_node(node_id, properties=props)
        quarantined.append(node_id)

    if quarantined:
        logger.warning(
            "SHACL gate quarantined %d node(s): %s",
            len(quarantined),
            ", ".join(quarantined[:20]),
        )

    return {
        "status": "completed",
        "conforms": bool(conforms) and not quarantined,
        "quarantined_count": len(quarantined),
        "quarantined_nodes": quarantined,
        "shapes_path": shapes_path,
        "report": report_text if quarantined else "",
    }


shacl_gate_phase = PipelinePhase(
    name="shacl_gate",
    # Runs after nodes are resolved/built, before the sync (commit) phase.
    deps=["resolve"],
    execute_fn=execute_shacl_gate,
)
