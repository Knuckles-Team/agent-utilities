"""LeanIX metamodel → OWL/RDF compiler (CONCEPT:KG-2.9).

Discovers the *live* LeanIX data model (every fact sheet type, its fields, and
its relations) and compiles it into a faithful OWL/RDF ontology so the enterprise
architecture is mirrored **natively** in the knowledge graph — not flattened to a
hand-written stub. The compiler is pure (no network, no live graph); a LeanIX
client is the caller's concern (see :func:`ea_clients.get_leanix_client`).

Two reasoning layers consume the output, and ``apply_leanix_metamodel`` feeds
both:

* **DL reasoners** (``pipeline/phases/owl_reasoning.py`` and
  ``maintenance/owl_closure.py``) reason over the static TTL files via
  rdflib/owlready2. ``knowledge_graph/ontology.ttl`` already
  ``owl:imports <…/kg/leanix>`` → ``ontology_leanix.ttl``, so we **regenerate**
  that file from the live metamodel (replacing the 4-class stub).
* **owl_bridge structural layer** reasons over the LPG gated by
  ``PROMOTABLE_NODE_TYPES`` — so we register the generated types via
  :func:`owl_bridge.register_promotable_node_types`.

Fact sheet types map to ArchiMate classes where a crosswalk exists (e.g.
``Application`` → ``:ApplicationComponent``); unmapped custom types get a bare
``owl:Class`` with no ``rdfs:subClassOf`` (mirroring the domain-free precedent in
``ontology_archimate.ttl``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# LeanIX field value type → XSD datatype (tolerant; unknown → xsd:string).
_XSD: dict[str, str] = {
    "STRING": "xsd:string",
    "TEXT": "xsd:string",
    "INTEGER": "xsd:integer",
    "LONG": "xsd:long",
    "DOUBLE": "xsd:double",
    "DECIMAL": "xsd:decimal",
    "BOOLEAN": "xsd:boolean",
    "DATE": "xsd:date",
    "DATE_TIME": "xsd:dateTime",
    "SINGLE_SELECT": "xsd:string",
    "MULTIPLE_SELECT": "xsd:string",
    "LIFECYCLE": "xsd:string",
    "PROJECT_STATUS": "xsd:string",
}

# LeanIX fact sheet type → ArchiMate/base class local name (rdfs:subClassOf),
# where a clean alignment exists. Custom types fall through to no parent.
DEFAULT_ARCHIMATE_CROSSWALK: dict[str, str] = {
    "Application": "ApplicationComponent",
    "ITComponent": "Node",
    "TechnicalStack": "SystemSoftware",
    "BusinessCapability": "BusinessCapability",
    "DataObject": "DataObject",
    "Process": "BusinessProcess",
    "BusinessContext": "BusinessFunction",
    "Interface": "ApplicationInterface",
    "Provider": "BusinessActor",
    "Objective": "Goal",
}

# Stable node-id prefixes for the well-known types (keeps ids compatible with the
# pre-existing extractor); other types fall back to a lowercased type name.
_ID_PREFIXES: dict[str, str] = {
    "Application": "app",
    "ITComponent": "itcomponent",
    "BusinessCapability": "capability",
    "DataObject": "dataobject",
}


def _upper_snake(name: str) -> str:
    """``relApplicationToITComponent`` → ``REL_APPLICATION_TO_IT_COMPONENT``."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", s)
    return s.upper()


def _id_prefix(fs_type: str) -> str:
    return _ID_PREFIXES.get(fs_type, fs_type.lower())


@dataclass
class ClassSpec:
    """One LeanIX fact sheet type as an OWL class."""

    local: str  # OWL class local name (== LeanIX type)
    label: str
    parent: str | None  # ArchiMate/base class local name, or None
    id_prefix: str


@dataclass
class ObjectPropertySpec:
    """One LeanIX relation as an OWL object property."""

    local: str  # OWL property local name (== LeanIX relation field)
    label: str
    domain: str
    range: str
    lpg_rel_type: str  # the UPPER_SNAKE edge label the extractor emits


@dataclass
class DatatypePropertySpec:
    """One LeanIX field as an OWL datatype property."""

    local: str
    label: str
    range: str  # an xsd:* type


@dataclass
class LeanixOntologySpec:
    """The compiled metamodel: OWL terms plus maps the extractor consumes."""

    classes: list[ClassSpec] = field(default_factory=list)
    object_properties: list[ObjectPropertySpec] = field(default_factory=list)
    datatype_properties: list[DatatypePropertySpec] = field(default_factory=list)
    # leanix type -> (owl class/label local, node-id prefix)
    type_map: dict[str, tuple[str, str]] = field(default_factory=dict)
    # relation field -> (lpg_rel_type, target_type)
    relation_map: dict[str, tuple[str, str]] = field(default_factory=dict)


def _fact_sheet_types(meta_model: dict[str, Any]) -> dict[str, dict]:
    """Extract ``{type: definition}`` from a data model (dict- or list-shaped)."""
    fs = meta_model.get("factSheets") if isinstance(meta_model, dict) else None
    if isinstance(fs, dict):
        return {k: v for k, v in fs.items() if isinstance(v, dict)}
    if isinstance(fs, list):
        out: dict[str, dict] = {}
        for t in fs:
            if isinstance(t, dict):
                key = t.get("type") or t.get("name")
                if key:
                    out[str(key)] = t
        return out
    return {}


def _fields_of(defn: dict) -> dict[str, str]:
    """``{fieldName: VALUE_TYPE}`` from a type definition (tolerant)."""
    fields = defn.get("fields")
    out: dict[str, str] = {}
    if isinstance(fields, dict):
        for name, fd in fields.items():
            vt = fd.get("type") if isinstance(fd, dict) else None
            out[str(name)] = str(vt or "STRING")
    elif isinstance(fields, list):
        for fd in fields:
            if isinstance(fd, dict) and fd.get("name"):
                out[str(fd["name"])] = str(fd.get("type") or "STRING")
    return out


def _relations_of(defn: dict) -> dict[str, str]:
    """``{relationField: targetFactSheetType}`` from a type definition (tolerant)."""
    rels = defn.get("relations")
    out: dict[str, str] = {}
    items: list[tuple[str, dict]] = []
    if isinstance(rels, dict):
        items = [(str(k), v) for k, v in rels.items() if isinstance(v, dict)]
    elif isinstance(rels, list):
        items = [
            (str(r["name"]), r)
            for r in rels
            if isinstance(r, dict) and r.get("name")
        ]
    for name, rd in items:
        if not name.startswith("rel"):
            continue
        target = (
            rd.get("targetFactSheetType")
            or rd.get("targetType")
            or rd.get("target")
            or ""
        )
        out[name] = str(target)
    return out


def compile_leanix_metamodel(
    meta_model: dict[str, Any],
    *,
    archimate_crosswalk: dict[str, str] | None = None,
) -> LeanixOntologySpec:
    """Compile a live LeanIX data model into a :class:`LeanixOntologySpec`."""
    crosswalk = archimate_crosswalk or DEFAULT_ARCHIMATE_CROSSWALK
    spec = LeanixOntologySpec()
    seen_dtp: set[str] = set()

    types = _fact_sheet_types(meta_model)
    for fs_type, defn in types.items():
        prefix = _id_prefix(fs_type)
        spec.classes.append(
            ClassSpec(
                local=fs_type,
                label=_humanize(fs_type),
                parent=crosswalk.get(fs_type),
                id_prefix=prefix,
            )
        )
        spec.type_map[fs_type] = (fs_type, prefix)

        for fname, vtype in _fields_of(defn).items():
            if fname in seen_dtp:
                continue
            seen_dtp.add(fname)
            spec.datatype_properties.append(
                DatatypePropertySpec(
                    local=fname,
                    label=_humanize(fname),
                    range=_XSD.get(vtype, "xsd:string"),
                )
            )

        for rfield, target in _relations_of(defn).items():
            lpg = _upper_snake(rfield)
            spec.object_properties.append(
                ObjectPropertySpec(
                    local=rfield,
                    label=_humanize(rfield),
                    domain=fs_type,
                    range=target or "owl:Thing",
                    lpg_rel_type=lpg,
                )
            )
            spec.relation_map[rfield] = (lpg, target)

    # Always provide a display-name datatype property (used by every node).
    if "factsheetName" not in seen_dtp:
        spec.datatype_properties.append(
            DatatypePropertySpec(
                local="factsheetName",
                label="factsheet name",
                range="xsd:string",
            )
        )
    return spec


def _humanize(camel: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", camel)
    return s[:1].upper() + s[1:]


_TTL_HEADER = """\
# GENERATED by agent_utilities.knowledge_graph.ontology.leanix_metamodel.
# Do not edit by hand — regenerate via the ``ontology_leanix_sync`` MCP action /
# POST /api/ontology/leanix/sync. CONCEPT:KG-2.9.
@prefix : <http://knuckles.team/kg#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<http://knuckles.team/kg/leanix> a owl:Ontology ;
    rdfs:label "LeanIX Enterprise Architecture Ontology (generated)" ;
    rdfs:comment "Faithful OWL mirror of the live LeanIX metamodel — every fact sheet type, relation, and field. CONCEPT:KG-2.9." ;
    owl:imports <http://knuckles.team/kg> .
"""


def export_leanix_ttl(spec: LeanixOntologySpec) -> str:
    """Serialize a compiled spec to Turtle (the regenerated ``ontology_leanix.ttl``)."""
    lines: list[str] = [_TTL_HEADER]

    for c in sorted(spec.classes, key=lambda x: x.local):
        block = [f":{c.local} a owl:Class ;", f'    rdfs:label "{c.label}" ;']
        if c.parent:
            block.append(f"    rdfs:subClassOf :{c.parent} ;")
        block.append('    rdfs:comment "LeanIX fact sheet type (generated)." .')
        lines.append("\n".join(block))

    for p in sorted(spec.object_properties, key=lambda x: (x.local, x.domain)):
        rng = p.range if p.range == "owl:Thing" else f":{p.range}"
        lines.append(
            f":{p.local} a owl:ObjectProperty ;\n"
            f'    rdfs:label "{p.label}" ;\n'
            f"    rdfs:domain :{p.domain} ;\n"
            f"    rdfs:range {rng} ."
        )

    for d in sorted(spec.datatype_properties, key=lambda x: x.local):
        lines.append(
            f":{d.local} a owl:DatatypeProperty ;\n"
            f'    rdfs:label "{d.label}" ;\n'
            f"    rdfs:range {d.range} ."
        )

    return "\n\n".join(lines) + "\n"


def default_ttl_path() -> Path:
    """The package ``ontology_leanix.ttl`` (imported by ``ontology.ttl``)."""
    return Path(__file__).resolve().parent.parent / "ontology_leanix.ttl"


def apply_leanix_metamodel(
    spec: LeanixOntologySpec,
    *,
    ttl_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Feed a compiled spec into both reasoning layers.

    Regenerates ``ontology_leanix.ttl`` (DL reasoners) and registers the
    generated class labels as promotable node types (owl_bridge structural
    layer). Returns a manifest; in ``dry_run`` mode nothing is written.
    """
    from ..core.owl_bridge import register_promotable_node_types

    path = Path(ttl_path) if ttl_path else default_ttl_path()
    ttl = export_leanix_ttl(spec)
    manifest: dict[str, Any] = {
        "classes": len(spec.classes),
        "object_properties": len(spec.object_properties),
        "datatype_properties": len(spec.datatype_properties),
        "ttl_path": str(path),
        "dry_run": dry_run,
        "promoted_types": [c.local for c in spec.classes],
    }
    if dry_run:
        manifest["ttl_preview"] = ttl
        return manifest

    register_promotable_node_types(c.local for c in spec.classes)
    path.write_text(ttl, encoding="utf-8")
    return manifest


def sync_leanix_ontology(
    client: Any = None,
    *,
    dry_run: bool = True,
    ttl_path: str | Path | None = None,
) -> dict[str, Any]:
    """Discover the live LeanIX metamodel and apply it as OWL (the action core).

    The single seam the ``ontology_leanix_sync`` MCP action and the
    ``POST /api/ontology/leanix/sync`` REST route both call. Resolves a LeanIX
    client when one is not injected, introspects the live data model, compiles
    it, and applies it to both reasoning layers (unless ``dry_run``).
    """
    if client is None:
        from ...ecosystem.ea_clients import get_leanix_client

        client = get_leanix_client()
    if client is None:
        return {
            "status": "skipped",
            "reason": "no LeanIX client configured (set LEANIX_URL / LEANIX_TOKEN)",
        }
    meta = client.meta_model()
    if not meta:
        return {
            "status": "skipped",
            "reason": "empty LeanIX metamodel (unreachable or unauthorized)",
        }
    spec = compile_leanix_metamodel(meta)
    manifest = apply_leanix_metamodel(spec, ttl_path=ttl_path, dry_run=dry_run)
    manifest["status"] = "completed"
    return manifest
