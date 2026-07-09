#!/usr/bin/env python3
"""Deterministic Connector Ontology Manifest generator (CONCEPT:AU-KG.ontology.connector-manifest-generator).

**ZERO LLM calls, no network.** For a given connector package (an ``agents/<pkg>``
checkout), projects the artifacts the connector ALREADY ships into a single
declarative ``connector_manifest.yml``:

  * ``<module>/ontology/*.ttl``            -> ``resources`` (owl:Class) +
                                               ``schema_mappings`` (owl:DatatypeProperty
                                               field/XSD) + resource ``relations``
                                               (owl:ObjectProperty, only where the ttl
                                               declares an ``rdfs:domain``).
  * ``<module>/connectors/mcp_source_presets.json`` -> ``sync`` (:data:`MCP_TOOL_PRESETS`
                                               shape) + ``identity`` + a synthetic
                                               ``events`` watermark per preset.
  * ``a2a.json``                           -> ``actions`` (from ``capabilities``).

Every field that cannot be derived losslessly from those artifacts (the ontology-class
crosswalk, PII/RLS policy) is filled with a documented heuristic default and flagged in
``review_todos`` — never silently guessed, never invented by an LLM.

Same input -> byte-identical output (pass ``--now`` to pin the provenance timestamp for
reproducible/test runs; a real run without it stamps the current UTC time).

Usage:
  python3 scripts/generate_connector_manifests.py --connector-root <path> [--output PATH]
  python3 scripts/generate_connector_manifests.py --all --agents-root <path> [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.ontology import ontology_integrity  # noqa: E402
from agent_utilities.knowledge_graph.ontology.connector_manifest import (  # noqa: E402
    DEFAULT_ARCHIMATE_CROSSWALK,
    PII_HEURISTIC_FIELD_NAMES,
    ActionSpec,
    ConnectorManifest,
    EventSpec,
    IdentitySpec,
    IntegrityInfo,
    PermissionsSpec,
    PolicySpec,
    ProvenanceSpec,
    ResourceRelation,
    ResourceSpec,
    SchemaMapping,
    SyncSpec,
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (  # noqa: E402
    compile_manifest,
    export_manifest_ttl,
)

_XSD_NS = "http://www.w3.org/2001/XMLSchema#"
_OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
_OWL_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#ObjectProperty"
_OWL_DATATYPE_PROPERTY = "http://www.w3.org/2002/07/owl#DatatypeProperty"
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
_RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
_RDFS_RANGE = "http://www.w3.org/2000/01/rdf-schema#range"
_RDFS_SUBCLASSOF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"


def _local(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[1]
    return uri.rsplit("/", 1)[-1]


def _humanize(camel: str) -> str:
    import re

    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", camel)
    return s[:1].upper() + s[1:] if s else s


def _xsd_curie(uri: str) -> str:
    if uri.startswith(_XSD_NS):
        return f"xsd:{uri[len(_XSD_NS) :]}"
    return "xsd:string"


def _find_module_dir(connector_root: Path) -> Path | None:
    """The one subdirectory shipping ``ontology/*.ttl`` (the connector's python package)."""
    candidates = sorted(
        d for d in connector_root.iterdir() if d.is_dir() and (d / "ontology").is_dir()
    )
    return candidates[0] if candidates else None


def _detect_ontology_source(graph: Any) -> str | None:
    """The ttl's own declared ``owl:Ontology`` IRI local slug (e.g. "servicenow"), if any.

    A connector's python package name (``servicenow-api``) commonly differs from the
    ontology domain slug its ttl already declares and the canonical ``ontology.ttl``
    already ``owl:imports`` (``servicenow``) — detecting it here (rather than assuming
    package name == IRI slug) is what makes the anti-sprawl "already wired" check land
    on the real, existing federated module instead of a false new-source guess.
    """
    import rdflib

    iris = sorted(
        str(s)
        for s in graph.subjects(
            predicate=rdflib.RDF.type,
            object=rdflib.URIRef("http://www.w3.org/2002/07/owl#Ontology"),
        )
    )
    for iri in iris:
        if iri.startswith("http://knuckles.team/kg/"):
            return _local(iri)
    return None


def _read_ontology(
    module_dir: Path,
) -> tuple[list[ResourceSpec], dict[str, SchemaMapping], list[str], str | None]:
    """Read every ``*.ttl`` in ``module_dir/ontology`` into resources + schema_mappings.

    Returns ``(resources, schema_mappings, review_todos, ontology_source)``, sorted
    deterministically by resource/local name.
    """
    import rdflib

    graph = rdflib.Graph()
    for ttl in sorted((module_dir / "ontology").glob("*.ttl")):
        graph.parse(str(ttl), format="turtle")

    ontology_source = _detect_ontology_source(graph)
    todos: list[str] = []

    class_uris = sorted(
        {
            str(s)
            for s in graph.subjects(
                predicate=rdflib.RDF.type, object=rdflib.URIRef(_OWL_CLASS)
            )
        }
    )
    class_locals = {_local(u) for u in class_uris}

    def _label(uri: str) -> str:
        for lbl in graph.objects(
            subject=rdflib.URIRef(uri), predicate=rdflib.URIRef(_RDFS_LABEL)
        ):
            return str(lbl)
        return _humanize(_local(uri))

    # datatype properties: global (domain-free) field vocabulary, matching the
    # existing fleet convention (ontology_leanix.ttl, servicenow.ttl, gitlab.ttl all
    # keep DatatypeProperty domain-free) — shared across every resource in this ttl set.
    fields: dict[str, str] = {}
    for uri in sorted(
        {
            str(s)
            for s in graph.subjects(
                predicate=rdflib.RDF.type, object=rdflib.URIRef(_OWL_DATATYPE_PROPERTY)
            )
        }
    ):
        rng = next(
            graph.objects(
                subject=rdflib.URIRef(uri), predicate=rdflib.URIRef(_RDFS_RANGE)
            ),
            None,
        )
        fields[_local(uri)] = _xsd_curie(str(rng)) if rng is not None else "xsd:string"

    # object properties: only attach to a resource when rdfs:domain is explicitly
    # declared AND resolves to a known class — never guess a domain (Wire-First: no
    # LLM/heuristic invention of structure the source ttl doesn't state).
    relations_by_domain: dict[str, list[ResourceRelation]] = {}
    for uri in sorted(
        {
            str(s)
            for s in graph.subjects(
                predicate=rdflib.RDF.type, object=rdflib.URIRef(_OWL_OBJECT_PROPERTY)
            )
        }
    ):
        domain = next(
            graph.objects(
                subject=rdflib.URIRef(uri), predicate=rdflib.URIRef(_RDFS_DOMAIN)
            ),
            None,
        )
        rng = next(
            graph.objects(
                subject=rdflib.URIRef(uri), predicate=rdflib.URIRef(_RDFS_RANGE)
            ),
            None,
        )
        target = (
            _local(str(rng))
            if rng is not None and _local(str(rng)) in class_locals
            else (_local(str(rng)) if rng is not None else "owl:Thing")
        )
        local = _local(uri)
        if domain is not None and _local(str(domain)) in class_locals:
            relations_by_domain.setdefault(_local(str(domain)), []).append(
                ResourceRelation(name=local, label=_label(uri), target=target)
            )
        else:
            todos.append(
                f"relation '{local}' has no declared rdfs:domain resolving to a known "
                f"resource in this connector's ontology — not attached to any resource; "
                f"verify its true domain manually."
            )

    resources: list[ResourceSpec] = []
    schema_mappings: dict[str, SchemaMapping] = {}
    for uri in class_uris:
        name = _local(uri)
        parent_ref = next(
            graph.objects(
                subject=rdflib.URIRef(uri), predicate=rdflib.URIRef(_RDFS_SUBCLASSOF)
            ),
            None,
        )
        crosswalk = (
            _local(str(parent_ref))
            if parent_ref is not None
            else DEFAULT_ARCHIMATE_CROSSWALK.get(name)
        )
        resources.append(
            ResourceSpec(
                name=name,
                label=_label(uri),
                id_prefix=name.lower(),
                relations=sorted(
                    relations_by_domain.get(name, []), key=lambda r: r.name
                ),
            )
        )
        schema_mappings[name] = SchemaMapping(
            ontology_class=crosswalk, fields=dict(sorted(fields.items()))
        )
        todos.append(
            f"schema_mappings.{name}.ontology_class = {crosswalk!r} is a heuristic "
            "identity crosswalk (subClassOf in the source ttl, else the default "
            "ArchiMate lookup by resource name) — verify manually."
        )

    return resources, schema_mappings, todos, ontology_source


def _read_sync(
    module_dir: Path,
) -> tuple[list[SyncSpec], IdentitySpec, list[EventSpec], PermissionsSpec]:
    presets_path = module_dir / "connectors" / "mcp_source_presets.json"
    sync: list[SyncSpec] = []
    identity = IdentitySpec()
    events: list[EventSpec] = []
    permissions = PermissionsSpec()
    if not presets_path.exists():
        return sync, identity, events, permissions

    data = json.loads(presets_path.read_text(encoding="utf-8"))
    for key in sorted(k for k in data if not k.startswith("_")):
        preset = data[key]
        if not isinstance(preset, dict):
            continue
        sync.append(
            SyncSpec(
                preset=key,
                server=str(preset.get("server", "")),
                tool=str(preset.get("tool", "")),
                action=preset.get("action"),
                records_path=preset.get("records_path"),
                id_field=preset.get("id_field"),
                title_field=preset.get("title_field"),
                text_field=preset.get("text_field"),
                updated_field=preset.get("updated_field"),
                pagination=preset.get("pagination"),
                doc_type=preset.get("doc_type"),
                raw=preset,
            )
        )
        doc_type = str(preset.get("doc_type") or key)
        for attr, bucket in (
            ("id_field", identity.id_field),
            ("title_field", identity.title_field),
            ("text_field", identity.text_field),
            ("updated_field", identity.updated_field),
        ):
            val = preset.get(attr)
            if val:
                bucket[doc_type] = str(val)
        if preset.get("updated_field"):
            events.append(
                EventSpec(
                    name=f"{key}.updated",
                    resource=doc_type,
                    description=(
                        f"Watermark event for '{key}' — advances on "
                        f"{preset['updated_field']} (mcp_source_presets.json)."
                    ),
                )
            )
        for field_name in sorted(preset):
            if field_name.startswith("acl_"):
                permissions.acl_fields.append(field_name)

    return sync, identity, events, permissions


def _read_actions(connector_root: Path) -> list[ActionSpec]:
    a2a_path = connector_root / "a2a.json"
    if not a2a_path.exists():
        return []
    data = json.loads(a2a_path.read_text(encoding="utf-8"))
    caps = data.get("capabilities") or []
    return [
        ActionSpec(
            id=str(c.get("id", "")),
            name=str(c.get("name", "")),
            description=str(c.get("description", "")),
        )
        for c in caps
        if isinstance(c, dict) and c.get("id")
    ]


def _pii_policy(schema_mappings: dict[str, SchemaMapping]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for resource, mapping in sorted(schema_mappings.items()):
        hits = sorted(
            f for f in mapping.fields if f.lower() in PII_HEURISTIC_FIELD_NAMES
        )
        if hits:
            out[resource] = hits
    return out


def build_manifest(
    connector_root: Path, *, now: datetime | None = None
) -> ConnectorManifest:
    """Build a :class:`ConnectorManifest` for one connector — pure, deterministic, offline."""
    connector = connector_root.name
    module_dir = _find_module_dir(connector_root)
    resources: list[ResourceSpec] = []
    schema_mappings: dict[str, SchemaMapping] = {}
    todos: list[str] = []
    sync: list[SyncSpec] = []
    identity = IdentitySpec()
    events: list[EventSpec] = []
    permissions = PermissionsSpec()
    source_artifacts: list[str] = []
    ontology_source: str | None = None

    if module_dir is not None:
        resources, schema_mappings, todos, ontology_source = _read_ontology(module_dir)
        source_artifacts.extend(
            str(p.relative_to(connector_root))
            for p in sorted((module_dir / "ontology").glob("*.ttl"))
        )
        sync, identity, events, permissions = _read_sync(module_dir)
        if (module_dir / "connectors" / "mcp_source_presets.json").exists():
            source_artifacts.append(
                str(
                    (module_dir / "connectors" / "mcp_source_presets.json").relative_to(
                        connector_root
                    )
                )
            )
    else:
        todos.append(
            "no <module>/ontology/*.ttl found for this connector — resources/"
            "schema_mappings could not be derived; ship an ontology module first."
        )

    actions = _read_actions(connector_root)
    if (connector_root / "a2a.json").exists():
        source_artifacts.append("a2a.json")

    policy = PolicySpec(pii_fields=_pii_policy(schema_mappings))
    if policy.pii_fields:
        todos.append(
            "policy.pii_fields was populated by a field-NAME heuristic "
            f"({sorted(PII_HEURISTIC_FIELD_NAMES)}) — verify against the actual data "
            "before relying on it for redaction/RLS."
        )
    todos.append(
        "policy.rls / policy.tenant_boundary are unset — no row-level-security "
        "predicate has been reviewed for this connector."
    )

    source_slug = ontology_source or connector
    placeholder = ConnectorManifest(
        connector=connector,
        ontology_source=(ontology_source or ""),
        resources=resources,
        actions=sorted(actions, key=lambda a: a.id),
        events=sorted(events, key=lambda e: e.name),
        identity=identity,
        permissions=permissions,
        schema_mappings=schema_mappings,
        sync=sync,
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="0" * 64)),
        policy=policy,
        review_todos=sorted(set(todos)),
    )

    spec = compile_manifest(placeholder)
    ttl = export_manifest_ttl(spec, source=source_slug)
    import rdflib

    g = rdflib.Graph()
    g.parse(data=ttl, format="turtle")
    digest, triple_count = ontology_integrity.canonical_hash(g)
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")

    provenance = ProvenanceSpec(
        generated_at=stamp,
        source_artifacts=sorted(source_artifacts),
        integrity=IntegrityInfo(hash=digest, triple_count=triple_count),
        signer=ontology_integrity.DEFAULT_SIGNER_ID,
        signature=ontology_integrity.sign(
            digest, signer_id=ontology_integrity.DEFAULT_SIGNER_ID
        ),
    )
    return placeholder.model_copy(update={"provenance": provenance})


def _to_yaml(manifest: ConnectorManifest) -> str:
    import yaml

    data = manifest.model_dump(mode="json", exclude_none=False)
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, width=100)


def write_manifest(
    connector_root: Path,
    output: Path,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
) -> ConnectorManifest:
    manifest = build_manifest(connector_root, now=now)
    text = _to_yaml(manifest)
    if dry_run:
        print(f"# --- {output} ---")
        print(text)
    else:
        output.write_text(text, encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--connector-root", type=Path, help="path to one connector repo (agents/<pkg>)"
    )
    ap.add_argument(
        "--connector", help="connector name, resolved as <agents-root>/<name>"
    )
    ap.add_argument(
        "--all", action="store_true", help="process every connector under --agents-root"
    )
    ap.add_argument("--agents-root", type=Path, help="the agents/ fleet root")
    ap.add_argument(
        "--output",
        type=Path,
        help="output path (single-connector mode; default <root>/connector_manifest.yml)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (--all mode); default: write into each connector root",
    )
    ap.add_argument(
        "--now", help="ISO-8601 UTC timestamp override, for reproducible runs"
    )
    ap.add_argument("--dry-run", action="store_true", help="print instead of write")
    args = ap.parse_args()

    now = (
        datetime.strptime(args.now, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        if args.now
        else None
    )

    roots: list[Path]
    if args.all:
        if not args.agents_root:
            ap.error("--all requires --agents-root")
        roots = sorted(
            d
            for d in args.agents_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    elif args.connector_root:
        roots = [args.connector_root]
    elif args.connector:
        if not args.agents_root:
            ap.error("--connector requires --agents-root")
        roots = [args.agents_root / args.connector]
    else:
        ap.error("one of --connector-root, --connector, or --all is required")
        return 2

    for root in roots:
        if not root.is_dir():
            print(f"skip: {root} is not a directory", file=sys.stderr)
            continue
        out = (
            args.output
            if (args.output and len(roots) == 1)
            else (
                (args.output_dir / f"{root.name}.connector_manifest.yml")
                if args.output_dir
                else (root / "connector_manifest.yml")
            )
        )
        manifest = write_manifest(root, out, now=now, dry_run=args.dry_run)
        print(
            f"generated {out}: {len(manifest.resources)} resources, {len(manifest.sync)} sync presets"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
