"""Connector Ontology Manifest compiler (CONCEPT:AU-KG.ontology.connector-manifest-compiler).

Generalizes ``ontology.leanix_metamodel``'s ``compile_leanix_metamodel`` /
``export_leanix_ttl`` / ``apply_leanix_metamodel`` (LeanIX-only) into a source-agnostic
pipeline over :class:`connector_manifest.ConnectorManifest`:

  ``compile_manifest`` — manifest -> :class:`~connector_manifest.OntologySpec` (OWL terms).
  ``export_manifest_ttl`` — typed spec -> EG ontology pack compiler.
  ``manifest_from_leanix_spec`` — makes LeanIX the **first caller** of this generalized
    compiler (proved lossless by the golden-file test in ``tests/``), without touching
    the existing production ``sync_leanix_ontology`` entry point.

Pure, deterministic, zero LLM calls — AU projects the manifest's declared fields;
EG owns Turtle syntax and semantic pack compilation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from epistemic_graph.ontology_pack import compile_ontology_pack

from .connector_manifest import (
    ConnectorManifest,
    OntologyClassSpec,
    OntologyDatatypePropertySpec,
    OntologyObjectPropertySpec,
    OntologySpec,
)

if TYPE_CHECKING:
    from .leanix_metamodel import LeanixOntologySpec

__all__ = [
    "compile_manifest",
    "export_manifest_ttl",
    "manifest_from_leanix_spec",
]


def _humanize(camel: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", camel)
    return s[:1].upper() + s[1:] if s else s


def compile_manifest(manifest: ConnectorManifest) -> OntologySpec:
    """Compile a :class:`ConnectorManifest` into a source-agnostic :class:`OntologySpec`."""
    spec = OntologySpec()
    seen_dtp: set[str] = set()

    for resource in manifest.resources:
        mapping = manifest.schema_mappings.get(resource.name)
        parent = mapping.ontology_class if mapping else None
        id_prefix = resource.id_prefix or resource.name.lower()
        spec.classes.append(
            OntologyClassSpec(
                local=resource.name,
                label=resource.label or _humanize(resource.name),
                parent=parent,
                id_prefix=id_prefix,
            )
        )
        spec.type_map[resource.name] = (resource.name, id_prefix)

        if mapping is not None:
            for fname, xsd in mapping.fields.items():
                if fname in seen_dtp:
                    continue
                seen_dtp.add(fname)
                spec.datatype_properties.append(
                    OntologyDatatypePropertySpec(
                        local=fname, label=_humanize(fname), range=xsd
                    )
                )

        for rel in resource.relations:
            lpg = rel.lpg_rel_type or _upper_snake(rel.name)
            spec.object_properties.append(
                OntologyObjectPropertySpec(
                    local=rel.name,
                    label=rel.label or _humanize(rel.name),
                    domain=resource.name,
                    range=rel.target,
                    lpg_rel_type=lpg,
                )
            )
            spec.relation_map[rel.name] = (lpg, rel.target)

    return spec


def _upper_snake(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", s)
    return s.upper()


def export_manifest_ttl(spec: OntologySpec, *, source: str) -> str:
    """Send typed declarations to EG's ontology pack compiler."""
    return compile_ontology_pack(
        source=source,
        classes=[
            {"local": c.local, "label": c.label, "parent": c.parent}
            for c in spec.classes
        ],
        object_properties=[
            {"local": p.local, "label": p.label, "domain": p.domain, "range": p.range}
            for p in spec.object_properties
        ],
        datatype_properties=[
            {"local": p.local, "label": p.label, "range": p.range}
            for p in spec.datatype_properties
        ],
    )


def manifest_from_leanix_spec(
    spec: LeanixOntologySpec, *, source: str = "leanix"
) -> ConnectorManifest:
    """Adapt an already-compiled :class:`LeanixOntologySpec` into a :class:`ConnectorManifest`.

    Makes LeanIX the **first caller** of the generalized manifest compiler
    (CONCEPT:AU-KG.ontology.connector-manifest-compiler) without touching the existing
    production ``sync_leanix_ontology``/``apply_leanix_metamodel`` entry point — this is
    the seam the golden-file regression test exercises to prove
    ``export_manifest_ttl(compile_manifest(...))`` reproduces ``export_leanix_ttl`` losslessly.
    """
    from .connector_manifest import (
        IntegrityInfo,
        ProvenanceSpec,
        ResourceRelation,
        ResourceSpec,
        SchemaMapping,
    )

    relations_by_domain: dict[str, list[ResourceRelation]] = {}
    for p in spec.object_properties:
        relations_by_domain.setdefault(p.domain, []).append(
            ResourceRelation(
                name=p.local, label=p.label, target=p.range, lpg_rel_type=p.lpg_rel_type
            )
        )

    fields: dict[str, str] = {d.local: d.range for d in spec.datatype_properties}

    resources = [
        ResourceSpec(
            name=c.local,
            label=c.label,
            id_prefix=c.id_prefix,
            relations=relations_by_domain.get(c.local, []),
        )
        for c in spec.classes
    ]
    # Matches leanix_metamodel's global (domain-free) datatype-property pool: every
    # resource's schema mapping carries the full shared field vocabulary.
    schema_mappings = {
        c.local: SchemaMapping(ontology_class=c.parent, fields=dict(fields))
        for c in spec.classes
    }

    placeholder = ProvenanceSpec(
        generated_by="manifest_from_leanix_spec",
        integrity=IntegrityInfo(hash="0" * 64),
    )
    return ConnectorManifest(
        connector=source,
        resources=resources,
        schema_mappings=schema_mappings,
        provenance=placeholder,
    )
