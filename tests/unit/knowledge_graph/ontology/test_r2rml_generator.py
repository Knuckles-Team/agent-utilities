"""Tests for CA-23's R2RML generator (CONCEPT:AU-KG.ontology.r2rml-generator).

Covers: deterministic Turtle generation per ``DEC-CA-06``'s generation-rules table,
the never-guess rule for a relation whose target does not resolve to a declared
resource, and the ``derive_stream_mapping``/``owl_bridge`` dict-parity comparison
CA-23-W05 requires -- including the measured NON-parity for the 3 sources the dict
covers today (documented, not silently dropped).
"""

from __future__ import annotations

import yaml

from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    ResourceRelation,
    ResourceSpec,
    SchemaMapping,
)
from agent_utilities.knowledge_graph.ontology.r2rml_generator import (
    derive_stream_mapping,
    generate_r2rml,
    generate_r2rml_report,
    unmapped_relations,
)

_GITLAB_MANIFEST_PATH = (
    "agent_utilities/knowledge_graph/ontology/connector_manifests/"
    "gitlab-api/connector_manifest.yml"
)
_SERVICENOW_MANIFEST_PATH = (
    "agent_utilities/knowledge_graph/ontology/connector_manifests/"
    "servicenow-api/connector_manifest.yml"
)


def _load_manifest(path: str) -> ConnectorManifest:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return ConnectorManifest.model_validate(data)


def _small_manifest() -> ConnectorManifest:
    return ConnectorManifest(
        connector="acme",
        resources=[
            ResourceSpec(
                name="Order",
                id_prefix="order",
                relations=[
                    ResourceRelation(name="placedBy", target="Person"),
                    # unresolved target -- must never get a guessed TriplesMap ref
                    ResourceRelation(name="shippedVia", target="owl:Thing"),
                ],
            ),
            ResourceSpec(name="Person", id_prefix="person"),
        ],
        schema_mappings={
            "Order": SchemaMapping(fields={"total": "xsd:decimal"}),
            "Person": SchemaMapping(fields={"email": "xsd:string"}),
        },
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="0" * 64)),
    )


# ── generation-rules table ──────────────────────────────────────────────────────


def test_generate_r2rml_emits_one_triples_map_per_resource():
    manifest = _small_manifest()
    report = generate_r2rml_report(manifest)
    assert report.triples_map_count == 2
    assert report.mapped_resources == ("Order", "Person")
    assert "<#TriplesMap_Order>" in report.turtle
    assert "<#TriplesMap_Person>" in report.turtle


def test_subject_map_template_uses_id_prefix():
    manifest = _small_manifest()
    ttl = generate_r2rml(manifest)
    assert 'rr:template "http://knuckles.team/kg/order/{id}"' in ttl
    assert 'rr:template "http://knuckles.team/kg/person/{id}"' in ttl


def test_subject_map_falls_back_to_lowercased_name_without_id_prefix():
    manifest = ConnectorManifest(
        connector="acme",
        resources=[ResourceSpec(name="Widget")],  # no id_prefix
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="0" * 64)),
    )
    ttl = generate_r2rml(manifest)
    assert 'rr:template "http://knuckles.team/kg/widget/{id}"' in ttl


def test_resolved_relation_becomes_predicate_object_map_with_parent_triples_map():
    manifest = _small_manifest()
    ttl = generate_r2rml(manifest)
    assert "rr:predicate :placedBy" in ttl
    assert "rr:parentTriplesMap <#TriplesMap_Person>" in ttl


def test_datatype_field_becomes_column_predicate_object_map():
    manifest = _small_manifest()
    ttl = generate_r2rml(manifest)
    assert 'rr:predicate :total ; rr:objectMap [ rr:column "total" ]' in ttl


def test_unresolved_relation_target_never_guessed():
    manifest = _small_manifest()
    report = generate_r2rml_report(manifest)
    assert "shippedVia" not in report.turtle
    assert report.unmapped_relations == ("Order.shippedVia -> owl:Thing",)
    assert unmapped_relations(manifest) == ["Order.shippedVia -> owl:Thing"]


def test_generation_is_deterministic():
    manifest = _small_manifest()
    assert generate_r2rml(manifest) == generate_r2rml(manifest)


def test_output_parses_as_valid_turtle():
    import rdflib

    manifest = _small_manifest()
    ttl = generate_r2rml(manifest)
    g = rdflib.Graph()
    g.parse(data=ttl, format="turtle")  # raises on malformed Turtle
    assert len(g) > 0


# ── real connector manifests (the 3 sources the dict covers) ───────────────────


def test_generate_r2rml_against_real_gitlab_manifest():
    manifest = _load_manifest(_GITLAB_MANIFEST_PATH)
    report = generate_r2rml_report(manifest)
    assert "Project" in report.mapped_resources
    assert "Pipeline" in report.mapped_resources
    assert report.triples_map_count == len(manifest.resources)
    # gitlab-api's unattached relations (authoredBy/belongsToProject/hasMilestone/
    # triggeredPipeline) never reached the manifest as ResourceRelation entries at
    # all -- nothing here for the generator to skip on their account. The ONE
    # relation this manifest DOES declare with an unresolved target is
    # PipelineRun.ranFor -> owl:Thing (never guessed, per DEC-CA-06).
    assert report.unmapped_relations == ("PipelineRun.ranFor -> owl:Thing",)


def test_generate_r2rml_against_real_servicenow_manifest():
    manifest = _load_manifest(_SERVICENOW_MANIFEST_PATH)
    report = generate_r2rml_report(manifest)
    assert "Incident" in report.mapped_resources
    assert report.triples_map_count == len(manifest.resources)


# ── owl_bridge dict-parity comparison (CA-23-W05) ───────────────────────────────

# The exact 3 entries of owl_bridge.py's hardcoded r2rml_mappings dict, reproduced
# here (not imported -- the dict is process-local to stream_api_to_graph) so this
# test is a documented, explicit comparison against the values CA-23-W05 must prove
# equivalent before the dict can be deleted.
_DICT_SERVICENOW_INCIDENT = {
    "class": "Incident",
    "id_field": "sys_id",
    "properties": ["short_description", "severity", "state", "description"],
    "edges": {
        "assigned_to": ("Person", "was_attributed_to"),
        "cmdb_ci": ("PlatformService", "monitors"),
    },
}
_DICT_GITLAB_PROJECT = {
    "class": "Repository",
    "id_field": "id",
    "properties": ["name", "path_with_namespace", "description"],
    "edges": {"owner": ("Person", "creator")},
}
_DICT_GITLAB_PIPELINE = {
    "class": "Pipeline",
    "id_field": "id",
    "properties": ["status", "ref", "sha"],
    "edges": {"project_id": ("Repository", "part_of")},
}


def test_derive_stream_mapping_returns_none_for_unknown_resource():
    manifest = _small_manifest()
    assert derive_stream_mapping(manifest, "NoSuchResource") is None


def test_derive_stream_mapping_servicenow_incident_class_and_id_field_match_dict():
    manifest = _load_manifest(_SERVICENOW_MANIFEST_PATH)
    derived = derive_stream_mapping(manifest, "Incident")
    assert derived is not None
    assert derived["class"] == _DICT_SERVICENOW_INCIDENT["class"]
    # dict id_field is the raw API field ("sys_id"); the manifest only carries the
    # node-id prefix convention -- these are DIFFERENT axes (an id FIELD name vs an
    # id PREFIX), not comparable 1:1. Document rather than silently assert equal.
    assert derived["id_field"] == "incident"  # the manifest's id_prefix, not sys_id


def test_derive_stream_mapping_does_not_reach_edge_parity_with_dict_servicenow():
    """MEASURED GAP (CA-23-W05): the live servicenow-api manifest's Incident resource
    declares zero relations -- the dict's assigned_to/cmdb_ci edges are not
    reproducible from this manifest today. Proves the gap explicitly rather than
    faking equivalence."""
    manifest = _load_manifest(_SERVICENOW_MANIFEST_PATH)
    derived = derive_stream_mapping(manifest, "Incident")
    assert derived is not None
    assert derived["edges"] == {}
    assert set(_DICT_SERVICENOW_INCIDENT["edges"]) - set(derived["edges"]) == {
        "assigned_to",
        "cmdb_ci",
    }


def test_derive_stream_mapping_does_not_reach_edge_parity_with_dict_gitlab_project():
    manifest = _load_manifest(_GITLAB_MANIFEST_PATH)
    derived = derive_stream_mapping(manifest, "Project")
    assert derived is not None
    # the manifest DOES declare a relation on Project ("partOfGroup"), but it is not
    # the dict's "owner" edge -- confirms the manifest's relation set is disjoint
    # from the dict's, not a subset/superset either direction.
    assert "owner" not in derived["edges"]


def test_derive_stream_mapping_does_not_reach_edge_parity_with_dict_gitlab_pipeline():
    manifest = _load_manifest(_GITLAB_MANIFEST_PATH)
    derived = derive_stream_mapping(manifest, "Pipeline")
    assert derived is not None
    assert "project_id" not in derived["edges"]
