"""Tests for ontology-aware schema discovery (CONCEPT:AU-KG.ontology.do-not-auto-merge)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.extraction.schema_discovery import (
    DiscoveredType,
    classify_candidate,
    discover_schema_extensions,
    discovery_report,
    generate_standalone_ontology,
    ontology_generation_report,
    parse_discovery_response,
    to_ttl_fragment,
)


def test_parse_discovery_response():
    raw = '```json\n{"entity_types":[{"name":"Drug"}],"relation_types":[{"name":"treats","domain":"Drug","range":"Disease"}]}\n```'
    ents, rels = parse_discovery_response(raw)
    assert ents == [{"name": "Drug"}]
    assert rels[0]["name"] == "treats"


def test_parse_discovery_bad_json():
    assert parse_discovery_response("garbage") == ([], [])
    assert parse_discovery_response("") == ([], [])


def test_classify_candidate_covered_synonym_missing():
    classes = {"organization", "person"}
    preds = {"works_for"}
    syn = {"vendor": "organization", "company": "organization"}
    # exact existing class
    assert (
        classify_candidate("Organization", "class", classes, preds, syn)[0] == "covered"
    )
    # synonym maps to existing
    assert classify_candidate("vendor", "class", classes, preds, syn)[0] == "synonym"
    # brand-new class
    assert (
        classify_candidate("ZorblaxWidget", "class", classes, preds, syn)[0]
        == "missing"
    )
    # existing predicate
    assert (
        classify_candidate("worksFor", "property", classes, preds, syn)[0] == "covered"
    )
    # new predicate
    assert classify_candidate("zaps", "property", classes, preds, syn)[0] == "missing"


def test_discover_schema_extensions_classifies(monkeypatch):
    def fake_llm(prompt: str) -> str:
        return (
            '{"entity_types":[{"name":"ZorblaxWidget","description":"a widget"}],'
            '"relation_types":[{"name":"zaps","domain":"ZorblaxWidget","range":"Target"}]}'
        )

    discovered = discover_schema_extensions(["some sample text"], "document", fake_llm)
    names = {d.name for d in discovered}
    assert "ZorblaxWidget" in names
    assert "zaps" in names
    # the novel widget is missing (not in the ontology)
    widget = next(d for d in discovered if d.name == "ZorblaxWidget")
    assert widget.classification == "missing"


def test_discover_no_llm_returns_empty():
    assert discover_schema_extensions(["text"], "document", None) == []


def test_to_ttl_fragment_only_missing():
    discovered = [
        DiscoveredType("ZorblaxWidget", "class", "a widget", classification="missing"),
        DiscoveredType("Organization", "class", classification="covered"),
        DiscoveredType(
            "zaps",
            "property",
            domain="ZorblaxWidget",
            range="Target",
            classification="missing",
        ),
    ]
    ttl = to_ttl_fragment(discovered)
    assert ":ZorblaxWidget a owl:Class" in ttl
    assert "RESERVE-PENDING" in ttl  # never hardcode a concept id
    assert ":zaps a owl:ObjectProperty" in ttl
    assert "rdfs:domain :ZorblaxWidget" in ttl
    assert "rdfs:range :Target" in ttl
    # covered classes are NOT proposed
    assert "Organization a owl:Class" not in ttl


def test_to_ttl_fragment_empty_when_nothing_missing():
    discovered = [DiscoveredType("Organization", "class", classification="covered")]
    assert to_ttl_fragment(discovered) == ""


def test_discovery_report_structure():
    discovered = [
        DiscoveredType("ZorblaxWidget", "class", classification="missing"),
        DiscoveredType("Organization", "class", classification="covered"),
    ]
    report = discovery_report(discovered)
    assert report["counts"]["missing"] == 1
    assert report["counts"]["covered"] == 1
    assert len(report["candidates"]) == 2
    assert "ZorblaxWidget" in report["ttl_proposal"]


# ── From-scratch generation (report row #13, against an EMPTY base) ─────────


def test_generate_standalone_ontology_never_diffs_live_ontology(monkeypatch):
    """Even a name the live ontology already covers is proposed in full.

    ``generate_standalone_ontology`` runs the identical LLM discovery path but
    with ``against_live_ontology=False``, so — unlike ``discover_schema_extensions``
    — it never loads ``extraction_schema``/``ontology_grounding`` to filter
    "covered" candidates out.
    """

    def fake_llm(prompt: str) -> str:
        return (
            '{"entity_types":[{"name":"Organization","description":"an org"}],'
            '"relation_types":[{"name":"employs","domain":"Organization","range":"Person"}]}'
        )

    discovered = generate_standalone_ontology(["some sample text"], "hr", fake_llm)
    names = {d.name for d in discovered}
    assert "Organization" in names
    assert "employs" in names
    # Every candidate classifies "missing" — a complete proposal, not a diff.
    assert all(d.classification == "missing" for d in discovered)


def test_generate_standalone_ontology_no_llm_returns_empty():
    assert generate_standalone_ontology(["text"], "document", None) == []


def test_ontology_generation_report_shapes_interfaces_and_link_types():
    discovered = [
        DiscoveredType(
            "VetClinic", "class", "a veterinary clinic", classification="missing"
        ),
        DiscoveredType(
            "treats",
            "property",
            "clinic treats an animal",
            domain="VetClinic",
            range="Animal",
            classification="missing",
        ),
    ]
    report = ontology_generation_report(discovered, domain_hint="veterinary")
    assert report["domain_hint"] == "veterinary"
    assert report["interfaces"] == [
        {"name": "VetClinic", "description": "a veterinary clinic", "properties": []}
    ]
    assert report["link_types"] == [
        {
            "name": "treats",
            "description": "clinic treats an animal",
            "source_type": "VetClinic",
            "target_type": "Animal",
        }
    ]
    assert report["counts"] == {"interfaces": 1, "link_types": 1}
    assert ":VetClinic a owl:Class" in report["ttl_proposal"]
    assert ":treats a owl:ObjectProperty" in report["ttl_proposal"]
