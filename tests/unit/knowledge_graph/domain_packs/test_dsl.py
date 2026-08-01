"""Pure evaluation tests for the mapping DSL (CONCEPT:AU-KG.ingest.mapping-dsl).

Proves the DSL's inspectability claim directly: given fragments, a mapping
produces EXACTLY the facts an operator can predict from reading it — computed
by :func:`evaluate_mapping` alone, with no engine, no pack loader, no I/O.
"""

from __future__ import annotations

import _fixtures

from agent_utilities.knowledge_graph.domain_packs.domain_pack import HeadingMapping
from agent_utilities.knowledge_graph.domain_packs.dsl import (
    MappingError,
    evaluate_mapping,
)
from agent_utilities.knowledge_graph.domain_packs.fragment_contract import (
    Artifact,
    Fragment,
)


def _case_artifact_and_fragments():
    case = _fixtures._synthetic_evaluation_case()
    artifact = Artifact(**case.artifact)
    fragments = [Fragment(**f) for f in case.fragments]
    return case, artifact, fragments


def test_frontmatter_table_link_mapping_produces_exact_predicted_facts():
    manifest = _fixtures.build_manifest(evaluation_cases=[])
    case, artifact, fragments = _case_artifact_and_fragments()

    result = evaluate_mapping(manifest, artifact, fragments)

    actual_entities = {frozenset(e.items()) for e in result.entity_dicts()}
    expected_entities = {frozenset(e.items()) for e in case.expect_entities}
    assert actual_entities == expected_entities

    actual_rels = {frozenset(r.items()) for r in result.relationship_dicts()}
    expected_rels = {frozenset(r.items()) for r in case.expect_relationships}
    assert actual_rels == expected_rels

    assert result.unmatched_fragment_ids == []


def test_every_produced_fact_is_traceable_to_its_rule():
    """Every fact names the rule that produced it — the inspectability the
    charter asks for ("read a mapping and predict what facts it will
    produce"). Edge-target/row/link facts additionally carry the exact
    fragment_id they came from."""
    manifest = _fixtures.build_manifest(evaluation_cases=[])
    _, artifact, fragments = _case_artifact_and_fragments()

    result = evaluate_mapping(manifest, artifact, fragments)

    assert all(e.rules for e in result.entities)
    assert all(r.rule and r.fragment_id for r in result.relationships)
    person_entities = [e for e in result.entities if e.node_type == "Person"]
    assert person_entities and all(e.fragment_ids for e in person_entities)


def test_unmatched_fragment_is_reported_not_silently_dropped():
    manifest = _fixtures.build_manifest(evaluation_cases=[])
    artifact = Artifact(artifact_id="md:x", source_path="x.md", content_hash="0" * 64)
    stray = Fragment(
        fragment_id="md:x#paragraph:0",
        artifact_id="md:x",
        fragment_type="paragraph",
        locator="paragraph[0]",
        content_hash="a" * 64,
        value="just prose, no rule targets this",
    )

    result = evaluate_mapping(manifest, artifact, [stray])

    assert result.entities == []
    assert result.relationships == []
    assert result.unmatched_fragment_ids == [stray.fragment_id]


def test_heading_mapping_creates_section_and_parent_edge():
    manifest = _fixtures.build_manifest(
        mappings=[
            HeadingMapping(
                heading_pattern="^Steps$",
                node_type="Document",
                id_template="{artifact_id}#section:{heading_path}",
                parent_relation="hasSection",
                parent_id_template="{artifact_id}",
            )
        ],
        evaluation_cases=[],
    )
    artifact = Artifact(artifact_id="md:h", source_path="h.md", content_hash="0" * 64)
    heading_fragment = Fragment(
        fragment_id="md:h#heading:Steps",
        artifact_id="md:h",
        fragment_type="heading_section",
        locator="heading[Steps]",
        content_hash="b" * 64,
        value="Steps",
        metadata={"heading": "Steps", "heading_path": "Steps", "level": 1},
    )

    result = evaluate_mapping(manifest, artifact, [heading_fragment])

    assert result.entity_dicts() == [
        {"id": "md:h#section:Steps", "node_type": "Document", "title": "Steps"}
    ]
    assert result.relationship_dicts() == [
        {"source": "md:h", "target": "md:h#section:Steps", "relationship": "hasSection"}
    ]


def test_template_referencing_unknown_field_raises_loudly():
    manifest = _fixtures.build_manifest(
        mappings=[
            HeadingMapping(
                heading_pattern=".*",
                node_type="Document",
                # references a context key the heading rule never provides
                id_template="{artifact_id}#{does_not_exist}",
            )
        ],
        evaluation_cases=[],
    )
    artifact = Artifact(artifact_id="md:e", source_path="e.md", content_hash="0" * 64)
    heading_fragment = Fragment(
        fragment_id="md:e#heading:Intro",
        artifact_id="md:e",
        fragment_type="heading_section",
        locator="heading[Intro]",
        content_hash="c" * 64,
        value="Intro",
        metadata={"heading": "Intro", "heading_path": "Intro"},
    )

    try:
        evaluate_mapping(manifest, artifact, [heading_fragment])
        raised = False
    except MappingError:
        raised = True
    assert raised, "an unresolvable id template must raise, never silently truncate"
