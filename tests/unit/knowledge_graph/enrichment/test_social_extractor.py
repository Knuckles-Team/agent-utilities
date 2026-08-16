"""Tests for the free-first deterministic social entity extractor.

CONCEPT:AU-KG.ingest.deterministic-social-entity-mining.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors.social import (
    StructuredEntities,
    extract_structured_entities,
    resolve_known_tools,
    to_kg_rows,
)


def test_extract_structured_entities_v2_shape() -> None:
    """Top-level ``entities.*`` (API v2-style) hashtags/mentions/urls extract."""
    record = {
        "entities": {
            "hashtags": [{"tag": "AI"}, {"tag": "buildinpublic"}],
            "user_mentions": [{"username": "OpenAI"}],
            "urls": [
                {"expanded_url": "https://github.com/foo/bar"},
                {"expanded_url": "https://vercel.com/docs"},
                {"expanded_url": "https://x.com/i/status/123"},
            ],
        }
    }
    entities = extract_structured_entities(record, exclude_url_hosts=("x.com",))

    assert entities.hashtags == ["ai", "buildinpublic"]
    assert entities.mentions == ["openai"]
    assert entities.urls == [
        "https://github.com/foo/bar",
        "https://vercel.com/docs",
    ]
    assert entities.tools == ["GitHub", "Vercel"]


def test_extract_structured_entities_legacy_v1_shape() -> None:
    """The ``legacy.entities.*`` wrapper (API v1.1-style) is tolerated identically,
    including the ``text``/``screen_name`` field-name drift from v2."""
    record = {
        "legacy": {
            "entities": {
                "hashtags": [{"text": "python"}],
                "user_mentions": [{"screen_name": "github"}],
                "urls": [{"url": "https://docs.github.com/api"}],
            }
        }
    }
    entities = extract_structured_entities(record)

    assert entities.hashtags == ["python"]
    assert entities.mentions == ["github"]
    # Registered-suffix match: docs.github.com -> github.com -> GitHub.
    assert entities.tools == ["GitHub"]


def test_extract_structured_entities_never_raises_on_malformed_input() -> None:
    """Schema-defensive: absent/malformed ``entities`` yields an empty result,
    never an exception — the free-first stage must not be able to break
    ingestion of a record whose shape doesn't match what we expect."""
    assert extract_structured_entities(None).is_empty()
    assert extract_structured_entities({}).is_empty()
    assert extract_structured_entities({"entities": "not-a-dict"}).is_empty()
    assert extract_structured_entities(
        {"entities": {"hashtags": "not-a-list"}}
    ).is_empty()
    assert extract_structured_entities(
        {"entities": {"hashtags": [{"tag": None}, "not-a-dict", {}]}}
    ).is_empty()


def test_resolve_known_tools_exact_and_suffix_match() -> None:
    urls = [
        "https://github.com/foo/bar",
        "https://docs.github.com/api",  # suffix match -> same tool, deduped
        "https://example.com/unknown",  # no match -> ignored
        "https://notion.so/page",
    ]
    assert resolve_known_tools(urls) == ["GitHub", "Notion"]


def test_resolve_known_tools_malformed_url_is_ignored() -> None:
    assert resolve_known_tools(["not a url", ""]) == []


def test_to_kg_rows_shape_and_provenance_stamping() -> None:
    entities = StructuredEntities(
        hashtags=["ai"], mentions=["openai"], tools=["GitHub"]
    )
    nodes, edges = to_kg_rows(entities, document_id="doc:1", confidence=0.9)

    nodes_by_id = {n["id"]: n for n in nodes}
    assert nodes_by_id["hashtag:ai"]["node_type"] == "Hashtag"
    assert nodes_by_id["mention:openai"]["node_type"] == "Mention"
    assert nodes_by_id["tool:github"]["node_type"] == "Tool"
    # Every node from this stage is stamped as such, distinguishable from any
    # later LLM-derived enrichment of the same document.
    assert all(n["extraction_stage"] == "deterministic" for n in nodes)
    assert all(n["confidence"] == 0.9 for n in nodes)

    edge_rels = {(e["source"], e["target"], e["relationship"]) for e in edges}
    assert ("doc:1", "hashtag:ai", "taggedWithHashtag") in edge_rels
    assert ("doc:1", "mention:openai", "mentionsHandle") in edge_rels
    assert ("doc:1", "tool:github", "referencesTool") in edge_rels


def test_to_kg_rows_empty_entities_yields_nothing() -> None:
    nodes, edges = to_kg_rows(StructuredEntities(), document_id="doc:1")
    assert nodes == []
    assert edges == []
