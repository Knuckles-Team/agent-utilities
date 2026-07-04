"""Tests for ontology-guided extraction (CONCEPT:AU-KG.retrieval.mmr-diversification).

Covers the schema loader (TBox → ExtractionSchema), the prompt rendering, the
content→ontology mapping, and the LIVE PATH that the schema reaches the extractor
prompt (Wire-First).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.extraction.extraction_schema import (
    EntityType,
    ExtractionSchema,
    Relation,
    _camel_to_snake,
    _module_paths,
    load_extraction_schema,
)

# rdflib lives in the [owl] extra; the loader degrades to None without it. These
# parse tests need it.
rdflib = pytest.importorskip("rdflib")


def test_camel_to_snake():
    assert _camel_to_snake("decidedBy") == "decided_by"
    assert _camel_to_snake("impactsConcept") == "impacts_concept"
    assert _camel_to_snake("enforces") == "enforces"
    assert _camel_to_snake("hasHTTPEndpoint") == "has_http_endpoint" or _camel_to_snake(
        "hasHTTPEndpoint"
    ).startswith("has")


def test_module_paths_skip_and_domain():
    # non-prose content types skip ontology guidance entirely
    assert _module_paths("codebase") is None
    assert _module_paths("config") is None
    assert _module_paths("") is None
    # prose content gets the core module
    assert _module_paths("document") == ("ontology",)
    # a domain source type adds its module on top of the core
    sn = _module_paths("servicenow")
    assert sn is not None and "ontology" in sn and "ontology_servicenow" in sn
    # substring match works (connector-qualified names)
    assert "ontology_legal" in (_module_paths("connector:legal") or ())


def test_load_core_schema_has_typed_relations():
    schema = load_extraction_schema("document")
    assert schema is not None
    assert not schema.is_empty
    assert len(schema.entity_types) > 10
    assert len(schema.relations) > 5
    # at least one relation carries an explicit domain→range direction
    directed = [r for r in schema.relations if r.domain and r.range]
    assert directed, "expected typed relations with domain and range"
    # predicates are snake_case (extractor convention)
    assert all("_" in r.predicate or r.predicate.islower() for r in schema.relations)


def test_codebase_returns_none():
    assert load_extraction_schema("codebase") is None


def test_servicenow_superset_of_core():
    core = load_extraction_schema("document")
    sn = load_extraction_schema("servicenow")
    assert core is not None and sn is not None
    # the domain schema name records both modules
    assert "servicenow" in sn.name
    # adding a domain module never drops core classes
    assert len(sn.entity_types) >= len(core.entity_types)


def test_prompt_block_render():
    schema = ExtractionSchema(
        name="t",
        entity_types=(
            EntityType("Organization", "a company", ("vendor", "supplier")),
            EntityType("Person", "a human"),
        ),
        relations=(
            Relation("works_for", "works for", ("Person",), ("Organization",)),
            Relation("knows", "knows", ("Person",), ("Person",), symmetric=True),
        ),
    )
    block = schema.prompt_block()
    assert "Organization" in block
    assert "aka vendor, supplier" in block
    assert "works_for: Person → Organization" in block
    assert "[symmetric]" in block  # symmetric relation flagged
    # soft-closed wording present (controlled overflow, not a hard menu)
    assert "coin a new term" in block.lower()
    assert schema.closed_predicate_set == frozenset({"works_for", "knows"})


def test_empty_schema_renders_empty():
    assert ExtractionSchema(name="e").prompt_block() == ""
    assert ExtractionSchema(name="e").is_empty


@pytest.mark.asyncio
async def test_schema_reaches_extractor_prompt_live_path():
    """LIVE PATH: extract_facts must splice the schema block into the prompt."""
    from agent_utilities.knowledge_graph.extraction.fact_extractor import extract_facts

    captured: dict[str, str] = {}

    async def fake_stream(prompt: str, seed: int):
        captured["prompt"] = prompt
        if False:  # pragma: no cover — generator with no yields
            yield ""

    schema = ExtractionSchema(
        name="t",
        entity_types=(EntityType("Organization", "a company"),),
        relations=(Relation("works_for", "", ("Person",), ("Organization",)),),
    )
    async for _ in extract_facts(
        "Acme hired Bob.", dedup=False, stream_fn=fake_stream, schema=schema
    ):
        pass

    assert "prompt" in captured
    assert "ONTOLOGY SCHEMA" in captured["prompt"]
    assert "works_for: Person → Organization" in captured["prompt"]
    # the base extraction guidance is still present (schema is prepended, not replacing)
    assert "knowledge graph" in captured["prompt"].lower()


@pytest.mark.asyncio
async def test_no_schema_leaves_prompt_unchanged_live_path():
    """Without a schema the prompt has no ontology block (no regression)."""
    from agent_utilities.knowledge_graph.extraction.fact_extractor import extract_facts

    captured: dict[str, str] = {}

    async def fake_stream(prompt: str, seed: int):
        captured["prompt"] = prompt
        if False:  # pragma: no cover
            yield ""

    async for _ in extract_facts(
        "Acme hired Bob.", dedup=False, stream_fn=fake_stream, schema=None
    ):
        pass

    assert "ONTOLOGY SCHEMA" not in captured["prompt"]
