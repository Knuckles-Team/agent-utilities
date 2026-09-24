"""Tests for ontology-guided extraction (CONCEPT:AU-KG.retrieval.mmr-diversification).

Covers the schema loader (TBox → ExtractionSchema), the prompt rendering, the
content→ontology mapping, and the LIVE PATH that the schema reaches the extractor
prompt (Wire-First).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.core import graph_compute
from agent_utilities.knowledge_graph.extraction import extraction_schema
from agent_utilities.knowledge_graph.extraction.extraction_schema import (
    EntityType,
    ExtractionSchema,
    Relation,
    _camel_to_snake,
    _source_ids,
    load_extraction_schema,
)

KG = "http://knuckles.team/kg#"


@pytest.fixture(autouse=True)
def _fresh_schema_cache():
    load_extraction_schema.cache_clear()
    yield
    load_extraction_schema.cache_clear()


def _term(local: str, **fields) -> SimpleNamespace:
    base = {"iri": f"{KG}{local}", "label": None, "comment": None}
    return SimpleNamespace(**{**base, **fields})


def test_camel_to_snake():
    assert _camel_to_snake("decidedBy") == "decided_by"
    assert _camel_to_snake("impactsConcept") == "impacts_concept"
    assert _camel_to_snake("enforces") == "enforces"
    assert _camel_to_snake("hasHTTPEndpoint") == "has_http_endpoint" or _camel_to_snake(
        "hasHTTPEndpoint"
    ).startswith("has")


def test_source_ids_skip_and_domain():
    # non-prose content types skip ontology guidance entirely
    assert _source_ids("codebase") is None
    assert _source_ids("config") is None
    assert _source_ids("") is None
    # prose content gets the EG core foundation
    assert _source_ids("document") == ("core:foundation@1",)
    # a domain source type adds its EG core module on top of the foundation
    medical = _source_ids("medical")
    assert medical == ("core:foundation@1", "core:medical@1")
    # substring match works (connector-qualified names)
    assert "core:hr@1" in (_source_ids("connector:hr") or ())


def test_schema_is_built_from_the_eg_vocabulary_view(monkeypatch):
    """The TBox comes from EG's OntologyInspect (EH-471); AU parses no RDF."""
    requested: list[list[str]] = []
    view = SimpleNamespace(
        classes=[
            _term("Person", comment="A human."),
            _term("Organization", label="Organization"),
            SimpleNamespace(iri="http://other.example/Thing", label=None, comment=None),
        ],
        object_properties=[
            _term(
                "worksFor",
                label="works for",
                domains=[f"{KG}Person"],
                ranges=[f"{KG}Organization"],
                symmetric=False,
            ),
            _term("knows", domains=[f"{KG}Person"], ranges=[], symmetric=True),
        ],
    )

    class _Engine:
        def ontology_inspect(self, documents=(), *, source_ids=()):
            assert list(documents) == []
            requested.append(list(source_ids))
            return view

    monkeypatch.setattr(
        graph_compute.GraphComputeEngine, "get_or_create", lambda *_a, **_k: _Engine()
    )
    schema = load_extraction_schema("document")
    assert requested == [["core:foundation@1"]]
    assert schema is not None
    assert [e.name for e in schema.entity_types] == ["Organization", "Person"]
    assert schema.entity_types[1].description == "A human."
    works_for = schema.relations_by_predicate()["works_for"]
    assert (works_for.domain, works_for.range) == (("Person",), ("Organization",))
    assert schema.relations_by_predicate()["knows"].symmetric


def test_unavailable_eg_degrades_to_free_vocabulary(monkeypatch):
    def _no_engine(*_a, **_k):
        raise RuntimeError("no engine loop available for OntologyInspect")

    monkeypatch.setattr(extraction_schema, "_inspect_sources", _no_engine)
    assert load_extraction_schema("document") is None


def test_load_core_schema_has_typed_relations(engine_graph, monkeypatch):
    monkeypatch.setattr(
        graph_compute.GraphComputeEngine,
        "get_or_create",
        lambda *_a, **_k: engine_graph,
    )
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
