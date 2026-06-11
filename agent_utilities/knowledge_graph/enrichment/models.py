"""Typed entities produced by the KG enrichment pipeline (CONCEPT:KG-2.8).

These are backend-agnostic value objects. The pipeline serialises them to graph
nodes/edges via the standard ``GraphBackend`` interface — no backend-specific
logic lives here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CodeEntity(BaseModel):
    """An application (non-test) symbol: function, class, or method.

    Class-only structural facts (``bases``/``methods``/``decorators``/
    ``is_abstract``) come from the Rust parser and drive design-pattern
    detection. They are empty for functions.
    """

    id: str
    name: str
    qualname: str
    kind: str  # function|method|constructor | class|interface|struct|enum|trait|...
    language: str = ""  # python|javascript|typescript|go|rust|java|c|cpp|csharp
    file_path: str
    line: int
    ast_hash: str
    is_test: bool = False
    calls: list[str] = Field(default_factory=list)  # callee names (for call graph)
    # Structural facts (classes)
    bases: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    decorators: list[str] = Field(default_factory=list)
    is_abstract: bool = False
    # Semantic enrichment (filled by later layers)
    patterns: list[str] = Field(default_factory=list)


class TestEntity(BaseModel):
    """A pytest test function with quality-signal metrics.

    The metrics are exactly what a "which tests need work" question needs, and
    are extracted statically so they can be stored as node properties and
    queried/reasoned over (rather than recomputed by an ad-hoc script).
    """

    __test__ = False  # not a pytest test class

    id: str
    name: str
    qualname: str
    file_path: str
    line: int
    ast_hash: str
    loc: int = 0
    assert_count: int = 0
    raises_count: int = (
        0  # `with pytest.raises(...)` / `pytest.fail` count as assertions
    )
    mock_count: int = 0  # Mock/MagicMock/AsyncMock/patch/mocker references
    fixture_count: int = 0  # injected fixture params (excl. self/cls)
    marks: list[str] = Field(default_factory=list)  # pytest.mark.* names
    is_skipped: bool = False  # has skip / skipif / xfail
    calls: list[str] = Field(
        default_factory=list
    )  # callee names (for COVERS resolution)

    @property
    def effective_assertions(self) -> int:
        return self.assert_count + self.raises_count


class EnrichmentEdge(BaseModel):
    """A typed relationship between two enrichment entities."""

    source: str
    target: str
    rel_type: str


class Concept(BaseModel):
    """A key idea/technique/claim extracted from a document or codebase.

    Concepts are the universal bridge across ingestion categories — the same
    Concept can be MENTIONED by a paper and REALIZED by code. (CONCEPT:KG-2.8)
    """

    id: str
    name: str
    summary: str = ""
    kind: str = "concept"  # concept | technique | claim | requirement | term
    source_ids: list[str] = Field(default_factory=list)  # docs/symbols mentioning it


class Insight(BaseModel):
    """A distilled, actionable observation extracted from a call/doc (KG-2.8).

    The "calls become operating intelligence" payoff: an insight is a reusable
    takeaway (an objection pattern, a positioning signal, a risk flag), not just
    a raw concept.
    """

    id: str
    title: str
    reasoning: str = ""
    confidence: float = 0.7
    source_ids: list[str] = Field(default_factory=list)


class Fact(BaseModel):
    """A discrete, checkable assertion extracted from a source (KG-2.8)."""

    id: str
    statement: str
    confidence: float = 0.7
    source_ids: list[str] = Field(default_factory=list)


class Framework(BaseModel):
    """A named mental model / repeatable method distilled from a source."""

    id: str
    name: str
    summary: str = ""
    steps: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class Playbook(BaseModel):
    """An executable, reusable procedure distilled from calls/docs (KG-2.8).

    Crosswalked to the ArchiMate ``BusinessProcess`` family so a playbook is
    queryable alongside Camunda/ServiceNow processes.
    """

    id: str
    name: str
    steps: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    expected_outcome: str = ""
    source_ids: list[str] = Field(default_factory=list)


class Document(BaseModel):
    """A non-code ingested artifact (paper, email, BRD, SOW, book, …).

    ``doc_type`` drives type-specific metadata extraction; ``metadata`` holds the
    extracted fields; ``concept_ids`` link to the concepts it mentions.
    """

    id: str
    title: str
    doc_type: str = "document"
    file_path: str = ""
    content_hash: str = ""
    # Full verbatim body text — retained so the document is faithfully
    # re-materialisable from the KG (e.g. distilled back into a skill-graph).
    # (CONCEPT:KG-2.7 — standardized document ingestion contract.)
    content: str = ""
    metadata: dict = Field(default_factory=dict)
    concept_ids: list[str] = Field(default_factory=list)


class Feature(BaseModel):
    """A cohesive cluster of code symbols (a community in the call graph).

    Discovered via the epistemic-graph engine's community detection — a feature
    is "how a capability is implemented across symbols". ``name``/``summary`` are
    filled by the LLM when available.
    """

    id: str
    name: str = ""
    summary: str = ""
    member_ids: list[str] = Field(default_factory=list)
    size: int = 0
    patterns: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """What a single-file extraction yields."""

    file_path: str
    content_hash: str
    code: list[CodeEntity] = Field(default_factory=list)
    tests: list[TestEntity] = Field(default_factory=list)
    edges: list[EnrichmentEdge] = Field(default_factory=list)


class GraphNode(BaseModel):
    """A backend-agnostic node a source extractor wants written.

    ``type`` is the label; ``props`` the remaining (scalar) properties. This is
    the uniform shape every enterprise/source extractor emits so new sources need
    no changes to shared pipeline/writer code. (CONCEPT:KG-2.9)
    """

    id: str
    type: str
    props: dict = Field(default_factory=dict)


class ExtractionBatch(BaseModel):
    """Uniform output of a source extractor: typed nodes + typed edges."""

    category: str = ""
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[EnrichmentEdge] = Field(default_factory=list)
