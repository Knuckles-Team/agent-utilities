"""Typed entities produced by the KG enrichment pipeline (CONCEPT:EG-KG.storage.nonblocking-checkpoint).

These are backend-agnostic value objects. The pipeline serialises them to graph
nodes/edges via the standard ``GraphBackend`` interface — no backend-specific
logic lives here.
"""

from __future__ import annotations

from enum import IntEnum

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


class EdgeRung(IntEnum):
    """The EH-270 ingestion cost-ladder position a fact was produced at,
    stamped onto every :class:`EnrichmentEdge` (EH-274).

    Ordered cheapest/most-certain (0) to most-expensive/least-certain (5),
    exactly as ruled for EH-270's cost ladder ("a higher rung runs only when
    every lower rung abstains and may never overwrite a deterministic
    fact"). Comparisons use plain ``int`` ordering (``IntEnum``), so "a
    higher rung" has exactly one meaning everywhere it's compared — see
    :func:`rung_may_overwrite`, the one place that comparison is made.

    ``UNKNOWN`` is a sentinel for edges nobody has classified yet (every
    edge persisted before EH-274, or a producer this lane's survey missed)
    — it is deliberately numbered *above* ``ASSERTED``, not below
    ``EXTRACTED``, so a naive ``rung <= EdgeRung.X`` comparison written
    elsewhere fails CLOSED (excludes the unclassified edge) rather than
    open. It must never be used as a stand-in for a real classification —
    see EH-274's ledger row: "a wrong provenance tag is worse than none."

    This ``rung`` is a DIFFERENT concept from the existing source-system
    provenance metadata (``source_system``/``domain`` stamped by
    :func:`.provenance.stamp_source`, or the lineage payload carried on a
    ``ChangeEnvelope.provenance`` dict) — those answer "which external
    system did this come from"; ``rung`` answers "how was this EDGE itself
    produced" (parsed, resolved, computed, modelled, embedded, or
    asserted).
    """

    EXTRACTED = 0  # rung 0: AST / a declared structural fact, read verbatim
    INFERRED = 1  # rung 1: symbol/type/identifier resolution
    DERIVED = 2  # rung 2: statistical/community/computed-aggregate
    MODELED = 3  # rung 3: classical ML/NER model output
    EMBEDDED = 4  # rung 4: vector-embedding similarity
    ASSERTED = 5  # rung 5: LLM extraction/generation
    UNKNOWN = (
        99  # sentinel: not yet classified — NEVER a fallback for real classification
    )


# Rung classification for the engine's type/scope-resolved edge stream
# (CONCEPT:EG-KG.compute.type-scope-resolved-call), keyed by the RAW WIRE
# ``edge_type`` string the engine returns. TWO producers consume this SAME
# engine RPC family independently -- ``extractors/code_test.py``'s
# ``entities_from_index_result`` (per-repo pipeline) and
# ``core/gitlab_indexer.py``'s ``map_index_result`` (whole-GitLab-instance
# sync) -- so this table has exactly ONE owner rather than two copies that
# could drift. ``calls``/``inherits``/``realizes`` all come out of the SAME
# cross-file type/scope resolution pass -> INFERRED (rung 1). ``similar_to``
# is the engine's MinHash code-similarity pass -- a statistical technique,
# not a resolution one -- so it is DERIVED (rung 2) "at best" per EH-274's
# own ledger row, never conflated with the resolved-symbol edges it happens
# to share a wire format with.
RESOLVED_EDGE_RUNG: dict[str, EdgeRung] = {
    "calls": EdgeRung.INFERRED,
    "inherits": EdgeRung.INFERRED,
    "realizes": EdgeRung.INFERRED,
    "similar_to": EdgeRung.DERIVED,
}


def rung_may_overwrite(existing: EdgeRung | None, new: EdgeRung) -> bool:
    """Monotone-safety gate (EH-274): may a write carrying ``new``'s rung
    replace an already-stored edge whose rung is ``existing``?

    The one rule, expressed exactly once so nothing downstream reimplements
    it as an if-chain: a fact already recorded at a MORE certain
    (lower-numbered) rung can never be replaced by one recorded at a LESS
    certain (higher-numbered) rung — the EH-270 cost-ladder guarantee,
    applied to edges instead of pipeline stages.

    ``existing=None``/``UNKNOWN`` means "nothing classified is on record
    yet" (a pre-EH-274 persisted edge, or an unclassified caller) — an
    explicitly classified incoming rung is always allowed to fill that gap,
    since a known rung is strictly more informative than none. The reverse
    is refused: a write must never REGRESS an already-classified edge back
    to ``UNKNOWN``.
    """
    if existing is None or existing is EdgeRung.UNKNOWN:
        return True
    if new is EdgeRung.UNKNOWN:
        return False
    return new <= existing


def dedupe_edges_by_rung(edges: list[EnrichmentEdge]) -> list[EnrichmentEdge]:
    """Collapse edges sharing a ``(source, target, rel_type)`` key to the
    single most-certain (lowest-rung) one, per :func:`rung_may_overwrite`.

    This is the monotone-safety guarantee applied at the ONE scope this
    in-process helper can enforce without a backend read-before-write: two
    edges for the same key arriving in the SAME write batch/call (e.g. a
    struct-edge pass and a resolver pass both touching one pair). A
    conflict against an edge already PERSISTED from a prior ingest run is a
    separate, larger problem — it needs a conditional write at the storage
    layer (the operational-authority backend's own upsert), tracked as a
    follow-up rather than solved here; see ``registry.write_batch``'s
    docstring. Edges with distinct keys pass through unchanged, in their
    original relative order.
    """
    best: dict[tuple[str, str, str], EnrichmentEdge] = {}
    order: list[tuple[str, str, str]] = []
    for e in edges:
        key = (e.source, e.target, e.rel_type)
        current = best.get(key)
        if current is None:
            order.append(key)
            best[key] = e
        elif rung_may_overwrite(current.rung, e.rung):
            best[key] = e
    return [best[k] for k in order]


class EnrichmentEdge(BaseModel):
    """A typed relationship between two enrichment entities.

    ``props`` carries optional scalar edge properties (e.g. the ``condition``
    expression on a BPMN sequence-flow ``FLOWS_TO`` edge, CONCEPT:AU-KG.ontology.descriptive-process-world-gains);
    empty for the common property-less case.

    ``rung`` (EH-274) is the EH-270 cost-ladder tier this edge's fact was
    produced at — see :class:`EdgeRung`. ``confidence`` is the quality
    signal for that fact, promoted from the pre-EH-274 convention of riding
    opportunistically in ``props["confidence"]``
    (``extractors/code_test.py``'s resolver output) into a first-class,
    modelled field; ``None`` is an explicit abstain (mirrors
    ``CandidateClaim.model_confidence`` — never fabricated when the
    producer has no real signal).
    """

    source: str
    target: str
    rel_type: str
    rung: EdgeRung = Field(
        default=EdgeRung.UNKNOWN,
        description="EH-270 cost-ladder rung this edge's fact was produced "
        "at (EH-274). UNKNOWN is only for producers nobody has classified "
        "yet — never a substitute for real classification.",
    )
    confidence: float | None = Field(
        default=None,
        description="Quality/confidence signal for this edge's fact. None "
        "is an explicit abstain, never fabricated.",
    )
    props: dict = Field(default_factory=dict)


class Concept(BaseModel):
    """A key idea/technique/claim extracted from a document or codebase.

    Concepts are the universal bridge across ingestion categories — the same
    Concept can be MENTIONED by a paper and REALIZED by code. (CONCEPT:EG-KG.storage.nonblocking-checkpoint)
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
    # (CONCEPT:AU-KG.ingest.standardized-document-ingestion — standardized document ingestion contract.)
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
    no changes to shared pipeline/writer code. (CONCEPT:AU-KG.ingest.enterprise-source-extractor)
    """

    id: str
    type: str
    props: dict = Field(default_factory=dict)


class ExtractionBatch(BaseModel):
    """Uniform output of a source extractor: typed nodes + typed edges."""

    category: str = ""
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[EnrichmentEdge] = Field(default_factory=list)
