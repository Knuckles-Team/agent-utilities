"""Tests for CONCEPT:AU-KG.enrichment.extraction-run-provenance /
CONCEPT:AU-KG.enrichment.extraction-outcome-taxonomy / CONCEPT:AU-KG.ingest.deterministic-extraction-default.

Covers three layers:

1. The pure item-classification / outcome-decision logic in ``extraction_run.py``.
2. The LIVE PATH through ``EntityClaimExtractor.extract_and_persist`` — every
   persisted entity/claim is bound to an ``ExtractionRun`` via
   ``WAS_GENERATED_BY``, and every one of the 7 typed outcomes is reachable and
   distinguishable.
3. The LIVE PATH through ``IngestionEngine._enrich_text`` — proving entity/claim
   extraction is now a DEFAULT stage of generic ingestion (not an analysis-only
   special case), and that the resource-priority cost bound produces
   ``budget_deferred`` telemetry instead of silently truncating.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.core.resource_priority import PriorityClass, priority_scope
from agent_utilities.knowledge_graph.ingestion.engine import IngestionEngine
from agent_utilities.knowledge_graph.kb.entity_claim_extractor import (
    EntityClaimExtractor,
)
from agent_utilities.knowledge_graph.kb.extraction_run import (
    ExtractionOutcome,
    OutcomeCounts,
    classify_claim,
    classify_entity,
    content_hash,
    decide_outcome,
)
from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

# --------------------------------------------------------------------------- #
# 1. Pure logic
# --------------------------------------------------------------------------- #


class TestDecideOutcome:
    def test_no_candidates_is_no_facts_found(self):
        assert decide_outcome(OutcomeCounts()) == ExtractionOutcome.NO_FACTS_FOUND

    def test_any_accepted_is_facts_found(self):
        counts = OutcomeCounts(accepted=1, rejected=3, quarantined=2)
        assert decide_outcome(counts) == ExtractionOutcome.FACTS_FOUND

    def test_needs_review_when_no_accepted(self):
        counts = OutcomeCounts(needs_review=2, rejected=1)
        assert decide_outcome(counts) == ExtractionOutcome.NEEDS_REVIEW

    def test_rejected_when_all_candidates_rejected(self):
        counts = OutcomeCounts(rejected=3)
        assert decide_outcome(counts) == ExtractionOutcome.REJECTED_BY_VALIDATION

    def test_quarantined_when_all_candidates_quarantined(self):
        counts = OutcomeCounts(quarantined=2)
        assert decide_outcome(counts) == ExtractionOutcome.QUARANTINED


class TestClassifyEntity:
    def test_valid_name_accepted(self):
        guard = PersistencePrivacyGuard()
        assert classify_entity("Jina AI", "an inference company", guard=guard) == (
            "accepted"
        )

    def test_numeric_only_name_rejected(self):
        guard = PersistencePrivacyGuard()
        assert classify_entity("12345", "", guard=guard) == "rejected"

    def test_empty_name_rejected(self):
        guard = PersistencePrivacyGuard()
        assert classify_entity("   ", "", guard=guard) == "rejected"

    def test_pii_in_description_quarantined(self):
        guard = PersistencePrivacyGuard()
        result = classify_entity(
            "Bob Smith", "reachable at bob.smith@example.com", guard=guard
        )
        assert result == "quarantined"


class TestClassifyClaim:
    def test_high_confidence_accepted(self):
        guard = PersistencePrivacyGuard()
        result = classify_claim(
            "the architecture should enforce typed schemas at every boundary",
            0.8,
            guard=guard,
        )
        assert result == "accepted"

    def test_low_confidence_needs_review(self):
        guard = PersistencePrivacyGuard()
        result = classify_claim(
            "the architecture should enforce typed schemas at every boundary",
            0.6,
            guard=guard,
        )
        assert result == "needs_review"

    def test_empty_text_rejected(self):
        guard = PersistencePrivacyGuard()
        assert classify_claim("   ", 0.9, guard=guard) == "rejected"

    def test_ssn_in_text_quarantined(self):
        guard = PersistencePrivacyGuard()
        result = classify_claim(
            "the account owner SSN is 123-45-6789 and must be verified",
            0.9,
            guard=guard,
        )
        assert result == "quarantined"


def test_content_hash_deterministic_and_sensitive_to_content():
    a = content_hash("hello world")
    b = content_hash("hello world")
    c = content_hash("hello mars")
    assert a == b
    assert a != c


# --------------------------------------------------------------------------- #
# 2. EntityClaimExtractor live path — ExtractionRun binding + outcome taxonomy
# --------------------------------------------------------------------------- #


class _FakeGraph:
    """Minimal in-memory double for ``EntityClaimExtractor``'s ``.graph`` needs
    (``add_node``/``__contains__``/``nodes(data=True)``) — a real
    ``GraphComputeEngine`` requires a live epistemic-graph engine this sandbox
    has none built for (mirrors ``test_second_brain_sync.py``'s own
    ``_FakeGraph``, kept hermetic and dependency-free for the same reason)."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}

    def add_node(self, node_id: str, **properties: Any) -> None:
        self._nodes[node_id] = dict(properties)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def nodes(self, data: bool = False):
        if data:
            return list(self._nodes.items())
        return list(self._nodes.keys())


class _MockEngine:
    """Mirrors ``test_entity_claim_extractor.py``'s ``mock_engine`` fixture,
    plus an ``edges`` log so ``WAS_GENERATED_BY`` provenance is assertable."""

    def __init__(self) -> None:
        self.graph = _FakeGraph()
        self.backend = None
        self.edges: list[tuple[str, str, str, dict[str, Any]]] = []

    def _serialize_node(self, node, label=""):
        data = node.model_dump()
        node_type = data.pop("type", None)
        if node_type is not None:
            data["node_type"] = getattr(node_type, "value", node_type)
        return data

    def _upsert_node(self, label, node_id, data):
        return None

    def link_nodes(self, source, target, edge_type, metadata=None):
        rel = edge_type.value if hasattr(edge_type, "value") else edge_type
        self.edges.append((source, target, rel, dict(metadata or {})))


def _node_types(engine: _MockEngine) -> dict[str, list[str]]:
    by_type: dict[str, list[str]] = {}
    for node_id, data in engine.graph.nodes(data=True):
        by_type.setdefault(str(data.get("node_type", "")).lower(), []).append(node_id)
    return by_type


def _generated_by_edges(engine: _MockEngine) -> list[tuple[str, str]]:
    return [
        (u, v)
        for u, v, rel, _props in engine.edges
        if str(rel).lower() == "was_generated_by"
    ]


class TestEntityClaimExtractorExtractionRunBinding:
    def test_facts_found_run_persisted_and_bound(self):
        engine = _MockEngine()
        engine.graph.add_node("doc:test", node_type="article", name="Test Doc")
        extractor = EntityClaimExtractor(engine)

        result = extractor.extract_and_persist(
            "See [[Inference Engine]] for details.", source_id="doc:test"
        )

        assert result.outcome == ExtractionOutcome.FACTS_FOUND.value
        assert result.run_id.startswith("extraction_run:")
        nodes = _node_types(engine)
        assert nodes.get("extraction_run")
        assert nodes["extraction_run"][0] == result.run_id

        generated_by = _generated_by_edges(engine)
        # The persisted entity must be bound to the run.
        assert any(target == result.run_id for _source, target in generated_by)

    def test_no_facts_found_when_extraction_yields_nothing(self):
        engine = _MockEngine()
        extractor = EntityClaimExtractor(engine)

        result = extractor.extract_and_persist(
            "plain text with nothing extractable here", source_id="doc:empty"
        )

        assert result.outcome == ExtractionOutcome.NO_FACTS_FOUND.value

    def test_extractor_unavailable_distinguishable_from_no_facts_found(self):
        """CONCEPT:AU-KG.enrichment.extraction-outcome-taxonomy — the exact
        misdiagnosis this taxonomy fixes: a missing engine must NEVER look like
        a clean zero-facts pass."""
        extractor = EntityClaimExtractor(None)  # type: ignore[arg-type]

        result = extractor.extract_and_persist(
            "See [[Inference Engine]] for details.", source_id="doc:no-engine"
        )

        assert result.outcome == ExtractionOutcome.EXTRACTOR_UNAVAILABLE.value
        assert result.outcome != ExtractionOutcome.NO_FACTS_FOUND.value

    def test_needs_review_claim_persisted_but_flagged(self):
        engine = _MockEngine()
        extractor = EntityClaimExtractor(engine)
        # "therefore/consequently/thus" pattern -> confidence 0.6, below the
        # 0.65 review threshold -> needs_review, not silently accepted.
        content = (
            "Therefore, the system should auto-fix recoverable issues before "
            "flagging errors to the operator for review."
        )

        result = extractor.extract_and_persist(content, source_id="doc:review")

        assert result.outcome_counts.get("needs_review", 0) >= 1
        claim_nodes = _node_types(engine).get("claim", [])
        assert claim_nodes  # the low-confidence claim IS persisted...
        claim_run_edge_props = [
            props
            for u, v, rel, props in engine.edges
            if u in claim_nodes and str(rel).lower() == "was_generated_by"
        ]
        assert any(
            props.get("review_state") == "needs_review"
            for props in claim_run_edge_props
        )

    def test_rejected_entity_never_persisted(self):
        engine = _MockEngine()
        extractor = EntityClaimExtractor(engine)
        # A purely-numeric wikilink target fails structural validation.
        result = extractor.extract_and_persist(
            "See [[12345]] for the reference number.", source_id="doc:rejected"
        )

        assert result.outcome_counts.get("rejected", 0) >= 1
        entity_nodes = _node_types(engine).get("entity", [])
        digest_id = f"entity:{__import__('hashlib').sha256(b'12345').hexdigest()[:32]}"
        assert digest_id not in entity_nodes

    def test_quarantined_claim_never_persisted(self):
        engine = _MockEngine()
        extractor = EntityClaimExtractor(engine)
        content = (
            "We recommend that the account owner SSN 123-45-6789 be verified "
            "before granting elevated access."
        )

        result = extractor.extract_and_persist(content, source_id="doc:quarantine")

        assert result.outcome_counts.get("quarantined", 0) >= 1
        # No claim node should carry the raw SSN text.
        for _node_id, data in engine.graph.nodes(data=True):
            assert "123-45-6789" not in str(data.get("claim_text", ""))


# --------------------------------------------------------------------------- #
# 3. IngestionEngine generic-ingestion wiring — default-on + cost bound
# --------------------------------------------------------------------------- #


class _RecordingBackend:
    def __init__(self) -> None:
        self.nodes: list[tuple[str, dict[str, Any]]] = []
        self.edges: list[tuple[str, str, dict[str, Any]]] = []

    def add_node(self, node_id: str, **props: Any) -> None:
        self.nodes.append((node_id, props))

    def add_edge(
        self, source: str, target: str, rel_type: str = "", **props: Any
    ) -> None:
        self.edges.append((source, target, {"rel_type": rel_type, **props}))


def _kg_engine() -> _MockEngine:
    return _MockEngine()


@pytest.fixture(autouse=True)
def _native_enrichment_slice(monkeypatch):
    """Same no-service seam ``test_ingestion_enrich.py`` uses for the native
    ChangeEnvelope commit at the end of ``_enrich_text``."""
    from agent_utilities.knowledge_graph.ingestion import envelope_ingest

    def _apply(_engine, _connector, entities, relationships=None, **_kwargs):
        return {"status": "success"}

    monkeypatch.setattr(envelope_ingest, "ingest_graph_slice", _apply)


@pytest.mark.asyncio
async def test_enrich_text_runs_entity_claim_extraction_by_default():
    """CONCEPT:AU-KG.ingest.deterministic-extraction-default — generic ingestion
    (``_enrich_text``, the seam every content type funnels through) now runs
    entity/claim extraction WITHOUT any special-case action, proving it is no
    longer reachable only from ``extract_claims``/second-brain sync."""
    kg = _kg_engine()
    eng = IngestionEngine(kg_engine=kg, backend=_RecordingBackend())

    out = await eng._enrich_text(
        "doc:generic",
        "See [[Inference Engine]] for details.",
        "document",
        enrich_concepts=False,
        enrich_facts=False,
    )

    assert out["entities"] >= 1
    assert out["extraction_outcomes"].get("facts_found", 0) >= 1
    # The ExtractionRun node landed directly in the KG engine's own graph.
    assert _node_types(kg).get("extraction_run")


@pytest.mark.asyncio
async def test_background_ingestion_defers_windows_beyond_budget(monkeypatch):
    """CONCEPT:AU-ORCH.scheduling.resource-priority-edict reused as the cost
    bound: under BACKGROUND_INGESTION priority, only the first N windows get
    the entity/claim pass; the rest are recorded BUDGET_DEFERRED — never
    silently dropped."""
    from agent_utilities.knowledge_graph.ingestion import engine as engine_mod

    monkeypatch.setattr(engine_mod, "_EXTRACTION_BACKGROUND_WINDOW_CAP", 1)

    kg = _kg_engine()
    eng = IngestionEngine(kg_engine=kg, backend=_RecordingBackend())
    monkeypatch.setattr(
        eng,
        "_enrichment_windows",
        lambda _text: ["window one text here", "window two text here", "window three"],
    )

    with priority_scope(PriorityClass.BACKGROUND_INGESTION):
        out = await eng._enrich_text(
            "doc:budget",
            "irrelevant — windows are patched above",
            "document",
            enrich_concepts=False,
            enrich_facts=False,
        )

    assert out["extraction_outcomes"].get("budget_deferred") == 2
    deferred_runs = [
        node_id
        for node_id, data in kg.graph.nodes(data=True)
        if str(data.get("node_type", "")).lower() == "extraction_run"
        and data.get("outcome") == "budget_deferred"
    ]
    assert len(deferred_runs) == 2


@pytest.mark.asyncio
async def test_interactive_priority_is_not_budget_capped(monkeypatch):
    """An INTERACTIVE-tagged call (e.g. the `extract_claims` MCP action's
    ambient context) gets the full window budget, unbounded by the
    background cost cap."""
    from agent_utilities.knowledge_graph.ingestion import engine as engine_mod

    monkeypatch.setattr(engine_mod, "_EXTRACTION_BACKGROUND_WINDOW_CAP", 1)

    kg = _kg_engine()
    eng = IngestionEngine(kg_engine=kg, backend=_RecordingBackend())
    monkeypatch.setattr(
        eng,
        "_enrichment_windows",
        lambda _text: ["window one text here", "window two text here"],
    )

    with priority_scope(PriorityClass.INTERACTIVE):
        out = await eng._enrich_text(
            "doc:interactive",
            "irrelevant — windows are patched above",
            "document",
            enrich_concepts=False,
            enrich_facts=False,
        )

    assert out["extraction_outcomes"].get("budget_deferred", 0) == 0


@pytest.mark.asyncio
async def test_extractor_unavailable_surfaced_through_enrich_text():
    """A broken engine binding must produce a typed `extractor_unavailable`
    outcome through the FULL generic-ingestion wiring, not a silent zero."""
    eng = IngestionEngine(kg_engine=object(), backend=_RecordingBackend())

    out = await eng._enrich_text(
        "doc:broken",
        "See [[Inference Engine]] for details.",
        "document",
        enrich_concepts=False,
        enrich_facts=False,
    )

    assert out["entities"] == 0
    assert out["extraction_outcomes"].get("extractor_unavailable", 0) >= 1


# --------------------------------------------------------------------------- #
# 4. Strict-schema (Kuzu/LadybugDB) typed columns (D-62-2)
# --------------------------------------------------------------------------- #


def test_claim_and_extraction_run_have_table_definitions() -> None:
    """``Claim``/``ExtractionRun`` get first-class typed columns on a
    strict-schema backend instead of only the ``GENERIC_NODE_COLUMNS``
    fallback (the fallback still works — this is fidelity, not correctness)."""
    from agent_utilities.models.schema_definition import SCHEMA

    table_names = {t.name for t in SCHEMA.nodes}
    assert "Claim" in table_names
    assert "ExtractionRun" in table_names


def test_claim_table_covers_claim_node_fields() -> None:
    from agent_utilities.models.schema_definition import SCHEMA

    claim_table = next(t for t in SCHEMA.nodes if t.name == "Claim")
    for field_name in (
        "claim_text",
        "confidence",
        "claim_type",
        "source_ids",
        "extracted_from",
        "domain",
        "is_verified",
    ):
        assert field_name in claim_table.columns, (
            f"Claim TableDefinition missing column {field_name!r}"
        )


def test_extraction_run_table_covers_extraction_run_node_fields() -> None:
    from agent_utilities.models.schema_definition import SCHEMA

    run_table = next(t for t in SCHEMA.nodes if t.name == "ExtractionRun")
    for field_name in (
        "source_id",
        "input_hash",
        "parser_version",
        "extractor_version",
        "schema_pack_ref",
        "model_ref",
        "graph_epoch",
        "calibration_policy",
        "thresholds",
        "outcome",
        "outcome_counts",
        "entities_count",
        "claims_count",
    ):
        assert field_name in run_table.columns, (
            f"ExtractionRun TableDefinition missing column {field_name!r}"
        )


def test_was_generated_by_covers_entity_and_claim_to_extraction_run() -> None:
    """The PROV-O ``wasGeneratedBy`` link ``link_generated_by`` writes
    (``child WAS_GENERATED_BY run``) is a declared connection on a
    strict-schema backend, for both persisted node types."""
    from agent_utilities.models.schema_definition import SCHEMA

    generated_by = next(e for e in SCHEMA.edges if e.type == "WAS_GENERATED_BY")
    connections = {(c["from"], c["to"]) for c in generated_by.connections}
    assert ("Entity", "ExtractionRun") in connections
    assert ("Claim", "ExtractionRun") in connections
