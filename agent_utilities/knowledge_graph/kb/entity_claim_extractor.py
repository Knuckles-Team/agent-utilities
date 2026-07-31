#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-KG.enrichment.entity-claim-extraction — Entity-Claim Extraction for MAGMA Epistemic View.

Extends the knowledge base pipeline to extract entities, claims, and
implicit relationships (BUILDS_ON, CONTRADICTS, EXEMPLIFIES) from
ingested documents. This fills the ``retrieve_epistemic_view()`` stub
in ``engine_query.py`` to complete the MAGMA view system.

Inspired by Understand-Anything's ``article-analyzer`` agent which
separates deterministic parsing from LLM-based implicit inference.

Usage::

    from agent_utilities.knowledge_graph.kb.entity_claim_extractor import (
        EntityClaimExtractor,
    )

    extractor = EntityClaimExtractor(engine)
    result = await extractor.extract_from_article(article_node)
    # result.entities, result.claims, result.relationships

See docs/pillars/architecture_c4.md §CONCEPT:AU-KG.enrichment.entity-claim-extraction
"""


import hashlib
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ...models.knowledge_graph import (
    ClaimNode,
    EntityNode,
    RegistryEdgeType,
    RegistryNodeType,
)
from ...security.persistence_privacy import PersistencePrivacyGuard
from .extraction_run import (
    OutcomeCounts,
    build_extraction_run,
    classify_claim,
    classify_entity,
    content_hash,
    decide_outcome,
    link_generated_by,
    persist_evidence_span,
    persist_extraction_run,
    record_unavailable_run,
)

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction Models
# ---------------------------------------------------------------------------


class ExtractedEntity(BaseModel):
    """An entity extracted from document content."""

    name: str
    entity_type: str  # person, tool, paper, organization, concept, technology
    description: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractedClaim(BaseModel):
    """A claim or assertion extracted from document content."""

    claim_text: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    claim_type: str = "assertion"  # assertion, decision, thesis, finding, opinion
    domain: str | None = None


class ExtractedRelationship(BaseModel):
    """An implicit relationship between extracted items."""

    source_name: str
    target_name: str
    relationship_type: str  # builds_on, contradicts, exemplifies, cites, authored_by
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """Complete result from entity-claim extraction."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
    source_id: str = ""
    #: The `ExtractionRun` (CONCEPT:AU-KG.enrichment.extraction-run-provenance)
    #: this result is bound to, and its typed outcome
    #: (CONCEPT:AU-KG.enrichment.extraction-outcome-taxonomy) — "" until
    #: `EntityClaimExtractor.extract_and_persist` populates them.
    run_id: str = ""
    outcome: str = ""
    outcome_counts: dict[str, int] = Field(default_factory=dict)
    #: Actual persisted-node counts by type (a subset of ``entities``/``claims``
    #: above — those raw lists include rejected/quarantined candidates that were
    #: classified but never written to the graph).
    entities_persisted: int = 0
    claims_persisted: int = 0


def claim_node_id(source_id: str, claim_text: str) -> str:
    """Content-addressed ``Claim`` node id for one ``(source_id, claim_text)`` pair.

    Mirrors the entity digest convention two lines below in
    :meth:`EntityClaimExtractor.extract_and_persist` (``entity:<sha256(name)>``)
    so claims are idempotent the SAME way entities already are: re-running
    extraction over unchanged content mints the identical id and upserts the
    existing ``ClaimNode`` instead of minting a duplicate. Exported so a caller
    that needs the id a just-extracted claim was persisted under (e.g. to
    propose it into :class:`~agent_utilities.knowledge_graph.research.
    claim_flywheel.ClaimFlywheel`) can recompute it without a follow-up query.
    """
    digest = hashlib.sha256(f"{source_id}:{claim_text}".encode()).hexdigest()[:32]
    return f"claim:{digest}"


# ---------------------------------------------------------------------------
# Deterministic Extraction (Phase 1 — no LLM needed)
# ---------------------------------------------------------------------------

# Patterns for deterministic entity extraction
_CITATION_PATTERN = re.compile(
    r"\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&|\band\b)\s+[A-Z][a-z]+)*"
    r"(?:,?\s*\d{4})?)\)",
)
_URL_PATTERN = re.compile(r"https?://[^\s\)\"'>]+")
_WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


def extract_deterministic(
    content: str,
    source_id: str,
) -> ExtractionResult:
    """Phase 1: Extract entities and relationships deterministically.

    Uses regex-based parsing for:
    - Citations (Author et al., YYYY)
    - URLs (technology/tool references)
    - Wikilinks ([[target]] or [[target|display]])
    - Markdown headers (section structure → domain concepts)

    This does NOT require an LLM — all extraction is deterministic.
    """
    entities: list[ExtractedEntity] = []
    claims: list[ExtractedClaim] = []
    relationships: list[ExtractedRelationship] = []
    seen_entities: set[str] = set()

    # 1. Extract citations
    for match in _CITATION_PATTERN.finditer(content):
        citation = match.group(1).strip()
        if citation and citation not in seen_entities:
            seen_entities.add(citation)
            entities.append(
                ExtractedEntity(
                    name=citation,
                    entity_type="paper",
                    description=f"Citation reference: {citation}",
                )
            )
            relationships.append(
                ExtractedRelationship(
                    source_name=source_id,
                    target_name=citation,
                    relationship_type="cites",
                    confidence=0.9,
                )
            )

    # 2. Extract wikilinks as entity references
    for match in _WIKILINK_PATTERN.finditer(content):
        target = match.group(1).strip()
        if target and target not in seen_entities:
            seen_entities.add(target)
            entities.append(
                ExtractedEntity(
                    name=target,
                    entity_type="concept",
                    description=f"Linked concept: {target}",
                )
            )
            relationships.append(
                ExtractedRelationship(
                    source_name=source_id,
                    target_name=target,
                    relationship_type="builds_on",
                    confidence=0.7,
                )
            )

    # 3. Extract key assertions from strong language patterns
    assertion_patterns = [
        (r"(?:must|shall|should|will)\s+(.{20,100}?)[.\n]", "decision", 0.8),
        (r"(?:we (?:recommend|propose|suggest))\s+(.{20,100}?)[.\n]", "thesis", 0.7),
        (r"(?:therefore|consequently|thus),?\s+(.{20,100}?)[.\n]", "finding", 0.6),
    ]

    for pattern, claim_type, confidence in assertion_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            claim_text = match.group(1).strip()
            if len(claim_text) > 20:
                claims.append(
                    ExtractedClaim(
                        claim_text=claim_text,
                        confidence=confidence,
                        claim_type=claim_type,
                    )
                )

    return ExtractionResult(
        entities=entities,
        claims=claims,
        relationships=relationships,
        source_id=source_id,
    )


# ---------------------------------------------------------------------------
# Entity-Claim Extractor (orchestrates Phase 1 + Phase 2)
# ---------------------------------------------------------------------------


class EntityClaimExtractor:
    """Extracts entities, claims, and epistemic relationships from documents.

    CONCEPT:AU-KG.enrichment.entity-claim-extraction — Entity-Claim Extraction / MAGMA Epistemic View

    Two-phase extraction inspired by Understand-Anything's article-analyzer:
      1. **Deterministic**: Regex-based entity extraction (citations, links, assertions)
      2. **Semantic** (optional): LLM-based implicit relationship detection

    Extracted items are persisted to the KG as ``EntityNode``, ``ClaimNode``,
    and epistemic edges (BUILDS_ON, CONTRADICTS, EXEMPLIFIES, CITES).

    Args:
        engine: The ``IntelligenceGraphEngine`` to persist results into.
    """

    def __init__(self, engine: IntelligenceGraphEngine, schema_pack=None) -> None:
        self.engine = engine
        # The active schema pack drives zero-LLM link inference; resolve the
        # process-active pack when not supplied so ingestion honours it without
        # every call site threading it (CONCEPT:AU-KG.research.zero-llm-pack-link).
        if schema_pack is None:
            try:
                from agent_utilities.models.schema_pack_loader import get_active_pack

                schema_pack = get_active_pack()
            except Exception:  # pragma: no cover - never block construction
                schema_pack = None
        self.schema_pack = schema_pack

    def extract_and_persist(
        self,
        content: str,
        source_id: str,
        article_id: str | None = None,
        domain: str | None = None,
    ) -> ExtractionResult:
        """Extract entities/claims from content and persist to the KG.

        Args:
            content: The text content to analyze.
            source_id: ID of the source document.
            article_id: Optional article node ID this was extracted from.
            domain: Optional domain label for categorization.

        Returns:
            ``ExtractionResult`` with extracted items, plus the
            ``ExtractionRun`` (CONCEPT:AU-KG.enrichment.extraction-run-provenance)
            it was bound to and its typed outcome
            (CONCEPT:AU-KG.enrichment.extraction-outcome-taxonomy).
        """
        started = time.monotonic()
        input_hash = content_hash(content)

        if self.engine is None or not hasattr(self.engine, "graph"):
            # CONCEPT:AU-KG.ingest.deterministic-extraction-default — the extractor
            # itself could not run at all. This must NEVER look like a clean
            # `no_facts_found` pass — see the ExtractionOutcome docstring.
            run = record_unavailable_run(
                self.engine, source_id=source_id, error="no engine bound"
            )
            return ExtractionResult(
                source_id=source_id,
                run_id=run.id,
                outcome=run.outcome,
                outcome_counts=run.outcome_counts,
            )

        # Phase 1: Deterministic extraction
        result = extract_deterministic(content, source_id)

        # Phase 1b: Zero-LLM, pack-driven typed-edge extraction (CONCEPT:AU-KG.research.zero-llm-pack-link).
        # The active pack's regex link-inference rules (ReDoS-bounded) materialise
        # domain edges (e.g. research-state supports/weakens/cites/uses_dataset).
        if self.schema_pack and getattr(self.schema_pack, "link_inference", None):
            try:
                from .link_inference import infer_links

                result.relationships.extend(
                    infer_links(content, source_id, self.schema_pack.link_inference)
                )
            except Exception as e:  # pragma: no cover - never block ingestion
                logger.debug("pack link inference failed: %s", e)

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        guard = PersistencePrivacyGuard()
        counts = OutcomeCounts()

        # Persist entities to KG. Every candidate is classified BEFORE it is
        # written (CONCEPT:AU-KG.enrichment.extraction-outcome-taxonomy) — a
        # rejected/quarantined entity is never persisted, so no unsafe or
        # malformed "fact" silently enters the graph (constraint 6).
        entity_id_map: dict[str, str] = {}
        persisted_entity_ids: list[str] = []
        for entity in result.entities:
            bucket = classify_entity(entity.name, entity.description, guard=guard)
            if bucket == "rejected":
                counts.rejected += 1
                continue
            if bucket == "quarantined":
                counts.quarantined += 1
                continue
            counts.accepted += 1

            digest = hashlib.sha256(entity.name.encode()).hexdigest()[:32]
            entity_id = f"entity:{digest}"
            entity_id_map[entity.name] = entity_id
            persisted_entity_ids.append(entity_id)

            node = EntityNode(
                id=entity_id,
                name=entity.name,
                type=RegistryNodeType.ENTITY,
                entity_type=entity.entity_type,
                description=entity.description,
                properties=entity.properties,
                timestamp=ts,
                importance_score=0.5,
            )
            graph_data = self.engine._serialize_node(node)
            self.engine.graph.add_node(entity_id, **graph_data)

            if self.engine.backend:
                data = self.engine._serialize_node(node, label="Entity")
                self.engine._upsert_node("Entity", entity_id, data)

            persist_evidence_span(
                source_id, content, entity.name, source_type=domain or "document"
            )

        # Persist claims to KG. Content-addressed (see `claim_node_id`) so
        # re-extracting unchanged content upserts the same ClaimNode rather
        # than minting a duplicate — the same idempotency entities have via
        # their own sha256(name) digest. Below-threshold claims are still
        # persisted (they are real candidates, not malformed input) but flagged
        # `needs_review` on their provenance edge rather than silently accepted.
        persisted_claims: list[tuple[str, ExtractedClaim, str]] = []
        for claim in result.claims:
            bucket = classify_claim(claim.claim_text, claim.confidence, guard=guard)
            if bucket == "rejected":
                counts.rejected += 1
                continue
            if bucket == "quarantined":
                counts.quarantined += 1
                continue
            if bucket == "needs_review":
                counts.needs_review += 1
            else:
                counts.accepted += 1

            claim_id = claim_node_id(source_id, claim.claim_text)
            persisted_claims.append((claim_id, claim, bucket))

            node = ClaimNode(  # type: ignore[assignment]
                id=claim_id,
                name=claim.claim_text[:80],
                claim_text=claim.claim_text,
                confidence=claim.confidence,
                claim_type=claim.claim_type,
                source_ids=[source_id],
                extracted_from=article_id,
                domain=domain,
                timestamp=ts,
                importance_score=claim.confidence * 0.7,
            )
            graph_data = self.engine._serialize_node(node)
            self.engine.graph.add_node(claim_id, **graph_data)

            if self.engine.backend:
                data = self.engine._serialize_node(node, label="Claim")
                self.engine._upsert_node("Claim", claim_id, data)

            # Link claim to its source
            if source_id in self.engine.graph:
                self.engine.link_nodes(
                    claim_id, source_id, RegistryEdgeType.WAS_DERIVED_FROM
                )

            persist_evidence_span(
                source_id,
                content,
                claim.claim_text,
                source_type=domain or "document",
                confidence=claim.confidence,
            )

        # Persist relationships to KG
        edge_type_map = {
            "builds_on": RegistryEdgeType.BUILDS_ON,
            "contradicts": RegistryEdgeType.CONTRADICTS,
            "exemplifies": RegistryEdgeType.EXEMPLIFIES,
            "cites": RegistryEdgeType.CITES,
            "authored_by": RegistryEdgeType.AUTHORED_BY,
            # Research-state / pack link-inference edges (CONCEPT:AU-KG.research.zero-llm-pack-link / KG-2.37)
            "supports_belief": RegistryEdgeType.SUPPORTS_BELIEF,
            "contradicts_belief": RegistryEdgeType.CONTRADICTS_BELIEF,
            "weakens": RegistryEdgeType.WEAKENS,
            "uses_dataset": RegistryEdgeType.USES_DATASET,
            "cites_source": RegistryEdgeType.CITES_SOURCE,
        }

        for rel in result.relationships:
            source = entity_id_map.get(rel.source_name, rel.source_name)
            target = entity_id_map.get(rel.target_name, rel.target_name)
            edge_type = edge_type_map.get(rel.relationship_type)
            # Generic fallback: any pack-declared edge_type *value* resolves to its
            # RegistryEdgeType member, so new pack verbs persist without a map edit.
            if edge_type is None:
                try:
                    edge_type = RegistryEdgeType(rel.relationship_type)
                except ValueError:
                    edge_type = None

            if (
                edge_type
                and source in self.engine.graph
                and target in self.engine.graph
            ):
                self.engine.link_nodes(
                    source,
                    target,
                    edge_type,
                    {"confidence": rel.confidence, "inferred": False},
                )
                counts.relationships += 1

        # Bind every persisted entity/claim to the run that produced it
        # (CONCEPT:AU-KG.enrichment.extraction-run-provenance) — no fact without
        # provenance. Built/persisted AFTER the writes above so its counts and
        # duration reflect what actually landed in the graph.
        outcome = decide_outcome(counts)
        schema_pack_ref = getattr(self.schema_pack, "name", "") or ""
        run = build_extraction_run(
            engine=self.engine,
            source_id=source_id,
            input_hash=input_hash,
            outcome=outcome,
            counts=counts,
            entities_count=len(persisted_entity_ids),
            claims_count=len(persisted_claims),
            schema_pack_ref=schema_pack_ref,
            started_at=started,
        )
        persist_extraction_run(self.engine, run)
        for entity_id in persisted_entity_ids:
            link_generated_by(self.engine, entity_id, run.id, review_state="accepted")
        for claim_id, claim, bucket in persisted_claims:
            link_generated_by(
                self.engine,
                claim_id,
                run.id,
                review_state=bucket,
                confidence=claim.confidence,
            )

        result.run_id = run.id
        result.outcome = run.outcome
        result.outcome_counts = run.outcome_counts
        result.entities_persisted = len(persisted_entity_ids)
        result.claims_persisted = len(persisted_claims)

        logger.info(
            "[CONCEPT:AU-KG.enrichment.entity-claim-extraction] Extracted %d entities, "
            "%d claims, %d relationships from %s — run=%s outcome=%s",
            len(persisted_entity_ids),
            len(persisted_claims),
            counts.relationships,
            source_id,
            run.id,
            run.outcome,
        )

        return result
