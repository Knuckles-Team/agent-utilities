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
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ...models.knowledge_graph import (
    ClaimNode,
    EntityNode,
    RegistryEdgeType,
    RegistryNodeType,
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
            ``ExtractionResult`` with extracted items.
        """
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

        # Persist entities to KG
        entity_id_map: dict[str, str] = {}
        for entity in result.entities:
            entity_id = f"entity:{hashlib.md5(entity.name.encode(), usedforsecurity=False).hexdigest()[:8]}"
            entity_id_map[entity.name] = entity_id

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
            self.engine.graph.add_node(entity_id, **node.model_dump())

            if self.engine.backend:
                data = self.engine._serialize_node(node, label="Entity")
                self.engine._upsert_node("Entity", entity_id, data)

        # Persist claims to KG
        for claim in result.claims:
            claim_id = f"claim:{uuid.uuid4().hex[:8]}"

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
            self.engine.graph.add_node(claim_id, **node.model_dump())

            if self.engine.backend:
                data = self.engine._serialize_node(node, label="Claim")
                self.engine._upsert_node("Claim", claim_id, data)

            # Link claim to its source
            if source_id in self.engine.graph:
                self.engine.link_nodes(
                    claim_id, source_id, RegistryEdgeType.WAS_DERIVED_FROM
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

        logger.info(
            "[CONCEPT:AU-KG.enrichment.entity-claim-extraction] Extracted %d entities, %d claims, %d relationships from %s",
            len(result.entities),
            len(result.claims),
            len(result.relationships),
            source_id,
        )

        return result
