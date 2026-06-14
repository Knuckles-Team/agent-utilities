#!/usr/bin/python
"""X Content Ingestion Bridge.

CONCEPT:ECO-4.0 — Social Content Ingestion

Bridges the X search tools (``x_search``, ``browse_x_post``) to the
Universal Knowledge Classifier and the Knowledge Graph. Handles:

- Parsing ``browse_x_post`` / ``x_search`` JSON output
- Running the ``UniversalKnowledgeClassifier`` for tier + evolution scoring
- Creating ``SocialPostNode`` instances in the KG with proper edges
- Detecting X Articles and routing them to ``KBIngestionEngine``
- Creating ``EvolutionCandidateNode`` for high-potential content

Usage::

    bridge = XIngestionBridge(graph=nx_graph)
    result = await bridge.ingest_browse_result(browse_json)
    print(result["action"])     # "ingest_and_evolve"
    print(result["node_id"])    # "social:x:2057129225593741768"

This module is designed to be called automatically by ``browse_x_post``
when ``auto_ingest=True`` (the default).
"""

import hashlib
import json
import logging
import re
import time
import typing
from typing import Any, Literal

# Rust-native graph compute — using GraphComputeEngine
from ...models.knowledge_graph import (
    EvolutionCandidateNode,
    RegistryEdgeType,
    RegistryNodeType,
    SocialPostNode,
)
from ..backends.base import GraphBackend
from .knowledge_classifier import KnowledgeClassification, UniversalKnowledgeClassifier

logger = logging.getLogger(__name__)

# Regex for detecting X Article URLs
_ARTICLE_URL_RE = re.compile(
    r"https?://(?:x\.com|twitter\.com)/\w+/articles?/\w+", re.IGNORECASE
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _social_post_id(platform: str, post_id: str) -> str:
    """Generate a deterministic KG node ID for a social post."""
    return f"social:{platform}:{post_id}"


def _person_id(handle: str, platform: str = "x") -> str:
    """Generate a deterministic KG node ID for a person."""
    return f"person:{platform}:{handle.lower().strip('@')}"


def _evolution_candidate_id(source_node_id: str) -> str:
    """Generate a deterministic KG node ID for an evolution candidate."""
    h = hashlib.sha256(source_node_id.encode()).hexdigest()[:12]
    return f"evo_candidate:{h}"


class XIngestionBridge:
    """Bridges X tool output → UniversalKnowledgeClassifier → Knowledge Graph.

    Args:
        graph: The GraphComputeEngine instance (shared with KG engine).
        backend: Optional graph backend for persistence.
        classifier: Optional pre-configured classifier instance.
        auto_evolve_threshold: Evolution potential score above which
            an ``EvolutionCandidateNode`` is created (default: 0.6).
    """

    def __init__(
        self,
        graph: Any,
        backend: GraphBackend | None = None,
        classifier: UniversalKnowledgeClassifier | None = None,
        auto_evolve_threshold: float = 0.6,
    ) -> None:
        self.graph = graph
        self.backend = backend
        self.classifier = classifier or UniversalKnowledgeClassifier()
        self.auto_evolve_threshold = auto_evolve_threshold

    async def ingest_browse_result(
        self,
        browse_json: str | dict,
        kg_context: list[str] | None = None,
    ) -> dict[str, Any]:
        """Ingest a ``browse_x_post`` result into the Knowledge Graph.

        Args:
            browse_json: JSON string or dict from ``browse_x_post``.
            kg_context: Optional existing KG topics for evolution matching.

        Returns:
            Dict with ingestion result: action, node_id, classification, etc.
        """
        data = json.loads(browse_json) if isinstance(browse_json, str) else browse_json

        if not data.get("success", False):
            return {
                "action": "skip",
                "reason": data.get("error", "Browse failed"),
                "node_id": None,
            }

        post_id = str(data.get("post_id", ""))
        username = str(data.get("username", "unknown"))
        answer = str(data.get("answer", ""))
        url = str(data.get("url", ""))
        citations = data.get("citations", []) + data.get("inline_citations", [])

        if not answer:
            return {"action": "skip", "reason": "Empty answer", "node_id": None}

        # Extract engagement metrics from the answer text (Grok embeds them)
        engagement = _extract_engagement(answer)

        # Build metadata for classifier
        metadata = {
            "author": f"@{username}",
            "platform": "x",
            "post_id": post_id,
            "url": url,
            "engagement": engagement,
            "citation_count": len(citations),
        }

        # Detect if this is an X Article (long-form content)
        is_article = len(answer) > 3000 or any(
            _ARTICLE_URL_RE.search(c.get("url", "")) for c in citations
        )
        source_type = "x_article" if is_article else "x_post"

        # Classify content
        classification = await self.classifier.classify(
            content=answer,
            source_type=source_type,
            metadata=metadata,
            kg_context=kg_context,
        )

        if classification.action == "skip":
            logger.info(
                "Skipping X post %s (tier=%s, importance=%.2f)",
                post_id,
                classification.content_tier,
                classification.importance_score,
            )
            return {
                "action": "skip",
                "reason": f"Classifier: {classification.content_tier}",
                "node_id": None,
                "classification": classification.model_dump(),
            }

        # Create SocialPost node
        node_id = _social_post_id("x", post_id)
        post_type = typing.cast(
            Literal["tweet", "article", "thread"], "article" if is_article else "tweet"
        )

        social_node = SocialPostNode(
            id=node_id,
            name=f"X post by @{username}",
            description=answer[:200],
            post_id=post_id,
            author_handle=username,
            platform="x",
            content_text=answer,
            post_url=url,
            post_type=post_type,
            engagement_metrics=engagement,
            citations=[
                {"url": c.get("url", ""), "title": c.get("title", "")}
                for c in citations
            ],
            importance_score=classification.importance_score,
            is_permanent=classification.is_permanent,
            evolution_potential=classification.evolution_potential,
            timestamp=_now(),
            metadata={
                "content_tier": classification.content_tier,
                "source_type": source_type,
                "concepts": classification.concepts,
            },
        )
        self.graph.add_node(node_id, **social_node.model_dump())
        logger.info(
            "Created SocialPost node: %s (tier=%s, evolution=%.2f)",
            node_id,
            classification.content_tier,
            classification.evolution_potential,
        )

        # Create Person node + CREATED_BY_PERSON edge
        person_id = _person_id(username)
        if person_id not in self.graph.nodes:
            self.graph.add_node(
                person_id,
                id=person_id,
                type=RegistryNodeType.PERSON,
                name=f"@{username}",
                description=f"X user @{username}",
                platform="x",
                importance_score=0.3,
                timestamp=_now(),
            )
        self.graph.add_edge(node_id, person_id, type=RegistryEdgeType.CREATED_BY_PERSON)

        # Link to extracted concepts via ABOUT edges
        for concept_name in classification.concepts:
            concept_id = f"kbc:social:{concept_name.lower().replace(' ', '-')[:40]}"
            if concept_id not in self.graph.nodes:
                self.graph.add_node(
                    concept_id,
                    id=concept_id,
                    type=RegistryNodeType.KB_CONCEPT,
                    name=concept_name,
                    description=f"Concept: {concept_name}",
                    importance_score=0.5,
                    timestamp=_now(),
                )
            self.graph.add_edge(node_id, concept_id, type=RegistryEdgeType.ABOUT)

        # Create EvolutionCandidate if evolution potential is high
        evo_node_id = None
        if classification.evolution_potential >= self.auto_evolve_threshold:
            evo_node_id = _evolution_candidate_id(node_id)
            evo_node = EvolutionCandidateNode(
                id=evo_node_id,
                name=f"Evolution candidate: {social_node.name}",
                description=classification.evolution_reasoning[:200],
                source_node_id=node_id,
                source_type=source_type,
                evolution_score=classification.evolution_potential,
                evolution_reasoning=classification.evolution_reasoning,
                matching_concepts=classification.matching_kg_topics,
                status="pending",
                importance_score=classification.evolution_potential,
                is_permanent=True,
                timestamp=_now(),
            )
            self.graph.add_node(evo_node_id, **evo_node.model_dump())
            self.graph.add_edge(
                evo_node_id,
                node_id,
                type=RegistryEdgeType.EVOLUTION_CANDIDATE_OF,
            )
            logger.info(
                "Created EvolutionCandidate: %s (score=%.2f) — %s",
                evo_node_id,
                classification.evolution_potential,
                classification.evolution_reasoning[:100],
            )

        # If this is an X Article, attempt to ingest the full article content
        article_node_id = None
        if is_article and classification.action in ("ingest", "ingest_and_evolve"):
            article_node_id = await self._ingest_article(
                node_id=node_id,
                url=url,
                citations=citations,
                classification=classification,
            )

        # Persist to backend if available
        if self.backend:
            self._persist_node(node_id)
            self._persist_node(person_id)
            if evo_node_id:
                self._persist_node(evo_node_id)

        return {
            "action": classification.action,
            "node_id": node_id,
            "post_type": post_type,
            "evolution_candidate_id": evo_node_id,
            "article_node_id": article_node_id,
            "classification": classification.model_dump(),
            # Verbatim post text so the ingestion seam can extract concepts +
            # canonical facts from it (KG-2.8 + KG-2.64).
            "content_text": answer,
            "title": f"X post by @{username}",
        }

    async def _ingest_article(
        self,
        node_id: str,
        url: str,
        citations: list[dict],
        classification: KnowledgeClassification,
    ) -> str | None:
        """Ingest an X Article's full content via KBIngestionEngine.

        X Articles are long-form posts (up to ~100K chars) that require
        fetching the full rendered content beyond what the API returns.

        This method:
        1. Finds the article URL from citations or the post URL
        2. Delegates to ``KBIngestionEngine.ingest_url()`` for full KB processing
        3. Links the SocialPost → Article via ``PROMOTES_RESEARCH`` edge

        Args:
            node_id: The SocialPost node ID in the KG.
            url: The original post URL.
            citations: Citation list from the browse result.
            classification: The classification result.

        Returns:
            The Article node ID if ingestion succeeded, None otherwise.
        """
        # Find the best article URL — check citations first, then use post URL
        article_url = url
        for c in citations:
            c_url = c.get("url", "")
            if _ARTICLE_URL_RE.search(c_url):
                article_url = c_url
                break

        kb_name = classification.suggested_kb_name or "x-articles"

        try:
            from .ingestion import KBIngestionEngine

            # Create an ingestion engine with our graph
            ingestion_engine = KBIngestionEngine(
                graph=self.graph,
                backend=self.backend,
            )

            await ingestion_engine.ingest_url(
                url=article_url,
                kb_name=kb_name,
                topic=f"X Article: {', '.join(classification.concepts[:3])}",
            )

            # Find the Article node created by the ingestion engine
            article_node_id = None
            for n in self.graph.nodes:
                n_data = self.graph.nodes[n]
                if (
                    n_data.get("type") == RegistryNodeType.ARTICLE
                    and n_data.get("metadata", {}).get("source_url") == article_url
                ):
                    article_node_id = n
                    break

            # Fall back to using the KB article pattern
            if not article_node_id:
                # KBIngestionEngine creates articles with predictable IDs
                for n in self.graph.nodes:
                    n_data = self.graph.nodes[n]
                    if n_data.get(
                        "type"
                    ) == RegistryNodeType.ARTICLE and kb_name in str(n):
                        article_node_id = n
                        break

            if article_node_id:
                # Link SocialPost → Article via PROMOTES_RESEARCH
                self.graph.add_edge(
                    node_id,
                    article_node_id,
                    type=RegistryEdgeType.PROMOTES_RESEARCH,
                )
                logger.info(
                    "Linked SocialPost %s → Article %s via PROMOTES_RESEARCH",
                    node_id,
                    article_node_id,
                )
                return article_node_id

            logger.info(
                "Article ingested into KB '%s' but node ID not resolved",
                kb_name,
            )
            return None

        except ImportError:
            logger.debug("KBIngestionEngine not available for article ingestion")
            return None
        except Exception as e:
            logger.warning("Article ingestion failed for %s: %s", article_url, e)
            return None

    async def ingest_search_results(
        self,
        search_json: str | dict,
        score_threshold: float = 0.5,
        kg_context: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Ingest multiple results from ``x_search``.

        Only ingests results above the importance score threshold.

        Args:
            search_json: JSON string or dict from ``x_search``.
            score_threshold: Minimum importance to persist.
            kg_context: Optional KG topics for evolution matching.

        Returns:
            List of ingestion results (same format as ingest_browse_result).
        """
        data = json.loads(search_json) if isinstance(search_json, str) else search_json

        if not data.get("success", False):
            return [{"action": "skip", "reason": "Search failed"}]

        # For x_search results, the answer is a single text with inline citations
        # We treat the entire result as one item
        result = await self.ingest_browse_result(
            browse_json=data, kg_context=kg_context
        )
        if (
            result.get("classification", {}).get("importance_score", 0)
            >= score_threshold
        ):
            return [result]
        return []

    def _persist_node(self, node_id: str) -> None:
        """Persist a single node to the graph backend."""
        if not self.backend or node_id not in self.graph.nodes:
            return
        data = dict(self.graph.nodes[node_id])
        node_type = data.get("type", "")
        if isinstance(node_type, RegistryNodeType):
            node_type = node_type.value

        table_map = {
            "social_post": "SocialPost",
            "person": "Person",
            "kb_concept": "KBConcept",
            "evolution_candidate": "EvolutionCandidate",
        }
        table = table_map.get(node_type)
        if not table:
            return

        try:
            fields = {
                k: v
                for k, v in data.items()
                if isinstance(v, str | int | float | bool) and k != "id"
            }
            set_clause = ", ".join(f"n.{k} = ${k}" for k in fields)
            query = f"MERGE (n:{table} {{id: $id}}) SET {set_clause}"
            self.backend.execute(query, {"id": node_id, **fields})
        except Exception as e:
            logger.debug("Backend persist failed for %s: %s", node_id, e)


def _extract_engagement(answer_text: str) -> dict[str, int]:
    """Extract engagement metrics from Grok's answer text.

    Grok typically embeds metrics like:
    ``Likes=100, Reposts=7, Quotes=1, Replies=1, Bookmarks=201, Views=47193``
    """
    metrics: dict[str, int] = {}
    patterns = [
        (r"Likes?\s*[=:]\s*(\d[\d,]*)", "likes"),
        (r"Reposts?\s*[=:]\s*(\d[\d,]*)", "reposts"),
        (r"Retweets?\s*[=:]\s*(\d[\d,]*)", "retweets"),
        (r"Quotes?\s*[=:]\s*(\d[\d,]*)", "quotes"),
        (r"Replies?\s*[=:]\s*(\d[\d,]*)", "replies"),
        (r"Bookmarks?\s*[=:]\s*(\d[\d,]*)", "bookmarks"),
        (r"Views?\s*[=:]\s*(\d[\d,]*)", "views"),
        (r"Impressions?\s*[=:]\s*(\d[\d,]*)", "impressions"),
    ]
    for pattern, name in patterns:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            val_str = match.group(1).replace(",", "")
            try:
                metrics[name] = int(val_str)
            except ValueError:
                pass
    return metrics
