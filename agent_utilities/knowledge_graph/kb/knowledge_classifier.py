#!/usr/bin/python
"""Universal Knowledge Classifier.

CONCEPT:KG-2.6 — Universal Knowledge Assimilation

LLM-backed classifier that evaluates ANY incoming content for KG ingestion.
Determines importance tier, evolution potential, and ingestion routing for
all knowledge sources: X posts, research papers, GitHub repos, documents,
and memories.

This is source-agnostic by design — the same scoring pipeline applies
whether the content comes from ``x_search``, ``scholarx``, ``github-mcp``,
or manual document ingestion.

Usage::

    classifier = UniversalKnowledgeClassifier()
    result = await classifier.classify(
        content="Monte Carlo permutation test for trading systems...",
        source_type="x_post",
        metadata={"author": "@phosphenq", "engagement": {"likes": 100}},
    )
    print(result.action)  # "ingest_and_evolve"
    print(result.evolution_potential)  # 0.85
"""

import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from ...core.config import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Classification Output Model
# --------------------------------------------------------------------------- #


class KnowledgeClassification(BaseModel):
    """Universal classification result for any incoming knowledge.

    Produced by ``UniversalKnowledgeClassifier.classify()`` and consumed by
    ingestion bridges (X, ScholarX, GitHub, etc.) to decide how to persist
    content into the Knowledge Graph.

    Attributes:
        importance_score: Overall importance (0.0–1.0). Drives decay/permanence.
        is_permanent: If True, node is hub-protected and won't decay.
        content_tier: Human-readable tier label.
        evolution_potential: How much this content could improve agent-utilities
            (0.0–1.0). Scored against existing KG concepts and the 5-pillar
            architecture (ORCH, KG, AHE, ECO, OS).
        evolution_reasoning: LLM-generated explanation of evolution opportunity.
        suggested_node_type: Target KG node type for ingestion.
        suggested_kb_name: Which KB namespace to ingest into (None = default).
        concepts: Key concepts extracted from the content.
        matching_kg_topics: Existing KG topics this content relates to.
        source_quality: Credibility/signal quality of the source (0.0–1.0).
        action: Ingestion action to take.
    """

    importance_score: float = Field(ge=0.0, le=1.0)
    is_permanent: bool = False
    content_tier: Literal[
        "ephemeral", "standard", "high_value", "critical"
    ] = "standard"
    evolution_potential: float = Field(default=0.0, ge=0.0, le=1.0)
    evolution_reasoning: str = ""
    suggested_node_type: str = "social_post"
    suggested_kb_name: str | None = None
    concepts: list[str] = Field(default_factory=list)
    matching_kg_topics: list[str] = Field(default_factory=list)
    source_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    action: Literal["ingest", "ingest_and_evolve", "decay", "skip"] = "decay"


# --------------------------------------------------------------------------- #
# Classification Prompts
# --------------------------------------------------------------------------- #

_CLASSIFIER_SYSTEM_PROMPT = """\
You are a knowledge triage agent for the agent-utilities ecosystem — an
autonomous self-evolving AI framework built on 5 pillars:

1. ORCH (Orchestration) — Multi-agent coordination, routing, topology
2. KG (Knowledge Graph) — Semantic storage, retrieval, distillation
3. AHE (Agentic Harness Engineering) — Self-improvement, testing, evolution
4. ECO (Ecosystem Peripherals) — MCP servers, API clients, messaging
5. OS (Agent Operating System) — Security, scheduling, identity, guardrails

Your job is to evaluate incoming knowledge and determine:
1. How important is this content? (0.0–1.0)
2. Could this content help agent-utilities evolve itself? (0.0–1.0)
3. What KG node type should receive it?
4. Should we ingest it, let it decay, or skip it entirely?

SCORING GUIDELINES:

Importance Tiers:
- "critical" (≥0.9): Breakthrough research, novel agent architectures, new
  MCP/tool paradigms, quantitative frameworks with implementations
- "high_value" (0.7–0.9): Technical deep-dives, framework docs, significant
  research papers, detailed implementation guides
- "standard" (0.4–0.7): Product launches, notable threads, interesting
  perspectives, developer updates
- "ephemeral" (≤0.3): Memes, promotional content, engagement bait, low-signal
  replies, generic news without technical depth

Evolution Potential Scoring:
- HIGH (≥0.7): Content directly about agent architectures, knowledge graphs,
  MCP tools, pydantic-ai, RAG techniques, LLM optimization, self-evolving
  systems, agentic workflows, multi-agent coordination
- MEDIUM (0.3–0.7): Adjacent topics — ML frameworks, graph databases,
  API design patterns, testing methodologies, security hardening
- LOW (<0.3): Unrelated domains — social commentary, marketing, politics,
  entertainment, general business

Node Type Mapping:
- Social posts (tweets) → "social_post"
- Long-form articles/X Articles → "article"
- Research papers → "article" with suggested_kb_name="research"
- GitHub repositories → "software_project"
- Documentation pages → "article" with suggested_kb_name="documentation"
- Ephemeral observations → "observation"

Always extract 3–10 key concepts from the content.
"""


# --------------------------------------------------------------------------- #
# Classifier Implementation
# --------------------------------------------------------------------------- #


class UniversalKnowledgeClassifier:
    """LLM-backed classifier that evaluates ANY content for KG ingestion.

    Uses the same LLM backend as the rest of the agent stack (LM Studio
    by default) via Pydantic AI structured output.

    Args:
        model: Override model ID (default: from AgentConfig).
        provider: Override provider (default: from AgentConfig).
        base_url: Override base URL (default: from AgentConfig).
        api_key: Override API key (default: from AgentConfig).
    """

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        _default_chat = config.default_chat_model
        self._model_str = model or os.environ.get("MODEL_ID", "gpt-4o-mini")
        self._provider = provider or os.environ.get("PROVIDER", "openai")
        self._base_url = base_url or (_default_chat.base_url if _default_chat else None)
        self._api_key = (
            api_key or (_default_chat.api_key if _default_chat else "") or ""
        )
        self._agent: Any = None

    def _get_agent(self):
        """Lazily create the Pydantic AI classification agent."""
        if self._agent is None:
            try:
                from pydantic_ai import Agent

                from ...core.model_factory import create_model

                model = create_model(
                    provider=self._provider,
                    model_id=self._model_str,
                    base_url=self._base_url,
                    api_key=self._api_key,
                )
                self._agent = Agent(
                    model=model,
                    output_type=KnowledgeClassification,
                    system_prompt=_CLASSIFIER_SYSTEM_PROMPT,
                )
            except Exception as e:
                logger.error("Cannot create classifier agent: %s", e)
                return None
        return self._agent

    async def classify(
        self,
        content: str,
        source_type: str,
        metadata: dict[str, Any] | None = None,
        kg_context: list[str] | None = None,
    ) -> KnowledgeClassification:
        """Classify content for KG ingestion.

        Args:
            content: The text content to classify.
            source_type: Origin type — one of: x_post, x_article,
                research_paper, github_repo, document, memory.
            metadata: Optional metadata (author, engagement, url, etc.).
            kg_context: Optional list of existing KG topic names/concepts
                for matching (improves evolution potential accuracy).

        Returns:
            A validated ``KnowledgeClassification`` with scoring and routing.
        """
        agent = self._get_agent()
        if not agent:
            return self._fallback_classify(content, source_type, metadata)

        meta = metadata or {}
        ctx_section = ""
        if kg_context:
            ctx_section = (
                "\n\nEXISTING KG TOPICS (match against these for evolution potential):\n"
                + "\n".join(f"- {t}" for t in kg_context[:30])
            )

        prompt = (
            f"SOURCE TYPE: {source_type}\n"
            f"METADATA: {_format_metadata(meta)}\n"
            f"{ctx_section}\n\n"
            f"CONTENT:\n{content[:12000]}"
        )

        try:
            result = await agent.run(prompt)
            classification = result.data
            logger.info(
                "Classified [%s] → tier=%s importance=%.2f evolution=%.2f action=%s",
                source_type,
                classification.content_tier,
                classification.importance_score,
                classification.evolution_potential,
                classification.action,
            )
            return classification
        except Exception as e:
            logger.error("Classification failed: %s", e)
            return self._fallback_classify(content, source_type, metadata)

    def _fallback_classify(
        self,
        content: str,
        source_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeClassification:
        """Heuristic fallback when LLM is unavailable.

        Uses keyword matching and content length to approximate importance.
        """
        meta = metadata or {}
        text_lower = content.lower()
        word_count = len(content.split())

        # Heuristic importance from content length and engagement
        importance = 0.3
        if word_count > 500:
            importance += 0.2
        if word_count > 2000:
            importance += 0.2

        likes = meta.get("engagement", {}).get("likes", 0)
        if isinstance(likes, int) and likes > 50:
            importance += 0.1

        # Heuristic evolution potential from keyword presence
        evolution_keywords = [
            "agent",
            "pydantic",
            "knowledge graph",
            "mcp",
            "rag",
            "llm",
            "self-evolving",
            "multi-agent",
            "orchestration",
            "embeddings",
            "tool-use",
            "function calling",
            "agentic",
        ]
        evolution_hits = sum(1 for kw in evolution_keywords if kw in text_lower)
        evolution_potential = min(1.0, evolution_hits * 0.15)

        # Determine tier
        if importance >= 0.9:
            tier: Literal[
                "ephemeral", "standard", "high_value", "critical"
            ] = "critical"
        elif importance >= 0.7:
            tier = "high_value"
        elif importance >= 0.4:
            tier = "standard"
        else:
            tier = "ephemeral"

        # Determine action
        if evolution_potential >= 0.7:
            action: Literal[
                "ingest", "ingest_and_evolve", "decay", "skip"
            ] = "ingest_and_evolve"
        elif importance >= 0.5:
            action = "ingest"
        elif importance >= 0.3:
            action = "decay"
        else:
            action = "skip"

        # Node type mapping
        node_type_map = {
            "x_post": "social_post",
            "x_article": "article",
            "research_paper": "article",
            "github_repo": "software_project",
            "document": "article",
            "memory": "observation",
        }

        return KnowledgeClassification(
            importance_score=min(1.0, importance),
            is_permanent=importance >= 0.7,
            content_tier=tier,
            evolution_potential=evolution_potential,
            evolution_reasoning="Heuristic fallback (LLM unavailable)",
            suggested_node_type=node_type_map.get(source_type, "article"),
            suggested_kb_name=None,
            concepts=[],
            matching_kg_topics=[],
            source_quality=0.5,
            action=action,
        )


def _format_metadata(meta: dict[str, Any]) -> str:
    """Format metadata dict into a readable string for the LLM prompt."""
    parts = []
    for k, v in meta.items():
        if isinstance(v, dict):
            inner = ", ".join(f"{ik}={iv}" for ik, iv in v.items())
            parts.append(f"{k}: {{{inner}}}")
        elif isinstance(v, list):
            parts.append(f"{k}: [{', '.join(str(x) for x in v[:5])}]")
        else:
            parts.append(f"{k}: {v}")
    return "; ".join(parts) if parts else "none"
