#!/usr/bin/python
"""KB Structured Extraction Engine.

Uses Pydantic AI with typed output_type= to extract structured articles,
facts, and concepts from raw document chunks. This ensures all LLM output
is validated by Pydantic before entering the knowledge graph.
"""

import logging
import os

from ...models.knowledge_base import (
    DocumentChunk,
    ExtractedArticle,
    ExtractedKBIndex,
    KBHealthReport,
)

logger = logging.getLogger(__name__)

_ARTICLE_SYSTEM_PROMPT = """\
You are a knowledge compiler. Your job is to read raw document chunks and compile
a well-structured wiki article.

Guidelines:
- Write clear, precise, factual content. Do not hallucinate.
- Extract concrete facts with their certainty level (1.0 = certain, 0.5 = likely).
- Identify key concepts this article covers (max 10, short names).
- Note backlinks: titles of other articles in this KB that are closely related.
- Tags should be lowercase, hyphen-separated keywords.
- Summary must be 2-3 sentences covering the core concept.
- Content should be full markdown with sections, code examples where relevant.
"""

_HEALTH_SYSTEM_PROMPT = """\
You are a knowledge base auditor. Review the provided articles and identify:
- Contradictions between facts
- Missing data or gaps in coverage
- Orphaned articles with no backlinks
- Stale or outdated content
- Opportunities for new articles

Provide a consistency_score from 0.0 (many issues) to 1.0 (no issues).
"""

_INDEX_SYSTEM_PROMPT = """\
You are a knowledge base indexer. Given a list of article titles and summaries,
produce a comprehensive index with:
- A 2-3 paragraph overview of the knowledge base
- One-liner summaries for each article
- Top-level concepts that span multiple articles
- 5-10 example queries this KB can answer (for agent discoverability)
"""


class KBExtractor:
    """LLM-backed structured extraction using Pydantic AI.

    All extraction results are Pydantic-validated before graph insertion.
    Uses the same provider/model as the rest of the agent stack by default.
    """

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self._model_str = model or os.environ.get("MODEL_ID", "gpt-4o-mini")
        self._provider = provider or os.environ.get("PROVIDER", "openai")
        self._base_url = base_url or os.environ.get("LLM_BASE_URL")
        self._api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self._article_agent = None
        self._health_agent = None
        self._index_agent = None

    def _get_model(self):
        """Lazily build the Pydantic AI model instance."""
        try:
            from agent_utilities.core.model_factory import create_model

            return create_model(
                provider=self._provider,
                model_id=self._model_str,
                base_url=self._base_url,
                api_key=self._api_key,
            )
        except Exception:
            # Fallback: simple string model identifier
            return self._model_str

    def _get_article_agent(self):
        """Lazily create the article extraction Pydantic AI agent."""
        if self._article_agent is None:
            try:
                from pydantic_ai import Agent

                self._article_agent = Agent(
                    model=self._get_model(),
                    output_type=ExtractedArticle,
                    system_prompt=_ARTICLE_SYSTEM_PROMPT,
                )
            except Exception as e:
                logger.error(f"Cannot create extraction agent: {e}")
                return None
        return self._article_agent

    def _get_health_agent(self):
        """Lazily create the health-check Pydantic AI agent."""
        if self._health_agent is None:
            try:
                from pydantic_ai import Agent

                self._health_agent = Agent(
                    model=self._get_model(),
                    output_type=KBHealthReport,
                    system_prompt=_HEALTH_SYSTEM_PROMPT,
                )
            except Exception as e:
                logger.error(f"Cannot create health agent: {e}")
                return None
        return self._health_agent

    def _get_index_agent(self):
        """Lazily create the index generation Pydantic AI agent."""
        if self._index_agent is None:
            try:
                from pydantic_ai import Agent

                self._index_agent = Agent(
                    model=self._get_model(),
                    output_type=ExtractedKBIndex,
                    system_prompt=_INDEX_SYSTEM_PROMPT,
                )
            except Exception as e:
                logger.error(f"Cannot create index agent: {e}")
                return None
        return self._index_agent

    async def extract_article(
        self,
        chunks: list[DocumentChunk],
        topic: str,
        existing_article: ExtractedArticle | None = None,
    ) -> ExtractedArticle | None:
        """Extract a structured article from document chunks.

        If existing_article is provided, only the new chunks are processed
        and merged with the existing article (incremental mode — saves tokens).

        Args:
            chunks: Document chunks to compile into an article.
            topic: KB topic context (e.g., "Pydantic AI documentation").
            existing_article: Optional existing article for incremental update.

        Returns:
            A validated ExtractedArticle or None on failure.
        """
        agent = self._get_article_agent()
        if not agent:
            return self._fallback_article(chunks, topic)

        # Build the prompt
        content = "\n\n---\n\n".join(c.content for c in chunks)
        if existing_article:
            prompt = (
                f"Topic: {topic}\n\n"
                f"EXISTING ARTICLE (update and improve it):\n"
                f"Title: {existing_article.title}\n"
                f"Content: {existing_article.content[:2000]}...\n\n"
                f"NEW SOURCE MATERIAL (integrate this):\n{content[:6000]}"
            )
        else:
            prompt = f"Topic: {topic}\n\nSource material:\n{content[:8000]}"

        try:
            result = await agent.run(prompt)
            return result.data
        except Exception as e:
            logger.error(f"Extraction failed for topic '{topic}': {e}")
            return self._fallback_article(chunks, topic)

    async def run_health_check(
        self, kb_id: str, kb_name: str, articles: list[dict]
    ) -> KBHealthReport | None:
        """Lint a knowledge base: find contradictions, gaps, orphans.

        Args:
            kb_id: The knowledge base ID.
            kb_name: Human-readable KB name.
            articles: List of article dicts with keys: id, title, summary, tags.

        Returns:
            A validated KBHealthReport or None on failure.
        """
        agent = self._get_health_agent()
        if not agent:
            return KBHealthReport(
                kb_id=kb_id,
                kb_name=kb_name,
                consistency_score=1.0,
                summary="Health check unavailable (LLM not configured).",
            )

        article_list = "\n".join(
            f"- [{a.get('title', a.get('id', 'unknown'))}]: {a.get('summary', '')[:200]}"
            for a in articles[:50]  # Cap at 50 articles per check
        )
        prompt = (
            f"Knowledge Base: {kb_name} (id: {kb_id})\n"
            f"Articles ({len(articles)} total):\n{article_list}"
        )

        try:
            result = await agent.run(prompt)
            report = result.data
            # Ensure KB identity is set
            report.kb_id = kb_id
            report.kb_name = kb_name
            return report
        except Exception as e:
            logger.error(f"Health check failed for KB '{kb_name}': {e}")
            return KBHealthReport(
                kb_id=kb_id,
                kb_name=kb_name,
                consistency_score=0.5,
                summary=f"Health check encountered an error: {e}",
            )

    async def generate_index(
        self, kb_id: str, articles: list[ExtractedArticle]
    ) -> ExtractedKBIndex | None:
        """Generate or refresh the KB discovery index.

        The index is what agents read first to understand what a KB contains
        and what questions it can answer.
        """
        agent = self._get_index_agent()
        if not agent:
            return None

        article_list = "\n".join(
            f"- {a.title}: {a.summary[:150]}" for a in articles[:100]
        )
        prompt = f"Knowledge Base ID: {kb_id}\nArticles:\n{article_list}"

        try:
            result = await agent.run(prompt)
            return result.data
        except Exception as e:
            logger.error(f"Index generation failed for KB '{kb_id}': {e}")
            return None

    # ------------------------------------------------------------------
    # Fallback: no-LLM extraction from chunks (for testing / offline use)
    # ------------------------------------------------------------------

    def _fallback_article(
        self, chunks: list[DocumentChunk], topic: str
    ) -> ExtractedArticle:
        """Create a minimal article without LLM (fallback mode)."""
        combined = " ".join(c.content for c in chunks[:3])
        title = topic or (chunks[0].source_path.split("/")[-1] if chunks else "Unknown")
        summary = combined[:300] + ("..." if len(combined) > 300 else "")
        return ExtractedArticle(
            title=title,
            summary=summary,
            content="\n\n".join(c.content for c in chunks),
            concepts=[topic] if topic else [],
            facts=[],
            backlinks=[],
            tags=[topic.lower().replace(" ", "-")] if topic else [],
        )
