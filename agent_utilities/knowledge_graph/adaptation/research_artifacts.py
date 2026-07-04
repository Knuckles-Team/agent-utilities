#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-KG.research.research-pipeline-runner — Research Artifact Generator.

Generates actionable LLM artifacts from KG-ingested research papers,
producing structured summaries, innovation extractions, concept linkages,
and periodic research digests.

Architecture:
    - **ResearchArtifactGenerator**: Creates rich markdown artifacts from
      ArticleNodes stored in the Knowledge Graph.
    - **ResearchArtifact**: Structured representation of a paper's
      actionable insights.
    - **DigestArtifact**: Periodic summary of research discoveries.

Integrates with:
    - CONCEPT:AU-KG.research.research-pipeline-runner (ResearchPipelineRunner): Paper ingestion source
    - CONCEPT:AU-KG.query.object-graph-mapper (IntelligenceGraphEngine): Graph traversal
    - CONCEPT:AU-KG.research.research-pipeline-runner (KGSourceResolver): Content materialization

See docs/pillars/2_epistemic_knowledge_graph.md §CONCEPT:AU-KG.research.research-pipeline-runner
"""


import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class ResearchArtifact(BaseModel):
    """Structured artifact from a single research paper.

    Attributes:
        article_id: KG ArticleNode ID.
        title: Paper title.
        summary: Structured summary of key contributions.
        key_contributions: List of main contributions.
        methods: Methodologies used.
        potential_applications: How this could apply to existing projects.
        concept_linkages: Connections to existing KG concepts.
        suggested_experiments: Agent-actionable experiment ideas.
        tags: Searchable topic tags.
        authors: Paper authors.
        source_url: Original paper URL.
        importance_score: Computed importance.
        timestamp: Generation timestamp.
    """

    article_id: str
    title: str
    summary: str = ""
    key_contributions: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    potential_applications: list[str] = Field(default_factory=list)
    concept_linkages: list[dict[str, str]] = Field(default_factory=list)
    suggested_experiments: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    source_url: str = ""
    importance_score: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


class DigestArtifact(BaseModel):
    """Periodic research digest summarizing discoveries.

    Attributes:
        period: 'daily', 'weekly', or 'monthly'.
        timestamp: Generation timestamp.
        paper_count: Total papers in digest.
        top_papers: Most important paper artifacts.
        domain_distribution: Count of papers per domain.
        emerging_themes: Detected emerging research themes.
        markdown: Pre-rendered markdown content.
    """

    period: str = "daily"
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    paper_count: int = 0
    top_papers: list[ResearchArtifact] = Field(default_factory=list)
    domain_distribution: dict[str, int] = Field(default_factory=dict)
    emerging_themes: list[str] = Field(default_factory=list)
    markdown: str = ""


class ResearchArtifactGenerator:
    """CONCEPT:AU-KG.research.research-pipeline-runner — Generates actionable LLM artifacts from KG-ingested research.

    Each artifact contains:
    - Structured summary (key contributions, methods, results)
    - Innovation extraction (what can we use?)
    - Concept linkages (how does this connect to existing KG knowledge?)
    - Suggested experiments (agent-actionable items)
    - Searchable tags and embeddings

    Args:
        engine: The IntelligenceGraphEngine to query.
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None):
        self.engine = engine

    def generate_paper_artifact(
        self,
        article_id: str,
        target_codebase: str | None = None,
    ) -> ResearchArtifact:
        """Generate a rich artifact for a single paper.

        Args:
            article_id: KG ArticleNode ID.
            target_codebase: Optional codebase to find applications for.

        Returns:
            ResearchArtifact with structured analysis.
        """
        if not self.engine or article_id not in self.engine.graph.nodes:
            return ResearchArtifact(article_id=article_id, title="Unknown")

        data = dict(self.engine.graph.nodes[article_id])
        title = data.get("name", "")
        content = data.get("content", "") or data.get("description", "")
        tags = data.get("tags", [])
        if isinstance(tags, str):
            import json

            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []

        # Extract key contributions from content
        contributions = self._extract_contributions(content)
        methods = self._extract_methods(content)
        applications = self._find_applications(content, target_codebase)
        linkages = self._find_concept_linkages(article_id)
        experiments = self._suggest_experiments(content, target_codebase)

        # Find authors via graph edges
        authors = self._get_authors(article_id)

        # Find source URL
        source_url = ""
        for succ in self.engine.graph.successors(article_id):
            succ_data = self.engine.graph.nodes.get(succ, {})
            if succ_data.get("url"):
                source_url = succ_data["url"]
                break

        return ResearchArtifact(
            article_id=article_id,
            title=title,
            summary=content[:500] if content else "",
            key_contributions=contributions,
            methods=methods,
            potential_applications=applications,
            concept_linkages=linkages,
            suggested_experiments=experiments,
            tags=tags if isinstance(tags, list) else [],
            authors=authors,
            source_url=source_url,
            importance_score=data.get("importance_score", 0.0),
        )

    def generate_digest(
        self,
        paper_ids: list[str],
        period: str = "daily",
    ) -> DigestArtifact:
        """Generate a periodic digest of research discoveries.

        Args:
            paper_ids: List of ArticleNode IDs to include.
            period: 'daily', 'weekly', or 'monthly'.

        Returns:
            DigestArtifact with rendered markdown.
        """
        artifacts: list[ResearchArtifact] = []
        domain_counts: dict[str, int] = {}

        for pid in paper_ids:
            artifact = self.generate_paper_artifact(pid)
            if artifact.title != "Unknown":
                artifacts.append(artifact)
                for tag in artifact.tags:
                    domain_counts[tag] = domain_counts.get(tag, 0) + 1

        # Sort by importance
        artifacts.sort(key=lambda a: a.importance_score, reverse=True)

        # Detect emerging themes (tags appearing 2+ times)
        themes = [t for t, c in domain_counts.items() if c >= 2]

        # Render markdown
        md = self._render_digest_markdown(artifacts, period, domain_counts, themes)

        return DigestArtifact(
            period=period,
            paper_count=len(artifacts),
            top_papers=artifacts[:10],
            domain_distribution=domain_counts,
            emerging_themes=themes,
            markdown=md,
        )

    def save_digest(
        self,
        digest: DigestArtifact,
        output_dir: str | Path | None = None,
    ) -> str:
        """Save a digest to disk as markdown.

        Args:
            digest: The digest to save.
            output_dir: Directory to save to. Defaults to ~/.scholarx/digests/.

        Returns:
            Path to the saved file.
        """
        if output_dir is None:
            output_dir = Path.home() / ".scholarx" / "digests"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        date_str = time.strftime("%Y-%m-%d")
        filename = f"digest_{digest.period}_{date_str}.md"
        filepath = output_dir / filename
        filepath.write_text(digest.markdown, encoding="utf-8")

        logger.info("[CONCEPT:AU-KG.research.research-pipeline-runner] Digest saved: %s", filepath)
        return str(filepath)

    def _extract_contributions(self, content: str) -> list[str]:
        """Extract key contributions from paper content."""
        contributions = []
        indicators = [
            "we propose",
            "we introduce",
            "we present",
            "our contribution",
            "we develop",
            "we design",
            "novel",
            "new approach",
            "we show",
        ]
        sentences = content.replace("\n", " ").split(". ")
        for sent in sentences:
            sent_lower = sent.lower()
            if any(ind in sent_lower for ind in indicators):
                clean = sent.strip()
                if 20 < len(clean) < 300:
                    contributions.append(clean + ".")
        return contributions[:5]

    def _extract_methods(self, content: str) -> list[str]:
        """Extract methodologies from paper content."""
        methods = []
        indicators = [
            "transformer",
            "attention",
            "reinforcement learning",
            "GNN",
            "graph neural",
            "contrastive",
            "self-supervised",
            "fine-tuning",
            "RLHF",
            "DPO",
            "curriculum",
            "distillation",
            "retrieval-augmented",
            "chain-of-thought",
            "tree-of-thought",
            "Monte Carlo",
        ]
        content_lower = content.lower()
        for ind in indicators:
            if ind.lower() in content_lower:
                methods.append(ind)
        return methods

    def _find_applications(
        self, content: str, target_codebase: str | None
    ) -> list[str]:
        """Find potential applications of the paper's ideas."""
        applications = []
        content_lower = content.lower()

        mapping = {
            "multi-agent": "Could enhance ORCH-1.0 specialist routing or ORCH-1.4 swarm orchestration",
            "knowledge graph": "Direct application to KG-2.0 graph engine or KG-2.2 ontology layer",
            "memory": "Could improve KG-2.1 tiered memory or KG-2.7 context compaction",
            "tool use": "Applicable to ECO-4.0 tool interface or ECO-4.1 MCP integration",
            "evaluation": "Could enhance AHE-3.12 EvalRunner or AHE-3.1 distillation",
            "safety": "Relevant to OS-5.4 prompt injection scanner or OS-5.7 guardrails",
            "reward": "Could improve AHE-3.10 decomposed reward signals",
            "planning": "Applicable to ORCH-1.1 HTN planning or AHE-3.9 horizon curriculum",
        }

        for keyword, application in mapping.items():
            if keyword in content_lower:
                applications.append(application)

        return applications[:5]

    def _find_concept_linkages(self, article_id: str) -> list[dict[str, str]]:
        """Find connections between this paper and existing KG concepts."""
        if not self.engine:
            return []

        linkages = []
        # Look at neighboring nodes
        for neighbor in self.engine.graph.successors(article_id):
            n_data = self.engine.graph.nodes.get(neighbor, {})
            n_type = n_data.get("type", "")
            if n_type in ("concept", "fact", "knowledge_base"):
                linkages.append(
                    {
                        "concept_id": neighbor,
                        "concept_name": n_data.get("name", neighbor),
                        "relationship": "related_to",
                    }
                )

        for predecessor in self.engine.graph.predecessors(article_id):
            p_data = self.engine.graph.nodes.get(predecessor, {})
            p_type = p_data.get("type", "")
            if p_type in ("concept", "fact", "knowledge_base"):
                linkages.append(
                    {
                        "concept_id": predecessor,
                        "concept_name": p_data.get("name", predecessor),
                        "relationship": "builds_on",
                    }
                )

        return linkages[:10]

    def _suggest_experiments(
        self, content: str, target_codebase: str | None
    ) -> list[str]:
        """Suggest actionable experiments based on paper content."""
        experiments = []
        content_lower = content.lower()

        if "benchmark" in content_lower:
            experiments.append(
                "Run the proposed benchmark against our existing pipeline to establish a baseline"
            )
        if "ablation" in content_lower:
            experiments.append(
                "Perform an ablation study on our existing implementation using the paper's methodology"
            )
        if "dataset" in content_lower:
            experiments.append(
                "Evaluate our system against the datasets mentioned in this paper"
            )
        if "improvement" in content_lower or "outperform" in content_lower:
            experiments.append(
                "Implement the core technique and measure performance delta against current baseline"
            )

        return experiments[:3]

    def _get_authors(self, article_id: str) -> list[str]:
        """Get author names from PersonNode edges."""
        if not self.engine:
            return []

        authors = []
        for succ in self.engine.graph.successors(article_id):
            edge_data = self.engine.graph.get_edge_data(article_id, succ)
            if edge_data:
                for _, edata in edge_data.items():
                    if edata.get("type") in ("AUTHORED", "authored"):
                        name = self.engine.graph.nodes.get(succ, {}).get("name", "")
                        if name:
                            authors.append(name)
        return authors

    def _render_digest_markdown(
        self,
        artifacts: list[ResearchArtifact],
        period: str,
        domain_counts: dict[str, int],
        themes: list[str],
    ) -> str:
        """Render a complete digest as markdown."""
        date_str = time.strftime("%Y-%m-%d")
        lines = [
            f"# Research Digest ({period.title()}) — {date_str}",
            "",
            f"**Papers analyzed**: {len(artifacts)}",
            "",
        ]

        if domain_counts:
            lines.append("## Domain Distribution")
            lines.append("")
            for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
                bar = "█" * count
                lines.append(f"- **{domain}**: {bar} ({count})")
            lines.append("")

        if themes:
            lines.append("## Emerging Themes")
            lines.append("")
            for theme in themes:
                lines.append(f"- {theme}")
            lines.append("")

        if artifacts:
            lines.append("## Papers")
            lines.append("")
            for i, art in enumerate(artifacts, 1):
                lines.append(f"### {i}. {art.title}")
                lines.append("")
                if art.authors:
                    lines.append(f"**Authors**: {', '.join(art.authors[:5])}")
                if art.importance_score > 0:
                    lines.append(f"**Importance**: {art.importance_score:.1f}")
                lines.append("")
                if art.summary:
                    lines.append(art.summary[:300])
                    lines.append("")
                if art.key_contributions:
                    lines.append("**Key Contributions:**")
                    for c in art.key_contributions:
                        lines.append(f"- {c}")
                    lines.append("")
                if art.potential_applications:
                    lines.append("**Potential Applications:**")
                    for a in art.potential_applications:
                        lines.append(f"- {a}")
                    lines.append("")
                if art.suggested_experiments:
                    lines.append("**Suggested Experiments:**")
                    for e in art.suggested_experiments:
                        lines.append(f"- {e}")
                    lines.append("")
                lines.append("---")
                lines.append("")

        return "\n".join(lines)
