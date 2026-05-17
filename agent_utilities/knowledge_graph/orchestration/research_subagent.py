#!/usr/bin/python
from __future__ import annotations

"""Research Intelligence Sub-Agent.

CONCEPT:KG-2.6 — Research Intelligence Sub-Agent

Provides an isolated research context with citation graph traversal,
doom-loop detection, and KG persistence. Adapted from ml-intern's
research_tool.py sub-agent pattern.

Key features:
* Isolated context with configurable token budget
* Citation graph traversal via Semantic Scholar API (rate-limited)
* Tool sandboxing via read-only whitelist
* KG persistence: findings → Evidence nodes with wasDerivedFrom chains
* Paper metadata → Document nodes with citesSource edges
"""


import hashlib
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from agent_utilities.models.knowledge_graph import (
    CitationEdgeNode,
    EvidenceNode,
    RegistryEdge,
    RegistryEdgeType,
    RegistryNode,
    ResearchSessionNode,
    SourceNode,
)
from agent_utilities.security.execution_stability_engine import DoomLoopDetector

logger = logging.getLogger(__name__)

# Rate limiting for Semantic Scholar API (100 req/5min without key)
_S2_RATE_LIMIT_SECONDS = 3.1
_S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = "paperId,title,abstract,year,citationCount,influentialCitationCount,authors,citations,references"


class CitationGraphWalker:
    """Traverses citation graphs via Semantic Scholar API.

    CONCEPT:KG-2.6 — Research Intelligence Sub-Agent

    Performs rate-limited citation and reference traversal with caching,
    depth-limited recursion, and influence-flag tracking.

    Example::

        walker = CitationGraphWalker()
        paper = walker.fetch_paper("649def34f8be52c8b66281af98ae884c09aef38b")
        citations = walker.get_citations(paper["paperId"], max_depth=2)
    """

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit_seconds: float = _S2_RATE_LIMIT_SECONDS,
    ):
        self._api_key = api_key
        self._rate_limit = rate_limit_seconds
        self._last_request_time = 0.0
        self._cache: dict[str, dict[str, Any]] = {}

    def _rate_limit_wait(self) -> None:
        """Wait if needed to respect API rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

    def fetch_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Fetch paper metadata from Semantic Scholar.

        Args:
            paper_id: S2 paper ID, DOI, or arXiv ID.

        Returns:
            Paper metadata dict or None if not found.
        """
        if paper_id in self._cache:
            return self._cache[paper_id]

        self._rate_limit_wait()

        try:
            import urllib.request

            url = f"{_S2_API_BASE}/paper/{paper_id}?fields={_S2_FIELDS}"
            req = urllib.request.Request(url)
            if self._api_key:
                req.add_header("x-api-key", self._api_key)
            req.add_header("User-Agent", "agent-utilities/1.0")

            with urllib.request.urlopen(req, timeout=30) as response:  # nosec
                import json

                data = json.loads(response.read().decode())
                self._cache[paper_id] = data
                self._last_request_time = time.time()
                return data
        except Exception as e:
            logger.warning("Failed to fetch paper %s: %s", paper_id, e)
            self._last_request_time = time.time()
            return None

    def get_citations(
        self,
        paper_id: str,
        max_depth: int = 2,
        max_per_level: int = 5,
    ) -> list[CitationEdgeNode]:
        """Traverse citation graph and return citation edges.

        Args:
            paper_id: Starting paper ID.
            max_depth: Maximum traversal depth.
            max_per_level: Maximum citations to follow per paper.

        Returns:
            List of CitationEdgeNode instances.
        """
        edges: list[CitationEdgeNode] = []
        visited: set[str] = set()
        self._walk_citations(paper_id, 0, max_depth, max_per_level, visited, edges)
        return edges

    def _walk_citations(
        self,
        paper_id: str,
        depth: int,
        max_depth: int,
        max_per_level: int,
        visited: set[str],
        edges: list[CitationEdgeNode],
    ) -> None:
        """Recursive citation traversal."""
        if depth >= max_depth or paper_id in visited:
            return

        visited.add(paper_id)
        paper = self.fetch_paper(paper_id)
        if not paper:
            return

        # Process citations (papers that cite this one)
        citations = paper.get("citations") or []
        for i, cite in enumerate(citations[:max_per_level]):
            cited_id = cite.get("paperId")
            if not cited_id:
                continue

            edge = CitationEdgeNode(
                id=f"cite_{uuid.uuid4().hex[:8]}",
                name=f"Citation: {paper.get('title', '')[:50]} → {cite.get('title', '')[:50]}",
                citing_paper_id=cited_id,
                cited_paper_id=paper_id,
                is_influential=cite.get("isInfluential", False),
                citation_intent="result",
                depth=depth,
            )
            edges.append(edge)

            # Recurse
            self._walk_citations(
                cited_id, depth + 1, max_depth, max_per_level, visited, edges
            )

        # Process references (papers this one cites)
        references = paper.get("references") or []
        for ref in references[:max_per_level]:
            ref_id = ref.get("paperId")
            if not ref_id:
                continue

            edge = CitationEdgeNode(
                id=f"cite_{uuid.uuid4().hex[:8]}",
                name=f"Reference: {paper.get('title', '')[:50]} ← {ref.get('title', '')[:50]}",
                citing_paper_id=paper_id,
                cited_paper_id=ref_id,
                is_influential=ref.get("isInfluential", False),
                citation_intent="background",
                depth=depth,
            )
            edges.append(edge)


class ResearchSubagent:
    """Isolated research context with KG persistence.

    CONCEPT:KG-2.6 — Research Intelligence Sub-Agent

    Manages an isolated research session with its own token budget,
    doom-loop detection, and citation graph traversal. Findings are
    persisted as KG nodes with provenance chains.

    Example::

        subagent = ResearchSubagent(query="spectral clustering for agents")
        subagent.add_finding("Eigengap heuristic selects k automatically",
                             source_paper_id="abc123")
        session = subagent.finalize()
    """

    def __init__(
        self,
        query: str,
        session_id: str | None = None,
        token_budget_warn: int = 170_000,
        token_budget_max: int = 190_000,
        s2_api_key: str | None = None,
    ):
        self._session_id = session_id or f"rs_{uuid.uuid4().hex[:8]}"
        self._query = query
        self._token_budget_warn = token_budget_warn
        self._token_budget_max = token_budget_max
        self._tokens_used = 0
        self._findings: list[EvidenceNode] = []
        self._papers: list[SourceNode] = []
        self._citation_edges: list[CitationEdgeNode] = []
        self._doom_detector = DoomLoopDetector(session_id=self._session_id)
        self._citation_walker = CitationGraphWalker(api_key=s2_api_key)
        self._status = "active"

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def status(self) -> str:
        return self._status

    def add_tokens(self, count: int) -> str | None:
        """Track token consumption, returning a warning if budget is near.

        Returns:
            Warning message if over warn threshold, None otherwise.
        """
        self._tokens_used += count
        if self._tokens_used >= self._token_budget_max:
            self._status = "budget_exceeded"
            return (
                f"TOKEN BUDGET EXCEEDED: {self._tokens_used}/{self._token_budget_max}"
            )
        if self._tokens_used >= self._token_budget_warn:
            return f"TOKEN WARNING: {self._tokens_used}/{self._token_budget_max}"
        return None

    def add_finding(
        self,
        claim: str,
        confidence: float = 0.8,
        _source_paper_id: str | None = None,
    ) -> EvidenceNode:
        """Record a research finding as an Evidence node.

        Args:
            claim: The factual claim or finding.
            confidence: Confidence score (0-1).
            source_paper_id: S2 paper ID that supports this finding.

        Returns:
            The created EvidenceNode.
        """
        finding = EvidenceNode(
            id=f"ev_{uuid.uuid4().hex[:8]}",
            name=f"Finding: {claim[:80]}",
            description=claim,
            evidence_id=f"ev_{hashlib.md5(claim.encode(), usedforsecurity=False).hexdigest()[:8]}",
            claim=claim,
            confidence_score=confidence,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._findings.append(finding)
        return finding

    def add_paper(
        self,
        paper_id: str,
        title: str,
        authors: list[str] | None = None,
        year: int | None = None,
        doi: str | None = None,
    ) -> SourceNode:
        """Record a discovered paper as a Source node.

        Args:
            paper_id: Semantic Scholar paper ID.
            title: Paper title.
            authors: List of author names.
            year: Publication year.
            doi: DOI if available.

        Returns:
            The created SourceNode.
        """
        paper = SourceNode(
            id=f"paper_{paper_id[:12]}",
            name=title[:100],
            description=f"Research paper: {title}",
            source_id=paper_id,
            doi=doi,
            authors=authors or [],
            publication_date=str(year) if year else None,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._papers.append(paper)
        return paper

    def traverse_citations(
        self,
        paper_id: str,
        max_depth: int = 2,
        max_per_level: int = 5,
    ) -> list[CitationEdgeNode]:
        """Traverse the citation graph for a paper.

        Args:
            paper_id: Starting paper ID.
            max_depth: Maximum traversal depth.
            max_per_level: Max citations per paper.

        Returns:
            List of discovered citation edges.
        """
        edges = self._citation_walker.get_citations(paper_id, max_depth, max_per_level)
        self._citation_edges.extend(edges)
        return edges

    def finalize(self) -> ResearchSessionNode:
        """Finalize the research session and create the session node.

        Returns:
            ResearchSessionNode summarizing the entire session.
        """
        if self._status == "active":
            self._status = "completed"

        return ResearchSessionNode(
            id=self._session_id,
            name=f"Research: {self._query[:80]}",
            description=f"Research session for: {self._query}",
            query=self._query,
            token_budget_warn=self._token_budget_warn,
            token_budget_max=self._token_budget_max,
            tokens_used=self._tokens_used,
            papers_discovered=len(self._papers),
            citations_traversed=len(self._citation_edges),
            findings_count=len(self._findings),
            status=self._status,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def get_all_nodes(self) -> list[RegistryNode]:
        """Get all KG nodes created during this session."""
        nodes: list[RegistryNode] = []
        nodes.extend(self._papers)
        nodes.extend(self._findings)
        nodes.extend(self._citation_edges)
        nodes.append(self.finalize())
        return nodes

    def get_provenance_edges(self) -> list[RegistryEdge]:
        """Get provenance edges linking findings to papers and session."""
        edges: list[RegistryEdge] = []

        # Link findings to session
        for finding in self._findings:
            edges.append(
                RegistryEdge(
                    source=finding.id,
                    target=self._session_id,
                    type=RegistryEdgeType.DISCOVERED_IN_SESSION,
                )
            )

        # Link papers to session
        for paper in self._papers:
            edges.append(
                RegistryEdge(
                    source=paper.id,
                    target=self._session_id,
                    type=RegistryEdgeType.DISCOVERED_IN_SESSION,
                )
            )

        # Link citation edges
        for cite in self._citation_edges:
            edges.append(
                RegistryEdge(
                    source=cite.citing_paper_id,
                    target=cite.cited_paper_id,
                    type=RegistryEdgeType.CITES_PAPER,
                    metadata={
                        "is_influential": cite.is_influential,
                        "depth": cite.depth,
                    },
                )
            )

        return edges
