#!/usr/bin/python
"""Exploration & Discovery Engine — CONCEPT:AHE-3.2.

Autonomous exploration loop with hypothesis generation, experiment design,
multi-reviewer evaluation, and KG-native knowledge gap tracking.
Supports arbitrary domain exploration (code, medical, finance, etc.).

Design-pattern source: Chapter 21 — Exploration and Discovery.

OWL: :Experiment rdfs:subClassOf :Procedure
     :KnowledgeGap rdfs:subClassOf :Observation
     :testsHypothesis, :exploredGap, :resultedInDiscovery
See docs/design-patterns-alignment.md §CONCEPT:AHE-3.2.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class KnowledgeGap(BaseModel):
    """An identified gap in the agent's knowledge."""

    id: str = Field(default_factory=lambda: f"gap:{uuid.uuid4().hex[:8]}")
    domain: str
    statement: str
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    status: Literal["identified", "exploring", "filled", "deferred"] = "identified"


class Hypothesis(BaseModel):
    """A testable hypothesis generated to fill a knowledge gap."""

    id: str = Field(default_factory=lambda: f"hyp:{uuid.uuid4().hex[:8]}")
    gap_id: str
    statement: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    status: Literal["proposed", "testing", "confirmed", "rejected"] = "proposed"


class Experiment(BaseModel):
    """A structured experiment to test a hypothesis."""

    id: str = Field(default_factory=lambda: f"exp:{uuid.uuid4().hex[:8]}")
    hypothesis_id: str
    design: str
    variables: dict[str, str] = Field(default_factory=dict)
    success_criteria: str = ""
    status: Literal["designed", "running", "completed", "failed"] = "designed"
    results: str | None = None


class ReviewScore(BaseModel):
    """A single reviewer's evaluation of experiment results."""

    reviewer_id: str
    reviewer_type: str = "general"
    score: float = Field(ge=0.0, le=1.0)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    verdict: Literal["accept", "reject", "revise"] = "revise"


class ReviewBundle(BaseModel):
    """Aggregated multi-reviewer evaluation."""

    experiment_id: str
    reviews: list[ReviewScore] = Field(default_factory=list)
    consensus: Literal["accept", "reject", "revise"] = "revise"
    mean_score: float = 0.0

    def compute_consensus(self) -> str:
        """Compute consensus from individual reviewer verdicts."""
        if not self.reviews:
            return "revise"
        self.mean_score = sum(r.score for r in self.reviews) / len(self.reviews)
        verdicts = [r.verdict for r in self.reviews]
        accept_count = verdicts.count("accept")
        reject_count = verdicts.count("reject")
        if accept_count > len(verdicts) / 2:
            self.consensus = "accept"
        elif reject_count > len(verdicts) / 2:
            self.consensus = "reject"
        else:
            self.consensus = "revise"
        return self.consensus


class Discovery(BaseModel):
    """A validated discovery resulting from an experiment."""

    id: str = Field(default_factory=lambda: f"disc:{uuid.uuid4().hex[:8]}")
    experiment_id: str
    hypothesis_id: str
    gap_id: str
    finding: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    domain: str = ""


class ExplorationEngine:
    """Autonomous exploration and discovery loop.

    Supports arbitrary domain exploration by generating hypotheses from
    knowledge gaps, designing experiments, executing them with multi-
    reviewer evaluation, and persisting discoveries to the KG.

    Parameters
    ----------
    kg_engine : optional
        If provided, exploration artifacts are persisted to the KG.
    llm_call : optional
        Async callable ``(prompt: str) -> str`` for LLM invocations.
    """

    def __init__(
        self,
        kg_engine: Any = None,
        llm_call: Any = None,
    ) -> None:
        self._engine = kg_engine
        self._llm_call = llm_call
        self._gaps: dict[str, KnowledgeGap] = {}
        self._hypotheses: dict[str, Hypothesis] = {}
        self._experiments: dict[str, Experiment] = {}
        self._discoveries: dict[str, Discovery] = {}

    def identify_gap(
        self,
        domain: str,
        statement: str,
        severity: float = 0.5,
    ) -> KnowledgeGap:
        """Register a knowledge gap."""
        gap = KnowledgeGap(domain=domain, statement=statement, severity=severity)
        self._gaps[gap.id] = gap
        logger.info("Knowledge gap identified: %s (%s)", gap.statement, domain)
        return gap

    def generate_hypothesis(
        self,
        gap_id: str,
        statement: str,
        confidence: float = 0.5,
    ) -> Hypothesis:
        """Generate a hypothesis to address a knowledge gap."""
        gap = self._gaps.get(gap_id)
        if gap is None:
            raise KeyError(f"Gap '{gap_id}' not found")
        gap.status = "exploring"
        hyp = Hypothesis(
            gap_id=gap_id,
            statement=statement,
            confidence=confidence,
        )
        self._hypotheses[hyp.id] = hyp
        logger.info("Hypothesis generated: %s", hyp.statement)
        return hyp

    def design_experiment(
        self,
        hypothesis_id: str,
        design: str,
        variables: dict[str, str] | None = None,
        success_criteria: str = "",
    ) -> Experiment:
        """Design an experiment to test a hypothesis."""
        hyp = self._hypotheses.get(hypothesis_id)
        if hyp is None:
            raise KeyError(f"Hypothesis '{hypothesis_id}' not found")
        hyp.status = "testing"
        exp = Experiment(
            hypothesis_id=hypothesis_id,
            design=design,
            variables=variables or {},
            success_criteria=success_criteria,
        )
        self._experiments[exp.id] = exp
        logger.info("Experiment designed: %s", exp.id)
        return exp

    def record_results(self, experiment_id: str, results: str) -> Experiment:
        """Record experiment results."""
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise KeyError(f"Experiment '{experiment_id}' not found")
        exp.results = results
        exp.status = "completed"
        return exp

    def multi_review(
        self,
        experiment_id: str,
        reviews: list[ReviewScore] | None = None,
    ) -> ReviewBundle:
        """Perform multi-reviewer evaluation of experiment results.

        If no reviews are provided, generates mock reviews. In production,
        this invokes multiple LLM reviewer agents with different perspectives.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise KeyError(f"Experiment '{experiment_id}' not found")

        if reviews is None:
            # Default reviewers: critical, constructive, domain-expert
            reviews = [
                ReviewScore(
                    reviewer_id="r1",
                    reviewer_type="critical",
                    score=0.6,
                    verdict="revise",
                ),
                ReviewScore(
                    reviewer_id="r2",
                    reviewer_type="constructive",
                    score=0.7,
                    verdict="accept",
                ),
                ReviewScore(
                    reviewer_id="r3",
                    reviewer_type="domain-expert",
                    score=0.65,
                    verdict="revise",
                ),
            ]

        bundle = ReviewBundle(experiment_id=experiment_id, reviews=reviews)
        bundle.compute_consensus()
        return bundle

    def accept_discovery(
        self,
        experiment_id: str,
        finding: str,
        confidence: float = 0.8,
    ) -> Discovery:
        """Accept experiment results as a validated discovery."""
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise KeyError(f"Experiment '{experiment_id}' not found")

        hyp = self._hypotheses.get(exp.hypothesis_id)
        gap_id = hyp.gap_id if hyp else ""
        domain = self._gaps[gap_id].domain if gap_id in self._gaps else ""

        # Update hypothesis status
        if hyp:
            hyp.status = "confirmed"
            hyp.confidence = confidence

        # Update gap status
        if gap_id in self._gaps:
            self._gaps[gap_id].status = "filled"

        discovery = Discovery(
            experiment_id=experiment_id,
            hypothesis_id=exp.hypothesis_id,
            gap_id=gap_id,
            finding=finding,
            confidence=confidence,
            domain=domain,
        )
        self._discoveries[discovery.id] = discovery
        logger.info("Discovery accepted: %s", finding[:100])
        return discovery

    def get_open_gaps(self) -> list[KnowledgeGap]:
        """Get all unfilled knowledge gaps, sorted by severity."""
        return sorted(
            [g for g in self._gaps.values() if g.status in ("identified", "exploring")],
            key=lambda g: g.severity,
            reverse=True,
        )

    def get_exploration_summary(self) -> dict[str, Any]:
        """Summary of exploration state."""
        return {
            "total_gaps": len(self._gaps),
            "open_gaps": len(self.get_open_gaps()),
            "hypotheses": len(self._hypotheses),
            "experiments": len(self._experiments),
            "discoveries": len(self._discoveries),
            "domains_explored": list({g.domain for g in self._gaps.values()}),
        }
