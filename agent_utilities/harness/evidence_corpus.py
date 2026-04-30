"""Layered Evidence Corpus Models.

CONCEPT:AU-012 — Agentic Harness Engineering (Experience Observability)

Provides the structured evidence corpus with progressive disclosure.
Instead of feeding raw Langfuse/OTel traces to the Evolve Agent
(which wastes tokens and causes hallucination), traces are distilled
into layered evidence with increasing detail:

    Layer 1 — Overview:      Benchmark-level summary (~1 page)
    Layer 2 — Per-task:      Root-cause analysis per failing task
    Layer 3 — Processed:     Lightly cleaned trajectory excerpt
    Layer 4 — Raw:           Full original trace (rarely needed)

The Evolve Agent reads Layer 1 first, then drills into Layer 2 only
for failure categories it decides to address, preserving token budget.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .manifest import ComponentType


class EvidenceLayer(StrEnum):
    """Progressive disclosure layers for trace evidence.

    Lower layers are cheaper (fewer tokens) and should be read first.
    The Evolve Agent drills down only when deeper analysis is needed.
    """

    OVERVIEW = "overview"  # Benchmark-level summary (~1 page)
    PER_TASK_REPORT = "per_task"  # Per-task root-cause analysis
    PROCESSED_TRACE = "processed"  # Lightly cleaned trajectory
    RAW_TRACE = "raw"  # Full original trace


class EvidenceEntry(BaseModel):
    """A single evidence entry for one task execution.

    Contains the distilled analysis of a single task's trajectory,
    including pass/fail classification, root-cause analysis, and
    component attribution for the Evolve Agent.

    Attributes:
        task_id: Identifier for the evaluated task.
        pass_fail: Whether the task execution succeeded.
        root_cause: Root-cause analysis for failures (None if passed).
        success_pattern: Pattern description for successes (None if failed).
        component_attribution: Which harness component is responsible.
        evidence_layer: The disclosure layer of this entry.
        content: The distilled evidence content.
        trace_id: Optional link to the original trace.
        score: Numeric score (0.0-1.0) for the task execution.
        tags: Categorization tags for clustering.
    """

    id: str = Field(default_factory=lambda: f"ev:{uuid.uuid4().hex[:8]}")
    task_id: str
    pass_fail: bool
    root_cause: str | None = None
    success_pattern: str | None = None
    component_attribution: ComponentType | None = None
    evidence_layer: EvidenceLayer = EvidenceLayer.PER_TASK_REPORT
    content: str = ""
    trace_id: str | None = None
    score: float = 0.0
    tags: list[str] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureCluster(BaseModel):
    """A cluster of related failures identified by semantic similarity.

    Failures are grouped by root cause so the Evolve Agent can address
    systemic issues rather than individual symptoms.

    Attributes:
        cluster_id: Unique identifier for this cluster.
        label: Human-readable label for the failure category.
        root_cause_summary: Aggregated root cause description.
        task_ids: Task IDs belonging to this cluster.
        component_attribution: Most likely responsible component.
        frequency: Number of tasks in this cluster.
        severity: Estimated severity (0.0-1.0).
    """

    cluster_id: str = Field(default_factory=lambda: f"clst:{uuid.uuid4().hex[:8]}")
    label: str
    root_cause_summary: str
    task_ids: list[str] = Field(default_factory=list)
    component_attribution: ComponentType | None = None
    frequency: int = 0
    severity: float = 0.0


class EvidenceCorpus(BaseModel):
    """A versioned corpus of evidence for one evolution round.

    This is the primary input to the Evolve Agent. It replaces
    direct trace reading with structured, token-efficient evidence.

    Structure:
        - ``overview``: ~1 page benchmark-level summary (always read)
        - ``entries``: Per-task evidence entries (read selectively)
        - ``failure_clusters``: Grouped failure categories
        - ``success_patterns``: Patterns from successful tasks

    Attributes:
        round_id: Links to the ChangeManifest round.
        overview: Benchmark-level overview document.
        entries: Individual task evidence entries.
        failure_clusters: Semantically clustered failure groups.
        success_patterns: Extracted success patterns for replication.
        total_tasks: Total number of tasks evaluated.
        pass_rate: Overall pass rate (0.0-1.0).
        benchmark_score: Aggregate benchmark score.
    """

    round_id: str = Field(default_factory=lambda: f"corpus:{uuid.uuid4().hex[:8]}")
    overview: str = ""
    entries: list[EvidenceEntry] = Field(default_factory=list)
    failure_clusters: list[FailureCluster] = Field(default_factory=list)
    success_patterns: list[str] = Field(default_factory=list)
    total_tasks: int = 0
    pass_rate: float = 0.0
    benchmark_score: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_failures(self) -> list[EvidenceEntry]:
        """Return only failing task entries."""
        return [e for e in self.entries if not e.pass_fail]

    def get_successes(self) -> list[EvidenceEntry]:
        """Return only passing task entries."""
        return [e for e in self.entries if e.pass_fail]

    def get_entries_by_component(
        self, component_type: ComponentType
    ) -> list[EvidenceEntry]:
        """Filter entries attributed to a specific component type."""
        return [e for e in self.entries if e.component_attribution == component_type]

    def get_top_failure_clusters(self, n: int = 5) -> list[FailureCluster]:
        """Return the top-N failure clusters by severity."""
        return sorted(self.failure_clusters, key=lambda c: c.severity, reverse=True)[:n]

    def generate_overview_text(self) -> str:
        """Generate a token-efficient overview for the Evolve Agent.

        This is the Layer 1 evidence that the Evolve Agent reads first.
        """
        lines = [
            f"# Evidence Corpus — Round {self.round_id}",
            "",
            "## Summary",
            f"- Total tasks: {self.total_tasks}",
            f"- Pass rate: {self.pass_rate:.1%}",
            f"- Benchmark score: {self.benchmark_score:.2f}",
            f"- Failure clusters: {len(self.failure_clusters)}",
            "",
        ]

        if self.failure_clusters:
            lines.append("## Top Failure Categories")
            for cluster in self.get_top_failure_clusters():
                lines.append(
                    f"- **{cluster.label}** ({cluster.frequency} tasks, "
                    f"severity={cluster.severity:.2f}): {cluster.root_cause_summary}"
                )
                if cluster.component_attribution:
                    lines.append(
                        f"  - Component: `{cluster.component_attribution.value}`"
                    )
            lines.append("")

        if self.success_patterns:
            lines.append("## Success Patterns")
            for pattern in self.success_patterns[:5]:
                lines.append(f"- {pattern}")

        self.overview = "\n".join(lines)
        return self.overview
