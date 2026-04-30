"""Automated Trace Distillation Pipeline.

CONCEPT:AU-012 — Agentic Harness Engineering (Experience Observability)

Transforms raw traces from a pluggable TraceBackend into a structured
EvidenceCorpus suitable for the Evolve Agent. This is the critical
pipeline that prevents the agent from reading raw Langfuse/OTel traces
(which wastes tokens, causes hallucination, and regresses to
trial-and-error).

Pipeline stages:
    1. Pull traces via pluggable TraceBackend (Langfuse or OTel)
    2. Classify pass/fail per task
    3. Cluster failures by root cause (using KG semantic search)
    4. Generate per-task analysis reports (using LLM summarizer)
    5. Aggregate into benchmark-level overview
    6. Store as versioned evidence artifacts
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..rlm.config import RLMConfig

from pydantic import BaseModel

from .evidence_corpus import (
    EvidenceCorpus,
    EvidenceEntry,
    EvidenceLayer,
    FailureCluster,
)
from .manifest import ComponentType
from .trace_backend import TraceBackend

logger = logging.getLogger(__name__)


class DistillationConfig(BaseModel):
    """Configuration for the trace distillation pipeline.

    Attributes:
        pass_threshold: Score threshold for pass/fail classification.
        min_cluster_size: Minimum failures to form a cluster.
        max_overview_tokens: Token budget for the overview layer.
        enable_llm_summarization: Whether to use LLM for summaries.
        evidence_output_dir: Directory for evidence artifact storage.
    """

    pass_threshold: float = 0.5
    min_cluster_size: int = 2
    max_overview_tokens: int = 2000
    enable_llm_summarization: bool = True
    evidence_output_dir: str = ".specify/evidence"


class TraceDistiller:
    """Agnostic trace distiller -> structured evidence corpus.

    Reads from a pluggable TraceBackend and produces a versioned
    EvidenceCorpus suitable for the Evolve Agent.

    Args:
        backend: The trace backend to ingest from.
        config: Distillation pipeline configuration.
        knowledge_engine: Optional KG engine for semantic clustering.
    """

    def __init__(
        self,
        backend: TraceBackend,
        config: DistillationConfig | None = None,
        knowledge_engine: Any = None,
    ) -> None:
        self.backend = backend
        self.config = config or DistillationConfig()
        self.knowledge_engine = knowledge_engine

    async def distill(self, round_id: str) -> EvidenceCorpus:
        """Run the full distillation pipeline for an evolution round.

        Args:
            round_id: The evolution round identifier.

        Returns:
            A complete EvidenceCorpus with all layers populated.
        """
        logger.info(f"TraceDistiller: Starting distillation for round {round_id}")

        # Stage 1: Pull traces
        traces = await self.backend.get_traces(round_id)
        logger.info(f"TraceDistiller: Retrieved {len(traces)} traces")

        if not traces:
            return EvidenceCorpus(
                round_id=round_id,
                overview="No traces found for this round.",
                total_tasks=0,
            )

        # Stage 2: Classify pass/fail
        entries = await self._classify_traces(traces)
        logger.info(
            f"TraceDistiller: Classified {len(entries)} entries "
            f"({sum(1 for e in entries if e.pass_fail)} pass, "
            f"{sum(1 for e in entries if not e.pass_fail)} fail)"
        )

        # Stage 3: Cluster failures
        failure_entries = [e for e in entries if not e.pass_fail]
        clusters = await self._cluster_failures(failure_entries)

        # Stage 4: Extract success patterns
        success_entries = [e for e in entries if e.pass_fail]
        success_patterns = self._extract_success_patterns(success_entries)

        # Stage 5: Build corpus
        corpus = EvidenceCorpus(
            round_id=round_id,
            entries=entries,
            failure_clusters=clusters,
            success_patterns=success_patterns,
            total_tasks=len(entries),
            pass_rate=sum(1 for e in entries if e.pass_fail) / max(len(entries), 1),
            benchmark_score=sum(e.score for e in entries) / max(len(entries), 1),
        )

        # Stage 6: Generate overview
        corpus.generate_overview_text()
        logger.info(
            f"TraceDistiller: Distillation complete. "
            f"Pass rate: {corpus.pass_rate:.1%}, Score: {corpus.benchmark_score:.2f}"
        )

        # Persist evidence artifacts
        await self._persist_evidence(corpus)

        return corpus

    async def _classify_traces(
        self, traces: list[dict[str, Any]]
    ) -> list[EvidenceEntry]:
        """Stage 2: Classify each trace as pass/fail and extract key info.

        Args:
            traces: Raw trace data from the backend.

        Returns:
            List of classified EvidenceEntry objects.
        """
        entries: list[EvidenceEntry] = []
        for trace in traces:
            score = self._extract_score(trace)
            passed = score >= self.config.pass_threshold
            error_msg = self._extract_error(trace)

            entry = EvidenceEntry(
                task_id=trace.get("name", trace.get("id", "unknown")),
                pass_fail=passed,
                root_cause=error_msg if not passed else None,
                score=score,
                trace_id=trace.get("id"),
                evidence_layer=EvidenceLayer.PER_TASK_REPORT,
                content=self._summarize_trace(trace),
                component_attribution=self._attribute_component(trace),
            )
            entries.append(entry)
        return entries

    async def _cluster_failures(
        self, failures: list[EvidenceEntry]
    ) -> list[FailureCluster]:
        """Stage 3: Group failures by root cause.

        Uses RLM for deep semantic clustering when trace count exceeds
        the configured ``ahe_trace_threshold`` (CONCEPT:AU-007 × AU-012).
        Falls back to keyword-based grouping otherwise.

        Args:
            failures: List of failing EvidenceEntry objects.

        Returns:
            Sorted list of FailureCluster objects (highest severity first).
        """
        if not failures:
            return []

        # Auto-trigger RLM for large trace sets
        from ..rlm.config import RLMConfig

        rlm_config = RLMConfig()
        if rlm_config.should_trigger(trace_count=len(failures)):
            logger.info(
                f"TraceDistiller: {len(failures)} failures exceeds RLM threshold "
                f"({rlm_config.ahe_trace_threshold}). Using RLM for deep clustering."
            )
            try:
                return await self._cluster_failures_rlm(failures, rlm_config)
            except Exception as e:
                logger.warning(
                    f"TraceDistiller: RLM clustering failed ({e}), "
                    f"falling back to keyword clustering."
                )

        return self._cluster_failures_keyword(failures)

    async def _cluster_failures_rlm(
        self,
        failures: list[EvidenceEntry],
        rlm_config: RLMConfig,
    ) -> list[FailureCluster]:
        """RLM-powered failure clustering for large trace sets.

        CONCEPT:AU-007 × AU-012 — RLM for AHE Experience Observability

        Delegates to an RLM sub-agent that programmatically loops over
        all failure entries, applies semantic grouping using the KG,
        and produces structured FailureCluster objects.

        Args:
            failures: List of failing EvidenceEntry objects.
            rlm_config: RLM configuration.

        Returns:
            Sorted list of FailureCluster objects.
        """
        from ..rlm.repl import RLMEnvironment

        # Serialize failures to JSON for the REPL context
        failures_data = [
            {
                "task_id": e.task_id,
                "root_cause": e.root_cause,
                "score": e.score,
                "component": e.component_attribution.value
                if e.component_attribution
                else None,
                "content": e.content[:200],
            }
            for e in failures
        ]

        env = RLMEnvironment(
            context=json.dumps(failures_data),
            config=rlm_config,
        )

        rlm_result = await env.run_full_rlm(
            "Analyze the failure entries in `context` (JSON array). "
            "Group them by semantic root cause similarity. "
            "For each cluster, produce a JSON object with: "
            "'label' (str), 'root_cause_summary' (str), "
            "'task_ids' (list[str]), 'component' (str or null), "
            "'frequency' (int), 'severity' (float 0-1). "
            "Output the clusters as a JSON array via FINAL_VAR('clusters', json_string)."
        )

        # Parse RLM output into FailureCluster objects
        try:
            cluster_data = json.loads(rlm_result)
            clusters = []
            for cd in cluster_data:
                comp = None
                if cd.get("component"):
                    try:
                        comp = ComponentType(cd["component"])
                    except (ValueError, KeyError):
                        pass
                clusters.append(
                    FailureCluster(
                        label=cd.get("label", "unknown"),
                        root_cause_summary=cd.get("root_cause_summary", ""),
                        task_ids=cd.get("task_ids", []),
                        component_attribution=comp,
                        frequency=cd.get("frequency", len(cd.get("task_ids", []))),
                        severity=cd.get("severity", 0.5),
                    )
                )
            return sorted(clusters, key=lambda c: c.severity, reverse=True)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"RLM cluster output not valid JSON: {e}")
            return self._cluster_failures_keyword(failures)

    def _cluster_failures_keyword(
        self, failures: list[EvidenceEntry]
    ) -> list[FailureCluster]:
        """Keyword-based failure clustering (original algorithm).

        Simple grouping by normalized root cause string. Used as
        fallback when RLM is unavailable or trace count is below
        the auto-trigger threshold.
        """
        clusters_map: dict[str, list[EvidenceEntry]] = {}
        for entry in failures:
            key = self._normalize_root_cause(entry.root_cause or "unknown")
            if key not in clusters_map:
                clusters_map[key] = []
            clusters_map[key].append(entry)

        clusters: list[FailureCluster] = []
        for label, group in clusters_map.items():
            if len(group) >= self.config.min_cluster_size:
                # Determine most common component attribution
                comp_counts: dict[ComponentType | None, int] = {}
                for e in group:
                    comp_counts[e.component_attribution] = (
                        comp_counts.get(e.component_attribution, 0) + 1
                    )
                top_comp = max(comp_counts, key=lambda k: comp_counts[k])

                cluster = FailureCluster(
                    label=label,
                    root_cause_summary=group[0].root_cause or label,
                    task_ids=[e.task_id for e in group],
                    component_attribution=top_comp,
                    frequency=len(group),
                    severity=1.0 - (sum(e.score for e in group) / len(group)),
                )
                clusters.append(cluster)

        return sorted(clusters, key=lambda c: c.severity, reverse=True)

    def _extract_success_patterns(self, successes: list[EvidenceEntry]) -> list[str]:
        """Stage 4: Extract reusable patterns from successful executions."""
        patterns: list[str] = []
        if successes:
            # Extract common success patterns
            comp_successes: dict[str, int] = {}
            for entry in successes:
                if entry.component_attribution:
                    comp_name = entry.component_attribution.value
                    comp_successes[comp_name] = comp_successes.get(comp_name, 0) + 1

            for comp, count in sorted(
                comp_successes.items(), key=lambda x: x[1], reverse=True
            ):
                patterns.append(
                    f"{comp} component contributed to {count} successful tasks"
                )
        return patterns

    def _extract_score(self, trace: dict[str, Any]) -> float:
        """Extract a normalized score from a trace."""
        # Try common score locations
        if "scores" in trace and trace["scores"]:
            scores = trace["scores"]
            if isinstance(scores, list):
                return float(scores[0].get("value", 0.0))
            if isinstance(scores, dict):
                return float(next(iter(scores.values()), 0.0))
        if "score" in trace:
            return float(trace["score"])
        if "status" in trace:
            return 1.0 if trace["status"] == "success" else 0.0
        return 0.0

    def _extract_error(self, trace: dict[str, Any]) -> str | None:
        """Extract error message from a trace."""
        for key in ("error", "statusMessage", "error_message", "exception"):
            if key in trace and trace[key]:
                return str(trace[key])[:500]
        return None

    def _summarize_trace(self, trace: dict[str, Any]) -> str:
        """Create a lightweight summary of a trace (Layer 2)."""
        parts = []
        if trace.get("name"):
            parts.append(f"Task: {trace['name']}")
        if trace.get("latency"):
            parts.append(f"Duration: {trace['latency']}ms")
        if trace.get("status"):
            parts.append(f"Status: {trace['status']}")
        if trace.get("input"):
            inp = str(trace["input"])[:200]
            parts.append(f"Input: {inp}")
        return " | ".join(parts)

    def _attribute_component(self, trace: dict[str, Any]) -> ComponentType | None:
        """Attempt to attribute a trace failure to a specific component type.

        Uses heuristics on the trace structure to determine which
        harness component is most likely responsible.
        """
        name = str(trace.get("name", "")).lower()
        error = str(trace.get("error", "")).lower()
        combined = f"{name} {error}"

        if any(k in combined for k in ("tool", "mcp", "function_call")):
            return ComponentType.TOOL_IMPLEMENTATION
        if any(k in combined for k in ("prompt", "system_prompt", "instruction")):
            return ComponentType.SYSTEM_PROMPT
        if any(k in combined for k in ("guard", "policy", "blocked", "denied")):
            return ComponentType.MIDDLEWARE
        if any(k in combined for k in ("skill", "capability")):
            return ComponentType.SKILL
        if any(k in combined for k in ("memory", "recall", "context")):
            return ComponentType.LONG_TERM_MEMORY

        return None

    def _normalize_root_cause(self, root_cause: str) -> str:
        """Normalize a root cause string for clustering."""
        # Simplified normalization — strip specific IDs and paths
        import re

        normalized = root_cause.lower().strip()
        normalized = re.sub(r"[0-9a-f]{8,}", "<id>", normalized)
        normalized = re.sub(r"/[\w/]+\.\w+", "<path>", normalized)
        # Truncate for clustering key
        return normalized[:100]

    async def _persist_evidence(self, corpus: EvidenceCorpus) -> None:
        """Persist the evidence corpus to disk as versioned artifacts."""
        output_dir = self.config.evidence_output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save full corpus
        corpus_path = os.path.join(output_dir, f"{corpus.round_id}.json")
        with open(corpus_path, "w") as f:
            f.write(corpus.model_dump_json(indent=2))

        # Save overview as readable markdown
        overview_path = os.path.join(output_dir, f"{corpus.round_id}_overview.md")
        with open(overview_path, "w") as f:
            f.write(corpus.overview)

        logger.info(f"TraceDistiller: Evidence persisted to {output_dir}")
