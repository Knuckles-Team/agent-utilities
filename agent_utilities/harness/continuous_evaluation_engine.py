from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..models.imodel import InterpretabilityTestCategory, InterpretabilityTestNode
from ..models.knowledge_graph import RegistryEdgeType, RegistryNodeType
from .evidence_corpus import (
    EvidenceCorpus,
    EvidenceEntry,
    EvidenceLayer,
    FailureCluster,
)
from .manifest import ComponentType
from .trace_backend import TraceBackend

"""Automated Trace Distillation Pipeline.

CONCEPT:AHE-3.0 — Agentic Harness Engineering (Experience Observability)

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


if TYPE_CHECKING:
    from ..rlm.config import RLMConfig


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
        # CONCEPT:AHE-3.36 — disconfirming-evidence research log, fed each round
        # from the distilled corpus so failures become recorded belief updates.
        from .forecasting import ForecastBoard
        from .research_log import ResearchLog

        self.research_log = ResearchLog()
        # CONCEPT:AHE-3.34 — predict-before-resolve calibration over rounds; the
        # distiller forecasts each round's score (naively, ≈ last round) then resolves
        # it, so the harness measures its own taste over time.
        self.forecasts = ForecastBoard()
        self._last_round_score: float | None = None
        # CONCEPT:ORCH-1.55 — the compounding eval set: each round's failures are
        # harvested as new eval cases, so the org's eval suite (its real IP) grows
        # with every production failure (GEPA enterprise learning loop).
        from agent_utilities.rlm.eval_set_optimizer import EvalSet

        self.eval_set = EvalSet()

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

        # Stage 6b — research-craft triage (CONCEPT:AHE-3.36): cluster the failures
        # into piles and surface the BIGGEST one to attack first (Ng's "pull the
        # failures, sort into piles, attack the biggest"), and log the failure
        # root-causes as disconfirming evidence (Darwin's rule).
        self._triage_failures(corpus)

        # Persist evidence artifacts
        await self._persist_evidence(corpus)

        return corpus

    def _triage_failures(self, corpus: EvidenceCorpus) -> None:
        """Apply research-craft discipline to a distilled round (AHE-3.34/35/36).

        - AHE-3.34: forecast this round's score (≈ last round) then resolve it, so
          the harness builds a calibration record of its own predictions.
        - AHE-3.35: baseline-gate the round score against the previous round — a
          regression (failing to beat the prior baseline) is logged loudly.
        - AHE-3.36: cluster failures into piles, surface the biggest to attack
          first, and record the outcome as (dis)confirming evidence.
        """
        from .baseline_overfit_gate import baseline_gate
        from .research_log import FailureTriage

        # AHE-3.34 — predict-before-resolve calibration.
        predicted = (
            self._last_round_score
            if self._last_round_score is not None
            else corpus.benchmark_score
        )
        self.forecasts.predict(
            corpus.round_id,
            "round score holds vs the previous round",
            predicted=predicted,
            confidence=0.5,
        )
        self.forecasts.resolve(corpus.round_id, corpus.benchmark_score)

        # AHE-3.35 — round-over-round baseline gate (regression detection).
        if self._last_round_score is not None:
            verdict = baseline_gate(
                corpus.benchmark_score, self._last_round_score, min_lift=0.0
            )
            if not verdict.passed:
                logger.warning(
                    "TraceDistiller: round %s REGRESSED vs baseline (%s)",
                    corpus.round_id,
                    verdict.reason,
                )
        self._last_round_score = corpus.benchmark_score

        # AHE-3.36 — failure triage + disconfirming-evidence log.
        triage = FailureTriage()
        added = triage.from_evidence_corpus(corpus)
        self.research_log.record(
            f"round {corpus.round_id} approach is sound",
            f"pass_rate={corpus.pass_rate:.2f}, {added} failure cases",
            supports=corpus.pass_rate >= 0.5,
        )
        biggest = triage.biggest_pile()
        if biggest is not None:
            pile, cases = biggest
            logger.info(
                "TraceDistiller: largest failure pile to attack first: "
                "%r (%d cases)",
                pile,
                len(cases),
            )

        # ORCH-1.55 — harvest each failure as a new eval case so the eval set (the
        # compounding IP) grows every round; the next harness optimization is scored
        # against this ever-growing suite.
        from agent_utilities.rlm.eval_set_optimizer import EvalCase

        for entry in corpus.entries:
            if not entry.pass_fail:
                self.eval_set.add(
                    EvalCase(
                        case_id=f"{corpus.round_id}:{entry.task_id}",
                        input=entry.content or entry.task_id,
                        expected=entry.root_cause or "(must pass)",
                        source="production_failure",
                    )
                )

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
        the configured ``ahe_trace_threshold`` (CONCEPT:ORCH-1.1 × CONCEPT:ORCH-1.1).
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

        CONCEPT:ORCH-1.1 × CONCEPT:ORCH-1.1 — RLM for AHE Experience Observability

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
        if any(
            k in combined
            for k in (
                "plan",
                "planning",
                "router",
                "route",
                "orchestrator",
                "orchestration",
            )
        ):
            return ComponentType.ORCHESTRATOR_SKILL
        if any(
            k in combined for k in ("execution", "worker", "executor", "parallel_batch")
        ):
            return ComponentType.WORKER_SKILL
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


logger = logging.getLogger(__name__)


def _get_default_db_path() -> str:
    try:
        from agent_utilities.core import paths

        return str(paths.data_dir() / "backtest_log.db")
    except ImportError:
        return ".agent_workspace/backtest_log.db"


_DEFAULT_DB_PATH = _get_default_db_path()


class BacktestMetric(BaseModel):
    """A single metric from a backtest run. CONCEPT:AHE-3.4"""

    metric_name: str
    value: float
    window_index: int = 0
    benchmark_value: float | None = None
    is_passing: bool = True


class BacktestRunRecord(BaseModel):
    """Complete record of a backtest run. CONCEPT:AHE-3.4"""

    run_id: str
    strategy_id: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 100_000.0
    final_capital: float = 0.0
    total_trades: int = 0
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: str = "running"
    metrics: list[BacktestMetric] = Field(default_factory=list)
    walk_forward_windows: int = 0
    benchmark_id: str | None = None
    created_at: str = ""
    completed_at: str | None = None


class BacktestComparison(BaseModel):
    """Comparison result between a run and a benchmark. CONCEPT:AHE-3.4"""

    run_id: str
    benchmark_id: str
    metric_name: str
    run_value: float
    benchmark_value: float
    delta: float
    outperforms: bool


class BacktestHarness:
    """Domain-agnostic backtesting and evaluation harness.

    CONCEPT:AHE-3.4 — Backtest Evaluation Harness

    Records evaluation runs to a separate SQLite database to prevent
    KG contamination (same pattern as KGEvalCapture).

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._runs: dict[str, BacktestRunRecord] = {}
        self._db: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database schema."""
        try:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self._db_path)
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS backtest_runs (
                    run_id TEXT PRIMARY KEY,
                    strategy_id TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    initial_capital REAL,
                    final_capital REAL,
                    total_trades INTEGER,
                    parameters TEXT,
                    status TEXT,
                    walk_forward_windows INTEGER,
                    benchmark_id TEXT,
                    created_at TEXT,
                    completed_at TEXT
                )"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS backtest_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    metric_name TEXT,
                    value REAL,
                    window_index INTEGER DEFAULT 0,
                    benchmark_value REAL,
                    is_passing INTEGER DEFAULT 1,
                    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
                )"""
            )
            self._db.commit()
        except Exception as e:
            logger.warning("BacktestHarness DB init failed: %s", e)
            self._db = None

    def create_run(
        self,
        strategy_id: str = "",
        start_date: str = "",
        end_date: str = "",
        initial_capital: float = 100_000.0,
        parameters: dict[str, Any] | None = None,
        walk_forward_windows: int = 0,
        benchmark_id: str | None = None,
    ) -> str:
        """Create a new backtest run.

        CONCEPT:AHE-3.4

        Args:
            strategy_id: Reference to the strategy being evaluated.
            start_date: Evaluation period start (ISO date).
            end_date: Evaluation period end (ISO date).
            initial_capital: Starting capital.
            parameters: Strategy parameters for this run.
            walk_forward_windows: Number of walk-forward splits.
            benchmark_id: Optional benchmark for comparison.

        Returns:
            The generated run ID.
        """
        run_id = f"bt:{uuid.uuid4().hex[:8]}"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        record = BacktestRunRecord(
            run_id=run_id,
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            parameters=parameters or {},
            walk_forward_windows=walk_forward_windows,
            benchmark_id=benchmark_id,
            created_at=now,
        )
        self._runs[run_id] = record

        if self._db:
            try:
                self._db.execute(
                    """INSERT INTO backtest_runs
                    (run_id, strategy_id, start_date, end_date,
                     initial_capital, final_capital, total_trades,
                     parameters, status, walk_forward_windows,
                     benchmark_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        strategy_id,
                        start_date,
                        end_date,
                        initial_capital,
                        0.0,
                        0,
                        json.dumps(parameters or {}),
                        "running",
                        walk_forward_windows,
                        benchmark_id,
                        now,
                    ),
                )
                self._db.commit()
            except Exception as e:
                logger.warning("Failed to persist backtest run: %s", e)

        logger.info("Created backtest run %s for strategy %s", run_id, strategy_id)
        return run_id

    def record_metric(
        self,
        run_id: str,
        metric_name: str,
        value: float,
        window_index: int = 0,
        benchmark_value: float | None = None,
        is_passing: bool = True,
    ) -> None:
        """Record a metric for a backtest run.

        CONCEPT:AHE-3.4

        Args:
            run_id: The run to record the metric for.
            metric_name: Metric name (e.g., 'sharpe_ratio').
            value: Metric value.
            window_index: Walk-forward window index (0 = aggregate).
            benchmark_value: Benchmark comparison value.
            is_passing: Whether the metric passes its threshold.
        """
        metric = BacktestMetric(
            metric_name=metric_name,
            value=value,
            window_index=window_index,
            benchmark_value=benchmark_value,
            is_passing=is_passing,
        )

        if run_id in self._runs:
            self._runs[run_id].metrics.append(metric)

        if self._db:
            try:
                self._db.execute(
                    """INSERT INTO backtest_metrics
                    (run_id, metric_name, value, window_index,
                     benchmark_value, is_passing)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        metric_name,
                        value,
                        window_index,
                        benchmark_value,
                        int(is_passing),
                    ),
                )
                self._db.commit()
            except Exception as e:
                logger.warning("Failed to persist metric: %s", e)

    def complete_run(
        self,
        run_id: str,
        final_capital: float = 0.0,
        total_trades: int = 0,
        status: str = "completed",
    ) -> BacktestRunRecord | None:
        """Complete a backtest run and finalize results.

        CONCEPT:AHE-3.4

        Args:
            run_id: The run to complete.
            final_capital: Ending capital.
            total_trades: Total trades executed.
            status: Final status (completed, failed).

        Returns:
            The completed run record, or None if not found.
        """
        record = self._runs.get(run_id)
        if not record:
            return None

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record.final_capital = final_capital
        record.total_trades = total_trades
        record.status = status
        record.completed_at = now

        if self._db:
            try:
                self._db.execute(
                    """UPDATE backtest_runs
                    SET final_capital = ?, total_trades = ?,
                        status = ?, completed_at = ?
                    WHERE run_id = ?""",
                    (final_capital, total_trades, status, now, run_id),
                )
                self._db.commit()
            except Exception as e:
                logger.warning("Failed to update backtest run: %s", e)

        logger.info(
            "Completed backtest run %s: capital=%.2f, trades=%d",
            run_id,
            final_capital,
            total_trades,
        )
        return record

    def get_run(self, run_id: str) -> BacktestRunRecord | None:
        """Retrieve a backtest run by ID.

        Args:
            run_id: The run ID.

        Returns:
            The run record or None.
        """
        return self._runs.get(run_id)

    def compare_to_benchmark(
        self,
        run_id: str,
        benchmark_metrics: dict[str, float],
    ) -> list[BacktestComparison]:
        """Compare a run's metrics against benchmark values.

        CONCEPT:AHE-3.4

        Args:
            run_id: The run to compare.
            benchmark_metrics: Dict of metric_name → benchmark_value.

        Returns:
            List of comparison results.
        """
        record = self._runs.get(run_id)
        if not record:
            return []

        comparisons: list[BacktestComparison] = []
        for metric in record.metrics:
            if metric.metric_name in benchmark_metrics:
                bv = benchmark_metrics[metric.metric_name]
                delta = metric.value - bv
                # For drawdown-like metrics, lower is better
                if "drawdown" in metric.metric_name.lower():
                    outperforms = metric.value < bv
                else:
                    outperforms = metric.value > bv

                comparisons.append(
                    BacktestComparison(
                        run_id=run_id,
                        benchmark_id=record.benchmark_id or "default",
                        metric_name=metric.metric_name,
                        run_value=metric.value,
                        benchmark_value=bv,
                        delta=delta,
                        outperforms=outperforms,
                    )
                )
        return comparisons

    def list_runs(
        self,
        strategy_id: str | None = None,
        limit: int = 50,
    ) -> list[BacktestRunRecord]:
        """List backtest runs, optionally filtered by strategy.

        Args:
            strategy_id: Filter by strategy ID.
            limit: Maximum runs to return.

        Returns:
            List of run records.
        """
        runs = list(self._runs.values())
        if strategy_id:
            runs = [r for r in runs if r.strategy_id == strategy_id]
        return runs[:limit]

    def purge(self) -> int:
        """Delete all backtest data from the database.

        Returns:
            Number of runs purged.
        """
        count = len(self._runs)
        self._runs.clear()
        if self._db:
            try:
                self._db.execute("DELETE FROM backtest_metrics")
                self._db.execute("DELETE FROM backtest_runs")
                self._db.commit()
            except Exception as e:
                logger.warning("Failed to purge backtest data: %s", e)
        return count


logger = logging.getLogger(__name__)

# Dimension weight defaults (configurable)
DEFAULT_WEIGHTS: dict[str, float] = {
    "correctness": 0.35,
    "completeness": 0.25,
    "relevance": 0.25,
    "safety": 0.15,
}


class EvaluationDimension(BaseModel):
    """A single evaluation dimension with score and evidence."""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=0.25, ge=0.0, le=1.0)
    rubric: str = ""
    evidence: str = ""


class EvaluationRubric(BaseModel):
    """Scoring rubric defining dimensions and their criteria."""

    id: str = "default"
    name: str = "Default Evaluation Rubric"
    dimensions: list[EvaluationDimension] = Field(
        default_factory=lambda: [
            EvaluationDimension(
                name="correctness",
                score=0.0,
                weight=0.35,
                rubric="Is the response factually accurate?",
            ),
            EvaluationDimension(
                name="completeness",
                score=0.0,
                weight=0.25,
                rubric="Does the response address all parts of the query?",
            ),
            EvaluationDimension(
                name="relevance",
                score=0.0,
                weight=0.25,
                rubric="Is the response directly relevant to the query?",
            ),
            EvaluationDimension(
                name="safety",
                score=0.0,
                weight=0.15,
                rubric="Is the response safe and appropriate?",
            ),
        ]
    )


class MultiDimensionalEvaluation(BaseModel):
    """Complete multi-dimensional evaluation result."""

    dimensions: list[EvaluationDimension] = Field(default_factory=list)
    composite_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evaluator: str = "llm-judge"
    rubric_id: str = "default"
    timestamp: float = Field(default_factory=time.time)
    session_id: str = ""
    query_preview: str = ""
    response_preview: str = ""

    def compute_composite(self) -> float:
        """Compute weighted composite score from dimensions."""
        if not self.dimensions:
            return 0.0
        total_weight = sum(d.weight for d in self.dimensions) or 1.0
        self.composite_score = (
            sum(d.score * d.weight for d in self.dimensions) / total_weight
        )
        return self.composite_score


class QualityAlert(BaseModel):
    """An alert triggered when quality degrades below threshold."""

    dimension: str
    current_score: float
    threshold: float
    trend: list[float] = Field(default_factory=list)
    message: str = ""
    severity: str = "warning"  # warning, critical


class EvaluationMonitor:
    """Continuous quality monitoring with trend analysis and alerting.

    Parameters
    ----------
    alert_threshold : float
        Score below which a quality alert is triggered (default: 0.6).
    critical_threshold : float
        Score below which a critical alert is triggered (default: 0.3).
    kg_engine : optional
        If provided, evaluations are persisted to the KG.
    """

    def __init__(
        self,
        alert_threshold: float = 0.6,
        critical_threshold: float = 0.3,
        kg_engine: Any = None,
    ) -> None:
        self._alert_threshold = alert_threshold
        self._critical_threshold = critical_threshold
        self._engine = kg_engine
        self._history: list[MultiDimensionalEvaluation] = []
        self._dimension_history: dict[str, list[float]] = defaultdict(list)

    def evaluate(
        self,
        query: str,
        response: str,
        dimension_scores: dict[str, float] | None = None,
        rubric: EvaluationRubric | None = None,
        evaluator: str = "llm-judge",
        session_id: str = "",
    ) -> MultiDimensionalEvaluation:
        """Create a multi-dimensional evaluation.

        Parameters
        ----------
        query : str
            The original query.
        response : str
            The agent's response.
        dimension_scores : dict[str, float] | None
            Pre-computed dimension scores (e.g. from LLM-as-Judge).
        rubric : EvaluationRubric | None
            Scoring rubric (defaults to standard 4-dimension rubric).
        evaluator : str
            Who performed the evaluation.
        session_id : str
            Session identifier.
        """
        rubric = rubric or EvaluationRubric()
        scores = dimension_scores or {}

        dimensions = []
        for dim in rubric.dimensions:
            score = scores.get(dim.name, dim.score)
            dimensions.append(
                EvaluationDimension(
                    name=dim.name,
                    score=score,
                    weight=dim.weight,
                    rubric=dim.rubric,
                )
            )

        evaluation = MultiDimensionalEvaluation(
            dimensions=dimensions,
            evaluator=evaluator,
            rubric_id=rubric.id,
            session_id=session_id,
            query_preview=query[:200],
            response_preview=response[:200],
        )
        evaluation.compute_composite()

        self.record_evaluation(evaluation)
        return evaluation

    def record_evaluation(self, evaluation: MultiDimensionalEvaluation) -> None:
        """Record an evaluation for trend tracking."""
        self._history.append(evaluation)
        for dim in evaluation.dimensions:
            self._dimension_history[dim.name].append(dim.score)

    def get_trend(self, dimension: str, lookback: int = 50) -> list[float]:
        """Get score trend for a dimension over recent evaluations."""
        scores = self._dimension_history.get(dimension, [])
        return scores[-lookback:]

    def get_composite_trend(self, lookback: int = 50) -> list[float]:
        """Get composite score trend over recent evaluations."""
        return [e.composite_score for e in self._history[-lookback:]]

    def check_alerts(self) -> list[QualityAlert]:
        """Check for quality degradation alerts across all dimensions."""
        alerts: list[QualityAlert] = []
        for dim_name, scores in self._dimension_history.items():
            if len(scores) < 3:
                continue
            recent_avg = sum(scores[-5:]) / len(scores[-5:])
            if recent_avg < self._critical_threshold:
                alerts.append(
                    QualityAlert(
                        dimension=dim_name,
                        current_score=recent_avg,
                        threshold=self._critical_threshold,
                        trend=scores[-10:],
                        severity="critical",
                        message=f"CRITICAL: {dim_name} score ({recent_avg:.2f}) "
                        f"below critical threshold ({self._critical_threshold})",
                    )
                )
            elif recent_avg < self._alert_threshold:
                alerts.append(
                    QualityAlert(
                        dimension=dim_name,
                        current_score=recent_avg,
                        threshold=self._alert_threshold,
                        trend=scores[-10:],
                        severity="warning",
                        message=f"WARNING: {dim_name} score ({recent_avg:.2f}) "
                        f"below alert threshold ({self._alert_threshold})",
                    )
                )
        return alerts

    async def persist_to_kg(self, evaluation: MultiDimensionalEvaluation) -> None:
        """Persist evaluation record to the Knowledge Graph."""
        if self._engine is None:
            return
        try:
            from agent_utilities.models.knowledge_graph import (
                EvaluationRecordNode,
                RegistryNodeType,
            )

            node = EvaluationRecordNode(
                id=f"eval:{evaluation.session_id}:{evaluation.timestamp}",
                type=RegistryNodeType.EVALUATION_RECORD,
                name=f"Evaluation: {evaluation.session_id}",
                correctness_score=next(
                    (d.score for d in evaluation.dimensions if d.name == "correctness"),
                    0.0,
                ),
                completeness_score=next(
                    (
                        d.score
                        for d in evaluation.dimensions
                        if d.name == "completeness"
                    ),
                    0.0,
                ),
                relevance_score=next(
                    (d.score for d in evaluation.dimensions if d.name == "relevance"),
                    0.0,
                ),
                safety_score=next(
                    (d.score for d in evaluation.dimensions if d.name == "safety"), 1.0
                ),
                composite_score=evaluation.composite_score,
                evaluator=evaluation.evaluator,
                rubric_id=evaluation.rubric_id,
                session_id=evaluation.session_id,
            )
            if hasattr(self._engine, "upsert_node"):
                self._engine.upsert_node(node.model_dump())
            logger.info("Persisted evaluation to KG: %.2f", evaluation.composite_score)
        except Exception:
            logger.debug("KG persistence skipped for evaluation")

    def summary(self) -> dict[str, Any]:
        """Return a summary of evaluation monitoring state."""
        return {
            "total_evaluations": len(self._history),
            "dimensions_tracked": list(self._dimension_history.keys()),
            "recent_composite": (
                self._history[-1].composite_score if self._history else None
            ),
            "composite_trend": self.get_composite_trend(10),
            "active_alerts": len(self.check_alerts()),
        }


# ---------------------------------------------------------------------------
# EvalRunner — Multi-Strategy Scoring (CONCEPT:AHE-3.1)
# ---------------------------------------------------------------------------
# Ported from MATE's eval_runner.py. Provides three concrete scoring
# strategies that execute automatically against test cases:
#   1. Exact Match — normalized string comparison
#   2. Semantic Similarity — cosine similarity via embedding model
#   3. LLM-as-Judge — structured JSON prompt for consistent scoring
#
# OWL synergy: Results persist as EvaluationRecordNode. OWL reasoning
# can infer `degradedPerformance` across sessions — a capability MATE
# lacks because it has no knowledge graph.
# ---------------------------------------------------------------------------


class EvalStrategy(StrEnum):
    """Evaluation strategy for scoring agent responses.

    CONCEPT:AHE-3.1 — Multi-Strategy Evaluation

    Ported from MATE's eval_runner.py pattern with three concrete
    strategies plus a composite mode that combines all three.
    """

    EXACT_MATCH = "exact_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    LLM_JUDGE = "llm_judge"
    COMPOSITE = "composite"
    #: CONCEPT:AHE-3.25 — judge a plain-English assertion (Opik "Test Suite" style).
    ASSERTION = "assertion"


class TestCase(BaseModel):
    """A single evaluation test case with expected output.

    CONCEPT:AHE-3.1 — Multi-Strategy Evaluation

    Mirrors MATE's test case schema but adds KG provenance fields
    for integration with the agent-utilities knowledge graph.
    """

    id: str = ""
    query: str
    expected_output: str
    agent_name: str = ""
    strategy: EvalStrategy = EvalStrategy.COMPOSITE
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    assertion: str = Field(
        default="",
        description=(
            "CONCEPT:AHE-3.25 — Plain-English regression assertions with verified-remediation auto-lock (Opik Test Suite style). "
            "Optional pass/fail assertion judged by LLM-as-judge; when set, it takes "
            "precedence over expected-output scoring."
        ),
    )


class EvalResult(BaseModel):
    """Result of evaluating a single test case.

    CONCEPT:AHE-3.1 — Multi-Strategy Evaluation

    Contains per-strategy scores and a final composite score.
    """

    test_case_id: str = ""
    query: str = ""
    expected_output: str = ""
    actual_output: str = ""
    strategy: EvalStrategy = EvalStrategy.COMPOSITE
    exact_match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_judge_score: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_judge_reasoning: str = ""
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)
    passed: bool = False
    timestamp: float = Field(default_factory=time.time)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRunner:
    """Multi-strategy evaluation runner.

    CONCEPT:AHE-3.1 — Multi-Strategy Evaluation

    Ported from MATE's ``EvalRunner`` with three scoring strategies:

    1. **Exact Match** — Normalized string comparison after lowercasing,
       stripping whitespace, and removing punctuation.

    2. **Semantic Similarity** — Cosine similarity between embeddings
       of expected and actual outputs.

    3. **LLM-as-Judge** — Structured JSON prompt that forces a
       consistent single-line output format with score and reasoning.
       Ported from MATE's ``llm_judge_eval`` prompt pattern.

    The runner integrates with the existing ``EvaluationMonitor`` for
    trend tracking and alerting, and persists results as
    ``EvaluationRecordNode`` in the Knowledge Graph.

    Parameters
    ----------
    monitor : EvaluationMonitor | None
        If provided, each eval result is also recorded in the monitor
        for trend tracking and alerting.
    pass_threshold : float
        Score at or above which a test case is considered passing.
    exact_weight : float
        Weight of exact match in composite scoring.
    semantic_weight : float
        Weight of semantic similarity in composite scoring.
    judge_weight : float
        Weight of LLM-as-Judge in composite scoring.
    """

    # Default LLM-as-Judge prompt — ported from MATE's structured judge
    # prompt that forces consistent single-line JSON output.
    LLM_JUDGE_PROMPT = (
        "You are an expert evaluator. Compare the actual output against "
        "the expected output for the given query.\n\n"
        "Query: {query}\n"
        "Expected Output: {expected}\n"
        "Actual Output: {actual}\n\n"
        "Rate the actual output on a scale of 0.0 to 1.0 where:\n"
        "- 1.0 = perfect match in meaning and completeness\n"
        "- 0.7 = mostly correct with minor omissions\n"
        "- 0.4 = partially correct but significant gaps\n"
        "- 0.0 = completely wrong or irrelevant\n\n"
        'Respond with ONLY a single JSON line: {{"score": <float>, "reasoning": "<brief explanation>"}}'
    )

    def __init__(
        self,
        monitor: EvaluationMonitor | None = None,
        pass_threshold: float = 0.7,
        exact_weight: float = 0.2,
        semantic_weight: float = 0.3,
        judge_weight: float = 0.5,
    ) -> None:
        self._monitor = monitor
        self._pass_threshold = pass_threshold
        self._exact_weight = exact_weight
        self._semantic_weight = semantic_weight
        self._judge_weight = judge_weight
        self._results: list[EvalResult] = []

    @property
    def results(self) -> list[EvalResult]:
        """All evaluation results collected so far."""
        return list(self._results)

    def run_eval(
        self,
        test_case: TestCase,
        actual_output: str,
        strategy: EvalStrategy | None = None,
    ) -> EvalResult:
        """Evaluate a single test case against actual output.

        Parameters
        ----------
        test_case : TestCase
            The test case with query and expected output.
        actual_output : str
            The agent's actual response.
        strategy : EvalStrategy | None
            Override the test case's default strategy.

        Returns
        -------
        EvalResult
            The evaluation result with per-strategy scores.
        """
        start = time.time()
        effective_strategy = strategy or test_case.strategy

        result = EvalResult(
            test_case_id=test_case.id,
            query=test_case.query,
            expected_output=test_case.expected_output,
            actual_output=actual_output,
            strategy=effective_strategy,
        )

        # A plain-English assertion is the test (Opik Test Suite semantics):
        # it takes precedence over expected-output scoring. CONCEPT:AHE-3.25.
        if test_case.assertion or effective_strategy == EvalStrategy.ASSERTION:
            score, reasoning = self._assertion_judge(
                test_case.assertion or test_case.expected_output,
                test_case.query,
                actual_output,
            )
            result.llm_judge_score = score
            result.llm_judge_reasoning = reasoning
            result.final_score = score
            result.passed = result.final_score >= self._pass_threshold
            result.duration_ms = (time.time() - start) * 1000
            result.timestamp = time.time()
            self._results.append(result)
            return result

        # Always compute exact match (it's free)
        result.exact_match_score = self._exact_match_eval(
            test_case.expected_output, actual_output
        )

        if effective_strategy == EvalStrategy.EXACT_MATCH:
            result.final_score = result.exact_match_score

        elif effective_strategy == EvalStrategy.SEMANTIC_SIMILARITY:
            result.semantic_similarity_score = self._semantic_similarity_eval(
                test_case.expected_output, actual_output
            )
            result.final_score = result.semantic_similarity_score

        elif effective_strategy == EvalStrategy.LLM_JUDGE:
            score, reasoning = self._llm_judge_eval(
                test_case.query, test_case.expected_output, actual_output
            )
            result.llm_judge_score = score
            result.llm_judge_reasoning = reasoning
            result.final_score = result.llm_judge_score

        elif effective_strategy == EvalStrategy.COMPOSITE:
            # Compute all strategies
            result.semantic_similarity_score = self._semantic_similarity_eval(
                test_case.expected_output, actual_output
            )
            score, reasoning = self._llm_judge_eval(
                test_case.query, test_case.expected_output, actual_output
            )
            result.llm_judge_score = score
            result.llm_judge_reasoning = reasoning
            # Weighted composite
            result.final_score = (
                self._exact_weight * result.exact_match_score
                + self._semantic_weight * result.semantic_similarity_score
                + self._judge_weight * result.llm_judge_score
            )

        result.passed = result.final_score >= self._pass_threshold
        result.duration_ms = (time.time() - start) * 1000
        result.timestamp = time.time()

        self._results.append(result)

        # Feed into EvaluationMonitor for trend tracking
        if self._monitor:
            self._monitor.evaluate(
                query=test_case.query,
                response=actual_output,
                dimension_scores={
                    "correctness": result.final_score,
                    "completeness": result.semantic_similarity_score,
                    "relevance": result.exact_match_score,
                    "safety": 1.0,  # not evaluated by EvalRunner
                },
                evaluator="eval_runner",
                session_id=test_case.id,
            )

        return result

    def run_batch(
        self,
        test_cases: list[TestCase],
        actual_outputs: list[str],
        strategy: EvalStrategy | None = None,
    ) -> list[EvalResult]:
        """Run evaluation on a batch of test cases.

        Parameters
        ----------
        test_cases : list[TestCase]
            List of test cases.
        actual_outputs : list[str]
            Corresponding actual outputs (must match length).
        strategy : EvalStrategy | None
            Override strategy for all cases.

        Returns
        -------
        list[EvalResult]
            Evaluation results for each test case.
        """
        if len(test_cases) != len(actual_outputs):
            raise ValueError(
                f"Mismatch: {len(test_cases)} test cases vs "
                f"{len(actual_outputs)} actual outputs"
            )
        return [
            self.run_eval(tc, output, strategy)
            for tc, output in zip(test_cases, actual_outputs, strict=False)
        ]

    def summary(self) -> dict[str, Any]:
        """Return aggregate statistics for all evaluations run so far."""
        if not self._results:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,  # nosec B105 - not a password
                "avg_score": 0.0,
            }
        passed = sum(1 for r in self._results if r.passed)
        return {
            "total": len(self._results),
            "passed": passed,
            "failed": len(self._results) - passed,
            "pass_rate": passed / len(self._results),
            "avg_score": sum(r.final_score for r in self._results) / len(self._results),
            "avg_duration_ms": sum(r.duration_ms for r in self._results)
            / len(self._results),
        }

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for exact matching."""
        import re as _re

        text = text.lower().strip()
        text = _re.sub(r"[^\w\s]", "", text)
        text = _re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _exact_match_eval(expected: str, actual: str) -> float:
        """Exact match with normalization.

        Returns 1.0 for exact match, or a partial score based on
        token overlap (Jaccard similarity) for near-matches.
        """
        norm_expected = EvalRunner._normalize(expected)
        norm_actual = EvalRunner._normalize(actual)

        if norm_expected == norm_actual:
            return 1.0

        # Jaccard similarity for partial credit
        expected_tokens = set(norm_expected.split())
        actual_tokens = set(norm_actual.split())
        if not expected_tokens and not actual_tokens:
            return 1.0
        if not expected_tokens or not actual_tokens:
            return 0.0
        intersection = expected_tokens & actual_tokens
        union = expected_tokens | actual_tokens
        return len(intersection) / len(union)

    @staticmethod
    def _semantic_similarity_eval(expected: str, actual: str) -> float:
        """Semantic similarity via cosine distance of embeddings.

        Falls back to token overlap if no embedding model is available.
        """
        try:
            from agent_utilities.core.embedding_utilities import (
                create_embedding_model,
            )

            model = create_embedding_model()
            if model is not None:
                expected_emb = model.get_text_embedding(expected)
                actual_emb = model.get_text_embedding(actual)
                # Cosine similarity
                dot = sum(a * b for a, b in zip(expected_emb, actual_emb, strict=False))
                norm_e = sum(a * a for a in expected_emb) ** 0.5
                norm_a = sum(a * a for a in actual_emb) ** 0.5
                if norm_e > 0 and norm_a > 0:
                    return max(0.0, min(1.0, dot / (norm_e * norm_a)))
        except Exception:  # nosec B110
            pass

        # Fallback: token overlap (same as exact match partial)
        return EvalRunner._exact_match_eval(expected, actual)

    @staticmethod
    def _run_llm(prompt: str) -> str | None:
        """One-shot LLM completion against the configured model (vLLM/OpenAI-style).

        A pydantic-ai ``Model`` is NOT directly runnable — ``run_sync`` lives on
        ``Agent`` — so the judge wraps the model in a minimal ``Agent``. Returns the
        text output, or ``None`` when no model is reachable (callers fall back).
        """
        try:
            from pydantic_ai import Agent

            from agent_utilities.core.model_factory import create_model

            result = Agent(create_model()).run_sync(prompt)
            return result.output if hasattr(result, "output") else str(result)
        except Exception as exc:  # pragma: no cover - model optional offline
            logger.debug("LLM unavailable for judge: %s", exc)
            return None

    @staticmethod
    def _llm_judge_eval(query: str, expected: str, actual: str) -> tuple[float, str]:
        """LLM-as-Judge evaluation with structured JSON output.

        Ported from MATE's ``llm_judge_eval`` — uses a structured prompt
        that forces a single-line JSON response with score and reasoning.

        Falls back to semantic similarity if no LLM is available.
        """
        try:
            import json as _json

            prompt = EvalRunner.LLM_JUDGE_PROMPT.format(
                query=query, expected=expected, actual=actual
            )
            response_text = EvalRunner._run_llm(prompt)
            if response_text is None:
                raise RuntimeError("no model available")

            # Parse JSON from response
            # Handle potential markdown code blocks
            clean = response_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = _json.loads(clean)
            score = float(parsed.get("score", 0.0))
            reasoning = str(parsed.get("reasoning", ""))
            return (max(0.0, min(1.0, score)), reasoning)

        except Exception as exc:
            logger.debug("LLM judge fallback (no model available): %s", exc)
            # Fallback: use semantic similarity as proxy
            fallback_score = EvalRunner._semantic_similarity_eval(expected, actual)
            return (
                fallback_score,
                f"LLM judge unavailable, using semantic fallback: {exc}",
            )

    ASSERTION_JUDGE_PROMPT = (
        "You are an expert evaluator. Decide whether the actual output satisfies "
        "the plain-English assertion for the given query.\n\n"
        "Query: {query}\n"
        "Assertion: {assertion}\n"
        "Actual Output: {actual}\n\n"
        'Respond with ONLY a single JSON line: {{"pass": <true|false>, '
        '"reasoning": "<brief explanation>"}}'
    )

    @staticmethod
    def _assertion_judge(assertion: str, query: str, actual: str) -> tuple[float, str]:
        """Judge a plain-English assertion, returning (1.0|0.0, reasoning).

        CONCEPT:AHE-3.25 — the Opik "Test Suite" check: a human-readable assertion
        is converted to an LLM-as-judge pass/fail. Falls back to a substring check
        when no model is available, so the seam works offline.
        """
        try:
            import json as _json

            prompt = EvalRunner.ASSERTION_JUDGE_PROMPT.format(
                query=query, assertion=assertion, actual=actual
            )
            response_text = EvalRunner._run_llm(prompt)
            if response_text is None:
                raise RuntimeError("no model available")
            clean = response_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = _json.loads(clean)
            passed = bool(parsed.get("pass", False))
            reasoning = str(parsed.get("reasoning", ""))
            return (1.0 if passed else 0.0, reasoning)
        except Exception as exc:
            logger.debug("assertion judge fallback (no model available): %s", exc)
            # Offline heuristic: pass if the assertion's salient words appear.
            words = [w for w in assertion.lower().split() if len(w) > 3]
            hit = sum(1 for w in words if w in actual.lower())
            ratio = hit / len(words) if words else 0.0
            return (
                1.0 if ratio >= 0.6 else 0.0,
                f"assertion judge unavailable, lexical fallback ratio={ratio:.2f}",
            )


if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test Case Definition
# ---------------------------------------------------------------------------


class InterpretabilityTestCase:
    """A single interpretability test case.

    Attributes:
        category: The test category.
        query: The quantitative question.
        ground_truth: The correct answer (string for flexible matching).
        tolerance: Numerical tolerance for grading (e.g., 0.05 = 5%).
    """

    def __init__(
        self,
        category: InterpretabilityTestCategory,
        query: str,
        ground_truth: str,
        tolerance: float = 0.05,
    ) -> None:
        self.category = category
        self.query = query
        self.ground_truth = ground_truth
        self.tolerance = tolerance


# ---------------------------------------------------------------------------
# Interpretability Grader
# ---------------------------------------------------------------------------


class InterpretabilityGrader:
    """LLM-based grading for interpretability test responses.

    CONCEPT:AHE-3.3 — LLM-Graded Interpretability Tests

    Uses the EvalRunner (CONCEPT:AHE-3.1) infrastructure for
    LLM-as-Judge scoring with numerical tolerance. Detects
    reward hacking when the model's ``__str__()`` directly
    contains the answer.

    Args:
        evaluator_model: Name of the LLM used for grading.
    """

    def __init__(self, evaluator_model: str = "default") -> None:
        self.evaluator_model = evaluator_model

    def grade(
        self,
        response: str,
        ground_truth: str,
        tolerance: float = 0.05,
    ) -> tuple[bool, str]:
        """Grade a single response against ground truth.

        Supports both numerical and categorical matching.

        Args:
            response: The LLM's response to the test question.
            ground_truth: The correct answer.
            tolerance: Relative tolerance for numerical answers.

        Returns:
            Tuple of (passed, reasoning).
        """
        response_clean = response.strip().lower()
        truth_clean = ground_truth.strip().lower()

        # Try numerical comparison first
        try:
            resp_val = float(response_clean.replace(",", ""))
            truth_val = float(truth_clean.replace(",", ""))

            if truth_val == 0.0:
                passed = abs(resp_val) <= tolerance
            else:
                relative_error = abs(resp_val - truth_val) / abs(truth_val)
                passed = relative_error <= tolerance

            return (
                passed,
                f"Numerical: response={resp_val:.4f}, truth={truth_val:.4f}, "
                f"tolerance={tolerance:.2%}, "
                f"error={abs(resp_val - truth_val) / max(abs(truth_val), 1e-10):.2%}",
            )
        except (ValueError, ZeroDivisionError):
            pass

        # Categorical/string comparison
        if response_clean == truth_clean:
            return True, "Exact string match"

        # Token overlap for partial matching
        resp_tokens = set(response_clean.split())
        truth_tokens = set(truth_clean.split())
        if truth_tokens and resp_tokens:
            overlap = len(resp_tokens & truth_tokens) / len(truth_tokens)
            if overlap >= 0.8:
                return True, f"Token overlap: {overlap:.0%}"

        return False, f"No match: '{response_clean}' vs '{truth_clean}'"

    def detect_reward_hacking(
        self,
        model_str: str,
        ground_truth: str,
    ) -> bool:
        """Detect if the model's __str__() directly contains the answer.

        Reward hacking occurs when a model's display representation
        is designed to directly recite test answers rather than
        expressing genuine structure. This is Pattern 3 from the paper.

        Args:
            model_str: The model's ``__str__()`` output.
            ground_truth: The test's correct answer.

        Returns:
            True if potential reward hacking detected.
        """
        truth_clean = ground_truth.strip()
        if len(truth_clean) < 3:
            return False
        return truth_clean in model_str


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


class InterpretabilityTestSuite:
    """Manages the full interpretability test protocol.

    CONCEPT:AHE-3.3 — LLM-Graded Interpretability Tests

    Implements the 6-category, 200-test protocol from arXiv:2605.03808:
        1. Feature Attribution (32 tests)
        2. Point Simulation (43 tests)
        3. Sensitivity Analysis (32 tests)
        4. Counterfactual (32 tests)
        5. Confidence Calibration (32 tests)
        6. Data Attribution (29 tests)

    The suite generates test cases, runs them against a model's
    ``__str__()`` output, and computes the aggregate agent
    interpretability score (pass rate).

    Args:
        grader: The grader to use for test evaluation.
        engine: Optional KG engine for persisting test results.
    """

    # Default test distribution per category (200 total)
    DEFAULT_TEST_COUNTS: dict[InterpretabilityTestCategory, int] = {
        InterpretabilityTestCategory.FEATURE_ATTRIBUTION: 32,
        InterpretabilityTestCategory.POINT_SIMULATION: 43,
        InterpretabilityTestCategory.SENSITIVITY_ANALYSIS: 32,
        InterpretabilityTestCategory.COUNTERFACTUAL: 32,
        InterpretabilityTestCategory.CONFIDENCE_CALIBRATION: 32,
        InterpretabilityTestCategory.DATA_ATTRIBUTION: 29,
    }

    def __init__(
        self,
        grader: InterpretabilityGrader | None = None,
        engine: IntelligenceGraphEngine | None = None,
    ) -> None:
        self._grader = grader or InterpretabilityGrader()
        self._engine = engine
        self._results: list[dict[str, Any]] = []

    @property
    def results(self) -> list[dict[str, Any]]:
        return list(self._results)

    def generate_feature_attribution_tests(
        self,
        feature_names: list[str],
        coefficients: list[float],
    ) -> list[InterpretabilityTestCase]:
        """Generate feature attribution test cases from known coefficients.

        Tests include:
            - Most important feature (largest absolute coefficient)
            - Feature ranking by importance
            - Sign of feature effects
            - Detection of irrelevant features (zero coefficient)

        Args:
            feature_names: List of feature names.
            coefficients: Corresponding coefficient values.

        Returns:
            List of test cases.
        """
        tests: list[InterpretabilityTestCase] = []
        if not feature_names or not coefficients:
            return tests

        paired = list(zip(feature_names, coefficients, strict=False))
        abs_sorted = sorted(paired, key=lambda x: abs(x[1]), reverse=True)

        # Most important feature
        tests.append(
            InterpretabilityTestCase(
                category=InterpretabilityTestCategory.FEATURE_ATTRIBUTION,
                query="Which feature has the largest absolute effect on the prediction?",
                ground_truth=abs_sorted[0][0],
            )
        )

        # Feature ranking (top-3)
        top3 = [name for name, _ in abs_sorted[:3]]
        tests.append(
            InterpretabilityTestCase(
                category=InterpretabilityTestCategory.FEATURE_ATTRIBUTION,
                query="List the top 3 most important features in order.",
                ground_truth=", ".join(top3),
            )
        )

        # Sign effects
        for name, coef in paired[:5]:
            sign = "positive" if coef > 0 else "negative" if coef < 0 else "zero"
            tests.append(
                InterpretabilityTestCase(
                    category=InterpretabilityTestCategory.FEATURE_ATTRIBUTION,
                    query=f"What is the sign of the effect of feature '{name}'?",
                    ground_truth=sign,
                )
            )

        # Irrelevant features (zero coefficients)
        zero_features = [name for name, coef in paired if coef == 0.0]
        if zero_features:
            tests.append(
                InterpretabilityTestCase(
                    category=InterpretabilityTestCategory.FEATURE_ATTRIBUTION,
                    query="Which features have zero effect on the prediction?",
                    ground_truth=", ".join(zero_features),
                )
            )

        return tests

    def generate_point_simulation_tests(
        self,
        inputs: list[dict[str, float]],
        outputs: list[float],
    ) -> list[InterpretabilityTestCase]:
        """Generate point simulation tests from known input-output pairs.

        Args:
            inputs: List of input feature dicts.
            outputs: Corresponding predicted outputs.

        Returns:
            List of test cases.
        """
        tests: list[InterpretabilityTestCase] = []
        for inp, out in zip(inputs, outputs, strict=False):
            features_str = ", ".join(f"{k}={v}" for k, v in inp.items())
            tests.append(
                InterpretabilityTestCase(
                    category=InterpretabilityTestCategory.POINT_SIMULATION,
                    query=f"What does the model predict for input: {features_str}?",
                    ground_truth=f"{out:.4f}",
                    tolerance=0.05,
                )
            )
        return tests

    def generate_sensitivity_tests(
        self,
        feature_names: list[str],
        sensitivities: dict[str, float],
    ) -> list[InterpretabilityTestCase]:
        """Generate sensitivity analysis tests.

        Args:
            feature_names: Feature names.
            sensitivities: Feature name → unit sensitivity value.

        Returns:
            List of test cases.
        """
        tests: list[InterpretabilityTestCase] = []
        for name in feature_names:
            if name not in sensitivities:
                continue
            sensitivity = sensitivities[name]
            direction = "increase" if sensitivity > 0 else "decrease"
            tests.append(
                InterpretabilityTestCase(
                    category=InterpretabilityTestCategory.SENSITIVITY_ANALYSIS,
                    query=(
                        f"If feature '{name}' increases by 1 unit, "
                        f"does the prediction increase or decrease?"
                    ),
                    ground_truth=direction,
                )
            )
            tests.append(
                InterpretabilityTestCase(
                    category=InterpretabilityTestCategory.SENSITIVITY_ANALYSIS,
                    query=(
                        f"By how much does the prediction change if "
                        f"feature '{name}' increases by 1 unit?"
                    ),
                    ground_truth=f"{sensitivity:.4f}",
                    tolerance=0.10,
                )
            )
        return tests

    def generate_counterfactual_tests(
        self,
        feature_names: list[str],
        counterfactuals: list[dict[str, Any]],
    ) -> list[InterpretabilityTestCase]:
        """Generate counterfactual test cases.

        Args:
            feature_names: Feature names.
            counterfactuals: List of dicts with 'target_output', 'feature',
                'required_value'.

        Returns:
            List of test cases.
        """
        tests: list[InterpretabilityTestCase] = []
        for cf in counterfactuals:
            tests.append(
                InterpretabilityTestCase(
                    category=InterpretabilityTestCategory.COUNTERFACTUAL,
                    query=(
                        f"What value of feature '{cf['feature']}' would produce "
                        f"a prediction of {cf['target_output']}?"
                    ),
                    ground_truth=f"{cf['required_value']:.4f}",
                    tolerance=0.10,
                )
            )
        return tests

    def run_test(
        self,
        model_str: str,
        test_case: InterpretabilityTestCase,
        llm_response: str,
    ) -> dict[str, Any]:
        """Run a single interpretability test and grade the response.

        Args:
            model_str: The model's ``__str__()`` output.
            test_case: The test case to evaluate.
            llm_response: The LLM's response to the test question.

        Returns:
            Dict with test results including passed, reasoning, etc.
        """
        passed, reasoning = self._grader.grade(
            llm_response,
            test_case.ground_truth,
            test_case.tolerance,
        )
        reward_hack = self._grader.detect_reward_hacking(
            model_str,
            test_case.ground_truth,
        )

        result = {
            "category": test_case.category.value,
            "query": test_case.query,
            "ground_truth": test_case.ground_truth,
            "llm_response": llm_response,
            "passed": passed and not reward_hack,
            "reasoning": reasoning,
            "reward_hacking_detected": reward_hack,
            "tolerance": test_case.tolerance,
            "timestamp": time.time(),
        }
        self._results.append(result)
        return result

    def run_suite(
        self,
        model_str: str,
        test_cases: list[InterpretabilityTestCase],
        llm_responses: list[str],
    ) -> dict[str, Any]:
        """Run a full suite of tests and compute aggregate score.

        Args:
            model_str: The model's ``__str__()`` output.
            test_cases: All test cases.
            llm_responses: Corresponding LLM responses.

        Returns:
            Dict with per-category and aggregate results.
        """
        if len(test_cases) != len(llm_responses):
            raise ValueError(
                f"Mismatch: {len(test_cases)} tests vs {len(llm_responses)} responses"
            )

        results = []
        for tc, resp in zip(test_cases, llm_responses, strict=False):
            results.append(self.run_test(model_str, tc, resp))

        return self.compute_agent_interpretability_score(results)

    def compute_agent_interpretability_score(
        self,
        results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compute aggregate interpretability score from test results.

        Returns the overall pass rate and per-category breakdown.

        Args:
            results: Test results. Defaults to all recorded results.

        Returns:
            Dict with ``overall_score``, ``per_category``, ``total_tests``,
            ``total_passed``, ``reward_hacking_count``.
        """
        pool = results if results is not None else self._results
        if not pool:
            return {
                "overall_score": 0.0,
                "per_category": {},
                "total_tests": 0,
                "total_passed": 0,
                "reward_hacking_count": 0,
            }

        total = len(pool)
        passed = sum(1 for r in pool if r.get("passed", False))
        hacked = sum(1 for r in pool if r.get("reward_hacking_detected", False))

        # Per-category breakdown
        categories: dict[str, dict[str, int]] = {}
        for r in pool:
            cat = r.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0}
            categories[cat]["total"] += 1
            if r.get("passed", False):
                categories[cat]["passed"] += 1

        per_category = {
            cat: {
                "total": counts["total"],
                "passed": counts["passed"],
                "score": counts["passed"] / counts["total"] if counts["total"] else 0.0,
            }
            for cat, counts in categories.items()
        }

        return {
            "overall_score": passed / total if total else 0.0,
            "per_category": per_category,
            "total_tests": total,
            "total_passed": passed,
            "reward_hacking_count": hacked,
        }

    def persist_results_to_kg(
        self,
        imodel_node_id: str,
        results: list[dict[str, Any]] | None = None,
    ) -> int:
        """Persist test results as InterpretabilityTestNode in the KG.

        Args:
            imodel_node_id: ID of the parent IModelNode.
            results: Results to persist. Defaults to all recorded.

        Returns:
            Number of nodes persisted.
        """
        if not self._engine:
            return 0

        pool = results if results is not None else self._results
        persisted = 0

        try:
            from ..knowledge_graph.core.ogm import KGMapper

            ogm = KGMapper(self._engine)

            for r in pool:
                node_id = f"itest:{uuid.uuid4().hex[:8]}"
                cat_str = r.get("category", "point_simulation")
                try:
                    cat = InterpretabilityTestCategory(cat_str)
                except ValueError:
                    cat = InterpretabilityTestCategory.POINT_SIMULATION

                node = InterpretabilityTestNode(
                    id=node_id,
                    type=RegistryNodeType.INTERPRETABILITY_TEST,
                    name=f"InterpTest: {cat_str}",
                    test_category=cat,
                    test_query=r.get("query", ""),
                    ground_truth=r.get("ground_truth", ""),
                    llm_response=r.get("llm_response", ""),
                    passed=r.get("passed", False),
                    tolerance=r.get("tolerance", 0.05),
                    evaluator_model=self._grader.evaluator_model,
                    imodel_node_id=imodel_node_id,
                    metadata={"concept": "AHE-3.16", "paper": "arXiv:2605.03808"},
                )
                ogm.upsert(node)
                ogm.upsert_edge(
                    imodel_node_id,
                    node_id,
                    RegistryEdgeType.TESTED_INTERPRETABILITY,
                    {"category": cat_str},
                )
                persisted += 1
        except Exception as exc:
            logger.warning("Failed to persist test results: %s", exc)

        return persisted
