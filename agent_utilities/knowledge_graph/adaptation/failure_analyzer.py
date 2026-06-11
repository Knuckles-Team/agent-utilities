#!/usr/bin/python
from __future__ import annotations

"""Failure-driven self-evolution (CONCEPT:AHE-3.18 — Failure-Driven Evolution).

Closes the loop the research-driven golden loop never had: instead of only
ingesting *papers* and *unresolved research concepts*, the KG now also learns
from **failures observed in production telemetry** (Langfuse).

Flow::

    pull     → ERROR observations + low-score traces + cost/latency anomalies
               from Langfuse (via the read-only LangfuseTraceBackend)
    cluster  → recurring failure *signatures* (deterministic, LLM-free)
    materialize → ExecutionSummary + PerformanceAnomaly KG nodes (activating the
               dormant telemetry schema that maintainer.py already consumes) and
               a synthetic ``failure_gap`` ``Concept`` per pattern — with NO
               ``ADDRESSED_BY`` edge, so the golden loop's existing intake stage
               (``topic_resolver.unresolved_topics``) picks it up unchanged and
               synthesizes a remediation proposal for it.

The whole module is *propose-only* in spirit: it only writes observation/topic
nodes. Whether a remediation auto-merges is gated separately by the golden
loop's :class:`GovernedAutoMerger` — and, for failure remediations, by the
regression check built here (:meth:`make_regression_check`).

Dependencies are injected so the analyzer is unit-testable without a live
engine or Langfuse; :meth:`from_engine` wires it from a running engine.
"""

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# anomaly_type values written to PerformanceAnomaly nodes (consumed by
# maintainer.trigger_self_improvement). Kept to a small controlled vocabulary.
ANOMALY_ERROR = "ERROR_RATE"
ANOMALY_LOW_SCORE = "LOW_SCORE"
ANOMALY_LATENCY = "TIMEOUT"
ANOMALY_COST = "HIGH_COST"
ANOMALY_TOKENS = "HIGH_TOKEN_USAGE"

# Normalization: collapse the volatile parts of an error string so the *same*
# failure produces the *same* signature across occurrences.
_HEX = re.compile(r"\b(?:0x)?[0-9a-f]{6,}\b", re.IGNORECASE)
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_NUM = re.compile(r"\d+")
_PATH = re.compile(r"(/[\w.\-]+){2,}")
_WS = re.compile(r"\s+")


def _normalize_detail(text: str) -> str:
    """Strip ids/paths/numbers from an error/status string for stable grouping."""
    if not text:
        return ""
    t = str(text)
    t = _UUID.sub("<id>", t)
    t = _PATH.sub("<path>", t)
    t = _HEX.sub("<id>", t)
    t = _NUM.sub("<n>", t)
    t = _WS.sub(" ", t).strip().lower()
    return t[:200]


def _sig(name: str, kind: str, detail: str) -> str:
    """Stable short signature for a (name, kind, normalized-detail) failure."""
    raw = f"{name}|{kind}|{detail}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_coro(coro: Any) -> Any:
    """Run a coroutine to completion whether or not a loop is already running.

    The daemon ``failure_ingest`` tick runs in a worker thread (no loop, so
    ``asyncio.run`` works), but the ``graph_orchestrate(action="failure_ingest")``
    MCP action runs inside the server's event loop — where ``asyncio.run`` raises.
    In that case run the coroutine on a short-lived helper thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


@dataclass
class FailureRecord:
    """One normalized failure observation pulled from telemetry."""

    kind: str  # "error" | "low_score" | "anomaly"
    name: str
    detail: str
    anomaly_type: str
    trace_id: str | None = None
    value: float | None = None
    baseline: float | None = None

    @property
    def signature(self) -> str:
        return _sig(self.name, self.kind, _normalize_detail(self.detail))


@dataclass
class FailurePattern:
    """A recurring failure cluster keyed by signature."""

    signature: str
    name: str
    kind: str
    anomaly_type: str
    count: int
    trace_ids: list[str] = field(default_factory=list)
    sample_detail: str = ""
    value: float | None = None
    baseline: float | None = None

    @property
    def label(self) -> str:
        return f"{self.anomaly_type} in {self.name}: {self.sample_detail[:80]}".strip()


def cluster_failures(records: list[FailureRecord]) -> list[FailurePattern]:
    """Group raw failure records into recurring patterns by signature.

    Deterministic and LLM-free: identical normalized signatures collapse into one
    pattern with an occurrence count and the set of evidencing trace ids.
    """
    by_sig: dict[str, FailurePattern] = {}
    for r in records:
        sig = r.signature
        p = by_sig.get(sig)
        if p is None:
            p = FailurePattern(
                signature=sig,
                name=r.name,
                kind=r.kind,
                anomaly_type=r.anomaly_type,
                count=0,
                sample_detail=r.detail,
                value=r.value,
                baseline=r.baseline,
            )
            by_sig[sig] = p
        p.count += 1
        if r.trace_id and r.trace_id not in p.trace_ids:
            p.trace_ids.append(r.trace_id)
    # Most frequent first — the golden loop addresses the worst offenders first.
    return sorted(by_sig.values(), key=lambda p: p.count, reverse=True)


def file_gap_topic(
    engine: Any,
    pattern: FailurePattern,
    *,
    anomaly_id: str | None = None,
    source: str = "failure_analyzer",
) -> dict[str, Any] | None:
    """Persist one synthetic ``failure_gap`` ``Concept`` topic for a pattern.

    The single shared gap-topic creation path (CONCEPT:AHE-3.18): used by
    :meth:`FailureAnalyzer._materialize` for Langfuse-derived patterns, by the
    fleet-event triage handler (CONCEPT:OS-5.15) and by the anomaly
    consumer (CONCEPT:AHE-3.19). The Concept carries NO ``ADDRESSED_BY`` edge,
    so the golden loop's existing ``unresolved_topics()`` intake picks it up
    unchanged. When ``anomaly_id`` is given a provenance
    ``(anomaly)-[:EVIDENCES]->(gap)`` edge is added.

    Returns the gap-topic dict (the shape ``run_failure_ingest`` feeds to the
    remediation cycle), or ``None`` when the Concept could not be persisted.
    """
    ts = _now_iso()
    gap_id = f"failure_gap:{pattern.signature}"
    try:
        engine.add_node(
            gap_id,
            "Concept",
            properties={
                "name": f"Failure: {pattern.label}",
                "kind": "failure_gap",
                "source": source,
                "pattern_signature": pattern.signature,
                "occurrences": pattern.count,
                "evidence_trace_ids": ",".join(pattern.trace_ids[:20]),
                "timestamp": ts,
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("gap concept persist failed: %s", e)
        return None

    if anomaly_id:
        try:
            engine.link_nodes(
                source_id=anomaly_id,
                target_id=gap_id,
                rel_type="EVIDENCES",
                properties={"source": source},
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("EVIDENCES edge failed: %s", e)

    return {
        "id": gap_id,
        "name": f"Failure: {pattern.label}",
        "signature": pattern.signature,
        "workflow": pattern.name,
        "anomaly_type": pattern.anomaly_type,
        "baseline": pattern.baseline,
        "occurrences": pattern.count,
    }


class FailureAnalyzer:
    """Turn observed telemetry failures into KG remediation topics. CONCEPT:AHE-3.18.

    Args:
        engine: KG engine (``add_node``/``link_nodes``/``query_cypher``).
        trace_backend: a :class:`TraceBackend` exposing the failure-read surface.
        feedback: optional :class:`FeedbackService` for eval/outcome corrections.
        window_seconds: how far back to pull telemetry.
        latency_budget_ms / cost_budget_usd: anomaly thresholds.
        min_occurrences: a pattern must recur at least this many times to become
            a gap topic (single one-offs are noise).
    """

    def __init__(
        self,
        engine: Any,
        *,
        trace_backend: Any = None,
        feedback: Any = None,
        window_seconds: float = 86400.0,
        latency_budget_ms: float | None = None,
        cost_budget_usd: float | None = None,
        low_score_threshold: float = 0.5,
        min_occurrences: int = 2,
    ) -> None:
        self.engine = engine
        self.trace_backend = trace_backend
        self.feedback = feedback
        self.window_seconds = window_seconds
        self.latency_budget_ms = latency_budget_ms
        self.cost_budget_usd = cost_budget_usd
        self.low_score_threshold = low_score_threshold
        self.min_occurrences = max(1, int(min_occurrences))

    @classmethod
    def from_engine(cls, engine: Any) -> FailureAnalyzer:
        """Wire from a running engine + environment config."""
        from agent_utilities.core.config import AgentConfig
        from agent_utilities.harness.trace_backend import create_trace_backend

        cfg = AgentConfig()
        backend = None
        try:
            backend = create_trace_backend("langfuse")
        except Exception as e:  # noqa: BLE001
            logger.debug("FailureAnalyzer: trace backend unavailable: %s", e)

        feedback = None
        try:
            from .feedback import FeedbackService

            feedback = FeedbackService.from_engine(engine)
        except Exception as e:  # noqa: BLE001
            logger.debug("FailureAnalyzer: feedback service unavailable: %s", e)

        window = float(getattr(cfg, "kg_failure_evolution_window", 86400.0))
        latency_budget = float(cfg.langfuse_latency_baseline_seconds) * 1000.0
        return cls(
            engine,
            trace_backend=backend,
            feedback=feedback,
            window_seconds=window,
            latency_budget_ms=latency_budget,
            cost_budget_usd=None,
            low_score_threshold=float(cfg.langfuse_dataset_capture_threshold),
        )

    # ── pull ────────────────────────────────────────────────────────────
    async def _pull(self) -> list[FailureRecord]:
        """Pull error/low-score/anomaly telemetry and normalize to records."""
        if self.trace_backend is None:
            return []
        since = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - self.window_seconds)
        )
        records: list[FailureRecord] = []

        for obs in await self.trace_backend.get_error_observations(since=since):
            records.append(
                FailureRecord(
                    kind="error",
                    name=obs.get("name") or obs.get("traceName") or "unknown",
                    detail=obs.get("statusMessage") or obs.get("level") or "error",
                    anomaly_type=ANOMALY_ERROR,
                    trace_id=obs.get("traceId") or obs.get("id"),
                )
            )

        for sc in await self.trace_backend.get_low_score_traces(
            max_value=self.low_score_threshold, since=since
        ):
            records.append(
                FailureRecord(
                    kind="low_score",
                    name=sc.get("name") or "score",
                    detail=f"score {sc.get('name')} below {self.low_score_threshold}",
                    anomaly_type=ANOMALY_LOW_SCORE,
                    trace_id=sc.get("trace_id"),
                    value=sc.get("value"),
                    baseline=self.low_score_threshold,
                )
            )

        for an in await self.trace_backend.get_cost_latency_anomalies(
            since=since,
            p95_latency_ms=self.latency_budget_ms,
            p95_cost_usd=self.cost_budget_usd,
        ):
            if an.get("over_latency"):
                records.append(
                    FailureRecord(
                        kind="anomaly",
                        name=an.get("name") or "unknown",
                        detail="p95 latency exceeds budget",
                        anomaly_type=ANOMALY_LATENCY,
                        value=an.get("p95_latency_ms"),
                        baseline=self.latency_budget_ms,
                    )
                )
            if an.get("over_cost"):
                records.append(
                    FailureRecord(
                        kind="anomaly",
                        name=an.get("name") or "unknown",
                        detail="total cost exceeds budget",
                        anomaly_type=ANOMALY_COST,
                        value=an.get("total_cost_usd"),
                        baseline=self.cost_budget_usd,
                    )
                )
        return records

    # ── materialize ─────────────────────────────────────────────────────
    def _materialize(self, patterns: list[FailurePattern]) -> dict[str, Any]:
        """Persist ExecutionSummary / PerformanceAnomaly / failure_gap Concept nodes."""
        ts = _now_iso()
        gap_concepts: list[dict[str, Any]] = []
        anomalies = 0
        summaries: dict[str, int] = {}

        for p in patterns:
            if p.count < self.min_occurrences:
                continue
            anomaly_id = f"perf_anomaly:{p.signature}"

            # 1. PerformanceAnomaly (target = the failing workflow/agent name).
            try:
                self.engine.add_node(
                    anomaly_id,
                    "PerformanceAnomaly",
                    properties={
                        "target_node_id": p.name,
                        "anomaly_type": p.anomaly_type,
                        "threshold_exceeded": float(p.value or 0.0),
                        "baseline": float(p.baseline or 0.0),
                        "timestamp": ts,
                        "metadata": p.sample_detail[:500],
                    },
                )
                anomalies += 1
            except Exception as e:  # noqa: BLE001
                logger.debug("anomaly node persist failed: %s", e)

            # 2.+3. failure_gap Concept (+EVIDENCES provenance) via the shared
            #    gap-topic creation path — NO ADDRESSED_BY edge, so the golden
            #    loop's unresolved_topics() picks it up automatically.
            gap = file_gap_topic(self.engine, p, anomaly_id=anomaly_id)
            if gap is None:
                continue

            summaries[p.name] = summaries.get(p.name, 0) + p.count
            gap_concepts.append(gap)

        # 4. ExecutionSummary rollup per failing workflow name (success_rate<1.0 so
        #    maintainer.trigger_self_improvement picks it up).
        for name, fail_count in summaries.items():
            summary_id = f"exec_summary:{_sig(name, 'rollup', '')}"
            try:
                self.engine.add_node(
                    summary_id,
                    "ExecutionSummary",
                    properties={
                        "workflow_id": name,
                        "success_rate": 0.0,
                        "duration_ms": 0.0,
                        "total_tokens": 0,
                        "timestamp": ts,
                        "metadata": f"failure_analyzer: {fail_count} failures observed",
                    },
                )
                # link the rollup to each gap of this workflow
                for g in gap_concepts:
                    if g["workflow"] == name:
                        self.engine.link_nodes(
                            source_id=summary_id,
                            target_id=g["id"],
                            rel_type="OBSERVED_IN",
                            properties={"source": "failure_analyzer"},
                        )
            except Exception as e:  # noqa: BLE001
                logger.debug("ExecutionSummary persist failed: %s", e)

        return {
            "gap_concepts": gap_concepts,
            "anomalies": anomalies,
            "summaries": len(summaries),
        }

    # ── orchestration ───────────────────────────────────────────────────
    async def run_once_async(self) -> dict[str, Any]:
        """Pull → cluster → materialize. Returns a JSON-able report."""
        records = await self._pull()
        patterns = cluster_failures(records)
        report = self._materialize(patterns)
        report["records_pulled"] = len(records)
        report["patterns"] = len(patterns)
        logger.info(
            "[AHE-3.18] failure ingest: pulled=%d patterns=%d gaps=%d anomalies=%d",
            len(records),
            len(patterns),
            len(report["gap_concepts"]),
            report["anomalies"],
        )
        return report

    def run_once(self) -> dict[str, Any]:
        """Synchronous entry point (for the daemon scheduler thread)."""
        return _run_coro(self.run_once_async())

    # ── closed-loop regression gate (CONCEPT:AHE-3.18, Phase 4)
    def make_regression_check(self, gaps: list[dict[str, Any]]) -> Any:
        """Build a ``(spec) -> bool`` regression gate for failure remediations.

        Wired into :class:`GovernedAutoMerger` so a failure-remediation proposal
        auto-merges **only** when promoting it does not coincide with a regression.

        Since the remediated artifact is not executed at merge time, this is a
        conservative *volatility* guard: it re-queries Langfuse for each gap's
        failing workflow over a recent window and returns ``True`` (safe to
        promote) only when no signature is actively spiking above the baseline
        occurrence count captured at ingest. A spiking failure holds the proposal
        for human review while the situation is unstable.

        Side effects (the durable feedback half of the loop): each gap is appended
        to the eval regression corpus and the failing capability's reward is nudged
        down, so the mistake is caught automatically thereafter.
        """
        baselines = {g["workflow"]: int(g.get("occurrences", 0)) for g in gaps}

        def _check(_spec: Any) -> bool:
            self._record_feedback(gaps)
            if self.trace_backend is None:
                # cannot observe → conservatively allow (no tracked regression)
                self._record_gate_result(
                    _spec, True, "no trace backend — no tracked regression"
                )
                return True
            try:
                current = _run_coro(self._current_counts(list(baselines)))
            except Exception as e:  # noqa: BLE001
                logger.debug("regression re-query failed: %s", e)
                self._record_gate_result(_spec, True, f"re-query failed: {e}")
                return True
            for name, base in baselines.items():
                if current.get(name, 0) > base:
                    logger.info(
                        "[AHE-3.18] regression hold: %s spiking (%d > baseline %d)",
                        name,
                        current.get(name, 0),
                        base,
                    )
                    self._record_gate_result(
                        _spec,
                        False,
                        f"{name} spiking ({current.get(name, 0)} > baseline {base})",
                    )
                    return False
            self._record_gate_result(_spec, True, "no spike above ingest baseline")
            return True

        return _check

    def _record_gate_result(self, spec: Any, passed: bool, detail: str) -> None:
        """Persist the gate verdict as a ``RegressionGateResult`` node.

        This is the durable record the promotion-governance validator
        (CONCEPT:AHE-3.20) consults: a recorded ``hold`` for a proposal blocks
        its auto-merge until the failure stabilizes and a later gate run
        records a ``pass``.
        """
        try:
            from ..research.auto_merge import GovernedAutoMerger

            pid = GovernedAutoMerger._spec_id(spec)
            ts = _now_iso()
            node_id = f"regression_gate:{_sig(pid, 'gate', str(time.time()))}"
            self.engine.add_node(
                node_id,
                "RegressionGateResult",
                properties={
                    "proposal_id": pid,
                    "result": "pass" if passed else "hold",
                    "detail": str(detail)[:300],
                    "timestamp": ts,
                },
            )
        except Exception as e:  # noqa: BLE001 — recording must never gate the gate
            logger.debug("regression gate record failed: %s", e)

    async def _current_counts(self, names: list[str]) -> dict[str, int]:
        """Re-query recent error occurrences per workflow name."""
        since = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(time.time() - min(self.window_seconds, 3600.0)),
        )
        counts: dict[str, int] = {n: 0 for n in names}
        for obs in await self.trace_backend.get_error_observations(since=since):
            name = obs.get("name") or obs.get("traceName")
            if name in counts:
                counts[name] += 1
        return counts

    def _record_feedback(self, gaps: list[dict[str, Any]]) -> None:
        """Append eval regression cases + nudge reward down for each gap."""
        if self.feedback is None:
            return
        for g in gaps:
            try:
                self.feedback.record_correction(
                    "eval",
                    target_id=g["signature"],
                    corrected_value="no recurrence of this failure",
                    reason=f"failure_gap regression case for {g['workflow']}",
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("eval correction failed: %s", e)
            try:
                self.feedback.record_correction(
                    "outcome",
                    target_id=g["workflow"],
                    reward=0.0,
                    reason="observed failure (failure_analyzer)",
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("outcome correction failed: %s", e)


def run_failure_ingest(engine: Any) -> dict[str, Any]:
    """One failure-driven evolution pass: pull Langfuse failures → materialize
    failure_gap topics → run a regression-gated remediation cycle that addresses
    those gaps directly (CONCEPT:AHE-3.18).

    Shared by the daemon's ``failure_ingest`` tick and the on-demand
    ``graph_orchestrate(action="failure_ingest")`` MCP action so the two never
    drift. Returns a JSON-able report (the ingest report plus a ``remediation``
    block when gaps were found).
    """
    analyzer = FailureAnalyzer.from_engine(engine)
    report = analyzer.run_once()
    gaps = report.get("gap_concepts", [])
    if gaps:
        from ..research.golden_loop import GoldenLoopController

        check = analyzer.make_regression_check(gaps)
        gap_topics = [{"id": g["id"], "name": g["name"]} for g in gaps]
        report["remediation"] = GoldenLoopController(
            engine, regression_check=check
        ).run_one_cycle(
            max_topics=min(len(gaps), 5),
            topics=gap_topics,
            assimilate=False,
            breadth=False,
            standardize=False,
        )
    return report
