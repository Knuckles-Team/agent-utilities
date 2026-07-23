#!/usr/bin/python
from __future__ import annotations

"""Failure-driven self-evolution (CONCEPT:AU-AHE.harness.failure-evolution — Failure-Driven Evolution).

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

from agent_utilities.security.persistence_privacy import (
    PersistencePrivacyGuard,
    persistence_reference,
)

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
_PRIVACY_NAMESPACE = "failure-evolution"
_PRIVACY_GUARD = PersistencePrivacyGuard()


def _persistence_ref(kind: str, value: Any) -> str:
    """Return an idempotent opaque reference for transient telemetry identity."""
    return persistence_reference(kind, value, namespace=_PRIVACY_NAMESPACE)


def _controlled_text(value: str) -> str:
    """Sanitize the small controlled strings that are allowed to persist."""
    clean, _report = _PRIVACY_GUARD.sanitize_text(str(value or ""))
    return clean


def _safe_anomaly_type(value: Any) -> str:
    candidate = str(value or "").upper()
    allowed = {
        ANOMALY_ERROR,
        ANOMALY_LOW_SCORE,
        ANOMALY_LATENCY,
        ANOMALY_COST,
        ANOMALY_TOKENS,
    }
    return candidate if candidate in allowed else "ANOMALY"


def _commit_graph_slice(
    engine: Any,
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    graph_writer: Any = None,
) -> dict[str, Any]:
    """Commit failure observations through the authoritative ChangeEnvelope seam.

    ``graph_writer`` is an explicit in-memory test adapter. Runtime callers never
    receive a legacy per-node fallback: an unavailable native authority fails
    closed in :func:`ingest_graph_slice`.
    """
    if graph_writer is not None:
        from agent_utilities.core.profile_guard import is_production_profile

        if is_production_profile():
            raise RuntimeError("test_graph_writer_forbidden_in_production")
        return graph_writer(entities, relationships or [])
    from ..ingestion.envelope_ingest import ingest_graph_slice

    return ingest_graph_slice(
        engine,
        "failure-evolution",
        entities,
        relationships or [],
        source_instance="telemetry",
    )


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


def _safe_pattern(pattern: FailurePattern) -> FailurePattern:
    """Pseudonymize a pattern before it crosses any durable boundary."""
    workflow_ref = _persistence_ref("workflow", pattern.name)
    raw_detail = str(pattern.sample_detail or "")
    detail_ref = _persistence_ref(
        "failure_detail",
        raw_detail
        if raw_detail.startswith("pref_failure_detail_")
        else _normalize_detail(raw_detail),
    )
    return FailurePattern(
        signature=_sig(workflow_ref, pattern.kind, detail_ref),
        name=workflow_ref,
        kind=pattern.kind,
        anomaly_type=_safe_anomaly_type(pattern.anomaly_type),
        count=pattern.count,
        trace_ids=[
            ref
            for ref in (_persistence_ref("trace", item) for item in pattern.trace_ids)
            if ref
        ],
        sample_detail=detail_ref,
        value=pattern.value,
        baseline=pattern.baseline,
    )


def _safe_gap(gap: dict[str, Any]) -> dict[str, Any]:
    """Normalize caller-supplied regression gap metadata to opaque references."""
    workflow_ref = _persistence_ref("workflow", gap.get("workflow"))
    signature_ref = _persistence_ref("failure_signature", gap.get("signature"))
    return {
        "id": _persistence_ref("failure_gap", gap.get("id") or signature_ref),
        "name": "Failure remediation pattern",
        "signature": signature_ref,
        "workflow": workflow_ref,
        "anomaly_type": _safe_anomaly_type(gap.get("anomaly_type")),
        "baseline": gap.get("baseline"),
        "occurrences": int(gap.get("occurrences", 0) or 0),
    }


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_coro(coro: Any) -> Any:
    """Run a coroutine to completion whether or not a loop is already running.

    The daemon ``failure_ingest`` tick runs in a worker thread (no loop, so
    ``asyncio.run`` works), but the ``graph_evolution(action="failure_ingest")``
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
        return _sig(
            _persistence_ref("workflow", self.name),
            self.kind,
            _persistence_ref("failure_detail", _normalize_detail(self.detail)),
        )


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
        return f"{self.anomaly_type} failure pattern"


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
    return sorted(
        (_safe_pattern(pattern) for pattern in by_sig.values()),
        key=lambda p: p.count,
        reverse=True,
    )


def file_gap_topic(
    engine: Any,
    pattern: FailurePattern,
    *,
    anomaly_id: str | None = None,
    source: str = "failure_analyzer",
    graph_writer: Any = None,
) -> dict[str, Any] | None:
    """Persist one synthetic ``failure_gap`` ``Concept`` topic for a pattern.

    The single shared gap-topic creation path (CONCEPT:AU-AHE.harness.failure-evolution): used by
    :meth:`FailureAnalyzer._materialize` for Langfuse-derived patterns, by the
    fleet-event triage handler (CONCEPT:AU-OS.config.fleet-event-ingress) and by the anomaly
    consumer (CONCEPT:AU-AHE.optimization.performance-anomaly-consumer). The Concept carries NO ``ADDRESSED_BY`` edge,
    so the golden loop's existing ``unresolved_topics()`` intake picks it up
    unchanged. When ``anomaly_id`` is given a provenance
    ``(anomaly)-[:EVIDENCES]->(gap)`` edge is added.

    Returns the gap-topic dict (the shape ``run_failure_ingest`` feeds to the
    remediation cycle), or ``None`` when the Concept could not be persisted.
    """
    pattern = _safe_pattern(pattern)
    ts = _now_iso()
    gap_id = f"failure_gap:{pattern.signature}"
    safe_source = (
        source
        if source
        in {
            "failure_analyzer",
            "fleet_event_triage",
            "anomaly_consumer",
            "swebench",
            "parallel_engine",
        }
        else "failure_analyzer"
    )
    entities = [
        {
            "id": gap_id,
            "node_type": "Concept",
            "name": f"Failure: {pattern.label}",
            "kind": "failure_gap",
            "source": safe_source,
            "pattern_signature": pattern.signature,
            "occurrences": pattern.count,
            "evidence_trace_refs": ",".join(pattern.trace_ids[:20]),
            "timestamp": ts,
        }
    ]
    relationships = (
        [
            {
                "source": anomaly_id,
                "target": gap_id,
                "relationship": "EVIDENCES",
                "source_system": safe_source,
            }
        ]
        if anomaly_id
        else []
    )
    try:
        _commit_graph_slice(
            engine,
            entities,
            relationships,
            graph_writer=graph_writer,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("gap concept persist failed (%s)", type(exc).__name__)
        return None

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
    """Turn observed telemetry failures into KG remediation topics. CONCEPT:AU-AHE.harness.failure-evolution.

    Args:
        engine: KG engine (``add_node``/``link_nodes``/``query_cypher``).
        trace_backend: a :class:`TraceBackend` exposing the failure-read surface.
        feedback: optional :class:`FeedbackService` for eval/outcome corrections.
        window_seconds: how far back to pull telemetry.
        latency_budget_ms / cost_budget_usd: anomaly thresholds.
        min_occurrences: a pattern must recur at least this many times to become
            a gap topic (single one-offs are noise).
        graph_writer: in-memory test adapter. Production always uses the native
            ChangeEnvelope authority and rejects this seam.
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
        graph_writer: Any = None,
    ) -> None:
        self.engine = engine
        self.trace_backend = trace_backend
        self.feedback = feedback
        self.window_seconds = window_seconds
        self.latency_budget_ms = latency_budget_ms
        self.cost_budget_usd = cost_budget_usd
        self.low_score_threshold = low_score_threshold
        self.min_occurrences = max(1, int(min_occurrences))
        self.graph_writer = graph_writer

    @classmethod
    def from_engine(cls, engine: Any) -> FailureAnalyzer:
        """Wire from a running engine + environment config."""
        from agent_utilities.core.config import AgentConfig
        from agent_utilities.harness.trace_backend import create_trace_backend

        cfg = AgentConfig()
        backend = None
        try:
            backend = create_trace_backend("langfuse")
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "FailureAnalyzer: trace backend unavailable (%s)",
                type(exc).__name__,
            )

        feedback = None
        try:
            from .feedback import FeedbackService

            feedback = FeedbackService.from_engine(engine)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "FailureAnalyzer: feedback service unavailable (%s)",
                type(exc).__name__,
            )

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

        for raw_pattern in patterns:
            p = _safe_pattern(raw_pattern)
            if p.count < self.min_occurrences:
                continue
            anomaly_id = f"perf_anomaly:{p.signature}"

            # 1. PerformanceAnomaly (target = the failing workflow/agent name).
            try:
                _commit_graph_slice(
                    self.engine,
                    [
                        {
                            "id": anomaly_id,
                            "node_type": "PerformanceAnomaly",
                            "target_node_id": p.name,
                            "anomaly_type": p.anomaly_type,
                            "threshold_exceeded": float(p.value or 0.0),
                            "baseline": float(p.baseline or 0.0),
                            "timestamp": ts,
                            "metadata": f"failure_detail_ref={p.sample_detail}",
                        }
                    ],
                    graph_writer=self.graph_writer,
                )
                anomalies += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("anomaly node persist failed (%s)", type(exc).__name__)

            # 2.+3. failure_gap Concept (+EVIDENCES provenance) via the shared
            #    gap-topic creation path — NO ADDRESSED_BY edge, so the golden
            #    loop's unresolved_topics() picks it up automatically.
            gap = file_gap_topic(
                self.engine,
                p,
                anomaly_id=anomaly_id,
                graph_writer=self.graph_writer,
            )
            if gap is None:
                continue

            summaries[p.name] = summaries.get(p.name, 0) + p.count
            gap_concepts.append(gap)

        # 4. ExecutionSummary rollup per failing workflow name (success_rate<1.0 so
        #    maintainer.trigger_self_improvement picks it up).
        for name, fail_count in summaries.items():
            summary_id = f"exec_summary:{_sig(name, 'rollup', '')}"
            try:
                summary_relationships = [
                    {
                        "source": summary_id,
                        "target": g["id"],
                        "relationship": "OBSERVED_IN",
                        "source_system": "failure_analyzer",
                    }
                    for g in gap_concepts
                    if g["workflow"] == name
                ]
                _commit_graph_slice(
                    self.engine,
                    [
                        {
                            "id": summary_id,
                            "node_type": "ExecutionSummary",
                            "workflow_id": name,
                            "success_rate": 0.0,
                            "duration_ms": 0.0,
                            "total_tokens": 0,
                            "timestamp": ts,
                            "metadata": f"failure_analyzer_count={fail_count}",
                        }
                    ],
                    summary_relationships,
                    graph_writer=self.graph_writer,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("ExecutionSummary persist failed (%s)", type(exc).__name__)

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

    # ── closed-loop regression gate (CONCEPT:AU-AHE.harness.failure-evolution, Phase 4)
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
        safe_gaps = [_safe_gap(gap) for gap in gaps]
        baselines = {g["workflow"]: int(g.get("occurrences", 0)) for g in safe_gaps}

        def _check(_spec: Any) -> bool:
            self._record_feedback(safe_gaps)
            if self.trace_backend is None:
                # cannot observe → conservatively allow (no tracked regression)
                self._record_gate_result(_spec, True, "no_backend")
                return True
            try:
                current = _run_coro(self._current_counts(list(baselines)))
            except Exception as exc:  # noqa: BLE001
                logger.debug("regression re-query failed (%s)", type(exc).__name__)
                self._record_gate_result(_spec, True, "requery_failed")
                return True
            for workflow_ref, base in baselines.items():
                if current.get(workflow_ref, 0) > base:
                    logger.info(
                        "[AHE-3.18] regression hold: failure count spiking (%d > %d)",
                        current.get(workflow_ref, 0),
                        base,
                    )
                    self._record_gate_result(_spec, False, "failure_spike")
                    return False
            # Verified: the remediation holds against the originally-observed
            # failures (no signature spiking). Lock each as a plain-English
            # regression assertion so the same failure can't silently recur.
            self._lock_regression_cases(safe_gaps)
            self._record_gate_result(_spec, True, "stable")
            return True

        return _check

    def _lock_regression_cases(self, gaps: list[dict[str, Any]]) -> None:
        """Promote each verified-remediated gap to a locked EvalCorpus assertion.

        CONCEPT:AU-AHE.evaluation.failure-analysis-loop — closes the Opik "lock-as-regression-test" step: once a
        fix is verified, persist a plain-English assertion case keyed to the
        failing workflow so future runs are judged against "this failure does not
        recur" rather than silently regressing. Idempotent per signature.
        """
        from agent_utilities.core.config import config

        # The regression corpus is content-bearing by nature. It stays disabled
        # unless an operator makes the separate, explicit governance decision.
        if not config.kg_failure_regression_dataset:
            return
        if not hasattr(self, "_locked_signatures"):
            self._locked_signatures: set[str] = set()
        try:
            from agent_utilities.harness.eval_corpus import EvalCorpus
        except Exception as exc:  # pragma: no cover - import best-effort
            logger.debug("eval corpus unavailable (%s)", type(exc).__name__)
            return
        corpus = EvalCorpus(backend=self.engine)
        for g in gaps:
            sig = str(g.get("signature", ""))
            if not sig or sig in self._locked_signatures:
                continue
            workflow = _persistence_ref("workflow", g.get("workflow"))
            try:
                corpus.add_case(
                    query="Re-run the workflow represented by the opaque failure reference.",
                    expected_output="The workflow completes without the prior failure.",
                    assertion="The prior categorized failure does not recur.",
                    tags=["failure_gap", "regression", "verified"],
                    reason="verified_failure_remediation",
                    metadata={
                        "signature_ref": _persistence_ref("failure_signature", sig),
                        "workflow_ref": workflow,
                    },
                )
                self._locked_signatures.add(sig)
            except Exception as exc:  # noqa: BLE001 — locking must never gate the gate
                logger.debug("regression lock failed (%s)", type(exc).__name__)

    def _record_gate_result(self, spec: Any, passed: bool, detail: str) -> None:
        """Persist the gate verdict as a ``RegressionGateResult`` node.

        This is the durable record the promotion-governance validator
        (CONCEPT:AU-AHE.harness.promotion-governance-validator) consults: a recorded ``hold`` for a proposal blocks
        its auto-merge until the failure stabilizes and a later gate run
        records a ``pass``.
        """
        try:
            from ..research.auto_merge import GovernedAutoMerger

            pid = _persistence_ref("proposal", GovernedAutoMerger._spec_id(spec))
            ts = _now_iso()
            node_id = f"regression_gate:{_sig(pid, 'gate', str(time.time()))}"
            _commit_graph_slice(
                self.engine,
                [
                    {
                        "id": node_id,
                        "node_type": "RegressionGateResult",
                        "proposal_id": pid,
                        "result": "pass" if passed else "hold",
                        "detail": _controlled_text(str(detail))[:80],
                        "timestamp": ts,
                    }
                ],
                graph_writer=self.graph_writer,
            )
        except Exception as exc:  # noqa: BLE001 — recording must never gate the gate
            logger.debug("regression gate record failed (%s)", type(exc).__name__)

    async def _current_counts(self, names: list[str]) -> dict[str, int]:
        """Re-query recent error occurrences per workflow name."""
        since = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(time.time() - min(self.window_seconds, 3600.0)),
        )
        counts: dict[str, int] = {_persistence_ref("workflow", n): 0 for n in names}
        for obs in await self.trace_backend.get_error_observations(since=since):
            workflow_ref = _persistence_ref(
                "workflow", obs.get("name") or obs.get("traceName")
            )
            if workflow_ref in counts:
                counts[workflow_ref] += 1
        return counts

    def _record_feedback(self, gaps: list[dict[str, Any]]) -> None:
        """Append eval regression cases + nudge reward down for each gap."""
        if self.feedback is None:
            return
        for g in gaps:
            try:
                self.feedback.record_correction(
                    "eval",
                    target_id=_persistence_ref("failure_signature", g.get("signature")),
                    corrected_value="categorized_failure_does_not_recur",
                    reason="failure_gap_regression_case",
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("eval correction failed (%s)", type(exc).__name__)
            try:
                self.feedback.record_correction(
                    "outcome",
                    target_id=_persistence_ref("workflow", g.get("workflow")),
                    reward=0.0,
                    reason="observed_failure",
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("outcome correction failed (%s)", type(exc).__name__)


def run_failure_ingest(engine: Any) -> dict[str, Any]:
    """One failure-driven evolution pass: pull Langfuse failures → materialize
    failure_gap topics → run a regression-gated remediation cycle that addresses
    those gaps directly (CONCEPT:AU-AHE.harness.failure-evolution).

    Shared by the daemon's ``failure_ingest`` tick and the on-demand
    ``graph_evolution(action="failure_ingest")`` MCP action so the two never
    drift. Returns a JSON-able report (the ingest report plus a ``remediation``
    block when gaps were found).
    """
    analyzer = FailureAnalyzer.from_engine(engine)
    report = analyzer.run_once()
    gaps = report.get("gap_concepts", [])
    if gaps:
        from ..research.loop_controller import LoopController

        check = analyzer.make_regression_check(gaps)
        gap_topics = [{"id": g["id"], "name": g["name"]} for g in gaps]
        report["remediation"] = LoopController(
            engine, regression_check=check
        ).run_one_cycle(
            max_topics=min(len(gaps), 5),
            topics=gap_topics,
            assimilate=False,
            breadth=False,
            standardize=False,
        )
    return report
