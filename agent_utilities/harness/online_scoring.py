#!/usr/bin/python
from __future__ import annotations

"""Online-scoring pipeline over live traces (CONCEPT:AU-AHE.harness.receives-trace-id-must).

Opik's keystone, KG-native: production **automation rules** AND regression **assertions**
run through ONE judge path. When a root trace completes, the
:class:`~agent_utilities.harness.trace_backend.KGTraceBackend` fires a fast hook that
defers scoring off the hot path; this sampler then:

1. applies a sample-rate + optional filter,
2. judges the trace's output against each registered automation rule (LLM-as-judge —
   the same live judge :class:`EvalRunner` uses) → ``OnlineScoreNode``,
3. judges any matching regression assertions (from the eval corpus) → ``AssertionResultNode``,
4. links every verdict ``SCORED_BY`` the trace, and on a FAILED assertion feeds it back
   into the eval corpus so the same break is caught from now on (the failing-trace →
   regression-test loop).

The judge runs OFF the traced call's hot path (a small thread pool), so a traced agent
run never pays scoring latency. The judge is the SAME code (`EvalRunner._assertion_judge`)
used for offline eval — prod monitoring and regression assertions share one path.
"""

import logging
import random
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.harness.continuous_evaluation_engine import EvalRunner
from agent_utilities.models.knowledge_graph import (
    AssertionResultNode,
    OnlineScoreNode,
    RegistryEdgeType,
)
from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

logger = logging.getLogger(__name__)


@dataclass
class AutomationRule:
    """A production online-scoring rule: judge every (sampled) trace's output against a
    plain-English ``criteria`` on a named ``dimension`` (e.g. hallucination, relevance)."""

    dimension: str
    criteria: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", self.dimension):
            raise ValueError("automation-rule dimension must be a bounded identifier")
        if not 1 <= len(self.criteria) <= 16_384:
            raise ValueError("automation-rule criteria exceeds the configured limit")


@dataclass
class Metric:
    """A user-defined Python metric (CONCEPT:AU-AHE.harness.onlinescorenode) run on every (sampled) trace inside
    a resource-bounded sandbox. ``source`` defines ``def metric(trace) -> float`` (0..1);
    ``trace`` is a dict view {input, output, status, spans, generations}."""

    name: str
    source: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", self.name):
            raise ValueError("metric name must be a bounded identifier")
        if not 1 <= len(self.source.encode("utf-8")) <= 64 * 1024:
            raise ValueError("metric source exceeds the configured limit")


async def _run_metric_isolated(source: str, trace: dict[str, Any]) -> float:
    """Execute one metric only across approved isolated RLM backends."""
    import json
    import math

    from agent_utilities.rlm.sandboxes.base import (
        SandboxEnv,
        SandboxRejected,
    )
    from agent_utilities.rlm.sandboxes.registry import default_sandboxes
    from agent_utilities.rlm.sandboxes.router import SandboxRouter
    from agent_utilities.rlm.telemetry import SandboxFatalError

    encoded_trace = json.dumps(trace, separators=(",", ":")).encode("utf-8")
    if len(encoded_trace) > 2 * 1024 * 1024:
        raise ValueError("metric trace view exceeds the configured limit")
    captured: dict[str, Any] = {}

    def final_var(name: str, value: Any) -> None:
        if name != "__metric_score__":
            raise ValueError("metric attempted to set an unknown output")
        captured[name] = value

    code = f"{source}\nFINAL_VAR('__metric_score__', metric(trace))"
    router = SandboxRouter(default_sandboxes())
    chain = router.select(code)
    env = SandboxEnv(vars={"trace": trace}, helpers={"FINAL_VAR": final_var})
    last_rejection = False
    for backend in chain:
        try:
            result = await backend.execute(code, env)
        except SandboxRejected:
            last_rejection = True
            continue
        if result.error:
            raise ValueError("metric execution failed")
        raw_score = captured.get("__metric_score__")
        if raw_score is None:
            raise ValueError("metric never set __metric_score__")
        value = float(raw_score)
        if not math.isfinite(value):
            raise ValueError("metric returned a non-finite score")
        return max(0.0, min(1.0, value))
    raise SandboxFatalError(
        "no approved isolated backend accepted the metric"
        if last_rejection
        else "no approved isolated metric backend is available"
    )


def _run_metric_sync(source: str, trace: dict[str, Any]) -> float:
    """Run the isolated async metric path from sync scoring code."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run_metric_isolated(source, trace))
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(
            lambda: asyncio.run(_run_metric_isolated(source, trace))
        ).result()


@dataclass
class OnlineScoringSampler:
    """Scores live traces through the shared LLM-judge path (CONCEPT:AU-AHE.harness.receives-trace-id-must)."""

    backend: (
        Any  # KGTraceBackend (provides get_trace + add_node/link_nodes via .backend)
    )
    rules: list[AutomationRule] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    eval_corpus: Any = None
    sample_rate: float = 1.0
    filter_fn: Callable[[Any], bool] | None = None
    judge: Callable[[str, str, str], tuple[float, str]] = EvalRunner._assertion_judge
    #: Auto-use the agentic tool-judge for traces over the size threshold (AHE-3.66).
    tool_judge: bool = True
    _pool: ThreadPoolExecutor | None = field(default=None, repr=False)
    _pending: threading.BoundedSemaphore = field(
        default_factory=lambda: threading.BoundedSemaphore(128), repr=False
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("online-scoring sample_rate must be between 0 and 1")

    def install(self) -> OnlineScoringSampler:
        """Wire this sampler to the backend's root-trace-complete hook (non-blocking)."""
        self._pool = self._pool or ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="online-score"
        )
        self.backend.on_trace_complete = self._schedule
        return self

    def _schedule(self, trace_id: str) -> None:
        """Fast hook: defer scoring to the pool so the traced call never blocks."""
        if self.sample_rate < 1.0 and random.random() > self.sample_rate:  # nosec B311 - sampling jitter, not cryptographic
            return
        pool = self._pool
        if pool is not None:
            if not self._pending.acquire(blocking=False):
                logger.warning("online scoring queue is full; dropping a sample")
                return

            def run_and_release() -> None:
                try:
                    self._safe_score(trace_id)
                finally:
                    self._pending.release()

            try:
                pool.submit(run_and_release)
            except Exception:
                self._pending.release()
                logger.debug("online scoring submission failed")
        else:  # no pool (test/no-install) — run inline
            self._safe_score(trace_id)

    def _safe_score(self, trace_id: str) -> None:
        try:
            self.score_trace(trace_id)
        except Exception as exc:  # pragma: no cover - scoring must never crash the host
            logger.debug("online score_trace failed (%s)", type(exc).__name__)

    def score_trace(self, trace_id: str) -> list[Any]:
        """Score one trace: automation rules + matching regression assertions.

        Returns the written ``OnlineScoreNode`` / ``AssertionResultNode`` objects.
        """
        entry = self.backend.get_trace(trace_id)
        if entry is None:
            return []
        privacy = PersistencePrivacyGuard()
        _, trace_id_report = privacy.sanitize_text(str(trace_id))
        if trace_id_report.changed:
            logger.debug("online scoring skipped an unsafe trace identifier")
            return []
        trace = entry["trace"]
        if self.filter_fn is not None and not self.filter_fn(trace):
            return []

        query = getattr(trace, "input", "") or getattr(trace, "name", "")
        actual = getattr(trace, "output", "") or ""
        written: list[Any] = []

        # Pick the judge ONCE per trace: a large trace navigates its span subgraph with
        # the tool-judge (CONCEPT:AU-AHE.harness.instead-context-stuffing-small) instead of context-stuffing; small traces use
        # the cheap inline judge. Both return (score, reasoning).
        from agent_utilities.harness.tool_judge import ToolEnabledJudge, should_use

        use_tools = self.tool_judge and should_use(entry)
        _tool = ToolEnabledJudge() if use_tools else None

        def judge_criteria(criteria: str) -> tuple[float, str]:
            if _tool is not None:
                return _tool.judge(entry, criteria)
            return self.judge(criteria, query, actual)

        # 1) Production automation rules → OnlineScoreNode (one judge path).
        for rule in self.rules:
            score, reasoning = judge_criteria(rule.criteria)
            node: Any = OnlineScoreNode(
                id=f"online_score:{trace_id}:{rule.dimension}",
                name=f"{rule.dimension} score",
                trace_id=trace_id,
                dimension=rule.dimension,
                score=score,
                reasoning=reasoning,
                evaluator="judge",
            )
            self._persist(node, trace_id)
            written.append(node)

        # 1b) Sandboxed user-defined Python metrics (CONCEPT:AU-AHE.harness.onlinescorenode) → OnlineScoreNode.
        for metric in self.metrics:
            score, reasoning = self._run_metric(metric, entry)
            node = OnlineScoreNode(
                id=f"online_score:{trace_id}:{metric.name}",
                name=f"{metric.name} metric",
                trace_id=trace_id,
                dimension=metric.name,
                score=score,
                reasoning=reasoning,
                evaluator=f"metric:{metric.name}",
            )
            self._persist(node, trace_id)
            written.append(node)

        # 2) Regression assertions (same judge) → AssertionResultNode; FAILED feeds back.
        for case in self._matching_cases(trace):
            assertion = getattr(case, "assertion", "") or getattr(
                case, "expected_output", ""
            )
            if not assertion:
                continue
            score, reasoning = judge_criteria(assertion)
            passed = score >= 0.5
            ar_node = AssertionResultNode(
                id=f"assertion_result:{trace_id}:{getattr(case, 'id', '')}",
                name="assertion result",
                trace_id=trace_id,
                case_id=getattr(case, "id", ""),
                assertion=assertion,
                status="passed" if passed else "failed",
                reasoning=reasoning,
            )
            self._persist(ar_node, trace_id)
            written.append(ar_node)
            if not passed and self.eval_corpus is not None:
                # The failing (prod) trace becomes/refreshes a regression case so the same
                # break is caught from now on (CONCEPT:AU-AHE.harness.receives-trace-id-must closes with AHE-3.61).
                try:
                    clean_feedback, _ = privacy.sanitize(
                        {
                            "query": query or trace_id,
                            "expected_output": actual,
                            "reason": f"online assertion failed: {assertion}",
                            "assertion": assertion,
                            "metadata": {"source_trace_id": trace_id},
                        }
                    )
                    self.eval_corpus.add_case(
                        query=clean_feedback["query"],
                        expected_output=clean_feedback["expected_output"],
                        tags=["regression", "online_failure"],
                        reason=clean_feedback["reason"],
                        assertion=clean_feedback["assertion"],
                        metadata=clean_feedback["metadata"],
                    )
                except Exception as exc:  # pragma: no cover
                    logger.debug("add_case feedback failed (%s)", type(exc).__name__)
        return written

    def _run_metric(self, metric: Metric, entry: dict[str, Any]) -> tuple[float, str]:
        """Run a user metric over a serialized trace view inside a bounded sandbox."""
        trace = entry["trace"]
        view = {
            "input": getattr(trace, "input", ""),
            "output": getattr(trace, "output", ""),
            "status": getattr(trace, "status", "ok"),
            "spans": [getattr(s, "name", "") for s in entry.get("spans", [])],
            "generations": [
                {
                    "model": getattr(g, "model", None),
                    "input_tokens": getattr(g, "input_tokens", 0),
                    "output_tokens": getattr(g, "output_tokens", 0),
                }
                for g in entry.get("generations", [])
            ],
        }
        try:
            score = _run_metric_sync(metric.source, view)
        except Exception as exc:  # noqa: BLE001 - scoring failure is contained
            return 0.0, f"metric:{metric.name} error:{type(exc).__name__}"
        return score, f"metric:{metric.name} ok"

    def _matching_cases(self, trace: Any) -> list[Any]:
        """Regression cases whose tags intersect the trace's tags (or untagged = all)."""
        if self.eval_corpus is None:
            return []
        try:
            cases = self.eval_corpus.load_cases()
        except Exception:  # pragma: no cover
            return []
        ttags = set(getattr(trace, "tags", []) or [])
        out = []
        for c in cases:
            ctags = set(getattr(c, "tags", []) or [])
            if not ctags or (ctags & ttags) or "regression" in ctags:
                out.append(c)
        return out

    def _persist(self, node: Any, trace_id: str) -> None:
        """Write the verdict node + a SCORED_BY edge to the durable backend."""
        be = getattr(self.backend, "backend", None)
        if be is None or not hasattr(be, "add_node"):
            return
        try:
            props = node.model_dump()
            props.pop("id", None)
            props["node_type"] = str(props.pop("type", ""))
            clean_props, _ = PersistencePrivacyGuard().sanitize(props)
            if not isinstance(clean_props, dict):
                return
            be.add_node(node.id, **clean_props)
            link = getattr(be, "link_nodes", None)
            if callable(link):
                link(trace_id, node.id, RegistryEdgeType.SCORED_BY)
        except Exception as exc:  # pragma: no cover - best-effort
            logger.debug("online-score persist failed (%s)", type(exc).__name__)


__all__ = ["AutomationRule", "Metric", "OnlineScoringSampler"]
