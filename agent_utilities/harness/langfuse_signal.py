#!/usr/bin/python
from __future__ import annotations

"""Langfuse-as-evolution-signal adapter (CONCEPT:AU-AHE.reward.langfuse-evolution-signal).

Our deployed Langfuse instance is not a destination the KG happens to export to —
it is production ground truth: real agent runs, real human/automated SCORES on
those runs, and curated DATASETS of known-good/known-bad tasks. This module turns
that ground truth into the inputs the existing AHE reward/reflection primitives
already consume, the same way DSPy treats traces + metrics as its optimization
signal:

- :func:`fetch_reward` — Langfuse SCORES → one normalized ``[0, 1]`` reward, for
  :meth:`~agent_utilities.knowledge_graph.research.claim_flywheel.ClaimFlywheel.record_outcome`
  (and any other durable-bandit consumer of
  :func:`~agent_utilities.graph.routing.enrichers.capability_designation.record_capability_outcome`).
- :func:`fetch_dataset_tasks` — a Langfuse DATASET → benchmark task rows a rollout
  can execute, via :meth:`~agent_utilities.harness.eval_corpus.EvalCorpus.add_from_langfuse_dataset`.
- :func:`blend_reward` — combine an existing internal reward with a Langfuse-sourced
  one (weighted average; degrades to the internal reward alone when Langfuse has no
  signal yet).

Failure-driven Reflect material (Langfuse TRACES → the golden loop's failure
intake) already exists and is NOT duplicated here — see
:mod:`agent_utilities.knowledge_graph.adaptation.failure_analyzer` (``run_failure_ingest``,
scheduled by ``engine_tasks._tick_failure_ingest``, opt-in via ``KG_FAILURE_EVOLUTION``).

Reuses the existing :class:`~agent_utilities.harness.trace_backend.LangfuseTraceBackend`
(itself a thin wrapper over the ``langfuse-agent`` package's ``LangfuseApi`` client) —
this module never talks to Langfuse's HTTP API directly and never re-implements a
client. Everything here is best-effort: Langfuse being unconfigured, unreachable, or
the optional ``langfuse`` extra not being installed degrades to ``None``/``[]``,
never raises into a caller's reward, promotion, or rollout path.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent_utilities.core.config import config

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_reward",
    "fetch_dataset_tasks",
    "fetch_low_score_traces",
    "blend_reward",
]


def _run_sync(coro: Any) -> Any:
    """Bridge an async trace-backend call into sync callers (e.g. ``ClaimFlywheel``).

    Mirrors ``online_scoring._run_metric_sync``'s loop-detection pattern: run
    directly when no loop is active, else hand the coroutine to a one-shot thread
    so a sync caller inside an already-running event loop never deadlocks.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def _langfuse_backend() -> Any:
    """The shared ``LangfuseTraceBackend``, or ``None`` when unconfigured/unavailable."""
    if not config.langfuse_secret_key_ref:
        return None
    try:
        from agent_utilities.harness.trace_backend import (
            LangfuseTraceBackend,
            create_trace_backend,
        )

        backend = create_trace_backend(backend_type="langfuse")
        return backend if isinstance(backend, LangfuseTraceBackend) else None
    except Exception as exc:  # noqa: BLE001 — optional dependency / unconfigured
        logger.debug("Langfuse trace backend unavailable (%s).", type(exc).__name__)
        return None


def fetch_reward(*, trace_id: str, name: str | None = None) -> float | None:
    """One trace's Langfuse SCORES, normalized to ``[0, 1]`` — or ``None`` (no signal yet).

    The synchronous entry point :meth:`~.claim_flywheel.ClaimFlywheel.record_outcome`
    and other sync reward consumers call. See
    :meth:`~agent_utilities.harness.trace_backend.LangfuseTraceBackend.get_trace_reward`
    for the normalization rule.
    """
    backend = _langfuse_backend()
    if backend is None:
        return None
    try:
        return _run_sync(backend.get_trace_reward(trace_id, name=name))
    except Exception as exc:  # noqa: BLE001 — reward fetch is best-effort
        logger.debug("Langfuse reward fetch failed (%s).", type(exc).__name__)
        return None


def fetch_dataset_tasks(dataset_name: str, *, limit: int = 50) -> list[dict[str, Any]]:
    """A Langfuse dataset's items as ``[{"id", "input", "expected_output"}, ...]``.

    ``[]`` when Langfuse is unconfigured, unreachable, or the dataset is empty —
    never raises, so a caller can unconditionally fold this into a rollout corpus.
    """
    backend = _langfuse_backend()
    if backend is None:
        return []
    try:
        return _run_sync(backend.list_dataset_items(dataset_name, limit=limit))
    except Exception as exc:  # noqa: BLE001 — dataset fetch is best-effort
        logger.debug("Langfuse dataset fetch failed (%s).", type(exc).__name__)
        return []


def fetch_low_score_traces(
    *, score_name: str | None = None, max_value: float = 0.5, limit: int = 50
) -> list[dict[str, Any]]:
    """Langfuse SCORES below ``max_value``, normalized to trace references.

    Reflect-stage (②) failure material: prior production episodes Langfuse has
    already scored poorly, via
    :meth:`~agent_utilities.harness.trace_backend.LangfuseTraceBackend.get_low_score_traces`
    — the SAME read
    :class:`~agent_utilities.knowledge_graph.adaptation.failure_analyzer.FailureAnalyzer`
    already uses for its own Reflect intake, reused here rather than re-implemented.
    ``[]`` when Langfuse is unconfigured, unreachable, or nothing is below
    threshold — never raises, so a caller can unconditionally fold this into a
    Reflect pass.
    """
    backend = _langfuse_backend()
    if backend is None:
        return []
    try:
        return _run_sync(
            backend.get_low_score_traces(
                score_name=score_name, max_value=max_value, limit=limit
            )
        )
    except Exception as exc:  # noqa: BLE001 — trace fetch is best-effort
        logger.debug("Langfuse low-score trace fetch failed (%s).", type(exc).__name__)
        return []


def blend_reward(
    internal_reward: float, langfuse_reward: float | None, *, weight: float = 0.5
) -> float:
    """Weighted-average an internal reward with a Langfuse-sourced one.

    ``weight`` is the Langfuse signal's share (default an even split). Returns
    ``internal_reward`` unchanged when ``langfuse_reward`` is ``None`` (no
    production signal yet) — the zero-overhead, zero-behavior-change default.
    """
    if langfuse_reward is None:
        return internal_reward
    w = max(0.0, min(1.0, float(weight)))
    blended = (1.0 - w) * internal_reward + w * langfuse_reward
    return max(0.0, min(1.0, blended))
