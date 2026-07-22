#!/usr/bin/python
from __future__ import annotations

"""Gateway self-telemetry → the shared health-intelligence kernels.

CONCEPT:AU-OS.observability.unified-health-kernels (gateway producer) /
AU-KG.identity.evidence-spine-convergence (``MetricWindow`` locus).

Wires two previously-dead pieces off ONE real, already-computed signal — the
per-request latency :class:`~agent_utilities.observability.gateway_metrics.
GatewayMetricsMiddleware` already measures for the Prometheus histogram, then
discards from AU's own reasoning once it is handed to ``.observe()``:

1. ``observability.health``'s anomaly-detection kernel (:func:`~.health.
   compute_baseline` / :func:`~.health.detect_anomaly`) plus
   :func:`~agent_utilities.observability.health_ingest.ingest_health_anomaly`
   had **zero live callers anywhere in AU** — a fully-built, generalized
   kernel with no producer wiring it. This module is that live caller.
2. ``MediaStore.store_metric_window_evidence`` (``eg_modality::EvidenceSpan::
   MetricWindow``) had no producer — an anomaly's own supporting trend window
   *is* the evidence for the claim "the gateway was anomalously slow here".

Bounded by design — NOT one evidence write per request (that would be the
SAME hot-path volume mismatch the seam audit declined for ``CodeSymbol``/
``TraceSpan``). Request durations are distilled into ONE trend row per
``_WINDOW_S`` window (:class:`~.health.HealthTrendBuffer`, the same
distill-don't-dump cadence every other health-kernel consumer uses), and a
KG write only fires when :func:`~.health.detect_anomaly` actually flags that
window against the gateway's own rolling baseline — a rare, meaningful event,
never a per-request write. Entirely best-effort: any failure here must never
affect the request/response path it's decoupled from (fire-and-forget via
``asyncio.create_task``, matching the *Native by default* "just happens"
wiring convention).
"""

import asyncio
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# 5-minute distilled trend window — matches the health kernel's own
# fan-manager-derived default cadence intent (frequent enough to catch a
# real regression, coarse enough to keep the KG write rate bounded).
_WINDOW_S = 300
# Bounded rolling baseline history (~4h at the default window) — a module
# constant, not a flag: there is no per-deployment value that should differ.
_HISTORY_MAX = 50

_ENTITY_ID = "gateway"
_SIGNAL = "request_duration_seconds"

_buffer: Any = None
_history: list[dict[str, Any]] = []
# Strong references to in-flight background checks — asyncio only holds a
# WEAK reference to a task via the event loop, so a task with no other
# referent can be garbage-collected mid-run. Standard fire-and-forget idiom:
# keep it here, discard on completion (`add_done_callback`).
_background_tasks: set[Any] = set()


def _get_buffer() -> Any:
    global _buffer
    if _buffer is None:
        from agent_utilities.observability.health import HealthTrendBuffer

        _buffer = HealthTrendBuffer(window_s=_WINDOW_S)
    return _buffer


def record_request_duration(duration_s: float) -> None:
    """Feed one request's latency into the gateway's health-trend buffer.

    Called from ``GatewayMetricsMiddleware`` with the SAME ``duration`` value
    it already computed for ``GATEWAY_REQUEST_DURATION.observe()`` — no new
    instrumentation, just a second, cheap (in-memory, no I/O) consumer of a
    value that already exists. Only rarely (once per ``_WINDOW_S`` window)
    does this produce a flushed trend; when it does, the anomaly-check + KG
    write is dispatched onto a background task so it never adds latency to
    the response the caller is still sending.
    """
    try:
        trend = _get_buffer().add(float(duration_s), at=time.time())
    except Exception as e:  # noqa: BLE001 — must never affect the response path
        logger.debug("gateway health sample skipped: %s", e)
        return
    if trend is not None:
        try:
            task = asyncio.create_task(_check_and_record(trend))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
        except RuntimeError:
            # No running loop (e.g. a sync caller/test) — drop the check
            # rather than block; the next window's flush will try again.
            pass


async def _check_and_record(trend: dict[str, Any]) -> None:
    """Best-effort: baseline + anomaly-detect the just-flushed window against
    the gateway's rolling trend history; on a genuine anomaly, write BOTH the
    ``:HealthAnomaly`` node (the dead kernel's first live caller) and a
    ``MetricWindow`` evidence locus (the seam's producer) carrying the exact
    triggering window. Never raises.
    """
    try:
        from agent_utilities.observability import health as h

        history = _history
        baseline = (
            h.compute_baseline(history, value_key="avg", peak_key="max")
            if len(history) >= h.DEFAULT_MIN_WINDOWS
            else None
        )
        anomaly = (
            h.detect_anomaly([trend], baseline, value_key="avg")
            if baseline is not None
            else None
        )
        history.append(trend)
        del history[:-_HISTORY_MAX]
        if anomaly is None:
            return

        from agent_utilities.observability import health_ingest

        await asyncio.to_thread(
            health_ingest.ingest_health_anomaly,
            _ENTITY_ID,
            _SIGNAL,
            anomaly,
            entity_type="Gateway",
        )

        from agent_utilities.knowledge_graph.memory.native_ingest import media_store

        store = media_store()
        data = json.dumps(trend, sort_keys=True).encode("utf-8")
        start_ms = int(float(trend.get("start_at") or 0.0) * 1000)
        end_ms = int(float(trend.get("end_at") or 0.0) * 1000)
        await asyncio.to_thread(
            store.store_metric_window_evidence,
            data,
            metric=f"{_ENTITY_ID}:{_SIGNAL}",
            start_ms=start_ms,
            end_ms=end_ms,
            mime_type="application/json",
            source="gateway-health",
        )
    except Exception as e:  # noqa: BLE001 — best-effort, never affects the request path
        logger.debug("gateway health anomaly check failed: %s", e)
