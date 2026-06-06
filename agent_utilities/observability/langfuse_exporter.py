#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:ECO-4.24 — Langfuse exporter for spans, token usage and traces.

The orchestration engine already opens an OpenTelemetry span around every graph
run (``orchestration/engine.py`` → ``tracer.start_as_current_span``) and the
4-bucket :class:`~agent_utilities.observability.token_tracker.TokenUsageRecord`
captures token usage. What was missing is a *sink*: a Langfuse exporter that
ships those spans/traces/token-usage to a Langfuse project for LLM observability.

This module provides that sink with **zero hard dependency**: the ``langfuse``
package is imported lazily, and the exporter only activates when
``LANGFUSE_PUBLIC_KEY`` + ``LANGFUSE_SECRET_KEY`` are set (default-on when keys
are present — wire-first, not opt-in). When the keys or the package are absent,
every method is a clean no-op so the live graph path is never affected.

Wiring: :func:`get_langfuse_exporter` returns a process-wide singleton; the
orchestration engine calls :meth:`LangfuseExporter.export_graph_run` from inside
its existing tracing block on every run when the exporter is enabled.
"""

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


class LangfuseExporter:
    """Optional Langfuse sink for graph-run traces + token usage. CONCEPT:ECO-4.24.

    Parameters
    ----------
    client:
        An injected Langfuse client (used by tests / advanced wiring). When
        ``None`` the exporter lazily constructs a real ``langfuse.Langfuse``
        from the ``LANGFUSE_*`` environment variables.
    public_key, secret_key, host:
        Explicit credentials; default to the ``LANGFUSE_PUBLIC_KEY`` /
        ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_BASE_URL`` env vars.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ) -> None:
        self._public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self._secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY", "")
        self._host = (
            host or os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST", "")
        )
        self._client = client
        self._probed = client is not None
        # Counters expose activity for health widgets + tests.
        self.exported_traces = 0
        self.exported_observations = 0

    # ── lifecycle ──────────────────────────────────────────────────────
    @property
    def configured(self) -> bool:
        """True when credentials are present (or a client was injected)."""
        return bool(self._client) or bool(self._public_key and self._secret_key)

    def _ensure_client(self) -> Any:
        """Lazily build the real Langfuse client; cache None if unavailable."""
        if self._client is not None:
            return self._client
        if self._probed:
            return None
        self._probed = True
        if not (self._public_key and self._secret_key):
            return None
        try:
            from langfuse import Langfuse  # type: ignore[import-not-found]

            kwargs: dict[str, Any] = {
                "public_key": self._public_key,
                "secret_key": self._secret_key,
            }
            if self._host:
                kwargs["host"] = self._host
            self._client = Langfuse(**kwargs)
            logger.info("Langfuse exporter active (host=%s)", self._host or "default")
        except Exception as exc:  # noqa: BLE001 — optional dep / offline
            logger.debug("Langfuse client unavailable, exporter no-ops: %s", exc)
            self._client = None
        return self._client

    @property
    def enabled(self) -> bool:
        """True when an actual client can be obtained (keys + dep present)."""
        return self._ensure_client() is not None

    # ── export API ─────────────────────────────────────────────────────
    def export_graph_run(
        self,
        *,
        run_id: str,
        query: str = "",
        status: str = "success",
        duration_ms: float = 0.0,
        token_usage: dict[str, int] | None = None,
        model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Export one graph run as a Langfuse trace. CONCEPT:ECO-4.24.

        Emits a ``trace`` (the graph run) with a nested ``generation``
        observation carrying token usage when present. Returns ``True`` when a
        trace was sent, ``False`` when the exporter is inactive (no keys/dep) —
        in which case it is a clean no-op.
        """
        client = self._ensure_client()
        if client is None:
            return False
        try:
            md = dict(metadata or {})
            md.update({"status": status, "duration_ms": duration_ms})
            trace = self._make_trace(client, run_id=run_id, query=query, metadata=md)
            self.exported_traces += 1
            if token_usage:
                self._make_generation(
                    client,
                    trace=trace,
                    run_id=run_id,
                    model=model,
                    token_usage=token_usage,
                )
                self.exported_observations += 1
            return True
        except Exception as exc:  # noqa: BLE001 — observability must never crash a run
            logger.debug("Langfuse export skipped for %s: %s", run_id, exc)
            return False

    def _make_trace(
        self, client: Any, *, run_id: str, query: str, metadata: dict[str, Any]
    ) -> Any:
        """Create a Langfuse trace via whichever API the client version exposes."""
        # Langfuse v2: client.trace(...); v3: client.start_span / context API.
        if hasattr(client, "trace"):
            return client.trace(
                id=run_id,
                name=f"graph_run:{run_id}",
                input=query[:2000],
                metadata=metadata,
            )
        if hasattr(client, "start_span"):
            return client.start_span(
                name=f"graph_run:{run_id}", input=query[:2000], metadata=metadata
            )
        return None

    def _make_generation(
        self,
        client: Any,
        *,
        trace: Any,
        run_id: str,
        model: str,
        token_usage: dict[str, int],
    ) -> None:
        """Attach a token-usage generation observation to ``trace``."""
        usage = {
            "input": int(token_usage.get("prompt", 0)),
            "output": int(token_usage.get("response", 0)),
            "total": int(
                token_usage.get("total") or sum(int(v) for v in token_usage.values())
            ),
        }
        if trace is not None and hasattr(trace, "generation"):
            trace.generation(name=f"llm:{run_id}", model=model, usage=usage)
        elif hasattr(client, "generation"):
            client.generation(
                trace_id=run_id, name=f"llm:{run_id}", model=model, usage=usage
            )

    def flush(self) -> None:
        """Flush buffered events to Langfuse (no-op when inactive)."""
        client = self._client
        if client is not None and hasattr(client, "flush"):
            try:
                client.flush()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Langfuse flush skipped: %s", exc)


# ── process-wide singleton (wired into the live graph path) ───────────────────
_EXPORTER: LangfuseExporter | None = None
_EXPORTER_BUILT = False


def get_langfuse_exporter() -> LangfuseExporter | None:
    """Return the process-wide exporter, or ``None`` when not configured.

    Default-on: when ``LANGFUSE_PUBLIC_KEY`` + ``LANGFUSE_SECRET_KEY`` are set,
    the orchestration engine receives a live exporter and ships every graph run.
    Absent keys → ``None`` so the live path skips export with no overhead.
    """
    global _EXPORTER, _EXPORTER_BUILT
    if _EXPORTER_BUILT:
        return _EXPORTER
    _EXPORTER_BUILT = True
    exporter = LangfuseExporter()
    _EXPORTER = exporter if exporter.configured else None
    return _EXPORTER


def set_langfuse_exporter(exporter: LangfuseExporter | None) -> None:
    """Install a specific exporter (used by tests to inject a fake client)."""
    global _EXPORTER, _EXPORTER_BUILT
    _EXPORTER = exporter
    _EXPORTER_BUILT = True


def reset_langfuse_exporter() -> None:
    """Reset the cached singleton so the next call re-probes the environment."""
    global _EXPORTER, _EXPORTER_BUILT
    _EXPORTER = None
    _EXPORTER_BUILT = False


def _now_ms() -> float:
    return time.time() * 1000.0
