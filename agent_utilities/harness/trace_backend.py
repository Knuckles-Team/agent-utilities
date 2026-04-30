"""Agnostic Trace Backend Abstraction.

CONCEPT:AU-012 — Agentic Harness Engineering (Experience Observability)

Provides a pluggable backend abstraction for trace ingestion, following
the same pattern as the database backend abstraction in
``knowledge_graph/backends/``.

Supported backends:
    - **LangfuseTraceBackend**: Imports ``langfuse_agent.langfuse_api``
      directly for zero-LLM-overhead trace extraction.
    - **OTelTraceBackend**: Ingests raw OTLP trace data from Logfire
      or any OpenTelemetry-compatible source.
    - **FileTraceBackend**: Reads pre-exported trace JSON files for
      offline/testing workflows.

Usage::

    from agent_utilities.harness.trace_backend import create_trace_backend

    # Auto-detect based on environment
    backend = create_trace_backend()

    # Or explicitly choose
    backend = create_trace_backend(backend_type="langfuse")
    traces = await backend.get_traces(round_id="round:abc123")
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class TraceBackend(ABC):
    """Abstract backend for trace ingestion (database-style abstraction).

    All trace backends must implement ``get_traces`` and ``get_trace_summary``
    to support the ``TraceDistiller`` pipeline.
    """

    @abstractmethod
    async def get_traces(self, round_id: str, **filters: Any) -> list[dict[str, Any]]:
        """Retrieve traces for a specific evolution round.

        Args:
            round_id: The evolution round identifier.
            **filters: Optional filters (e.g., status, date range).

        Returns:
            List of trace dictionaries with standardized schema.
        """

    @abstractmethod
    async def get_trace_summary(self, trace_id: str) -> dict[str, Any]:
        """Get a lightweight summary of a single trace.

        This should be token-efficient — only key metrics and outcomes,
        not the full trace payload.

        Args:
            trace_id: The unique trace identifier.

        Returns:
            Summary dict with keys: id, name, status, duration_ms,
            input_tokens, output_tokens, score, error.
        """

    @abstractmethod
    async def get_trace_scores(self, trace_ids: list[str]) -> dict[str, float]:
        """Batch-fetch scores for multiple traces.

        Args:
            trace_ids: List of trace identifiers.

        Returns:
            Mapping of trace_id → score (0.0-1.0).
        """

    async def health_check(self) -> bool:
        """Check if the backend is available and configured.

        Returns:
            True if the backend can serve requests.
        """
        return True


class LangfuseTraceBackend(TraceBackend):
    """Langfuse trace backend — imports langfuse_agent.langfuse_api directly.

    This backend calls the Langfuse API through the ``langfuse-agent``
    project's API client, bypassing the MCP/LLM layer for zero-overhead
    trace ingestion.
    """

    def __init__(self) -> None:
        self._api: Any = None

    def _get_api(self) -> Any:
        """Lazy-load the Langfuse API client."""
        if self._api is None:
            try:
                from langfuse_agent.langfuse_api import LangfuseAPI

                self._api = LangfuseAPI()
                logger.info("LangfuseTraceBackend: API client initialized.")
            except ImportError:
                raise ImportError(
                    "langfuse-agent package is required for LangfuseTraceBackend. "
                    "Install with: pip install langfuse-agent"
                ) from None
        return self._api

    async def get_traces(self, round_id: str, **filters: Any) -> list[dict[str, Any]]:
        """Retrieve traces from Langfuse for an evolution round.

        Uses the ``tags`` filter to match traces tagged with the round_id,
        or falls back to session-based filtering.
        """
        api = self._get_api()
        try:
            # Try to get traces tagged with this round
            if hasattr(api, "get_traces_for_round"):
                return await api.get_traces_for_round(round_id, **filters)

            # Fallback: use standard trace listing with tag filter
            traces = api.list_traces(tags=[round_id], **filters)
            return traces if isinstance(traces, list) else []
        except Exception as e:
            logger.error(f"LangfuseTraceBackend: Failed to get traces: {e}")
            return []

    async def get_trace_summary(self, trace_id: str) -> dict[str, Any]:
        """Get a lightweight trace summary from Langfuse."""
        api = self._get_api()
        try:
            if hasattr(api, "get_trace_summary"):
                return await api.get_trace_summary(trace_id)

            # Fallback: fetch full trace and extract summary fields
            trace = api.get_trace(trace_id)
            return {
                "id": trace.get("id", trace_id),
                "name": trace.get("name", ""),
                "status": trace.get("status", "unknown"),
                "duration_ms": trace.get("latency", 0),
                "input_tokens": trace.get("input", {}).get("token_count", 0),
                "output_tokens": trace.get("output", {}).get("token_count", 0),
                "score": trace.get("scores", [{}])[0].get("value", 0.0)
                if trace.get("scores")
                else 0.0,
                "error": trace.get("statusMessage"),
            }
        except Exception as e:
            logger.error(f"LangfuseTraceBackend: Failed to get trace summary: {e}")
            return {"id": trace_id, "error": str(e)}

    async def get_trace_scores(self, trace_ids: list[str]) -> dict[str, float]:
        """Batch-fetch scores from Langfuse."""
        scores: dict[str, float] = {}
        for tid in trace_ids:
            summary = await self.get_trace_summary(tid)
            scores[tid] = summary.get("score", 0.0)
        return scores

    async def health_check(self) -> bool:
        """Check Langfuse API availability."""
        try:
            self._get_api()
            return True
        except Exception:
            return False


class OTelTraceBackend(TraceBackend):
    """OpenTelemetry/Logfire trace backend for raw OTLP trace data.

    Reads from a configurable OTLP endpoint or exported trace files.
    Supports the same standardized schema as LangfuseTraceBackend.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        export_dir: str | None = None,
    ) -> None:
        self.endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        self.export_dir = export_dir

    async def get_traces(self, round_id: str, **filters: Any) -> list[dict[str, Any]]:
        """Retrieve traces from OTel/Logfire.

        If an export directory is configured, reads from JSON files.
        Otherwise, queries the OTLP endpoint.
        """
        if self.export_dir:
            return self._read_exported_traces(round_id)
        logger.warning(
            "OTelTraceBackend: OTLP query not yet implemented. "
            "Set export_dir for file-based ingestion."
        )
        return []

    def _read_exported_traces(self, round_id: str) -> list[dict[str, Any]]:
        """Read exported trace JSON files from disk."""
        import glob
        import json

        if not self.export_dir:
            return []

        traces: list[dict[str, Any]] = []
        pattern = os.path.join(self.export_dir, f"*{round_id}*.json")
        for path in glob.glob(pattern):
            try:
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    traces.extend(data)
                elif isinstance(data, dict):
                    traces.append(data)
            except Exception as e:
                logger.warning(f"OTelTraceBackend: Failed to read {path}: {e}")
        return traces

    async def get_trace_summary(self, trace_id: str) -> dict[str, Any]:
        """Stub: OTel trace summary extraction."""
        return {"id": trace_id, "status": "unknown", "error": "not_implemented"}

    async def get_trace_scores(self, trace_ids: list[str]) -> dict[str, float]:
        """Stub: scores not natively available in OTel."""
        return {tid: 0.0 for tid in trace_ids}


class FileTraceBackend(TraceBackend):
    """File-based trace backend for testing and offline workflows.

    Reads pre-exported trace data from a directory of JSON files.
    """

    def __init__(self, trace_dir: str) -> None:
        self.trace_dir = trace_dir

    async def get_traces(self, round_id: str, **filters: Any) -> list[dict[str, Any]]:
        """Load traces from JSON files in the trace directory."""
        import json

        traces: list[dict[str, Any]] = []
        trace_file = os.path.join(self.trace_dir, f"{round_id}.json")
        if os.path.exists(trace_file):
            with open(trace_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                traces = data
            elif isinstance(data, dict):
                traces = [data]
        return traces

    async def get_trace_summary(self, trace_id: str) -> dict[str, Any]:
        """Load a single trace summary from file."""
        import json

        summary_file = os.path.join(self.trace_dir, f"{trace_id}_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file) as f:
                return json.load(f)
        return {"id": trace_id, "error": "not_found"}

    async def get_trace_scores(self, trace_ids: list[str]) -> dict[str, float]:
        """Load scores from file-based traces."""
        scores: dict[str, float] = {}
        for tid in trace_ids:
            summary = await self.get_trace_summary(tid)
            scores[tid] = summary.get("score", 0.0)
        return scores


def create_trace_backend(
    backend_type: str | None = None, **kwargs: Any
) -> TraceBackend:
    """Factory function for creating trace backends.

    Auto-detects the best backend based on environment if ``backend_type``
    is not specified:
        1. If ``LANGFUSE_SECRET_KEY`` is set → LangfuseTraceBackend
        2. If ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set → OTelTraceBackend
        3. If ``trace_dir`` kwarg is provided → FileTraceBackend
        4. Otherwise → FileTraceBackend with default path

    Args:
        backend_type: Explicit backend choice ("langfuse", "otel", "file").
        **kwargs: Backend-specific configuration.

    Returns:
        Configured TraceBackend instance.
    """
    if backend_type == "langfuse":
        return LangfuseTraceBackend()
    elif backend_type == "otel":
        return OTelTraceBackend(**kwargs)
    elif backend_type == "file":
        return FileTraceBackend(trace_dir=kwargs.get("trace_dir", "."))

    # Auto-detect
    if os.environ.get("LANGFUSE_SECRET_KEY"):
        logger.info("TraceBackend: Auto-detected Langfuse credentials.")
        return LangfuseTraceBackend()
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        logger.info("TraceBackend: Auto-detected OTel endpoint.")
        return OTelTraceBackend(**kwargs)

    # Default to file-based
    trace_dir = kwargs.get("trace_dir", ".")
    logger.info(f"TraceBackend: Defaulting to FileTraceBackend ({trace_dir}).")
    return FileTraceBackend(trace_dir=trace_dir)
