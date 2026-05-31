from __future__ import annotations

"""Agnostic Trace Backend Abstraction.

CONCEPT:AHE-3.0 — Agentic Harness Engineering (Experience Observability)

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


import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from agent_utilities.core.config import config

logger = logging.getLogger(__name__)


class TraceBackend(ABC):
    """Abstract backend for trace ingestion (database-style abstraction).

    All trace backends must implement ``get_traces`` and ``get_trace_summary``
    to support the ``TraceDistiller`` pipeline.
    """

    @abstractmethod
    async def get_traces(self, round_id: str, **_filters: Any) -> list[dict[str, Any]]:
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
                from langfuse_agent.api_client import LangfuseApi

                if not config.langfuse_public_key or not config.langfuse_secret_key:
                    raise ValueError(
                        "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set in configuration."
                    )
                host = config.langfuse_host or "https://cloud.langfuse.com"

                self._api = LangfuseApi(
                    public_key=config.langfuse_public_key,
                    secret_key=config.langfuse_secret_key,
                    host=host,
                )
                logger.info("LangfuseTraceBackend: API client initialized.")
            except ImportError:
                raise ImportError(
                    "langfuse-agent package is required for LangfuseTraceBackend. "
                    "Install with: pip install langfuse-agent"
                ) from None
        return self._api

    async def get_traces(self, round_id: str, **_filters: Any) -> list[dict[str, Any]]:
        """Retrieve traces from Langfuse for an evolution round.

        Uses the `tags` filter to match traces tagged with the round_id.
        """
        api = self._get_api()
        try:
            # LangfuseApi.observations_get_many expects trace/observation queries
            # Wait, api_client.py provides `observations_get_many` which can filter by tag if supported,
            # or `legacy_observations_v1_get_many`. Actually we can filter by `traceName` or `tags`.
            # Let's pass the tag or traceName as filter.
            response = api.observations_get_many(
                type="TRACE", filter=f"tags='{round_id}'", limit=100
            )
            return response.get("data", [])
        except Exception as e:
            logger.error(f"LangfuseTraceBackend: Failed to get traces: {e}")
            return []

    async def get_trace_summary(self, trace_id: str) -> dict[str, Any]:
        """Get a lightweight trace summary from Langfuse."""
        api = self._get_api()
        try:
            response = api.observations_get_many(
                trace_id=trace_id, limit=1, fields="core,basic,metrics"
            )
            data = response.get("data", [])
            if not data:
                return {"id": trace_id, "error": "not_found"}

            trace = data[0]
            return {
                "id": trace.get("id", trace_id),
                "name": trace.get("name", ""),
                "status": trace.get("statusMessage") or "unknown",
                "duration_ms": trace.get("latency", 0),
                "input_tokens": trace.get("usageDetails", {}).get("input", 0)
                if trace.get("usageDetails")
                else 0,
                "output_tokens": trace.get("usageDetails", {}).get("output", 0)
                if trace.get("usageDetails")
                else 0,
                "score": 0.0,  # Would need a separate score fetch
                "error": trace.get("statusMessage"),
            }
        except Exception as e:
            logger.error(f"LangfuseTraceBackend: Failed to get trace summary: {e}")
            return {"id": trace_id, "error": str(e)}

    async def submit_score(
        self, trace_id: str, name: str, value: float, comment: str | None = None
    ) -> bool:
        """Submit an evaluation score for a trace."""
        api = self._get_api()
        try:
            payload = {"traceId": trace_id, "name": name, "value": value}
            if comment:
                payload["comment"] = comment
            api.legacy_score_v1_create(payload)
            return True
        except Exception as e:
            logger.error(f"LangfuseTraceBackend: Failed to submit score: {e}")
            return False

    async def add_to_dataset(
        self,
        dataset_name: str,
        trace_id: str,
        input_data: Any = None,
        expected_output: Any = None,
    ) -> bool:
        """Add a trace to a Langfuse dataset for continuous learning."""
        api = self._get_api()
        try:
            payload = {
                "datasetName": dataset_name,
                "sourceTraceId": trace_id,
                "input": input_data,
                "expectedOutput": expected_output,
            }
            api.dataset_items_create(payload)
            return True
        except Exception as e:
            logger.error(
                f"LangfuseTraceBackend: Failed to add trace {trace_id} to dataset {dataset_name}: {e}"
            )
            return False

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

    async def get_traces(self, round_id: str, **_filters: Any) -> list[dict[str, Any]]:
        """Retrieve traces from OTel/Logfire.

        If an export directory is configured, reads from JSON files.
        Otherwise, queries the OTLP endpoint.
        """
        if self.export_dir:
            return self._read_exported_traces(round_id)
        
        if self.endpoint:
            import httpx
            try:
                query_url = self.endpoint.replace("4318", "16686").replace("4317", "16686").rstrip("/")
                if not query_url.endswith("/api/traces"):
                    query_url += "/api/traces"
                if not query_url.startswith("http"):
                    query_url = f"http://{query_url}"

                params = {"service": "agent-utilities", "tags": f'{{"round_id":"{round_id}"}}'}
                async with httpx.AsyncClient() as client:
                    response = await client.get(query_url, params=params, timeout=5.0)
                    if response.status_code == 200:
                        data = response.json()
                        traces: list[dict[str, Any]] = []
                        for trace in data.get("data", []):
                            traces.append({
                                "id": trace.get("traceID"),
                                "round_id": round_id,
                                "spans": trace.get("spans", []),
                            })
                        return traces
            except Exception as e:
                logger.warning(f"Failed to query live OTLP endpoint: {e}")
        
        logger.warning(
            "OTelTraceBackend: OTLP query could not resolve endpoint or query failed. "
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
        """Get a lightweight trace summary from OTel/Logfire."""
        # 1. First search in exported directory files
        if self.export_dir:
            import glob
            import json

            for path in glob.glob(os.path.join(self.export_dir, "*.json")):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    trace_list = data if isinstance(data, list) else [data]
                    for t in trace_list:
                        if isinstance(t, dict) and (
                            t.get("id") == trace_id
                            or t.get("traceId") == trace_id
                            or t.get("trace_id") == trace_id
                        ):
                            return self._format_otel_summary(t, trace_id)
                except Exception as e:
                    logger.debug(f"Failed to parse trace file {path}: {e}")

        # 2. If endpoint is configured, try querying the endpoint (e.g. Jaeger API or local server)
        if self.endpoint:
            import httpx

            try:
                url = f"{self.endpoint.rstrip('/')}/api/traces/{trace_id}"
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=5.0)
                    if resp.status_code == 200:
                        t_data = resp.json()
                        if (
                            "data" in t_data
                            and isinstance(t_data["data"], list)
                            and t_data["data"]
                        ):
                            return self._format_otel_summary(
                                t_data["data"][0], trace_id
                            )
            except Exception as e:
                logger.debug(
                    f"Failed to fetch trace from endpoint {self.endpoint}: {e}"
                )

        return {"id": trace_id, "status": "unknown", "error": "trace_not_found"}

    def _format_otel_summary(
        self, trace: dict[str, Any], trace_id: str
    ) -> dict[str, Any]:
        """Format an OTel trace dict into a standard summary structure."""
        name = trace.get("name") or trace.get("traceName") or ""
        duration = (
            trace.get("duration")
            or trace.get("latency")
            or trace.get("duration_ms")
            or 0
        )
        status = trace.get("status") or trace.get("statusMessage") or "unknown"

        usage = trace.get("usageDetails") or trace.get("usage") or {}
        input_tokens = usage.get("input") or usage.get("prompt_tokens") or 0
        output_tokens = usage.get("output") or usage.get("completion_tokens") or 0

        score = trace.get("score") or trace.get("value") or 0.0
        if not score and "scores" in trace:
            scores_dict = trace["scores"]
            if isinstance(scores_dict, dict) and scores_dict:
                score = list(scores_dict.values())[0]
            elif isinstance(scores_dict, list) and scores_dict:
                score = scores_dict[0].get("value", 0.0)

        return {
            "id": trace_id,
            "name": name,
            "status": status,
            "duration_ms": duration,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "score": float(score),
            "error": None if status in ("ok", "success") else status,
        }

    async def get_trace_scores(self, trace_ids: list[str]) -> dict[str, float]:
        """Fetch scores from OTel/Logfire traces."""
        scores: dict[str, float] = {}
        for tid in trace_ids:
            summary = await self.get_trace_summary(tid)
            scores[tid] = summary.get("score", 0.0)
        return scores


class FileTraceBackend(TraceBackend):
    """File-based trace backend for testing and offline workflows.

    Reads pre-exported trace data from a directory of JSON files.
    """

    def __init__(self, trace_dir: str) -> None:
        self.trace_dir = trace_dir

    async def get_traces(self, round_id: str, **_filters: Any) -> list[dict[str, Any]]:
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
    if config.langfuse_secret_key or os.environ.get("LANGFUSE_SECRET_KEY"):
        logger.info("TraceBackend: Auto-detected Langfuse credentials.")
        return LangfuseTraceBackend()
    if config.otel_exporter_otlp_endpoint or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    ):
        logger.info("TraceBackend: Auto-detected OTel endpoint.")
        return OTelTraceBackend(**kwargs)

    # Default to file-based
    trace_dir = kwargs.get("trace_dir", ".")
    logger.info(f"TraceBackend: Defaulting to FileTraceBackend ({trace_dir}).")
    return FileTraceBackend(trace_dir=trace_dir)
