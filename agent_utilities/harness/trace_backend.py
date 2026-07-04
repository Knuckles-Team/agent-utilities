from __future__ import annotations

"""Agnostic Trace Backend Abstraction.

CONCEPT:AU-AHE.harness.harness-evolution — Agentic Harness Engineering (Experience Observability)

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

from agent_utilities.core.config import config, setting

logger = logging.getLogger(__name__)


def _first(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-None value among ``keys`` (else ``default``)."""
    for k in keys:
        v = row.get(k)
        if v is not None:
            return v
    return default


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

    # ── failure-read surface (CONCEPT:AU-AHE.harness.failure-evolution)
    # Default no-ops so a backend that has no failure feed (OTel/File without
    # fixtures) degrades gracefully; ``LangfuseTraceBackend`` overrides them.
    async def get_error_observations(
        self, *, since: str | None = None, level: str = "ERROR", limit: int = 100
    ) -> list[dict[str, Any]]:
        """Return error/warning observations since ``since`` (ISO-8601).

        Each record carries at least ``id``, ``traceId``, ``name``,
        ``level``, ``statusMessage``.
        """
        return []

    async def get_low_score_traces(
        self,
        *,
        score_name: str | None = None,
        max_value: float = 0.5,
        since: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return scores below ``max_value`` (normalized to trace references)."""
        return []

    async def get_cost_latency_anomalies(
        self,
        *,
        since: str | None = None,
        p95_latency_ms: float | None = None,
        p95_cost_usd: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return per-name cost/latency rollups exceeding the given p95 budgets."""
        return []


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

    # ── failure-read surface (CONCEPT:AU-AHE.harness.failure-evolution)
    async def get_error_observations(
        self, *, since: str | None = None, level: str = "ERROR", limit: int = 100
    ) -> list[dict[str, Any]]:
        """Pull ERROR/WARNING observations from Langfuse since ``since``.

        Uses the stable ``/api/public/observations`` endpoint (the ``v2``
        observations route is absent on older self-hosted versions and 404s).
        """
        api = self._get_api()
        try:
            resp = api.legacy_observations_v1_get_many(
                level=level,
                from_start_time=since,
                limit=limit,
            )
            return resp.get("data", []) or []
        except Exception as e:  # noqa: BLE001
            logger.error("LangfuseTraceBackend: get_error_observations failed: %s", e)
            return []

    async def get_low_score_traces(
        self,
        *,
        score_name: str | None = None,
        max_value: float = 0.5,
        since: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Pull scores below ``max_value`` and normalize to trace references."""
        api = self._get_api()
        try:
            resp = api.scores_get_many(
                operator="<",
                value=max_value,
                name=score_name,
                from_timestamp=since,
                limit=limit,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("LangfuseTraceBackend: get_low_score_traces failed: %s", e)
            return []
        out: list[dict[str, Any]] = []
        for s in resp.get("data", []) or []:
            out.append(
                {
                    "trace_id": s.get("traceId"),
                    "observation_id": s.get("observationId"),
                    "name": s.get("name"),
                    "value": s.get("value"),
                    "comment": s.get("comment"),
                }
            )
        return out

    async def get_cost_latency_anomalies(
        self,
        *,
        since: str | None = None,
        p95_latency_ms: float | None = None,
        p95_cost_usd: float | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregate per-name p95 latency + cost via the v2 metrics API.

        Returns one normalized row per observation ``name`` whose p95 latency or
        total cost exceeds the supplied budget. When no budget is supplied the
        row is still returned (caller decides) — anomaly classification lives in
        the analyzer, this method only shapes the metrics response.
        """
        import json
        import time

        api = self._get_api()
        # The metrics API requires BOTH fromTimestamp and toTimestamp; a null
        # toTimestamp is a 400. Default the window to the last 24h when unset.
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        from_ts = since or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 86400)
        )
        query: dict[str, Any] = {
            "view": "observations",
            "dimensions": [{"field": "name"}],
            "metrics": [
                {"measure": "latency", "aggregation": "p95"},
                {"measure": "totalCost", "aggregation": "sum"},
                {"measure": "totalTokens", "aggregation": "sum"},
                {"measure": "count", "aggregation": "count"},
            ],
            "filters": [],
            "fromTimestamp": from_ts,
            "toTimestamp": now,
        }
        try:
            # ``/api/public/metrics`` (the v2 ``/api/public/v2/metrics`` alias is
            # absent on older self-hosted versions and 404s); the query schema is
            # identical across both routes.
            resp = api.legacy_metrics_v1_metrics(json.dumps(query))
        except Exception as e:  # noqa: BLE001
            logger.error(
                "LangfuseTraceBackend: get_cost_latency_anomalies failed: %s", e
            )
            return []
        out: list[dict[str, Any]] = []
        for row in resp.get("data", []) or []:
            # Langfuse keys aggregated measures as ``{aggregation}_{measure}``;
            # fall back to the bare measure name across API versions.
            lat = _first(row, "p95_latency", "latency", default=0.0)
            cost = _first(row, "sum_totalCost", "totalCost", default=0.0)
            tokens = _first(row, "sum_totalTokens", "totalTokens", default=0)
            count = _first(row, "count_count", "count", default=0)
            p95_val = float(lat or 0.0)
            cost_val = float(cost or 0.0)
            row_out = {
                "name": row.get("name") or "unknown",
                "p95_latency_ms": p95_val,
                "total_cost_usd": cost_val,
                "total_tokens": int(tokens or 0),
                "count": int(count or 0),
            }
            over_latency = p95_latency_ms is not None and p95_val > p95_latency_ms
            over_cost = p95_cost_usd is not None and cost_val > p95_cost_usd
            if (
                over_latency
                or over_cost
                or (p95_latency_ms is None and p95_cost_usd is None)
            ):
                row_out["over_latency"] = over_latency
                row_out["over_cost"] = over_cost
                out.append(row_out)
        return out

    # ── dataset-based regression (CONCEPT:AU-AHE.harness.failure-evolution, Phase 4)
    async def create_regression_dataset(self, name: str, description: str = "") -> bool:
        """Create (idempotent) a Langfuse dataset used for remediation regression."""
        api = self._get_api()
        try:
            api.datasets_create({"name": name, "description": description})
            return True
        except Exception as e:  # noqa: BLE001
            # Already-exists is benign — the dataset is reused.
            logger.debug("create_regression_dataset(%s) note: %s", name, e)
            return False

    async def get_dataset_run(self, dataset_name: str, run_name: str) -> dict[str, Any]:
        """Fetch a completed dataset run (for post-merge regression comparison)."""
        api = self._get_api()
        try:
            return api.datasets_get_run(dataset_name, run_name) or {}
        except Exception as e:  # noqa: BLE001
            logger.error("get_dataset_run(%s/%s) failed: %s", dataset_name, run_name, e)
            return {}

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
        self.endpoint = endpoint or setting("OTEL_EXPORTER_OTLP_ENDPOINT", "")
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
                query_url = (
                    self.endpoint.replace("4318", "16686")
                    .replace("4317", "16686")
                    .rstrip("/")
                )
                if not query_url.endswith("/api/traces"):
                    query_url += "/api/traces"
                if not query_url.startswith("http"):
                    query_url = f"http://{query_url}"

                params = {
                    "service": "agent-utilities",
                    "tags": f'{{"round_id":"{round_id}"}}',
                }
                async with httpx.AsyncClient() as client:
                    response = await client.get(query_url, params=params, timeout=5.0)
                    if response.status_code == 200:
                        data = response.json()
                        traces: list[dict[str, Any]] = []
                        for trace in data.get("data", []):
                            traces.append(
                                {
                                    "id": trace.get("traceID"),
                                    "round_id": round_id,
                                    "spans": trace.get("spans", []),
                                }
                            )
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

    # ── failure-read surface from JSON fixtures (CONCEPT:AU-AHE.harness.failure-evolution)
    def _read_fixture(self, name: str) -> list[dict[str, Any]]:
        import json

        path = os.path.join(self.trace_dir, name)
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            logger.debug("FileTraceBackend: failed to read %s: %s", path, e)
            return []
        if isinstance(data, dict):
            data = data.get("data", [])
        return data if isinstance(data, list) else []

    async def get_error_observations(
        self, *, since: str | None = None, level: str = "ERROR", limit: int = 100
    ) -> list[dict[str, Any]]:
        return self._read_fixture("error_observations.json")[:limit]

    async def get_low_score_traces(
        self,
        *,
        score_name: str | None = None,
        max_value: float = 0.5,
        since: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = self._read_fixture("low_scores.json")
        return [r for r in rows if (r.get("value") is None or r["value"] < max_value)][
            :limit
        ]

    async def get_cost_latency_anomalies(
        self,
        *,
        since: str | None = None,
        p95_latency_ms: float | None = None,
        p95_cost_usd: float | None = None,
    ) -> list[dict[str, Any]]:
        return self._read_fixture("cost_latency_anomalies.json")


class KGTraceBackend(TraceBackend):
    """KG-native trace sink (CONCEPT:AU-OS.config.model-factory-passthrough) — the moat over an opaque trace store.

    Every trace persists as a ``TraceNode → SpanNode/GenerationNode`` subgraph
    (HAS_SPAN/HAS_GENERATION edges) in the SAME OWL/RDF engine that holds the
    ecosystem, so traces are graph-queryable: "every FAILED trace whose root cause
    chains to capability X", "which prompt version regressed which dimension" — joins
    Opik's ClickHouse cannot express. Per-generation cost is resolved from the engine's
    own pricing catalog (ECO-4.40), not a vendored price table.

    ``backend`` is the KG facade (duck-typed: ``add_node(id, **props)`` +
    ``link_nodes(src, dst, rel)``). An in-memory index mirrors what was emitted so the
    ``TraceDistiller`` read surface (``get_traces``/``get_trace_summary``/
    ``get_trace_scores``) works even before a graph round-trip — exactly the
    best-effort-persist + in-memory pattern :class:`EvalCorpus` uses.
    """

    def __init__(self, backend: Any = None, *, max_traces: int = 2000) -> None:
        self.backend = backend
        self._traces: dict[str, dict[str, Any]] = {}
        self._max_traces = max_traces  # bound in-memory mirror (oldest evicted)
        # Optional fast hook fired when a ROOT trace completes (set by the
        # OnlineScoringSampler, CONCEPT:AU-AHE.harness.receives-trace-id-must). Receives the trace_id; must only
        # schedule/enqueue — never run the judge inline (keeps the hot path fast).
        self.on_trace_complete: Any = None

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """The full in-memory entry ``{trace, spans, generations}`` for a trace id."""
        return self._traces.get(trace_id)

    def _evict_if_needed(self) -> None:
        # FIFO retention so an always-on in-memory mirror can't grow unbounded.
        while len(self._traces) > self._max_traces:
            self._traces.pop(next(iter(self._traces)))

    def record_event(
        self,
        *,
        trace_id: str,
        span_id: str,
        name: str,
        is_root: bool,
        kind: str = "general",  # general | llm | tool
        parent_span_id: str | None = None,
        session_id: str | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tags: list[str] | None = None,
        input_text: str = "",
        output_text: str = "",
    ) -> None:
        """Incrementally record ONE trace/span/generation event (CONCEPT:AU-OS.config.model-factory-passthrough).

        The decorator path (``@trace``/``@generation``) emits events one at a time — a
        root trace, then child spans/generations — so this upserts the TraceNode bucket
        and appends the child node, persisting + linking each immediately. The
        always-on tracing sink uses this; ``emit_trace`` remains the batch path.
        """
        from agent_utilities.models.knowledge_graph import (
            GenerationNode,
            RegistryEdgeType,
            SpanNode,
            TraceNode,
        )

        entry = self._traces.get(trace_id)
        new_trace = entry is None
        if entry is None:
            trace = TraceNode(
                id=trace_id,
                name=name if is_root else "trace",
                session_id=session_id,
                tags=list(tags or []),
            )
            entry = {"trace": trace, "spans": [], "generations": []}
            self._traces[trace_id] = entry
            self._evict_if_needed()
        assert entry is not None  # set above when new_trace; else the cache hit
        trace = entry["trace"]
        if error:
            trace.status = "error"
        if is_root:
            trace.latency_ms = latency_ms
            if input_text:
                trace.input = input_text[:4000]
            if output_text:
                trace.output = output_text[:4000]

        # Persist/refresh the trace node on creation OR when the root completes (status
        # flip / input-output now known), so the snapshot reflects the final state.
        if (
            (new_trace or error or is_root)
            and self.backend is not None
            and hasattr(self.backend, "add_node")
        ):
            try:
                self.backend.add_node(trace_id, **self._node_props(trace))
            except Exception as exc:  # pragma: no cover - best-effort
                logger.debug("KGTraceBackend trace persist failed: %s", exc)

        if is_root:
            # Root span = trace finished. Fire the completion hook (best-effort, fast —
            # the hook only schedules/enqueues; it must NOT run the judge inline, so a
            # traced call never pays scoring latency). CONCEPT:AU-AHE.harness.receives-trace-id-must.
            cb = getattr(self, "on_trace_complete", None)
            if callable(cb):
                try:
                    cb(trace_id)
                except Exception as exc:  # pragma: no cover - best-effort
                    logger.debug("on_trace_complete hook failed: %s", exc)
            return

        if kind == "llm":
            node: Any = GenerationNode(
                id=span_id,
                name=name,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                error=error,
            )
            node.total_cost_usd = self._cost_usd(model, input_tokens, output_tokens)
            entry["generations"].append(node)
            trace.total_cost_usd += node.total_cost_usd
            trace.input_tokens += input_tokens
            trace.output_tokens += output_tokens
            edge = RegistryEdgeType.HAS_GENERATION
        else:
            node = SpanNode(
                id=span_id,
                name=name,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                span_kind=kind,
                latency_ms=latency_ms,
                error=error,
            )
            entry["spans"].append(node)
            edge = RegistryEdgeType.HAS_SPAN

        if self.backend is not None and hasattr(self.backend, "add_node"):
            try:
                self.backend.add_node(span_id, **self._node_props(node))
                link = getattr(self.backend, "link_nodes", None)
                if callable(link):
                    link(parent_span_id or trace_id, span_id, edge)
            except Exception as exc:  # pragma: no cover - best-effort
                logger.debug("KGTraceBackend event persist failed: %s", exc)

    @staticmethod
    def _node_props(node: Any) -> dict[str, Any]:
        d = node.model_dump() if hasattr(node, "model_dump") else dict(node)
        d.pop("id", None)
        d["type"] = str(d.get("type", ""))
        return d

    @staticmethod
    def _cost_usd(model: str | None, input_tokens: int, output_tokens: int) -> float:
        """Resolve $ cost from the shared pricing catalog (no vendored table)."""
        if not model:
            return 0.0
        try:
            from agent_utilities.pricing import get_pricing_catalog

            cost, priced = get_pricing_catalog().cost_for(
                model, input_tokens=input_tokens, output_tokens=output_tokens
            )
            return float(cost) if priced and cost is not None else 0.0
        except Exception:  # pragma: no cover - pricing is best-effort
            return 0.0

    def emit_trace(
        self,
        trace: Any,
        spans: list[Any] | None = None,
        generations: list[Any] | None = None,
    ) -> None:
        """Persist a trace subgraph (best-effort graph write + in-memory index).

        ``trace`` is a ``TraceNode``; ``spans``/``generations`` are ``SpanNode`` /
        ``GenerationNode`` instances. Generation cost is (re)computed from the pricing
        catalog if unset. Idempotent on the in-memory index (re-emit overwrites).
        """
        spans = spans or []
        generations = generations or []
        # Fill in $ cost for any generation that didn't carry one.
        for g in generations:
            if getattr(g, "total_cost_usd", 0.0) in (0.0, None):
                g.total_cost_usd = self._cost_usd(
                    getattr(g, "model", None),
                    getattr(g, "input_tokens", 0),
                    getattr(g, "output_tokens", 0),
                )
        # Roll up trace-level cost/tokens from its generations.
        trace.total_cost_usd = sum(
            getattr(g, "total_cost_usd", 0.0) for g in generations
        )
        trace.input_tokens = sum(getattr(g, "input_tokens", 0) for g in generations)
        trace.output_tokens = sum(getattr(g, "output_tokens", 0) for g in generations)

        self._traces[trace.id] = {
            "trace": trace,
            "spans": list(spans),
            "generations": list(generations),
        }

        if self.backend is not None and hasattr(self.backend, "add_node"):
            try:
                self._persist(trace, spans, generations)
            except Exception as exc:  # pragma: no cover - persistence best-effort
                logger.debug("KGTraceBackend persist failed: %s", exc)

    def _persist(self, trace: Any, spans: list[Any], generations: list[Any]) -> None:
        from agent_utilities.models.knowledge_graph import RegistryEdgeType

        def _props(node: Any) -> dict[str, Any]:
            d = node.model_dump() if hasattr(node, "model_dump") else dict(node)
            d.pop("id", None)
            d["type"] = str(d.get("type", ""))
            return d

        self.backend.add_node(trace.id, **_props(trace))
        link = getattr(self.backend, "link_nodes", None)
        for s in spans:
            self.backend.add_node(s.id, **_props(s))
            if callable(link):
                link(trace.id, s.id, RegistryEdgeType.HAS_SPAN)
        for g in generations:
            self.backend.add_node(g.id, **_props(g))
            if callable(link):
                parent = getattr(g, "parent_span_id", None) or trace.id
                link(parent, g.id, RegistryEdgeType.HAS_GENERATION)

    def _summarize(self, entry: dict[str, Any]) -> dict[str, Any]:
        t = entry["trace"]
        return {
            "id": t.id,
            "name": getattr(t, "name", ""),
            "status": getattr(t, "status", "ok"),
            "duration_ms": getattr(t, "latency_ms", None),
            "input_tokens": getattr(t, "input_tokens", 0),
            "output_tokens": getattr(t, "output_tokens", 0),
            "total_cost_usd": getattr(t, "total_cost_usd", 0.0),
            "score": getattr(t, "metadata", {}).get("score", 0.0)
            if isinstance(getattr(t, "metadata", {}), dict)
            else 0.0,
            "error": next(
                (g.error for g in entry["generations"] if getattr(g, "error", None)),
                None,
            ),
        }

    async def get_traces(self, round_id: str, **_filters: Any) -> list[dict[str, Any]]:
        return [
            self._summarize(e)
            for e in self._traces.values()
            if not round_id
            or getattr(e["trace"], "metadata", {}).get("round_id") == round_id
        ]

    async def get_trace_summary(self, trace_id: str) -> dict[str, Any]:
        entry = self._traces.get(trace_id)
        return (
            self._summarize(entry) if entry else {"id": trace_id, "error": "not_found"}
        )

    async def get_trace_scores(self, trace_ids: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for tid in trace_ids:
            s = await self.get_trace_summary(tid)
            out[tid] = float(s.get("score", 0.0) or 0.0)
        return out


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
    elif backend_type == "kg":
        return KGTraceBackend(backend=kwargs.get("backend"))
    elif backend_type == "file":
        return FileTraceBackend(trace_dir=kwargs.get("trace_dir", "."))

    # Auto-detect. Langfuse/OTel, when configured, become fan-out sinks; otherwise the
    # KG-native backend is the default so traces are graph-queryable (CONCEPT:AU-OS.config.model-factory-passthrough).
    # An explicit ``trace_dir`` still selects the file backend for offline fixtures.
    if config.langfuse_secret_key or setting("LANGFUSE_SECRET_KEY"):
        logger.info("TraceBackend: Auto-detected Langfuse credentials.")
        return LangfuseTraceBackend()
    if config.otel_exporter_otlp_endpoint or setting("OTEL_EXPORTER_OTLP_ENDPOINT"):
        logger.info("TraceBackend: Auto-detected OTel endpoint.")
        return OTelTraceBackend(**kwargs)
    if "trace_dir" in kwargs:
        logger.info("TraceBackend: file fixtures (%s).", kwargs["trace_dir"])
        return FileTraceBackend(trace_dir=kwargs["trace_dir"])

    logger.info("TraceBackend: Defaulting to KG-native backend (OS-5.68).")
    return KGTraceBackend(backend=kwargs.get("backend"))
