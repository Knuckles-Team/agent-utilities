"""The :class:`GatewayClient` — typed, reusable access to the agent-utilities gateway.

CONCEPT:ECO-4.37 — Surface Gateway Client SDK

One client for every surface. Unary calls go through
:class:`agent_utilities.http.AsyncBaseApiClient` (auth, 429 backoff, redaction);
the streaming channel reuses that client's underlying ``httpx`` connection so a
single pool serves both. Methods return the *body* of the gateway's response
envelope (the SDK unwraps ``{"status_code", "data", ...}`` for the caller).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agent_utilities.http import AsyncBaseApiClient, TokenAuth

logger = logging.getLogger("agent_utilities.gateway_client")

# The default agent server (web-UI + enhanced routes + ``/stream``) listens here;
# the bare REST gateway (``python -m agent_utilities``) defaults to ``:9000``. Pass
# ``base_url`` explicitly to target whichever the deployment exposes.
DEFAULT_GATEWAY_URL = "http://localhost:8000"


class GatewayClient:
    """Async client for the agent-utilities gateway REST + streaming surface.

    Args:
        base_url: Gateway origin (e.g. ``http://gateway.arpa:9000``).
        token: Optional bearer token; when set, every request carries a
            server-validated JWT (OS-5.14). Omit for the zero-auth local default.
        timeout: Per-request timeout for unary calls (seconds).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_GATEWAY_URL,
        *,
        token: str | None = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        auth = TokenAuth(token=token) if token else None
        self._api = AsyncBaseApiClient(
            base_url=self.base_url, auth=auth, timeout=timeout, transport=transport
        )

    @staticmethod
    def _body(envelope: dict[str, Any]) -> Any:
        """Unwrap the ``AsyncBaseApiClient`` response envelope to its JSON body."""
        return envelope.get("data")

    # --- Agents / health / commands (the "enhanced" surface the UIs use) ----- #

    async def list_agents(self) -> list[dict[str, Any]]:
        """List the specialists/agents the gateway exposes (``/api/enhanced/agents``)."""
        body = self._body(await self._api.get("/api/enhanced/agents"))
        if isinstance(body, dict) and body.get("status") == "ok":
            return body.get("agents", [])
        return []

    async def maintenance_status(self) -> dict[str, Any]:
        """Gateway maintenance/health snapshot (``/api/enhanced/maintenance/status``)."""
        body = self._body(await self._api.get("/api/enhanced/maintenance/status"))
        return body if isinstance(body, dict) else {}

    async def autocomplete(self, query: str) -> list[str]:
        """Slash-command autocomplete suggestions for ``query``."""
        body = self._body(
            await self._api.get(
                "/api/enhanced/commands/autocomplete", params={"query": query}
            )
        )
        return body.get("suggestions", []) if isinstance(body, dict) else []

    async def execute_command(self, command: str) -> dict[str, Any]:
        """Execute a slash ``command`` and return its result + client actions."""
        body = self._body(
            await self._api.post(
                "/api/enhanced/commands/execute", json={"command": command}
            )
        )
        if not isinstance(body, dict):
            return {"result": ""}
        return {
            "result": body.get("response_markdown", ""),
            "client_actions": body.get("client_actions", []),
        }

    # --- Knowledge graph ----------------------------------------------------- #

    async def graph_query(self, cypher: str, **params: Any) -> dict[str, Any]:
        """Run a Cypher ``cypher`` query against the KG (``/api/graph/query``)."""
        body = self._body(
            await self._api.post("/api/graph/query", json={"cypher": cypher, **params})
        )
        return body if isinstance(body, dict) else {"data": body}

    async def graph_search(self, query: str, *, top_k: int = 10) -> dict[str, Any]:
        """Semantic search over the KG (``/api/graph/search``)."""
        body = self._body(
            await self._api.post(
                "/api/graph/search", json={"query": query, "top_k": top_k}
            )
        )
        return body if isinstance(body, dict) else {"results": body}

    # --- Fleet supervisory plane (OS-5.10 / OS-5.15 / OS-5.24) --------------- #

    async def fleet_topology(self) -> dict[str, Any]:
        """Fleet/worker placement topology (``/api/fleet/topology``)."""
        body = self._body(await self._api.get("/api/fleet/topology"))
        return body if isinstance(body, dict) else {}

    async def fleet_approvals(self) -> list[dict[str, Any]]:
        """Pending ActionPolicy approvals awaiting a human decision."""
        body = self._body(await self._api.get("/api/fleet/approvals"))
        if isinstance(body, dict):
            return body.get("approvals", [])
        return body if isinstance(body, list) else []

    async def grant_approval(self, approval_id: str) -> dict[str, Any]:
        """Grant a pending approval by id (``/api/fleet/approvals/grant``)."""
        body = self._body(
            await self._api.post(
                "/api/fleet/approvals/grant", json={"approval_id": approval_id}
            )
        )
        return body if isinstance(body, dict) else {}

    async def post_fleet_event(
        self, payload: dict[str, Any], *, source: str | None = None
    ) -> dict[str, Any]:
        """Ingest a monitoring event (``POST /api/fleet/events``, OS-5.15)."""
        endpoint = "/api/fleet/events"
        if source:
            endpoint = f"{endpoint}?source={source}"
        body = self._body(await self._api.post(endpoint, json=payload))
        return body if isinstance(body, dict) else {}

    # --- Observability / usage / cost (CONCEPT:ECO-4.41) -------------------- #

    async def usage_summary(self, **filters: Any) -> dict[str, Any]:
        """Token/cost/cache totals (``/api/observability/summary``)."""
        body = self._body(
            await self._api.get("/api/observability/summary", params=filters or None)
        )
        return body if isinstance(body, dict) else {}

    async def usage_by_model(self, **filters: Any) -> list[dict[str, Any]]:
        """Cost+tokens grouped by model."""
        body = self._body(
            await self._api.get("/api/observability/by-model", params=filters or None)
        )
        return body if isinstance(body, list) else []

    async def usage_by_project(self, **filters: Any) -> list[dict[str, Any]]:
        body = self._body(
            await self._api.get(
                "/api/observability/by-project", params=filters or None
            )
        )
        return body if isinstance(body, list) else []

    async def usage_by_agent(self, **filters: Any) -> list[dict[str, Any]]:
        body = self._body(
            await self._api.get("/api/observability/by-agent", params=filters or None)
        )
        return body if isinstance(body, list) else []

    async def analytics_tools(self, **filters: Any) -> list[dict[str, Any]]:
        """Tool/skill/db call frequency + success rate."""
        body = self._body(
            await self._api.get(
                "/api/observability/analytics/tools", params=filters or None
            )
        )
        return body if isinstance(body, list) else []

    async def analytics_activity(self, **filters: Any) -> list[dict[str, Any]]:
        """Day-of-week × hour activity heatmap."""
        body = self._body(
            await self._api.get(
                "/api/observability/analytics/activity", params=filters or None
            )
        )
        return body if isinstance(body, list) else []

    async def analytics_session_shape(self, **filters: Any) -> dict[str, Any]:
        body = self._body(
            await self._api.get(
                "/api/observability/analytics/session-shape", params=filters or None
            )
        )
        return body if isinstance(body, dict) else {}

    async def usage_sessions(
        self, *, limit: int = 100, **filters: Any
    ) -> list[dict[str, Any]]:
        params = {"limit": limit, **filters}
        body = self._body(
            await self._api.get("/api/observability/sessions", params=params)
        )
        return body if isinstance(body, list) else []

    async def usage_top_sessions(
        self, *, limit: int = 20, **filters: Any
    ) -> list[dict[str, Any]]:
        params = {"limit": limit, **filters}
        body = self._body(
            await self._api.get("/api/observability/top-sessions", params=params)
        )
        return body if isinstance(body, list) else []

    async def usage_session_detail(self, session_id: str) -> dict[str, Any]:
        body = self._body(
            await self._api.get(f"/api/observability/sessions/{session_id}")
        )
        return body if isinstance(body, dict) else {}

    async def usage_search(
        self, query: str, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        body = self._body(
            await self._api.get(
                "/api/observability/search", params={"q": query, "limit": limit}
            )
        )
        return body if isinstance(body, list) else []

    async def usage_traces(self, **filters: Any) -> dict[str, Any]:
        """Langfuse trace links (gated server-side on credentials)."""
        body = self._body(
            await self._api.get("/api/observability/traces", params=filters or None)
        )
        return body if isinstance(body, dict) else {"enabled": False, "traces": []}

    # --- Streaming (AG-UI Server-Sent Events over ``/stream``) -------------- #

    async def stream(
        self,
        query: str,
        *,
        mode: str = "ask",
        topology: str = "basic",
        timeout: float = 120.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream an agent run, yielding each parsed SSE event as a dict.

        Event shapes match the gateway's AG-UI channel — e.g. ``{"type":
        "thought", ...}``, ``{"type": "call_tool", "tool": ...}``, ``{"type":
        "final_output", "content": ...}``. Malformed ``data:`` lines are skipped.
        """
        # Reuse the AsyncBaseApiClient's pooled httpx client for the stream so a
        # single connection pool serves both unary and streaming calls.
        http = self._api._client
        async with http.stream(
            "POST",
            "/stream",
            json={"query": query, "mode": mode, "topology": topology},
            timeout=timeout,
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    yield json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

    # --- Lifecycle ---------------------------------------------------------- #

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._api.aclose()

    async def __aenter__(self) -> GatewayClient:
        return self

    async def __aexit__(self, *_exc_info: Any) -> None:
        await self.aclose()
