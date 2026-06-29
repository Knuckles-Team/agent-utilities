#!/usr/bin/python
from __future__ import annotations

"""Pluggable memory-backend transport for the MemoryData bake-off (CONCEPT:AHE-3.71).

The MemoryData benchmark dispatches every memorize/query call through an agent adapter.
To run that adapter *out of process* against a live ``graph-os`` engine **and** keep the
whole sweep unit-testable while the engine is down, the adapter never talks to the engine
directly — it talks to a :class:`MemoryBackendClient`. Two transports implement it:

* :class:`MockBackendClient` — a deterministic, dependency-free in-memory store with a
  trivial token-overlap recall scorer and an echo synthesizer. Used by the offline tests
  and any dry run (``transport="mock"`` is the default so the suite passes with nothing
  deployed).
* :class:`GraphOSRestClient` — calls the graph-os REST surface (``/graph/*``) over
  ``httpx``. A connection failure raises :class:`BackendUnavailable` so one unreachable
  cell never aborts the bake-off.

``build_client(config)`` is the factory the adapter uses; ``config["transport"]`` selects
the implementation.
"""

import re
import time
from abc import ABC, abstractmethod
from typing import Any

__all__ = [
    "BackendUnavailable",
    "MemoryBackendClient",
    "MockBackendClient",
    "GraphOSRestClient",
    "EngineBackendClient",
    "build_client",
]


class BackendUnavailable(RuntimeError):
    """Raised when a backend transport cannot reach its engine (network/connection error).

    The bake-off catches this per cell so an unreachable engine scores 0 for that cell
    rather than crashing the whole sweep.
    """


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    """Lowercase word/number tokens — the shared lexical unit for the mock scorer."""
    return _TOKEN_RE.findall((text or "").lower())


class MemoryBackendClient(ABC):
    """Abstract memory transport the adapter drives (CONCEPT:AHE-3.71).

    A backend stores context chunks (``ingest_memory``), retrieves them for a query
    (``recall``), optionally answers with citations (``synthesize``), and can clear a
    namespace (``reset``). Both the in-memory mock and the live REST client honour this
    same contract so the adapter is transport-agnostic.
    """

    @abstractmethod
    def ingest_memory(
        self, text: str, context_id: int | str, event_time: float | None = None
    ) -> None:
        """Store one memory chunk under ``context_id`` (optionally bi-temporally stamped)."""

    @abstractmethod
    def recall(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        as_of: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to ``top_k`` memories ``[{id, text, score, ...}]`` ranked for ``query``."""

    @abstractmethod
    def synthesize(self, query: str, domain: str = "code", intent: str = "how") -> dict[str, Any]:
        """Return a cited answer ``{answer, citations, ...}`` for ``query`` (context-plane style)."""

    @abstractmethod
    def reset(self, namespace: str) -> None:
        """Drop every memory stored under ``namespace`` (per-run isolation)."""


class MockBackendClient(MemoryBackendClient):
    """Deterministic in-memory backend for tests and dry runs (CONCEPT:AHE-3.71).

    ``recall`` scores stored chunks by token-overlap with the query (a normalized
    intersection count), so a planted fact is reliably retrieved without any model or
    network. ``synthesize`` echoes the best-matching chunks as a cited answer. State is a
    plain dict keyed by namespace, so runs are reproducible and isolated.
    """

    def __init__(self, namespace: str = "default") -> None:
        self.namespace = namespace
        self._store: dict[str, list[dict[str, Any]]] = {}
        self._counter = 0

    def _bucket(self) -> list[dict[str, Any]]:
        return self._store.setdefault(self.namespace, [])

    def ingest_memory(
        self, text: str, context_id: int | str, event_time: float | None = None
    ) -> None:
        self._counter += 1
        self._bucket().append(
            {
                "id": f"{self.namespace}:{context_id}:{self._counter}",
                "text": str(text or ""),
                "context_id": context_id,
                "event_time": event_time if event_time is not None else time.time(),
            }
        )

    def _score(self, query_tokens: set[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        mem_tokens = set(_tokens(text))
        if not mem_tokens:
            return 0.0
        overlap = len(query_tokens & mem_tokens)
        return overlap / len(query_tokens)

    def recall(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        as_of: float | None = None,
    ) -> list[dict[str, Any]]:
        q_tokens = set(_tokens(query))
        rows = self._bucket()
        if as_of is not None:
            rows = [r for r in rows if r.get("event_time", 0.0) <= as_of]
        scored = [
            {"id": r["id"], "text": r["text"], "score": self._score(q_tokens, r["text"]), "mode": mode}
            for r in rows
        ]
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[: max(0, top_k)]

    def synthesize(self, query: str, domain: str = "code", intent: str = "how") -> dict[str, Any]:
        top = self.recall(query, mode="hybrid", top_k=3)
        cited = [t for t in top if t["score"] > 0]
        answer = " ".join(t["text"] for t in cited) if cited else ""
        return {
            "answer": answer,
            "citations": [t["id"] for t in cited],
            "domain": domain,
            "intent": intent,
        }

    def reset(self, namespace: str) -> None:
        self._store.pop(namespace, None)


class GraphOSRestClient(MemoryBackendClient):
    """Live transport over the graph-os REST surface (``/graph/*``, CONCEPT:AHE-3.71).

    Maps the abstract operations onto the served endpoints:

    * ``recall`` → ``POST /graph/search`` (semantic/hybrid retrieval),
    * ``synthesize`` → ``POST /graph/analyze`` with ``action=explain`` (context plane),
    * ``ingest_memory`` → ``POST /graph/ingest_sessions`` (memory add),
    * ``reset`` → ``POST /graph/ingest_sessions`` with ``action=reset``.

    Base URL and bearer token come from ``GRAPHOS_BASE_URL`` (default
    ``http://127.0.0.1:8000``) and ``GRAPHOS_TOKEN``. Any connection error is re-raised as
    :class:`BackendUnavailable` so the sweep degrades gracefully instead of crashing.
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        namespace: str = "default",
        timeout: float = 30.0,
    ) -> None:
        from agent_utilities.core.config import setting

        self.base_url = (
            base_url or setting("GRAPHOS_BASE_URL", "http://127.0.0.1:8000")
        ).rstrip("/")
        self.token = (
            token if token is not None else setting("GRAPHOS_TOKEN", "")
        ) or None
        self.namespace = namespace
        self.timeout = timeout
        self._client: Any | None = None

    def _http(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - httpx is a declared dep
                raise BackendUnavailable(f"httpx not available: {exc}") from exc
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout)
        return self._client

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailable(f"httpx not available: {exc}") from exc
        try:
            resp = self._http().post(path, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise BackendUnavailable(f"graph-os request to {path} failed: {exc}") from exc

    def ingest_memory(
        self, text: str, context_id: int | str, event_time: float | None = None
    ) -> None:
        payload: dict[str, Any] = {
            "namespace": self.namespace,
            "context_id": context_id,
            "text": str(text or ""),
        }
        if event_time is not None:
            payload["event_time"] = event_time
        self._post("/graph/ingest_sessions", payload)

    def recall(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        as_of: float | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "namespace": self.namespace,
        }
        if as_of is not None:
            payload["as_of"] = as_of
        data = self._post("/graph/search", payload)
        results = data.get("results") or data.get("matches") or data.get("hits") or []
        rows: list[dict[str, Any]] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            rows.append(
                {
                    "id": r.get("id") or r.get("node_id") or r.get("uid"),
                    "text": r.get("text") or r.get("content") or r.get("summary") or "",
                    "score": float(r.get("score", 0.0) or 0.0),
                    "mode": mode,
                }
            )
        return rows[:top_k]

    def synthesize(self, query: str, domain: str = "code", intent: str = "how") -> dict[str, Any]:
        data = self._post(
            "/graph/analyze",
            {"action": "explain", "query": query, "domain": domain, "intent": intent},
        )
        answer = data.get("answer") or data.get("output") or ""
        return {
            "answer": answer,
            "citations": data.get("citations", []),
            "domain": domain,
            "intent": intent,
        }

    def reset(self, namespace: str) -> None:
        self._post("/graph/ingest_sessions", {"action": "reset", "namespace": namespace})


class EngineBackendClient(MemoryBackendClient):
    """In-process transport over the *live* graph-os engine (CONCEPT:AHE-3.71).

    Unlike :class:`GraphOSRestClient` (which assumes a ``/graph/*`` REST gateway), this
    transport talks to the engine the same way the MCP tools do — via
    ``agent_utilities.mcp.kg_server._get_engine()``, which connects to the running
    ``epistemic-graph`` daemon over its UDS socket as a *client* (role != host, so it never
    contends for the host write lock). This is the path that actually exercises the deployed
    memory stack:

    * ``ingest_memory`` → ``engine.add_node`` of a namespaced ``:MemoryChunk`` node,
    * ``recall`` → the per-mode engine retriever (``search_hybrid`` for hybrid/rerank/as-of,
      ``search_memories`` for memory, the latent-topology RAG for latent),
    * ``synthesize`` → the context-plane synthesizer when available, else recall+join.

    Any engine acquisition / call failure raises :class:`BackendUnavailable` so one bad cell
    degrades to a 0-score instead of crashing the sweep. The heavy engine import graph is
    loaded lazily on first use, keeping the module import (and the offline tests) dependency
    free.
    """

    def __init__(self, namespace: str = "default", *, top_k_floor: int = 1) -> None:
        self.namespace = namespace
        self._engine: Any | None = None
        self._counter = 0
        self._top_k_floor = top_k_floor

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        import os

        # Connect as a read/write client of the running daemon, never as the host.
        os.environ.setdefault("KG_DAEMON_ROLE", "client")
        try:
            from agent_utilities.mcp import kg_server

            engine = kg_server._get_engine()
        except Exception as exc:  # noqa: BLE001 - any import/connect failure → unavailable
            raise BackendUnavailable(f"graph-os engine unavailable: {exc}") from exc
        if engine is None:
            raise BackendUnavailable("graph-os engine not active")
        self._engine = engine
        return engine

    def ingest_memory(
        self, text: str, context_id: int | str, event_time: float | None = None
    ) -> None:
        engine = self._get_engine()
        self._counter += 1
        node_id = f"memorydata:{self.namespace}:{context_id}:{self._counter}"
        props = {
            "name": f"{self.namespace} chunk {context_id}/{self._counter}",
            "description": str(text or ""),
            "text": str(text or ""),
            "namespace": self.namespace,
            "context_id": str(context_id),
            "source": "memorydata-bakeoff",
        }
        if event_time is not None:
            props["event_time"] = event_time
        try:
            engine.add_node(node_id, "MemoryChunk", props)
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"engine.add_node failed: {exc}") from exc

    @staticmethod
    def _row_text(row: Any) -> str:
        if isinstance(row, dict):
            for key in ("text", "content", "summary", "description", "name"):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            # Join name + description when both present (engine node shape).
            name = str(row.get("name") or "").strip()
            desc = str(row.get("description") or "").strip()
            joined = " ".join(p for p in (name, desc) if p)
            if joined:
                return joined
            return ""
        return str(row or "")

    @staticmethod
    def _row_id(row: Any) -> Any:
        if isinstance(row, dict):
            return row.get("id") or row.get("node_id") or row.get("uid")
        return None

    @staticmethod
    def _row_score(row: Any) -> float:
        if isinstance(row, dict):
            try:
                return float(row.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def recall(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        as_of: float | None = None,
    ) -> list[dict[str, Any]]:
        engine = self._get_engine()
        k = max(self._top_k_floor, int(top_k))
        try:
            if mode == "memory":
                results = engine.search_memories(query=query, top_k=k)
            elif mode == "latent":
                from agent_utilities.knowledge_graph.retrieval.latent_topology_rag import (
                    LatentTopologicalRAG,
                )

                results = LatentTopologicalRAG(engine).retrieve(query, top_k=k)
            else:
                # hybrid / rerank / bi-temporal as-of all route through hybrid search.
                kwargs: dict[str, Any] = {"query": query, "top_k": k}
                if as_of is not None:
                    import datetime as _dt

                    kwargs["as_of"] = _dt.datetime.fromtimestamp(
                        as_of, tz=_dt.UTC
                    ).isoformat()
                results = engine.search_hybrid(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"engine recall ({mode}) failed: {exc}") from exc

        rows: list[dict[str, Any]] = []
        for row in results or []:
            rows.append(
                {
                    "id": self._row_id(row),
                    "text": self._row_text(row),
                    "score": self._row_score(row),
                    "mode": mode,
                }
            )
        return rows[:k]

    def synthesize(self, query: str, domain: str = "code", intent: str = "how") -> dict[str, Any]:
        engine = self._get_engine()
        # Prefer the context-plane synthesizer (the live "explain" surface) when present.
        try:
            from agent_utilities.knowledge_graph.context_plane import (  # type: ignore
                synthesize_context,
            )

            result = synthesize_context(engine, query, domain=domain, intent=intent)
            answer = ""
            citations: list[Any] = []
            if isinstance(result, dict):
                answer = str(result.get("answer") or result.get("output") or "")
                citations = result.get("citations") or []
            else:
                answer = str(result or "")
            if answer:
                return {"answer": answer, "citations": citations, "domain": domain, "intent": intent}
        except Exception:  # noqa: BLE001 - fall through to recall+join
            pass
        memories = self.recall(query, mode="hybrid", top_k=3)
        answer = " ".join(m["text"] for m in memories if m.get("text")).strip()
        return {
            "answer": answer,
            "citations": [m["id"] for m in memories if m.get("id")],
            "domain": domain,
            "intent": intent,
        }

    def reset(self, namespace: str) -> None:
        # Best-effort: the engine has no namespace-scoped bulk delete on the client path,
        # so a reset is a no-op here. Runs isolate by unique node ids instead.
        return None


def build_client(config: dict[str, Any]) -> MemoryBackendClient:
    """Construct a backend client from ``config`` (CONCEPT:AHE-3.71).

    ``config["transport"]`` selects the implementation — ``"mock"`` (default, offline),
    ``"rest"`` (graph-os ``/graph/*`` gateway), or ``"engine"`` (the *live* in-process
    engine over its UDS socket). ``config["namespace"]`` isolates a run's memories.
    """
    transport = (config or {}).get("transport", "mock")
    namespace = (config or {}).get("namespace", "default")
    if transport == "rest":
        return GraphOSRestClient(
            base_url=config.get("base_url"),
            token=config.get("token"),
            namespace=namespace,
        )
    if transport == "engine":
        return EngineBackendClient(namespace=namespace)
    if transport == "mock":
        return MockBackendClient(namespace=namespace)
    raise ValueError(
        f"unknown transport {transport!r} (expected 'mock', 'rest', or 'engine')"
    )
