#!/usr/bin/python
from __future__ import annotations

"""MemoryData agent adapter for the graph-os memory stack (CONCEPT:AU-AHE.harness.hardening-transparency-surface).

MemoryData (``utils/agent.py::AgentWrapper``) drives every method through two calls:

* **memorize** — ``agent.send_message(chunk, memorizing=True, context_id=<int>)`` stores a
  context chunk; the return value is ignored.
* **query** — ``agent.send_message(question, memorizing=False, query_id=..., context_id=...,
  eval_metadata=...)`` must return a dict shaped exactly like
  ``AgentWrapper._create_standard_response``:
  ``{"output", "input_len", "output_len", "memory_construction_time", "query_time_len"}``.

:class:`GraphOSMemoryMethod` implements that contract over a pluggable
:class:`~agent_utilities.harness.memorydata.client.MemoryBackendClient`. The ``retrieval``
knob in the agent config selects one of :data:`RETRIEVAL_CONFIGS` — each maps to a graph-os
retrieval surface (semantic/hybrid, bi-temporal as-of, context-plane synthesis, latent,
memory-facts, or graph rerank). Scoring helpers are reused from the gateway benchmark router
when importable, with a dependency-free fallback so the adapter works offline.
"""

import time
from typing import Any

from agent_utilities.harness.memorydata.client import build_client

__all__ = ["GraphOSMemoryMethod", "RETRIEVAL_CONFIGS"]


# ── retrieval presets ─────────────────────────────────────────────────────────────────
# Each name maps a MemoryData config to one graph-os retrieval surface. ``mode`` is the
# recall mode passed to the backend; ``uses_synthesize`` routes the query through the
# context-plane explain path instead of raw recall; ``uses_as_of`` enables bi-temporal
# point-in-time recall; ``maintenance`` is a free-form hint for future consolidation hooks.
RETRIEVAL_CONFIGS: dict[str, dict[str, Any]] = {
    "graphos_semantic_hnsw": {"mode": "hybrid"},
    "graphos_bitemporal_asof": {"mode": "hybrid", "uses_as_of": True},
    "graphos_context_plane": {
        "uses_synthesize": True,
        "domain": "code",
        "intent": "how",
    },
    "graphos_latent": {"mode": "latent"},
    "graphos_rlm_facts": {"mode": "memory"},
    "graphos_graph_rerank": {"mode": "rerank"},
}


# ── scoring helpers (gateway reuse w/ offline fallback) ─────────────────────────────────
def _scoring_helpers() -> tuple[Any, Any]:
    """Return ``(normalize_answer, judge_binary)`` — the gateway pair if importable.

    Imported lazily so the heavy gateway/engine import graph never loads at module import.
    Falls back to a local minimal normalizer + exact/substring judge when the gateway (or
    its engine deps) is unavailable.
    """
    try:
        from agent_utilities.server.routers.benchmark import (  # type: ignore
            judge_binary,
            normalize_answer,
        )

        return normalize_answer, judge_binary
    except Exception:  # noqa: BLE001 - any import/engine failure → local fallback
        return _local_normalize_answer, _local_judge_binary


def _local_normalize_answer(text: str) -> str:
    """Lowercase, drop articles + punctuation, squeeze whitespace (LongMemEval-style)."""
    import re

    t = (text or "").lower().strip()
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _local_judge_binary(predicted: str, gold: str) -> bool:
    """Exact-match (after normalization) with substring containment — offline fallback."""
    g = _local_normalize_answer(gold)
    p = _local_normalize_answer(predicted)
    if not g:
        return False
    return g == p or g in p or (len(p) >= 2 and p in g)


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) — avoids a tiktoken/tokenizer dependency."""
    return max(0, len(text or "") // 4)


class GraphOSMemoryMethod:
    """Graph-os memory method conforming to MemoryData's agent contract (CONCEPT:AU-AHE.harness.hardening-transparency-surface).

    Construct with ``agent_config`` (carrying ``agent_name``, ``retrieval``, ``top_k``,
    ``transport`` …) and ``dataset_config``; ``send_message`` then stores chunks and answers
    queries through the configured graph-os retrieval surface, returning the exact 5-key
    standard response MemoryData expects.
    """

    def __init__(
        self,
        agent_config: dict[str, Any],
        dataset_config: dict[str, Any] | None = None,
        load_agent_from: str | None = None,
    ) -> None:
        self.agent_config = dict(agent_config or {})
        self.dataset_config = dict(dataset_config or {})
        self.load_agent_from = load_agent_from

        self.agent_name = self.agent_config.get("agent_name", "graphos")
        self.retrieval = self.agent_config.get("retrieval", "graphos_semantic_hnsw")
        if self.retrieval not in RETRIEVAL_CONFIGS:
            raise ValueError(
                f"unknown retrieval config {self.retrieval!r}; "
                f"expected one of {sorted(RETRIEVAL_CONFIGS)}"
            )
        self.spec = RETRIEVAL_CONFIGS[self.retrieval]
        self.top_k = int(self.agent_config.get("top_k", 10))

        sub = (
            self.dataset_config.get("sub_dataset")
            or self.dataset_config.get("dataset")
            or "run"
        )
        self.namespace = f"{self.agent_name}:{sub}"

        client_config = {
            "transport": self.agent_config.get("transport", "mock"),
            "namespace": self.namespace,
            "base_url": self.agent_config.get("base_url"),
            "token": self.agent_config.get("token"),
        }
        self.client = build_client(client_config)

        self._normalize_answer, self._judge_binary = _scoring_helpers()

    # -- the MemoryData contract -----------------------------------------------------------
    def send_message(
        self,
        message: Any,
        memorizing: bool = False,
        query_id: Any | None = None,
        context_id: Any | None = None,
        eval_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Memorize a chunk (``memorizing=True``) or answer a query, MemoryData-style.

        Returns the standard 5-key response in both branches; the memorize branch returns an
        empty ``output`` with a measured ``memory_construction_time``.
        """
        text = self._coerce_text(message)

        if memorizing:
            t0 = time.perf_counter()
            self.client.ingest_memory(
                text, context_id=context_id if context_id is not None else 0
            )
            dt = time.perf_counter() - t0
            return self._standard_response("", 0, 0, dt, 0.0)

        t0 = time.perf_counter()
        if self.spec.get("uses_synthesize"):
            result = self.client.synthesize(
                text,
                domain=self.spec.get("domain", "code"),
                intent=self.spec.get("intent", "how"),
            )
            output = str(result.get("answer", "") or "")
        else:
            as_of = (
                self._resolve_as_of(eval_metadata)
                if self.spec.get("uses_as_of")
                else None
            )
            memories = self.client.recall(
                text,
                mode=self.spec.get("mode", "hybrid"),
                top_k=self.top_k,
                as_of=as_of,
            )
            output = self._compose_answer(memories)
        query_time = time.perf_counter() - t0

        return self._standard_response(
            output,
            _estimate_tokens(text),
            _estimate_tokens(output),
            0.0,
            query_time,
        )

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _coerce_text(message: Any) -> str:
        """Normalize a structured MemoryData chunk (or plain string) into text."""
        if isinstance(message, dict):
            return str(message.get("storage_text") or message.get("text") or "")
        return str(message if message is not None else "")

    @staticmethod
    def _resolve_as_of(eval_metadata: dict[str, Any] | None) -> float | None:
        """Pull a point-in-time cutoff from eval metadata for bi-temporal recall."""
        if not eval_metadata:
            return None
        for key in ("as_of", "as_of_time", "valid_at", "tx_to"):
            val = eval_metadata.get(key)
            if isinstance(val, int | float):
                return float(val)
        return None

    def _compose_answer(
        self, memories: list[dict[str, Any]], max_chars: int = 2000
    ) -> str:
        """Join the recalled memory texts into one answer string, trimmed to ``max_chars``."""
        parts: list[str] = []
        used = 0
        for mem in memories:
            chunk = str(mem.get("text", "") or "").strip()
            if not chunk:
                continue
            if used + len(chunk) > max_chars and parts:
                break
            parts.append(chunk)
            used += len(chunk)
        return " ".join(parts).strip()

    @staticmethod
    def _standard_response(
        output: str,
        input_len: int,
        output_len: int,
        mem_time: float,
        query_time: float,
    ) -> dict[str, Any]:
        """Build the exact response dict MemoryData's ``_create_standard_response`` returns."""
        return {
            "output": output,
            "input_len": input_len,
            "output_len": output_len,
            "memory_construction_time": mem_time,
            "query_time_len": query_time,
        }
