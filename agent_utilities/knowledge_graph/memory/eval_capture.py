#!/usr/bin/python
from __future__ import annotations

"""KG Eval Capture — Regression testing for Knowledge Graph changes.

CONCEPT:AHE-3.1 — Eval & Distillation

Records real queries and their retrieved results natively to the Knowledge Graph
as EvaluationRecordNode entries, enabling replay-based regression testing.

Controlled by the ``KG_EVAL_CAPTURE`` environment variable (default: disabled).
"""


import json
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ...models.knowledge_graph import EvaluationRecordNode

logger = logging.getLogger(__name__)

# Feature gate — disabled by default
_EVAL_CAPTURE_ENABLED = os.environ.get("KG_EVAL_CAPTURE", "").lower() in (
    "true",
    "1",
    "yes",
)


class EvalReplayResult(BaseModel):
    total_queries: int = 0
    mean_jaccard_at_k: float = 0.0
    top_1_stability: float = 0.0
    mean_latency_delta_ms: float = 0.0
    regressions: list[dict[str, Any]] = Field(default_factory=list)


class KGEvalCapture:
    """Lightweight eval harness for Knowledge Graph retrieval regression testing.

    CONCEPT:AHE-3.1 — Eval & Distillation

    Stores query-result pairs as EvaluationRecordNode entries directly in the KG.
    Provides replay functionality to measure retrieval drift after KG changes.
    """

    def __init__(self, knowledge_engine: Any, enabled: bool | None = None) -> None:
        self.ke = knowledge_engine
        self.enabled = enabled if enabled is not None else _EVAL_CAPTURE_ENABLED

    def capture(
        self,
        query: str,
        method: str,
        result_node_ids: list[str],
        scores: list[float] | None = None,
        latency_ms: float | None = None,
        schema_pack: str | None = None,
    ) -> None:
        if not self.enabled or not self.ke:
            return

        import uuid

        try:
            record = EvaluationRecordNode(  # type: ignore[call-arg]
                id=f"eval:{uuid.uuid4().hex[:8]}",
                name=f"Eval: {query[:20]}",
                query=query,
                method=method,
                result_node_ids=result_node_ids,
                evidence=json.dumps(scores) if scores else "",
                latency_ms=latency_ms,
                schema_pack=schema_pack,
                evaluator="kg_capture",
            )
            self.ke.ogm.save(record)
        except Exception as e:
            logger.debug("Eval capture write failed: %s", e)

    def replay(
        self,
        search_fn: Callable[[str], list[dict[str, Any]]],
        k: int = 10,
        regression_threshold: float = 0.5,
    ) -> EvalReplayResult:
        if not self.ke:
            return EvalReplayResult()

        records = self.ke.ogm.find(
            EvaluationRecordNode, properties={"evaluator": "kg_capture"}
        )
        if not records:
            return EvalReplayResult()

        jaccard_scores: list[float] = []
        top_1_matches: list[bool] = []
        latency_deltas: list[float] = []
        regressions: list[dict[str, Any]] = []

        for record in records:
            if not record.query or not record.result_node_ids:
                continue

            query = record.query
            original_ids = record.result_node_ids[:k]
            original_latency = record.latency_ms

            original_set = set(original_ids)

            start = time.perf_counter()
            current_results = search_fn(query)
            current_latency = (time.perf_counter() - start) * 1000

            current_ids = [r.get("id", "") for r in current_results[:k]]
            current_set = set(current_ids)

            if original_set or current_set:
                intersection = original_set & current_set
                union = original_set | current_set
                jaccard = len(intersection) / len(union) if union else 1.0
            else:
                jaccard = 1.0

            jaccard_scores.append(jaccard)

            top_1_match = (
                bool(original_ids)
                and bool(current_ids)
                and original_ids[0] == current_ids[0]
            )
            top_1_matches.append(top_1_match)

            if original_latency is not None:
                latency_deltas.append(current_latency - original_latency)

            if jaccard < regression_threshold:
                regressions.append(
                    {
                        "query": query,
                        "jaccard_at_k": round(jaccard, 4),
                        "original_ids": original_ids,
                        "current_ids": current_ids,
                    }
                )

        total = len(jaccard_scores)
        return EvalReplayResult(
            total_queries=total,
            mean_jaccard_at_k=round(sum(jaccard_scores) / total, 4) if total else 0.0,
            top_1_stability=round(sum(top_1_matches) / total, 4) if total else 0.0,
            mean_latency_delta_ms=round(sum(latency_deltas) / len(latency_deltas), 2)
            if latency_deltas
            else 0.0,
            regressions=regressions,
        )

    def count(self) -> int:
        if not self.ke:
            return 0
        records = self.ke.ogm.find(
            EvaluationRecordNode, properties={"evaluator": "kg_capture"}
        )
        return len(records)
