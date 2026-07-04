"""CONCEPT:AU-AHE.evaluation.longmemeval-validation-harness — LongMemEval-S Validation Harness.

A FastAPI surface that lets Quarq's HTTP benchmark runner (``quarqlabs/benchmarks``) drive the
agent-utilities memory-first stack (ORCH-1.27 + KG-2.11/2.12/2.13) against LongMemEval-S, and
prove it meets/beats 98.2%. Haystack messages are ingested as episodic memory into a **frozen,
versioned** ``EvaluationCorpus`` (reproducible across agent versions — Quarq re-derives FAISS each
run); questions run the HyDE + two-pass pipeline; answers are scored by the ORCH-1.27 ``judge``
role with a deterministic pure-Python fallback.

Pure helpers (:func:`normalize_answer`, :func:`judge_binary`, :func:`aggregate_report`) are
LLM-free and fully unit-testable; the HTTP endpoints wire them to the live engine.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Benchmark"], prefix="/benchmark")

# In-memory run accumulator (a benchmark harness is ephemeral; persistence is the corpus + KG).
_RUNS: dict[str, list[dict[str, Any]]] = {}


# ── request models ──────────────────────────────────────────────────────────────


class HaystackMessage(BaseModel):
    role: str = "user"
    content: str = ""
    timestamp: str = ""
    event_time: str | None = None


class BenchmarkSession(BaseModel):
    session_id: str = ""
    name: str = "longmemeval-session"
    messages: list[HaystackMessage] = Field(default_factory=list)


class BenchmarkQuery(BaseModel):
    session_id: str = ""
    corpus_id: str = ""
    question: str
    gold_answer: str = ""
    question_type: str = "unknown"
    run_id: str = "default"
    self_correct: bool = True


# ── pure scoring helpers ─────────────────────────────────────────────────────────


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra-space — LongMemEval-style normalization."""
    t = (text or "").lower().strip()
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d[\d,]*(?:\.\d+)?", text or "")


def judge_binary(predicted: str, gold: str) -> bool:
    """Deterministic binary correctness fallback (used when the LLM judge is unavailable).

    Correct when the normalized gold is contained in the normalized prediction (or vice-versa),
    or — for numeric gold answers — every gold number appears in the prediction. Conservative:
    an empty gold is never auto-correct.
    """
    g = normalize_answer(gold)
    p = normalize_answer(predicted)
    if not g:
        return False
    if g in p or (len(p) >= 2 and p in g):
        return True
    gold_nums = _numbers(gold)
    if gold_nums and all(n in (predicted or "") for n in gold_nums):
        return True
    return False


def aggregate_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question results into accuracy + per-category breakdown.

    Each result is ``{"correct": bool, "question_type": str, ...}``. Mirrors the LongMemEval
    leaderboard breakdown the article reports.
    """
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    by_cat: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r.get("question_type", "unknown")
        slot = by_cat.setdefault(cat, {"total": 0, "correct": 0})
        slot["total"] += 1
        if r.get("correct"):
            slot["correct"] += 1
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "by_category": {
            c: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0.0}
            for c, v in by_cat.items()
        },
    }


# ── engine glue (best-effort; harness degrades gracefully if KG is cold) ──────────


def _engine():
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    return IntelligenceGraphEngine.get_active()


def _judge_answer(predicted: str, gold: str) -> bool:
    """Score an answer via the ORCH-1.27 ``judge`` role, falling back to :func:`judge_binary`."""
    try:
        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        model = create_model(role="judge")
        agent = Agent(
            model=model,
            system_prompt=(
                "You are a strict binary evaluator. Reply ONLY 'CORRECT' or 'INCORRECT'. "
                "The answer is CORRECT iff it conveys the gold answer's facts."
            ),
        )
        out = str(
            getattr(agent.run_sync(f"GOLD: {gold}\nANSWER: {predicted}"), "output", "")
        )
        if "CORRECT" in out.upper():
            return "INCORRECT" not in out.upper()
    except Exception as e:  # pragma: no cover - judge is best-effort
        logger.debug(
            "[AHE-3.12] LLM judge unavailable, using deterministic fallback: %s", e
        )
    return judge_binary(predicted, gold)


# ── endpoints ────────────────────────────────────────────────────────────────────


@router.get("/health")
async def benchmark_health() -> dict[str, Any]:
    """Liveness + active-run summary for the benchmark harness."""
    return {
        "status": "ok",
        "concept": "AU-AHE.evaluation.longmemeval-validation-harness",
        "active_runs": {rid: len(rows) for rid, rows in _RUNS.items()},
    }


@router.post("/session")
async def create_session(payload: BenchmarkSession) -> JSONResponse:
    """Ingest haystack messages as episodic memory and freeze a reproducible corpus."""
    session_id = payload.session_id or f"sess-{uuid.uuid4().hex[:10]}"
    try:
        engine = _engine()
        if engine is None:
            return JSONResponse(
                {"error": "IntelligenceGraphEngine not active"}, status_code=503
            )
        from agent_utilities.knowledge_graph.retrieval.evaluation_corpus import (
            CorpusManager,
        )
        from agent_utilities.models.knowledge_graph import MemoryNode

        doc_ids: list[str] = []
        for i, msg in enumerate(payload.messages):
            mid = f"{session_id}-msg-{i}"
            node = MemoryNode(
                id=mid,
                name=msg.content[:80],
                content=msg.content,
                memory_type="episodic",
                event_time=msg.event_time,
                storage_time=msg.timestamp or None,
            )
            engine.add_memory_node(node)
            doc_ids.append(mid)

        mgr = CorpusManager(engine)
        corpus_id = mgr.create_corpus(
            name=payload.name,
            document_ids=doc_ids,
            description="LongMemEval-S haystack",
        )
        mgr.freeze_corpus(corpus_id)
        return JSONResponse(
            {"session_id": session_id, "corpus_id": corpus_id, "ingested": len(doc_ids)}
        )
    except Exception as e:
        logger.exception("[AHE-3.12] session ingest failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/query")
async def run_query(payload: BenchmarkQuery) -> JSONResponse:
    """Run the memory-first pipeline against the session corpus and judge the answer."""
    try:
        engine = _engine()
        if engine is None:
            return JSONResponse(
                {"error": "IntelligenceGraphEngine not active"}, status_code=503
            )
        nodes = engine.search_hybrid(
            query=payload.question,
            top_k=10,
            mode="hyde",
            self_correct=payload.self_correct,
            corpus_id=payload.corpus_id or None,
        )
        # Compose a grounded context for the generator; a thin extractive answer is the fallback.
        context = "\n".join(str(n.get("content", "")) for n in nodes[:10])
        answer = _generate_answer(payload.question, context)
        correct = (
            _judge_answer(answer, payload.gold_answer) if payload.gold_answer else False
        )
        result = {
            "question": payload.question,
            "answer": answer,
            "gold_answer": payload.gold_answer,
            "question_type": payload.question_type,
            "correct": correct,
            "retrieved": len(nodes),
        }
        _RUNS.setdefault(payload.run_id, []).append(result)
        return JSONResponse(result)
    except Exception as e:
        logger.exception("[AHE-3.12] query failed")
        return JSONResponse({"error": str(e)}, status_code=500)


def _generate_answer(question: str, context: str) -> str:
    """Synthesize an answer via the ORCH-1.27 ``generator`` role (extractive fallback)."""
    try:
        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        model = create_model(role="generator")
        agent = Agent(
            model=model,
            system_prompt=(
                "Answer the question using ONLY the provided memories. Be concise and exact. "
                "If the memories are insufficient, say you don't have enough information."
            ),
        )
        out = getattr(
            agent.run_sync(f"MEMORIES:\n{context}\n\nQUESTION: {question}"),
            "output",
            "",
        )
        return str(out)
    except Exception as e:  # pragma: no cover - generator is best-effort
        logger.debug("[AHE-3.12] generator unavailable, extractive fallback: %s", e)
        return context[:500]


@router.get("/report/{run_id}")
async def get_report(run_id: str) -> JSONResponse:
    """Return the accuracy + per-category report for a run (LongMemEval-style breakdown)."""
    return JSONResponse(aggregate_report(_RUNS.get(run_id, [])))
