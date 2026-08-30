#!/usr/bin/python
from __future__ import annotations

"""Drive the graph-os retrieval configs across MemoryData families (CONCEPT:AU-AHE.harness.when-outcome-names-agent).

A *bake-off* runs each retrieval config (:data:`RETRIEVAL_CONFIGS`) over each task family —
memorizing the family's context chunks, answering its queries, and scoring exact-match plus
ROUGE-L. The loop mirrors :mod:`agent_utilities.rlm.benchmarks.runner`: every (config ×
family × task) cell is independent, a failed cell scores 0 and the sweep continues, and each
cell records mean memorize/query latency so a config's cost shows up next to its accuracy.

A *family* is a dict::

    {"tag": "membench-update", "name": "MemBench-update",
     "context_chunks": ["...", "..."],
     "tasks": [{"task": "recall", "queries": [{"question": "...", "answer": "..."}]}]}

so the harness is corpus-agnostic — a real MemoryData preset and a synthetic fixture share
the same shape.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_utilities.harness.memorydata.adapter import (
    RETRIEVAL_CONFIGS,
    GraphOSMemoryMethod,
)

if TYPE_CHECKING:
    from agent_utilities.harness.memorydata.router_method import GraphOSRouterMethod

__all__ = ["BakeoffResult", "run_bakeoff", "rouge_l", "ROUTER_CONFIG"]

# The router is a meta-config: it is NOT a single retrieval surface in
# ``RETRIEVAL_CONFIGS`` but a :class:`GraphOSRouterMethod` that picks a surface per query
# from the family tag. ``run_bakeoff`` recognizes this name and drives the router method so
# the bake-off can pit "route per family" against every single config (CONCEPT:AU-AHE.harness.callers-feed-back-per).
ROUTER_CONFIG = "graphos_router"


@dataclass
class BakeoffResult:
    """One (config × family × task) cell of the bake-off (CONCEPT:AU-AHE.harness.when-outcome-names-agent)."""

    config: str
    family: str
    task: str
    n: int
    exact_match: float
    rouge_l: float
    judge_score: float
    mean_query_s: float
    mean_mem_s: float
    notes: str = ""
    is_router: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 over whitespace tokens (longest-common-subsequence based).

    A tiny dependency-free implementation used when ``rouge_score`` is not installed.
    """
    pred = (prediction or "").lower().split()
    ref = (reference or "").lower().split()
    if not pred or not ref:
        return 0.0
    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of token lists ``a`` and ``b``."""
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _rouge_scorer() -> Callable[[str, str], float]:
    """Return a ROUGE-L scorer — the ``rouge_score`` package if present, else the local LCS."""
    try:
        from rouge_score import rouge_scorer  # type: ignore

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        def _score(prediction: str, reference: str) -> float:
            return scorer.score(reference or "", prediction or "")["rougeL"].fmeasure

        return _score
    except Exception:  # noqa: BLE001 - optional dep absent → local implementation
        return rouge_l


def _iter_queries(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a task record into a list of ``{question, answer}`` query dicts."""
    queries = task.get("queries")
    if queries:
        return list(queries)
    if "question" in task:
        return [{"question": task["question"], "answer": task.get("answer", "")}]
    return []


def _build_agent(
    config: str,
    family_tag: str,
    client_transport: str,
    model: str,
    top_k: Any,
    is_router: bool,
) -> GraphOSRouterMethod | GraphOSMemoryMethod:
    """Construct the adapter for one config/family cell."""
    agent_config = {
        "agent_name": f"{model}_{config}",
        "retrieval": config if not is_router else None,
        "transport": client_transport,
        "top_k": top_k,
    }
    dataset_config = {
        "sub_dataset": str(family_tag),
        "dataset": "memorydata",
    }
    if is_router:
        from agent_utilities.harness.memorydata.router_method import GraphOSRouterMethod

        agent_config.pop("retrieval", None)
        return GraphOSRouterMethod(
            agent_config=agent_config,
            dataset_config=dataset_config,
            family_tag=str(family_tag),
        )
    return GraphOSMemoryMethod(
        agent_config=agent_config,
        dataset_config=dataset_config,
    )


def _memorize_context(
    agent: GraphOSRouterMethod | GraphOSMemoryMethod, context_chunks: Any
) -> list[float]:
    """Memorize every context chunk and return its measured latency."""
    mem_times: list[float] = []
    for idx, chunk in enumerate(context_chunks):
        resp = agent.send_message(chunk, memorizing=True, context_id=idx)
        mem_times.append(float(resp.get("memory_construction_time", 0.0)))
    return mem_times


def _score_queries(
    agent: GraphOSRouterMethod | GraphOSMemoryMethod,
    queries: list[dict[str, Any]],
    rouge: Callable[[str, str], float],
    judge_fn: Callable[[str, str], bool] | None,
) -> tuple[int, float, int, list[float]]:
    """Answer queries and return exact, ROUGE, judge, and latency totals."""
    em_hits = 0
    rouge_sum = 0.0
    judge_hits = 0
    query_times: list[float] = []
    for query in queries:
        question = query.get("question", "")
        gold = query.get("answer", "")
        resp = agent.send_message(
            question,
            memorizing=False,
            query_id=query.get("query_id"),
            context_id=query.get("context_id"),
            eval_metadata=query.get("eval_metadata"),
        )
        output = str(resp.get("output", "") or "")
        query_times.append(float(resp.get("query_time_len", 0.0)))
        if _exact_hit(output, gold):
            em_hits += 1
        rouge_sum += rouge(output, gold)
        if judge_fn is not None and judge_fn(output, gold):
            judge_hits += 1
    return em_hits, rouge_sum, judge_hits, query_times


def _error_result(
    config: str,
    family_tag: str,
    task_name: str,
    n: int,
    is_router: bool,
    exc: Exception,
) -> BakeoffResult:
    """Build the zeroed result used when one bakeoff cell fails."""
    return BakeoffResult(
        config=config,
        family=family_tag,
        task=task_name,
        n=n,
        exact_match=0.0,
        rouge_l=0.0,
        judge_score=0.0,
        mean_query_s=0.0,
        mean_mem_s=0.0,
        notes=f"error: {type(exc).__name__}: {exc}",
        is_router=is_router,
    )


def _scored_result(
    config: str,
    family_tag: str,
    task_name: str,
    n: int,
    is_router: bool,
    judge_fn: Callable[[str, str], bool] | None,
    em_hits: int,
    rouge_sum: float,
    judge_hits: int,
    query_times: list[float],
    mem_times: list[float],
) -> BakeoffResult:
    """Build the rounded result for a successfully completed bakeoff cell."""
    return BakeoffResult(
        config=config,
        family=family_tag,
        task=task_name,
        n=n,
        exact_match=round(em_hits / n, 4) if n else 0.0,
        rouge_l=round(rouge_sum / n, 4) if n else 0.0,
        judge_score=round(judge_hits / n, 4) if (n and judge_fn) else 0.0,
        mean_query_s=round(sum(query_times) / len(query_times), 6)
        if query_times
        else 0.0,
        mean_mem_s=round(sum(mem_times) / len(mem_times), 6) if mem_times else 0.0,
        is_router=is_router,
    )


def _run_cell(
    config: str,
    family_tag: str,
    family: dict[str, Any],
    task: dict[str, Any],
    context_chunks: Any,
    client_transport: str,
    judge_fn: Callable[[str, str], bool] | None,
    model: str,
    rouge: Callable[[str, str], float],
    is_router: bool,
) -> BakeoffResult:
    """Run one independent config/family/task cell."""
    task_name = task.get("task") or task.get("name") or "task"
    queries = _iter_queries(task)
    n = len(queries)
    try:
        agent = _build_agent(
            config,
            family_tag,
            client_transport,
            model,
            family.get("top_k", 10),
            is_router,
        )
        mem_times = _memorize_context(agent, context_chunks)
        em_hits, rouge_sum, judge_hits, query_times = _score_queries(
            agent, queries, rouge, judge_fn
        )
    except Exception as exc:  # noqa: BLE001 - failed cell scores 0, sweep continues
        return _error_result(config, str(family_tag), str(task_name), n, is_router, exc)
    return _scored_result(
        config,
        str(family_tag),
        str(task_name),
        n,
        is_router,
        judge_fn,
        em_hits,
        rouge_sum,
        judge_hits,
        query_times,
        mem_times,
    )


def _run_config(
    config: str,
    families: list[dict[str, Any]],
    client_transport: str,
    judge_fn: Callable[[str, str], bool] | None,
    model: str,
    rouge: Callable[[str, str], float],
) -> list[BakeoffResult]:
    """Run one known config across all family/task cells."""
    is_router = config == ROUTER_CONFIG
    results: list[BakeoffResult] = []
    for family in families:
        family_tag = family.get("tag") or family.get("name") or "family"
        context_chunks = family.get("context_chunks", [])
        for task in family.get("tasks", []):
            results.append(
                _run_cell(
                    config,
                    str(family_tag),
                    family,
                    task,
                    context_chunks,
                    client_transport,
                    judge_fn,
                    model,
                    rouge,
                    is_router,
                )
            )
    return results


def run_bakeoff(
    configs: list[str],
    families: list[dict[str, Any]],
    client_transport: str = "mock",
    judge: Callable[[str, str], bool] | None = None,
    *,
    model: str = "graphos",
) -> list[BakeoffResult]:
    """Run ``configs`` over ``families`` and return one :class:`BakeoffResult` per cell.

    For each (config × family × task) the adapter is freshly instantiated, the family's
    ``context_chunks`` are memorized, then each query is answered and scored with exact-match
    + ROUGE-L (and ``judge`` if provided). A cell that errors scores 0 and the sweep
    continues — every cell is independent.
    """
    rouge = _rouge_scorer()
    judge_fn = judge
    results: list[BakeoffResult] = []

    for config in configs:
        is_router = config == ROUTER_CONFIG
        if config not in RETRIEVAL_CONFIGS and not is_router:
            results.append(
                BakeoffResult(
                    config=config,
                    family="*",
                    task="*",
                    n=0,
                    exact_match=0.0,
                    rouge_l=0.0,
                    judge_score=0.0,
                    mean_query_s=0.0,
                    mean_mem_s=0.0,
                    notes=f"unknown config {config!r}",
                )
            )
            continue
        results.extend(
            _run_config(
                config,
                families,
                client_transport,
                judge_fn,
                model,
                rouge,
            )
        )
    return results


def _exact_hit(prediction: str, gold: str) -> bool:
    """Normalized exact-match / containment — the deterministic correctness floor."""
    g = (gold or "").strip().lower()
    p = (prediction or "").strip().lower()
    if not g:
        return False
    return g == p or g in p
