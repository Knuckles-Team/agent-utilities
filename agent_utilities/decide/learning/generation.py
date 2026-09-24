"""EH-397: corpus re-embedding as a governed embedding-GENERATION swap.

Stored vectors are never mutated in place and a store's space never changes,
so a re-embed with a fine-tuned model is a new generation of the graph:

1. under a capacity lease, the embedding model is fine-tuned through the
   substrate trainer (EH-347) -- AU owns models and training;
2. the corpus is re-embedded into a SHADOW graph with the new model (a fresh
   store; the engine builds its ANN generation like any other graph's);
3. EG dual-serves the log's independently judged runs against both
   generations -- AU embeds each run's query in both spaces, EG probes both as
   the caller may see them -- and issues a receipt (coverage, sign test,
   top-1 score PSI);
4. only a passing receipt measured against the generation active NOW moves
   the logical graph's pointer; rollback returns to the previous generation,
   which the swap never deletes.

The retrieval path resolves a logical graph through that pointer
(:func:`active_generation`), so activation and rollback are pointer moves the
next query sees.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from agent_utilities.decide.learning.ops import q16, result_of, retrieval_op
from agent_utilities.decide.learning.session import LearningSession, current_session

logger = logging.getLogger(__name__)

#: How long a resolved generation pointer is reused before it is read again.
POINTER_TTL_S = 30.0
#: The judged runs one evaluation replays at most (EG's own bound).
MAX_EVAL_ITEMS = 1024
_JUDGED_SQL = (
    "SELECT DISTINCT d.record_id FROM decisions d JOIN evaluations e "
    "ON d.record_id = e.record_id WHERE d.question_id = '{question}' "
    "AND e.success = true ORDER BY d.record_id LIMIT {limit}"
)


class Embedder(Protocol):
    model_name: str

    def get_text_embedding(self, text: str) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    """One swap: the logical graph, its active generation and the shadow's."""

    logical: str
    active_graph: str
    active_space: str
    shadow_graph: str
    shadow_space: str
    question_id: str = "au.retrieval.plan"
    top_k: int = 20
    min_eval_items: int = 50
    max_score_psi_q16: int = 1 << 14


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    """Where a swap stopped, and EG's receipt/pointer when it got that far."""

    activated: bool
    stage: str
    detail: str = ""
    receipt: Mapping[str, Any] | None = None
    pointer: Mapping[str, Any] | None = None


#: Fine-tune the base embedder under the held lease; the trained model's name,
#: or ``None`` when the substrate did not produce one (recorded/skipped).
Trainer = Callable[[str], str | None]
#: Re-embed ``active_graph``'s corpus into ``shadow_graph`` with ``model``;
#: returns how many rows were embedded.
Reembedder = Callable[[str, str, str], Awaitable[int]]
#: The capacity lease the training and the re-embed run under.
CapacityLease = Callable[[], AbstractContextManager[Any]]


def _cell(cell: Any) -> Any:
    return cell.get("value") if isinstance(cell, Mapping) else cell


def _query_param(entry: Mapping[str, Any]) -> str | None:
    params = ((entry.get("record") or {}).get("inputs") or {}).get("params") or []
    for param in params:
        value = param.get("value") if isinstance(param, Mapping) else None
        if isinstance(value, Mapping) and param.get("name") == "query":
            return str(value.get("value"))
    return None


async def judged_queries(
    session: LearningSession, question_id: str, limit: int = MAX_EVAL_ITEMS
) -> list[tuple[str, str]]:
    """``(record_id, query text)`` of the question's successfully evaluated runs
    the caller may read (EG's SQL view, then each record's ``query`` param)."""
    sql = _JUDGED_SQL.format(question=question_id.replace("'", "''"), limit=int(limit))
    view = await session.asend({"op": "query", "tenant_id": session.tenant, "sql": sql})
    body = getattr(view, "payload", view) or {}
    ids = [str(_cell(row[0])) for row in body.get("rows") or [] if row]
    out: list[tuple[str, str]] = []
    for record_id in ids:
        op = {"op": "get", "tenant_id": session.tenant, "record_id": record_id}
        entry = await session.asend(op)
        text = _query_param(getattr(entry, "payload", entry) or {})
        if text:
            out.append((record_id, text))
    return out


def eval_items(
    queries: Sequence[tuple[str, str]], active: Embedder, shadow: Embedder
) -> list[dict[str, Any]]:
    """Each judged query embedded in BOTH spaces (EG has no model)."""
    return [
        {
            "record_id": record_id,
            "active_q16": q16(active.get_text_embedding(text)),
            "shadow_q16": q16(shadow.get_text_embedding(text)),
        }
        for record_id, text in queries
    ]


def eval_request(plan: GenerationPlan, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "logical": plan.logical,
        "active_graph": plan.active_graph,
        "active_space": plan.active_space,
        "shadow_graph": plan.shadow_graph,
        "shadow_space": plan.shadow_space,
        "items": items,
        "top_k": plan.top_k,
        "min_eval_items": plan.min_eval_items,
        "max_score_psi_q16": plan.max_score_psi_q16,
    }


@dataclass
class GenerationSwap:
    """Train -> re-embed -> dual-serve -> activate, over EG's governed pointer.

    The session's principal must hold EG's ``admin:decision-head`` action.
    """

    session: LearningSession
    train: Trainer
    reembed: Reembedder
    lease: CapacityLease
    embedder_for: Callable[[str], Embedder]

    async def _ask(self, action: str, kind: str, **fields: Any) -> Mapping[str, Any]:
        op = retrieval_op(self.session.tenant, action, **fields)
        return result_of(await self.session.asend(op), kind)

    async def _build_shadow(self, plan: GenerationPlan, base_model: str) -> str | None:
        with self.lease():
            model = self.train(base_model)
            if model is None:
                return None
            embedded = await self.reembed(plan.active_graph, plan.shadow_graph, model)
        logger.info(
            "shadow generation %s: %d rows embedded", plan.shadow_graph, embedded
        )
        return model

    async def evaluate(
        self, plan: GenerationPlan, base_model: str, shadow_model: str
    ) -> Mapping[str, Any]:
        queries = await judged_queries(self.session, plan.question_id)
        items = eval_items(
            queries, self.embedder_for(base_model), self.embedder_for(shadow_model)
        )
        return await self._ask(
            "evaluate_generation", "generation", request=eval_request(plan, items)
        )

    async def run(self, plan: GenerationPlan, base_model: str) -> GenerationOutcome:
        shadow_model = await self._build_shadow(plan, base_model)
        if shadow_model is None:
            return GenerationOutcome(False, "train", "the substrate produced no model")
        evaluated = await self.evaluate(plan, base_model, shadow_model)
        receipt = evaluated.get("receipt") or {}
        if not receipt.get("passed"):
            return GenerationOutcome(False, "eval", "the receipt did not pass", receipt)
        pointer = await self._ask(
            "activate_generation",
            "pointer",
            logical=plan.logical,
            shadow_graph=plan.shadow_graph,
            receipt_digest=evaluated["receipt_digest"],
        )
        _forget(plan.logical)
        return GenerationOutcome(True, "activated", receipt=receipt, pointer=pointer)

    async def rollback(self, logical: str) -> Mapping[str, Any]:
        pointer = await self._ask("rollback_generation", "pointer", logical=logical)
        _forget(logical)
        return pointer


#: logical graph -> (resolved generation, when it was read).
_RESOLVED: dict[str, tuple[str, float]] = {}


def _forget(logical: str) -> None:
    _RESOLVED.pop(logical, None)


def resolve_generation(
    logical: str, *, now: Callable[[], float] = time.monotonic
) -> str:
    """The graph ``logical`` resolves to: its generation pointer's target, or
    itself. Read through EG at most once per :data:`POINTER_TTL_S`; with no
    session (or an unreadable pointer) the logical graph itself."""
    cached = _RESOLVED.get(logical)
    if cached is not None and now() - cached[1] < POINTER_TTL_S:
        return cached[0]
    session = current_session()
    if session is None:
        return logical
    op = retrieval_op(session.tenant, "generation_status", logical=logical)
    try:
        pointer = result_of(session.send(op), "pointer")
    except Exception as exc:
        logger.warning("generation pointer of %s unreadable: %s", logical, exc)
        return logical
    target = str(((pointer.get("active") or {}).get("target")) or logical)
    _RESOLVED[logical] = (target, now())
    return target


def active_generation(graph: Any) -> Any:
    """``graph`` routed to its logical graph's active generation (a named-graph
    view of the same transport); ``graph`` itself when it is its own."""
    name = getattr(graph, "graph_name", None)
    if graph is None or not name or not hasattr(graph, "for_graph"):
        return graph
    target = resolve_generation(str(name))
    return graph if target == name else graph.for_graph(target)


__all__ = [
    "POINTER_TTL_S",
    "CapacityLease",
    "Embedder",
    "GenerationOutcome",
    "GenerationPlan",
    "GenerationSwap",
    "Reembedder",
    "Trainer",
    "active_generation",
    "eval_items",
    "eval_request",
    "judged_queries",
    "resolve_generation",
]
