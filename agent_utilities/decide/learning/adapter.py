"""EH-396: promote a query-side embedding adapter through EG's governed path.

EG fits the adapter (a small bounded low-rank head on the QUERY vector only)
from independently judged retrieval runs, evaluates the quantised body on a
held-out share and pins both by digest. This promoter only moves it through
EG's gates -- fit, then activate with the passing receipt that qualified
exactly that body -- and rolls back. It fits nothing itself, and nothing it
does can widen what a query may see: EG applies the adapter inside the probe,
after row-level security has fixed the candidate set.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent_utilities.decide.learning.ops import ALL_TIME
from agent_utilities.decide.learning.session import LearningSession


@dataclass(frozen=True, slots=True)
class AdapterPlan:
    """What to fit: the graph, AU's space identity, the bounds and the gate."""

    graph: str
    space_digest: str
    rank: int = 4
    max_gain_q16: int = 1 << 14
    holdout_per_mille: int = 250
    min_eval_items: int = 50
    question_id: str | None = "au.retrieval.plan"


@dataclass(frozen=True, slots=True)
class AdapterPromotion:
    """The pointer after a promotion, or the step that stopped it and why."""

    promoted: bool
    stage: str
    detail: str = ""
    receipt: Mapping[str, Any] | None = None
    pointer: Mapping[str, Any] | None = None


def fit_request(plan: AdapterPlan) -> dict[str, Any]:
    return {
        "graph": plan.graph,
        "space_digest": plan.space_digest,
        "question_id": plan.question_id,
        "window": dict(ALL_TIME),
        "rank": plan.rank,
        "max_gain_q16": plan.max_gain_q16,
        "holdout_per_mille": plan.holdout_per_mille,
        "min_eval_items": plan.min_eval_items,
    }


@dataclass
class AdapterPromoter:
    """Fit -> activate-with-receipt -> (rollback), over ``DecisionLog.retrieval``.

    The session's principal must hold EG's ``admin:decision-head`` action.
    """

    session: LearningSession

    async def promote(self, plan: AdapterPlan) -> AdapterPromotion:
        fitted = await self.session.ask(
            "fit_adapter", "fitted", request=fit_request(plan)
        )
        receipt = fitted.get("receipt") or {}
        if not receipt.get("passed"):
            return AdapterPromotion(
                False, "eval", "the held-out receipt did not pass", receipt
            )
        pointer = await self.session.ask(
            "activate_adapter",
            "pointer",
            graph=plan.graph,
            adapter_digest=fitted["adapter_digest"],
            receipt_digest=fitted["receipt_digest"],
        )
        return AdapterPromotion(True, "activated", receipt=receipt, pointer=pointer)

    async def rollback(self, graph: str) -> Mapping[str, Any]:
        return await self.session.ask("rollback_adapter", "pointer", graph=graph)

    async def status(self, graph: str) -> Mapping[str, Any]:
        return await self.session.ask("adapter_status", "pointer", graph=graph)


__all__ = ["AdapterPlan", "AdapterPromoter", "AdapterPromotion", "fit_request"]
