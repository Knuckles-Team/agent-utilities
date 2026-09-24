"""EH-040: post-run promotion of a fitted decision head.

After runs have been evaluated, a point's head is re-fitted from EG's own
decision log and promoted only through EG's governed path
(DECIDE-LAYER-DESIGN §4.2, invariant 9):

1. ``DecisionFit`` (admin job) fits a draft head from the logged, independently
   evaluated records of one question -- a Blob CAS draft, never a component;
2. ``DecisionEval`` (admin job) evaluates that exact draft (by content digest)
   and issues a receipt; every failed promotion gate is named;
3. only a PASSED receipt is handed to the publisher, which publishes the draft
   as a ``DecisionHead`` component carrying the receipt digest -- EG refuses a
   head publish without a matching receipt. The publisher is the process that
   owns the admin mutation context (graph-os); AU never mints one.

Nothing here decides anything: it only moves a head through EG's gates.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.layers.clients import generated

#: Publishes ``(draft head body, receipt digest)``; returns the head's pin.
HeadPublisher = Callable[[Mapping[str, Any], str], Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class PromotionPlan:
    """What to fit and how; wire values are EG's (snake_case enums)."""

    question_id: str
    feature_schema: Mapping[str, Any]
    policy: Mapping[str, Any] = field(default_factory=lambda: {"policy": "default"})
    head_kind: str = "listwise_logistic"
    label_regime: str = "bandit_label"
    estimators: tuple[str, ...] = ("clipped_ips", "snips")
    window: Mapping[str, int] = field(
        default_factory=lambda: {"from_ms": 0, "to_ms": 2**63 - 1}
    )
    optimiser: Mapping[str, Any] = field(
        default_factory=lambda: {
            "max_iterations": 200,
            "tolerance": {"scale": "q32", "value": 1 << 12},
            "seed": 0,
        }
    )


@dataclass(frozen=True, slots=True)
class Promotion:
    """What happened: the published head, or the step that stopped it and why."""

    promoted: bool
    stage: str
    detail: str = ""
    failed_gates: tuple[str, ...] = ()
    head: Mapping[str, Any] | None = None


def _logged(plan: PromotionPlan) -> dict[str, Any]:
    return {"source": "logged", "question_id": plan.question_id}


def _fit_request(plan: PromotionPlan, tenant: str, key: str) -> dict[str, Any]:
    return {
        "tenant_id": tenant,
        "idempotency_key": f"{key}:fit",
        "head_kind": plan.head_kind,
        "feature_schema": dict(plan.feature_schema),
        "policy": dict(plan.policy),
        "label_regime": plan.label_regime,
        "window": dict(plan.window),
        "optimiser": dict(plan.optimiser),
        "source": _logged(plan),
    }


def _eval_request(
    plan: PromotionPlan, tenant: str, key: str, fit: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "tenant_id": tenant,
        "idempotency_key": f"{key}:eval",
        "candidate": {
            "candidate": "draft_artifact",
            "sha256": fit["draft_sha256"],
            "length": fit["draft_length"],
        },
        "policy": dict(plan.policy),
        "estimators": list(plan.estimators),
        "window": dict(plan.window),
        "source": _logged(plan),
    }


def _output(job: Any) -> Mapping[str, Any] | None:
    job = getattr(job, "payload", job)
    state = job.get("state") if isinstance(job, Mapping) else None
    if not isinstance(state, Mapping) or state.get("state") != "succeeded":
        return None
    output = state.get("output")
    return output if isinstance(output, Mapping) else None


def _state_of(job: Any) -> str:
    job = getattr(job, "payload", job)
    state = job.get("state") if isinstance(job, Mapping) else None
    return str(state) if state is not None else "missing"


@dataclass
class HeadPromoter:
    """Fit -> evaluate -> publish-with-receipt, over EG's generated job senders."""

    client: Any
    tenant: str
    publish: HeadPublisher
    graph: str | None = None

    async def _submit(self, sender: str, request: Mapping[str, Any]) -> Any:
        send = generated("coordination", sender)
        op = {"op": {"op": "submit", "request": dict(request)}}
        return await send(self.client, op, self.graph)

    async def promote(self, plan: PromotionPlan, key: str) -> Promotion:
        fit = _output(
            await self._submit(
                "send_decision_fit", _fit_request(plan, self.tenant, key)
            )
        )
        if fit is None or fit.get("output") != "fit":
            return Promotion(False, "fit", "the fit job did not succeed")
        job = await self._submit(
            "send_decision_eval", _eval_request(plan, self.tenant, key, fit)
        )
        output = _output(job)
        receipt = (output or {}).get("receipt")
        if not isinstance(receipt, Mapping):
            return Promotion(
                False, "eval", f"the eval job did not succeed: {_state_of(job)}"
            )
        if not receipt.get("passed"):
            gates = tuple(str(g) for g in receipt.get("failed_gates") or ())
            return Promotion(False, "eval", "the receipt did not pass", gates)
        head = await self.publish(fit["draft"], str(receipt["receipt_digest"]))
        return Promotion(True, "published", head=head)


__all__ = ["HeadPromoter", "HeadPublisher", "Promotion", "PromotionPlan"]
