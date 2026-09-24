"""The Loop engine's side of the graph-driven work market (EH-348).

EG owns the market: the canonical Gap paired with its one native WorkItem, the
versioned derived ``WorkOffer`` and its deterministic utility rate, the fenced
claim, and the engine-read outcome evidence. This module is what the Loop engine
DOES with it, through the typed EG clients only (``engine.client.gaps`` /
``.work_market`` / ``.work_items``):

* :func:`price_gap` proposes/refreshes a live Gap's offer inputs from the Gap's own
  recorded evidence (``WorkOfferPut``); EG computes the rate.
* :func:`claim_gap_work` admits execution of ONE named Gap WorkItem through the
  native fenced ``ClaimWorkItem`` -- two claimants of one item cannot both win.
* :func:`settle_gap_work` asks EG to record the WorkItem's terminal outcome on the
  Gap (``GapSettle``); the caller never states the outcome.
* :func:`reconcile_market` is the bounded recovery sweep for missed events:
  settle live Gaps whose WorkItem finished, cancel the never-run WorkItem of a
  closed Gap, price unpriced live Gaps. A sweep, not a heartbeat that invents work.

Nothing here ranks, sorts or selects offers: ranking legal work is the ``Decide``
layer's (design §3.1). Every clock is an argument, so a test replays exactly.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from .gaps import (
    GapAuthorityUnavailable,
    gap_tenant,
    iter_gaps,
    settle_gap,
)

#: How long a Gap reopened after a failed attempt waits before it may be tried again.
FAILURE_COOLDOWN_MS = 60 * 60 * 1000
#: Most evidence digests one offer cites (EG's own bound).
_MAX_OFFER_EVIDENCE = 32
#: A WorkItem status EG no longer schedules.
_TERMINAL_WORK = frozenset({"succeeded", "failed", "cancelled", "dead_letter"})

#: Per-source pricing priors: (probability of closure ppm, expected cost microunits,
#: blast radius). A table so every discovery track is priced by one rule; an unknown
#: source takes the conservative default.
_PRICING: dict[str, tuple[int, int, str]] = {
    "audit": (600_000, 20_000, "local"),
    "failure": (500_000, 40_000, "repository"),
    "runtime": (450_000, 40_000, "repository"),
    "skill": (400_000, 30_000, "local"),
    "research": (300_000, 80_000, "repository"),
}
_DEFAULT_PRICING = (350_000, 60_000, "repository")


def _namespace(engine: Any, name: str) -> Any:
    namespace = getattr(getattr(engine, "client", None), name, None)
    if namespace is None:
        raise GapAuthorityUnavailable(f"connected engine has no typed {name} surface")
    return namespace


def offer_inputs(gap: Mapping[str, Any]) -> dict[str, Any]:
    """The deterministic ``WorkOffer`` inputs for a Gap view, from its own evidence.

    Utility scales with severity; closure probability, cost and blast radius come
    from the source prior; a Gap reopened after a failed attempt cools down from
    the time EG recorded that outcome.
    """
    closure_ppm, cost, blast = _PRICING.get(str(gap.get("source")), _DEFAULT_PRICING)
    evidence = [e for e in gap.get("evidence") or [] if isinstance(e, Mapping)]
    outcomes = [
        int(e.get("recorded_at_ms") or 0)
        for e in evidence
        if e.get("kind") == "work_item_outcome"
    ]
    return {
        "expected_utility_micros": int(gap.get("severity_ppm") or 0) * 10,
        "probability_of_closure_ppm": closure_ppm,
        "expected_cost_microunits": cost,
        "cost_uncertainty_ppm": 250_000,
        "blast_radius": blast,
        "reversible": True,
        "required_capabilities": [],
        "repository_scope": [],
        "cooldown_until_ms": max(outcomes) + FAILURE_COOLDOWN_MS if outcomes else 0,
        "depends_on_gap_ids": [],
        "evidence_digests": [str(e["digest"]) for e in evidence][-_MAX_OFFER_EVIDENCE:],
    }


def needs_price(gap: Mapping[str, Any]) -> bool:
    """A live Gap whose current WorkItem carries no offer yet."""
    offer = gap.get("offer") or {}
    live = gap.get("status") in ("open", "specified")
    return bool(live and offer.get("work_item_id") != gap.get("work_item_id"))


def price_gap(engine: Any, gap: Mapping[str, Any]) -> str:
    """Record the Gap's derived offer (``WorkOfferPut``, CAS on its offer version)."""
    answer = _namespace(engine, "work_market").put_offer(
        tenant=gap_tenant(),
        gap_id=str(gap["gap_id"]),
        expected_offer_version=int(gap.get("offer_version") or 0),
        offer=offer_inputs(gap),
        idempotency_key=f"offer:{gap['gap_id']}:{gap.get('offer_version') or 0}",
    )
    return str(answer["outcome"])


def claim_gap_work(
    engine: Any,
    gap: Mapping[str, Any],
    *,
    worker: str,
    now_ms: int,
    lease_ms: int = 15 * 60 * 1000,
) -> dict[str, Any] | None:
    """Claim ONE named Gap WorkItem through the native fenced claim, or ``None``.

    The item is named by the caller -- in production the committed ``Decide``
    decision that selected it; this module never picks one. A concurrent claimant
    of the same item gets ``None``: the engine admits exactly one lease.
    """
    answer = _namespace(engine, "work_items").claim(
        {
            "schema_version": "1",
            "tenant_ref": gap_tenant(),
            "work_item_id": str(gap["work_item_id"]),
            "queue_ref": None,
            "resource_class": None,
            "fairness_group": None,
            "worker_ref": worker,
            "now_ms": int(now_ms),
            "lease_ms": int(lease_ms),
            "max_tenant_in_flight": 64,
        }
    )
    return answer if answer.get("claimed") else None


def settle_gap_work(engine: Any, gap_id: str) -> str:
    """Record the Gap's WorkItem outcome on the Gap; EG reads it from the WorkItem row."""
    return settle_gap(engine, gap_id)


def _work_status(engine: Any, gap: Mapping[str, Any]) -> str:
    view = _namespace(engine, "work_items").get(
        tenant=gap_tenant(), work_item_id=str(gap["work_item_id"])
    )
    return str((view or {}).get("status") or "")


def _cancel_unrun(engine: Any, gap: Mapping[str, Any], now_ms: int) -> None:
    _namespace(engine, "work_items").cancel(
        tenant=gap_tenant(),
        work_item_id=str(gap["work_item_id"]),
        idempotency_key=f"gap-closed:{gap['work_item_id']}",
        now_ms=int(now_ms),
        reason_ref="gap_closed",
    )


def _reconcile_one(engine: Any, gap: Mapping[str, Any], now_ms: int) -> str:
    """One Gap's recovery step; returns what it did (``settled``/``cancelled``/
    ``priced``/``none``)."""
    live = gap.get("status") in ("open", "specified")
    finished = _work_status(engine, gap) in _TERMINAL_WORK
    if finished:
        return (
            "settled"
            if settle_gap(engine, str(gap["gap_id"])) != "unchanged"
            else "none"
        )
    if not live:
        _cancel_unrun(engine, gap, now_ms)
        return "cancelled"
    if needs_price(gap):
        return "priced" if price_gap(engine, gap) == "applied" else "none"
    return "none"


def reconcile_market(engine: Any, *, now_ms: int, limit: int = 200) -> dict[str, int]:
    """The bounded recovery sweep over the tenant's Gaps (at most ``limit``)."""
    counts = {"examined": 0, "settled": 0, "cancelled": 0, "priced": 0, "none": 0}
    for gap in iter_gaps(engine):
        if counts["examined"] >= limit:
            break
        counts["examined"] += 1
        counts[_reconcile_one(engine, gap, now_ms)] += 1
    return counts


def run_market_stage(engine: Any, *, now_ms: int | None = None) -> dict[str, Any]:
    """The Loop engine's ``work_market`` stage: one bounded :func:`reconcile_market`.

    Skips (and says so) when the connected engine serves no typed Gap surface; there
    is no fallback path.
    """
    if getattr(getattr(engine, "client", None), "gaps", None) is None:
        return {"skipped": "connected engine has no typed Gap surface"}
    stamp = int(time.time() * 1000) if now_ms is None else int(now_ms)
    return dict(reconcile_market(engine, now_ms=stamp))


__all__ = [
    "FAILURE_COOLDOWN_MS",
    "claim_gap_work",
    "needs_price",
    "offer_inputs",
    "price_gap",
    "reconcile_market",
    "run_market_stage",
    "settle_gap_work",
]
