"""A deterministic in-memory stand-in for EG's work-market surfaces (the harness-evolution spec's graph-driven work-market requirement).

``FakeWorkMarket`` mirrors the engine rules the AU callers depend on, exactly as
``eg_types::work_market`` states them: one Gap per ``(tenant, gap_id)`` created
together with its WorkItem; only unseen evidence changes a Gap and only it reopens a
closed one (new generation, new WorkItem); a transition follows the legal-edge table
under revision CAS; ``settle`` reads the WorkItem's own status; an offer cites only
held evidence, is version-CAS'd and gets the fixed-point utility rate; a claim of a
named WorkItem admits exactly one lease. Tenants never see each other's rows.

Attach with :func:`attach_market` (sets ``engine.client``) and run inside
:func:`tests.unit.fleet_autonomy_fakes.verified_fleet_session` — every Gap call binds
the ambient verified tenant.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

_LEGAL = {
    ("open", "specified"),
    ("open", "resolved"),
    ("open", "deferred"),
    ("specified", "resolved"),
    ("specified", "deferred"),
}
_SETTLE = {
    "succeeded": ("resolved", "resolved"),
    "failed": ("deferred", "deferred"),
    "cancelled": ("deferred", "deferred"),
    "dead_letter": ("deferred", "deferred"),
}
_BUCKETS = ((850_000, 0), (600_000, 1), (300_000, 2))
_COST_FLOOR = 1_000


def _digest(*parts: str) -> str:
    return "sha256:" + hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _bucket(severity_ppm: int) -> int:
    return next((b for floor, b in _BUCKETS if severity_ppm >= floor), 3)


class FakeWorkMarket:
    """Gaps + their WorkItems + offers, keyed per tenant; ``now_ms`` is injected."""

    def __init__(self, now_ms: int = 1_000) -> None:
        self.now_ms = now_ms
        self.gap_rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.items: dict[str, dict[str, Any]] = {}
        self.gaps = SimpleNamespace(
            upsert=self.upsert,
            transition=self.transition,
            settle=self.settle,
            get=self.get,
            list=self.list,
        )
        self.work_market = SimpleNamespace(put_offer=self.put_offer)
        self.work_items = SimpleNamespace(
            claim=self.claim, get=self.get_item, cancel=self.cancel, finish=self.finish
        )

    # ── Gaps ──────────────────────────────────────────────────────────────
    def _item(self, tenant: str, gap_id: str, generation: int, kind: str) -> str:
        item_id = f"work-item:gap:{_digest(tenant, gap_id, str(generation))[7:23]}"
        self.items[item_id] = {
            "tenant": tenant,
            "kind": kind,
            "status": "ready",
            "lease": None,
        }
        return item_id

    def _fresh(self, tenant: str, request: dict[str, Any]) -> dict[str, Any]:
        severity = int(request["severity_ppm"])
        return {
            "gap_id": request["gap_id"],
            "source": request["source"],
            "signature": request["signature"],
            "statement": request["statement"],
            "domain": request.get("domain", ""),
            "severity_ppm": severity,
            "priority_bucket": _bucket(severity),
            "status": "open",
            "generation": 1,
            "work_item_id": self._item(
                tenant, request["gap_id"], 1, request["work_kind"]
            ),
            "work": {
                "kind": request["work_kind"],
                "max_attempts": request["max_attempts"],
            },
            "evidence": [],
            "evidence_count": 0,
            "concept_ids": list(request.get("concept_ids") or []),
            "spec_refs": [],
            "offer": None,
            "offer_version": 0,
            "created_at_ms": self.now_ms,
            "updated_at_ms": self.now_ms,
            "revision": 0,
        }

    def _record(
        self, gap: dict[str, Any], digest: str, kind: str, reference: str
    ) -> None:
        gap["evidence"].append(
            {
                "digest": digest,
                "kind": kind,
                "reference": reference,
                "generation": gap["generation"],
                "recorded_at_ms": self.now_ms,
            }
        )
        gap["evidence_count"] += 1
        gap["updated_at_ms"] = self.now_ms

    def _store(self, tenant: str, gap: dict[str, Any]) -> dict[str, Any]:
        gap["revision"] += 1
        self.gap_rows[(tenant, gap["gap_id"])] = gap
        return dict(gap)

    def upsert(self, *, tenant: str, **request: Any) -> dict[str, Any]:
        key = (tenant, request["gap_id"])
        gap = self.gap_rows.get(key)
        created = gap is None
        gap = self._fresh(tenant, request) if gap is None else gap
        held = {e["digest"] for e in gap["evidence"]}
        fresh = [e for e in request["evidence"] if e["digest"] not in held]
        if not created and not fresh:
            return {
                "outcome": "unchanged",
                "gap": dict(gap),
                "work_item_created": False,
            }
        reopened = not created and gap["status"] in ("resolved", "deferred")
        if reopened:
            gap["generation"] += 1
            gap["work_item_id"] = self._item(
                tenant, gap["gap_id"], gap["generation"], request["work_kind"]
            )
            gap["status"], gap["offer"] = "open", None
        for entry in fresh:
            self._record(gap, entry["digest"], entry["kind"], entry["reference"])
        gap["severity_ppm"] = max(gap["severity_ppm"], int(request["severity_ppm"]))
        gap["priority_bucket"] = _bucket(gap["severity_ppm"])
        outcome = "created" if created else "reopened" if reopened else "merged"
        return {
            "outcome": outcome,
            "gap": self._store(tenant, gap),
            "work_item_created": created or reopened,
        }

    def get(self, *, tenant: str, gap_id: str) -> dict[str, Any] | None:
        gap = self.gap_rows.get((tenant, gap_id))
        return None if gap is None else dict(gap)

    def list(
        self,
        *,
        tenant: str,
        status: str | None = None,
        source: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        keys = sorted(k for k in self.gap_rows if k[0] == tenant)
        start = int(cursor or 0)
        page = keys[start : start + limit]
        gaps = [
            dict(self.gap_rows[k])
            for k in page
            if status in (None, self.gap_rows[k]["status"])
            and source in (None, self.gap_rows[k]["source"])
        ]
        more = start + limit < len(keys)
        return {"gaps": gaps, "next_cursor": str(start + limit) if more else None}

    def transition(
        self,
        *,
        tenant: str,
        gap_id: str,
        expected_revision: int,
        to: str,
        reference: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        gap = self.gap_rows.get((tenant, gap_id))
        if gap is None:
            return {"outcome": "not_found", "gap": None}
        if (gap["status"], to) not in _LEGAL or gap["revision"] != expected_revision:
            return {"outcome": "conflict", "gap": dict(gap)}
        if to == "specified":
            gap["spec_refs"].append(reference)
        else:
            self._record(gap, _digest(to, gap_id, reference), to, reference)
        gap["status"] = to
        return {"outcome": "applied", "gap": self._store(tenant, gap)}

    def settle(
        self, *, tenant: str, gap_id: str, idempotency_key: str
    ) -> dict[str, Any]:
        gap = self.gap_rows.get((tenant, gap_id))
        if gap is None:
            return {"outcome": "not_found", "gap": None}
        item_id = gap["work_item_id"]
        effect = _SETTLE.get(self.items[item_id]["status"])
        if effect is None:
            return {"outcome": "pending", "gap": dict(gap)}
        digest = _digest("work_item_outcome", item_id, self.items[item_id]["status"])
        if digest in {e["digest"] for e in gap["evidence"]}:
            return {"outcome": "unchanged", "gap": dict(gap)}
        self._record(gap, digest, "work_item_outcome", item_id)
        outcome = "recorded"
        if gap["status"] in ("open", "specified"):
            outcome, gap["status"] = effect
        return {"outcome": outcome, "gap": self._store(tenant, gap)}

    # ── Offers ────────────────────────────────────────────────────────────
    def put_offer(
        self,
        *,
        tenant: str,
        gap_id: str,
        expected_offer_version: int,
        offer: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        gap = self.gap_rows.get((tenant, gap_id))
        if gap is None:
            return {"outcome": "not_found", "gap": None}
        held = {e["digest"] for e in gap["evidence"]}
        if not offer["evidence_digests"] or not set(offer["evidence_digests"]) <= held:
            raise RuntimeError("work offer cites evidence the Gap does not hold")
        live = gap["status"] in ("open", "specified")
        if not live or gap["offer_version"] != expected_offer_version:
            return {"outcome": "conflict", "gap": dict(gap)}
        gap["offer_version"] += 1
        cost = max(int(offer["expected_cost_microunits"]), _COST_FLOOR)
        gap["offer"] = {
            "version": gap["offer_version"],
            "generation": gap["generation"],
            "work_item_id": gap["work_item_id"],
            "offer": dict(offer),
            "utility_rate": offer["expected_utility_micros"]
            * offer["probability_of_closure_ppm"]
            // cost,
            "stage": "eg/work-offer-utility-rate/v1",
            "offered_at_ms": self.now_ms,
        }
        return {"outcome": "applied", "gap": self._store(tenant, gap)}

    # ── WorkItems (the native claim fence) ────────────────────────────────
    def claim(self, request: dict[str, Any]) -> dict[str, Any]:
        item = self.items.get(request["work_item_id"])
        if (
            item is None
            or item["tenant"] != request["tenant_ref"]
            or item["status"] != "ready"
        ):
            return {"claimed": False, "reason": "empty", "work_item_id": None}
        item["status"], item["lease"] = "leased", request["worker_ref"]
        return {
            "claimed": True,
            "reason": "claimed",
            "work_item_id": request["work_item_id"],
            "lease_holder_ref": request["worker_ref"],
        }

    def get_item(self, *, tenant: str, work_item_id: str) -> dict[str, Any] | None:
        item = self.items.get(work_item_id)
        if item is None or item["tenant"] != tenant:
            return None
        return {
            "work_item_id": work_item_id,
            "kind": item["kind"],
            "status": item["status"],
        }

    def cancel(self, *, tenant: str, work_item_id: str, **_: Any) -> dict[str, Any]:
        item = self.items[work_item_id]
        if item["tenant"] == tenant and item["status"] in ("ready", "submitted"):
            item["status"] = "cancelled"
        return {"status": item["status"]}

    def finish(self, work_item_id: str, worker: str, status: str) -> None:
        """Test helper standing in for the fenced ``commit_result`` of the lease holder."""
        item = self.items[work_item_id]
        assert item["lease"] == worker and item["status"] == "leased"
        item["status"], item["lease"] = status, None


def attach_market(engine: Any, market: FakeWorkMarket | None = None) -> FakeWorkMarket:
    """Expose ``market`` as ``engine.client``'s typed EG namespaces."""
    market = market or FakeWorkMarket()
    client = getattr(engine, "client", None)
    if client is None:
        engine.client = SimpleNamespace()
        client = engine.client
    client.gaps = market.gaps
    client.work_market = market.work_market
    client.work_items = market.work_items
    return market
