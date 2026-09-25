"""Emit privacy-normalized usage events to epistemic-graph durability.

The GraphOS usage control plane supplies a signed EG client.  This adapter
does not construct credentials or keep a local durability fallback.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from agent_utilities.security.persistence_privacy import persistence_reference

from .models import ParsedSessionBundle
from .privacy import normalize_bundle

if TYPE_CHECKING:
    from epistemic_graph.client import EpistemicGraphClient


class UsageFactEmitter:
    def __init__(self, client: EpistemicGraphClient, reference_key: bytes) -> None:
        self._client = client
        self._reference_key = reference_key

    async def emit_bundle(self, bundle: ParsedSessionBundle) -> int:
        """Append each usage event under the signed tenant, with no SQL write."""

        from epistemic_graph.usage_facts import UsageEventFact, UsageFactStore

        tenant = self._client._effective_verified_context()["tenant"]
        if not tenant or bundle.session.tenant_id != tenant:
            raise PermissionError("usage bundle tenant must match signed authority")
        normalized = normalize_bundle(bundle)
        store = UsageFactStore(self._client, self._reference_key)
        inserted = 0
        for ordinal, event in enumerate(normalized.usage_events):
            occurred_at = event.occurred_at or normalized.session.started_at
            if not occurred_at:
                raise ValueError("usage event timestamp required")
            original = bundle.usage_events[ordinal]
            event_ref = event.dedup_key or persistence_reference(
                "usage_dedup", f"{bundle.session.id}:{ordinal}", namespace=tenant
            )
            cost = (
                int(Decimal(str(event.cost_usd)) * 1_000_000)
                if event.cost_usd is not None
                else None
            )
            fact = UsageEventFact(
                event_ref=event_ref,
                run_ref=normalized.session.id,
                origin=event.origin,
                occurred_at=occurred_at,
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                cache_creation_tokens=event.cache_creation_input_tokens,
                cache_read_tokens=event.cache_read_input_tokens,
                reasoning_tokens=event.reasoning_tokens,
                cost_microusd=cost,
                model_ref=(
                    persistence_reference("model", original.model)
                    if original.model
                    else None
                ),
            )
            inserted += int(await store.append_event(fact))
        return inserted
