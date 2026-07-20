"""Cost computation for usage events (CONCEPT:AU-OS.observability.usage-analytics-store / ECO-4.40).

Fills ``cost_usd``/``cost_status``/``cost_source`` on a usage event from the
pricing catalog when the source agent did not already report a cost.
"""

from __future__ import annotations

from agent_utilities.pricing import get_pricing_catalog

from .models import UsageEvent


def price_event(event: UsageEvent) -> UsageEvent:
    """Return ``event`` with cost populated. Agent-reported cost is preserved."""
    if event.cost_usd is not None and event.cost_status == "from_agent":
        return event
    catalog = get_pricing_catalog()
    cost, priced = catalog.cost_for(
        event.model,
        input_tokens=event.input_tokens,
        output_tokens=event.output_tokens,
        cache_creation_tokens=event.cache_creation_input_tokens,
        cache_read_tokens=event.cache_read_input_tokens,
    )
    if priced:
        event.cost_usd = cost
        event.cost_status = "catalog"
        event.cost_source = "catalog"
    elif event.cost_usd is None:
        event.cost_status = "unpriced"
        event.cost_source = "none"
    return event
