"""Emerald-Exchange source extractor (CONCEPT:KG-2.9).

Reads the trading account state from an Emerald exchange backend into canonical
quant OWL nodes: account → :Account, a synthetic :Portfolio, open positions →
:Position (HELD_IN the portfolio). Stamped externalToolId + domain="emerald".

The backend (``emerald_exchange.backends.create_backend``) is injected; its
read methods (``get_account``/``get_positions``) return dataclasses, so values
are read tolerantly (object attr or dict key). Read-only: no orders here — order
flow is the *write* path and is high-stakes / propose-only (see the sink).
"""

from __future__ import annotations

from typing import Any

from ...core import owl_bridge
from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "emerald"
_DOMAIN = "emerald"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _call(client: Any, name: str) -> Any:
    m = getattr(client, name, None)
    try:
        return m() if callable(m) else None
    except Exception:  # noqa: BLE001 - tolerant of disconnected backend
        return None


def extract(config: Any) -> ExtractionBatch:
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    account = _call(client, "get_account")
    exchange = str(_attr(account, "exchange", "emerald") or "emerald")
    portfolio_id = f"emerald:portfolio:{exchange}"
    nodes.append(
        GraphNode(
            id=portfolio_id,
            type="Portfolio",
            props={"exchange": exchange, "externalToolId": exchange, "domain": _DOMAIN},
        )
    )
    if account is not None:
        nodes.append(
            GraphNode(
                id=f"emerald:account:{exchange}",
                type="Account",
                props={
                    "equity": _attr(account, "equity"),
                    "cash": _attr(account, "cash"),
                    "buying_power": _attr(account, "buying_power"),
                    "currency": _attr(account, "currency"),
                    "externalToolId": f"account:{exchange}",
                    "domain": _DOMAIN,
                },
            )
        )

    positions = _call(client, "get_positions") or []
    for pos in positions if isinstance(positions, list) else []:
        symbol = _attr(pos, "symbol")
        if not symbol:
            continue
        node_id = f"emerald:pos:{exchange}:{symbol}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="Position",
                props={
                    "symbol": symbol,
                    "qty": _attr(pos, "qty"),
                    "avg_entry_price": _attr(pos, "avg_entry_price"),
                    "current_price": _attr(pos, "current_price"),
                    "unrealized_pnl": _attr(pos, "unrealized_pnl"),
                    "side": _attr(pos, "side"),
                    "externalToolId": f"{exchange}:{symbol}",
                    "domain": _DOMAIN,
                },
            )
        )
        edges.append(
            EnrichmentEdge(source=node_id, target=portfolio_id, rel_type="HELD_IN")
        )

    owl_bridge.register_promotable_node_types({n.type for n in nodes})
    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY, extract, description="Emerald-Exchange (account/portfolio/positions) → KG"
)
