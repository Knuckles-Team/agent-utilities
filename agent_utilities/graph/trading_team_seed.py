"""Curated trading TeamConfig seed — CONCEPT:AU-AHE.harness.trading-team-seed.

A blessed multi-specialist trading team mapped onto the Finance Pipeline topology
(router → alpha → risk → execution → attribution) and the emerald-exchange tool
surface. Seeded with a high ``success_rate`` so ``compose_team`` reuses it as a
proven team rather than re-synthesising the roster on every trading query. The
``risk-manager`` carries the ``critic`` capability (adversarial sizing veto) and
the ``execution-specialist`` is the SOLE holder of order/derivative/crypto/
prediction tools, so paper-first execution gating concentrates at one node.
"""

from __future__ import annotations

import logging
from typing import Any

from ..models.knowledge_graph import TeamConfigNode

logger = logging.getLogger(__name__)

TRADING_TEAM_ID = "teamcfg:trading_paper_v1"


def build_trading_team_config() -> TeamConfigNode:
    """Return the curated paper-first multi-asset trading TeamConfig."""
    return TeamConfigNode(
        id=TRADING_TEAM_ID,
        name="Paper-First Multi-Asset Trading Team",
        task_pattern=(
            "multi-asset paper-first trade research, validation, sizing and "
            "staged execution across equities, crypto, prediction markets, "
            "and derivatives"
        ),
        specialist_ids=[
            "market-analyst",
            "alpha-strategist",
            "risk-manager",
            "execution-specialist",
            "attribution-analyst",
        ],
        tool_assignments={
            "market-analyst": [
                "emerald_market_data",
                "emerald_fundamentals",
                "emerald_wallet_intel",
                "graph_search",
            ],
            "alpha-strategist": [
                "emerald_signals",
                "emerald_strategy",
                "emerald_statarb",
                "ontology_query",
                "graph_query",
            ],
            "risk-manager": ["emerald_risk", "emerald_portfolio", "graph_query"],
            "execution-specialist": [
                "emerald_orders",
                "emerald_derivatives",
                "emerald_crypto",
                "emerald_prediction_markets",
                "emerald_market_making",
            ],
            "attribution-analyst": [
                "emerald_portfolio",
                "emerald_debate",
                "graph_write",
            ],
        },
        prompt_template_ids=[
            "prompt:trading.market_analyst",
            "prompt:trading.alpha_strategist",
            "prompt:trading.risk_manager",
            "prompt:trading.execution_specialist",
            "prompt:trading.attribution_analyst",
        ],
        capability_overrides={
            "alpha-strategist": ["rlm"],  # large data → RLM auto-attach
            "risk-manager": ["critic"],  # adversarial veto on sizing
        },
        # Curated/blessed baseline: above the 0.72 reuse threshold so it is reused
        # immediately, then updated by real outcomes as the team runs.
        success_rate=0.75,
        usage_count=0,
        reuse_threshold=0.72,
        origin="local",
    )


def seed_trading_team(engine: Any) -> str | None:
    """Write the curated trading TeamConfig into the KG via ``engine``.

    Idempotent on node id. Returns the node id on success, ``None`` when no
    engine/graph is available (so callers can no-op offline).
    """
    node = build_trading_team_config()
    graph = getattr(engine, "graph", None)
    if graph is None:
        logger.debug("seed_trading_team: no graph on engine; skipped")
        return None
    try:
        graph.add_node(node.id, **node.model_dump())
        upsert = getattr(engine, "_upsert_node", None)
        serialize = getattr(engine, "_serialize_node", None)
        if getattr(engine, "backend", None) and upsert and serialize:
            upsert("TeamConfig", node.id, serialize(node, label="TeamConfig"))
    except Exception as exc:  # noqa: BLE001 — best effort seed
        logger.warning("seed_trading_team failed: %s", exc)
        return None
    return node.id
