"""Focused graph-native domain operations not exposed by engine subclients."""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_text


def register_domain_ops_tools(mcp: Any) -> None:
    """Register persisted enterprise, finance, and ML/RLM domain operations."""

    @mcp.tool(
        name="graph_domain_ops",
        description=(
            "Run graph-native domain mutations. Actions: 'allocate_budget' creates a "
            "business-unit payment budget; 'fit_markov_regime' fits and persists a strategy "
            "regime model; 'register_rlm_actor' creates an RLM learning actor."
        ),
        tags=["graph-os", "domain", "enterprise", "finance", "ml"],
    )
    def graph_domain_ops(
        action: str = Field(
            default="allocate_budget",
            description="allocate_budget | fit_markov_regime | register_rlm_actor",
        ),
        target_id: str = Field(
            default="", description="Business-unit, strategy, or actor name/id."
        ),
        amount: float = Field(default=10_000.0, description="Budget amount."),
        currency: str = Field(default="USD", description="Budget currency."),
        returns_json: str = Field(
            default="[]", description="JSON list of returns for fit_markov_regime."
        ),
        asset_class: str = Field(default="equities", description="Regime asset class."),
        bull_threshold: float | None = Field(default=None),
        bear_threshold: float | None = Field(default=None),
        window: int | None = Field(default=None),
        method: str = Field(default="rolling_sum"),
        learning_rate: float = Field(default=0.01),
        discount_factor: float = Field(default=0.99),
    ) -> str:
        engine = kg_server._get_engine()
        if engine is None:
            return "Error: IntelligenceGraphEngine not active."
        try:
            if action == "allocate_budget":
                if not target_id:
                    raise ValueError("target_id is required for allocate_budget")
                allocate = getattr(engine, "allocate_budget", None)
                if not callable(allocate):
                    raise RuntimeError(
                        "active engine does not expose the enterprise budget capability"
                    )
                budget_id = allocate(target_id, float(amount), currency)
                return json.dumps(
                    {
                        "budget_id": budget_id,
                        "business_unit_id": target_id,
                        "amount": float(amount),
                        "currency": currency,
                    },
                    default=str,
                )

            if action == "fit_markov_regime":
                if not target_id:
                    raise ValueError("target_id is required for fit_markov_regime")
                returns = json.loads(returns_json) if returns_json else []
                if not isinstance(returns, list) or not returns:
                    raise ValueError("returns_json must contain a non-empty list")
                fit = getattr(engine, "fit_markov_regime", None)
                if not callable(fit):
                    raise RuntimeError(
                        "active engine does not expose the persisted regime capability"
                    )
                matrix_id = fit(
                    returns=[float(value) for value in returns],
                    strategy_id=target_id,
                    asset_class=asset_class,
                    bull_threshold=bull_threshold,
                    bear_threshold=bear_threshold,
                    window=window,
                    method=method,
                )
                return json.dumps({"matrix_id": matrix_id, "status": "fitted"})

            if action == "register_rlm_actor":
                if not target_id:
                    raise ValueError("target_id is required for register_rlm_actor")
                register = getattr(engine, "register_rlm_actor", None)
                if not callable(register):
                    raise RuntimeError(
                        "active engine does not expose the RLM actor capability"
                    )
                actor_id = register(
                    name=target_id,
                    learning_rate=float(learning_rate),
                    discount_factor=float(discount_factor),
                )
                return json.dumps({"actor_id": actor_id, "status": "registered"})

            return f"Error: Unknown graph_domain_ops action '{action}'"
        except PermissionError:
            raise
        except Exception as exc:
            return public_error_text(exc)

    kg_server.REGISTERED_TOOLS["graph_domain_ops"] = graph_domain_ops
    kg_server.ACTION_TOOL_ROUTES["graph_domain_ops"] = "/graph/domain-ops"
