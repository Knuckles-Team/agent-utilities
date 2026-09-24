"""Focused graph-native domain operations not exposed by engine subclients."""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_text


def register_domain_ops_tools(mcp: Any) -> None:
    """Register persisted enterprise and ML/RLM domain operations.

    Finance regime fitting left with the rest of AU's finance math (EH-423 /
    AUD-30): epistemic-graph owns regime detection and Markov transition
    matrices (``FinanceDetectRegimes`` / ``FinanceMarkovTransitionMatrix``).
    """

    @mcp.tool(
        name="graph_domain_ops",
        description=(
            "Run graph-native domain mutations. Actions: 'allocate_budget' creates a "
            "business-unit payment budget; 'register_rlm_actor' creates an RLM "
            "learning actor."
        ),
        tags=["graph-os", "domain", "enterprise", "ml"],
    )
    def graph_domain_ops(
        action: str = Field(
            default="allocate_budget",
            description="allocate_budget | register_rlm_actor",
        ),
        target_id: str = Field(
            default="", description="Business-unit or actor name/id."
        ),
        amount: float = Field(default=10_000.0, description="Budget amount."),
        currency: str = Field(default="USD", description="Budget currency."),
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
