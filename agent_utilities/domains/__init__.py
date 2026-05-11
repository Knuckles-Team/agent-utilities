from __future__ import annotations

"""
Multi-domain root module for agent-utilities.

CONCEPT:KG-2.6 — Multi-Domain Architecture

This module houses all domain-specific integrations and provides a
domain registry for the ServiceRegistry and KGTeamComposer to discover
domain-specific capabilities at runtime.
"""


__all__ = ["finance", "hr", "medical", "law", "government", "DOMAIN_REGISTRY"]


# Domain registry mapping domain names to their capabilities
DOMAIN_REGISTRY: dict[str, dict[str, str]] = {
    "finance": {
        "alpha_factors": "agent_utilities.domains.finance.alpha_factors",
        "risk_management": "agent_utilities.domains.finance.risk_manager",
        "portfolio_optimization": "agent_utilities.domains.finance.portfolio_optimizer",
        "versioned_orders": "agent_utilities.domains.finance.versioned_orders",
        "market_data": "agent_utilities.domains.finance.market_data",
        "payments": "agent_utilities.domains.finance.payments",
        "profit_attribution": "agent_utilities.domains.finance.profit_attribution",
        "streaming": "agent_utilities.domains.finance.streaming",
        "kronos_forecaster": "agent_utilities.domains.finance.kronos_forecaster",
        "trading_swarm": "agent_utilities.domains.finance.trading_swarm",
        "visual_ta": "agent_utilities.domains.finance.visual_ta",
        "market_feeds": "agent_utilities.domains.finance.market_feeds",
        "strategy_export": "agent_utilities.domains.finance.strategy_export",
        "research_autopilot": "agent_utilities.domains.finance.research_autopilot",
        "strategy_sharing": "agent_utilities.domains.finance.strategy_sharing",
    },
}


def get_domain_capabilities(domain: str) -> list[str]:
    """Get available capabilities for a given domain.

    Args:
        domain: The domain name (e.g., 'finance').

    Returns:
        List of capability names available in that domain.
    """
    return list(DOMAIN_REGISTRY.get(domain, {}).keys())


def list_domains() -> list[str]:
    """List all registered domains."""
    return list(DOMAIN_REGISTRY.keys())
