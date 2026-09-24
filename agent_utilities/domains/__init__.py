from __future__ import annotations

"""
Multi-domain root module for agent-utilities.

CONCEPT:AU-KG.domains.multi-domain-architecture — Multi-Domain Architecture

This module houses all domain-specific integrations and provides a
domain registry for the ServiceRegistry and KGTeamComposer to discover
domain-specific capabilities at runtime.
"""


__all__ = ["finance", "hr", "medical", "law", "government", "DOMAIN_REGISTRY"]


# Domain registry mapping domain names to their capabilities
DOMAIN_REGISTRY: dict[str, dict[str, str]] = {
    "finance": {
        "kronos_forecaster": "agent_utilities.domains.finance.kronos_forecaster",
        "trading_swarm": "agent_utilities.domains.finance.trading_swarm",
        "research_autopilot": "agent_utilities.domains.finance.research_autopilot",
        "flip_explainer": "agent_utilities.domains.finance.flip_explainer",
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
