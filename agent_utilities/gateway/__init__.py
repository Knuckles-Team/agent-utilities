"""Gateway — Unified service dashboard and aggregation for Agent-OS.

CONCEPT:GW-1.0 — Gateway Service Dashboard

Provides the widget registry, data aggregation, and API layer that all
three frontends (agent-webui, agent-terminal-ui, geniusbot) use to render
Homepage-style service dashboards.

Replaces the former standalone ``service-dashboard-core`` package by folding
its genuinely new functionality (UI models, parallel aggregator, widget ABC)
into the canonical ``agent-utilities`` infrastructure.

Usage::

    from agent_utilities.gateway import Aggregator, ConfigManager
    from agent_utilities.gateway.models import WidgetData, DashboardLayout
    from agent_utilities.gateway.registry import get_registry
"""

from agent_utilities.gateway.models import (
    DashboardLayout,
    ServiceCategory,
    ServiceConfig,
    ServiceGroup,
    WidgetData,
    WidgetField,
    WidgetRegistration,
)
from agent_utilities.gateway.registry import Registry, get_registry

__all__ = [
    "Aggregator",
    "ConfigManager",
    "DashboardLayout",
    "Registry",
    "ServiceCategory",
    "ServiceConfig",
    "ServiceGroup",
    "WidgetData",
    "WidgetField",
    "WidgetRegistration",
    "get_registry",
]

__version__ = "0.1.0"
