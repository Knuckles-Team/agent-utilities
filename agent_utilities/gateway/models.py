"""Pydantic models for the service dashboard.

CONCEPT:GW-1.0 — Gateway Service Dashboard

These models define the data contract between the backend aggregator
and all three frontends. Mirrors Homepage's widget/block/container pattern
but with full Pydantic typing.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ServiceCategory(str, Enum):
    """Categories for grouping services on the dashboard."""

    INFRASTRUCTURE = "Infrastructure"
    DEVOPS = "DevOps"
    MEDIA = "Media"
    PRODUCTIVITY = "Productivity"
    LIFESTYLE = "Lifestyle"
    SECURITY = "Security"
    COMMUNICATION = "Communication"
    OBSERVABILITY = "Observability"
    BUSINESS = "Business"
    DATA_SCIENCE = "Data & Research"
    CUSTOM = "Custom"


class WidgetField(BaseModel):
    """A single metric field displayed in a widget card.

    Mirrors Homepage's ``Block`` component — a label + formatted value pair.
    """

    key: str = Field(description="Machine key for the field, e.g. 'running'")
    label: str = Field(description="Human-readable label, e.g. 'Running'")
    format: str = Field(
        default="number",
        description="Display format: number, percent, bytes, duration, text, status",
    )
    suffix: str = Field(default="", description="Optional suffix like 'ms', 'GB'")
    highlight: bool = Field(
        default=False, description="Whether to apply conditional highlighting"
    )


class WidgetData(BaseModel):
    """Data returned from a widget's fetch_data() method.

    Mirrors Homepage's useWidgetAPI response shape.
    """

    fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs of metric data, keyed by WidgetField.key",
    )
    status: str = Field(
        default="ok", description="Service status: ok, error, unreachable, unknown"
    )
    error: str | None = Field(
        default=None, description="Error message if status is not ok"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    raw: dict[str, Any] | None = Field(
        default=None, description="Optional raw response data for advanced views"
    )


class ServiceConfig(BaseModel):
    """Configuration for a single service instance.

    Loaded from ``services.yaml`` or auto-discovered from ``mcp_config.json``.
    Mirrors Homepage's per-service YAML structure.
    """

    id: str = Field(description="Unique service ID, e.g. 'portainer-1'")
    name: str = Field(description="Display name, e.g. 'Portainer'")
    widget_type: str = Field(
        description="Widget type key from registry, e.g. 'portainer'"
    )
    url: str = Field(default="", description="Service base URL")
    icon: str = Field(
        default="", description="Icon identifier (Lucide name, URL, or emoji)"
    )
    description: str = Field(default="", description="Short description")
    category: ServiceCategory = Field(default=ServiceCategory.CUSTOM)

    # Auth — mirrors the env-var pattern from each agent-package's auth.py
    api_key: str = Field(default="", description="API key or token")
    username: str = Field(default="")
    password: str = Field(default="")
    env_prefix: str = Field(
        default="",
        description="Env var prefix for auto-resolving credentials, e.g. 'PORTAINER'",
    )

    # Widget display
    fields: list[str] | None = Field(
        default=None,
        description="Specific fields to show (None = all available)",
    )
    refresh_interval: int = Field(
        default=30, description="Polling interval in seconds"
    )
    websocket: bool = Field(
        default=False, description="Use WebSocket for real-time updates if available"
    )

    # Layout
    column_span: int = Field(default=1, description="Grid column span (1-4)")
    row_span: int = Field(default=1, description="Grid row span")
    visible: bool = Field(default=True, description="Whether the widget is shown")
    order: int = Field(default=0, description="Sort order within group")

    # Service link
    href: str = Field(
        default="", description="URL to open when clicking the service card header"
    )
    target: str = Field(default="_blank", description="Link target (_blank, _self)")


class ServiceGroup(BaseModel):
    """A named group of services displayed as a section.

    Mirrors Homepage's YAML group structure.
    """

    name: str = Field(description="Group name, e.g. 'Infrastructure'")
    services: list[ServiceConfig] = Field(default_factory=list)
    order: int = Field(default=0)
    collapsed: bool = Field(default=False)
    icon: str = Field(default="")


class DashboardLayout(BaseModel):
    """Full dashboard configuration — groups, settings, theme.

    Persisted to YAML at the XDG config path.
    """

    groups: list[ServiceGroup] = Field(default_factory=list)
    columns: int = Field(default=4, description="Number of grid columns")
    theme: str = Field(default="system", description="Theme: system, dark, light, glass")
    card_size: str = Field(default="medium", description="Card size: small, medium, large")
    show_search: bool = Field(default=True)
    show_status_indicators: bool = Field(default=True)
    auto_refresh: bool = Field(default=True)
    refresh_interval: int = Field(
        default=30, description="Global default refresh interval in seconds"
    )


class WidgetRegistration(BaseModel):
    """Metadata about a registered widget type."""

    widget_type: str
    display_name: str
    icon: str
    category: ServiceCategory
    description: str
    available_fields: list[WidgetField]
    supports_websocket: bool = False
    env_prefix: str = Field(
        default="",
        description="Default env var prefix for credential auto-discovery",
    )
    default_url: str = Field(
        default="", description="Default URL to try for auto-discovery"
    )
