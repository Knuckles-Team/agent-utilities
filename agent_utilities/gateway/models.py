"""Pydantic models for the service dashboard.

CONCEPT:AU-OS.config.gateway-service-dashboard — Gateway Service Dashboard

These models define the data contract between the backend aggregator
and all three frontends. Mirrors Homepage's widget/block/container pattern
but with full Pydantic typing.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

_CREDENTIAL_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_SECRET_REF_PREFIXES = ("env://", "secret://", "vault://")


class ServiceCategory(StrEnum):
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

    id: str = Field(min_length=1, max_length=128, description="Unique service ID")
    name: str = Field(min_length=1, max_length=256, description="Display name")
    widget_type: str = Field(
        min_length=1,
        max_length=128,
        description="Widget type key from registry, e.g. 'portainer'",
    )
    url: str = Field(default="", max_length=8192, description="Service base URL")
    icon: str = Field(
        default="",
        max_length=512,
        description="Icon identifier (Lucide name, URL, or emoji)",
    )
    description: str = Field(
        default="", max_length=2048, description="Short description"
    )
    category: ServiceCategory = Field(default=ServiceCategory.CUSTOM)

    # Runtime-only compatibility fields. These may be supplied to an in-memory
    # ServiceConfig by a trusted embedding application, but Pydantic excludes
    # them from every serialization surface and ConfigManager never reads them
    # from persistent YAML. Durable configuration must use ``credential_refs``.
    api_key: str = Field(default="", max_length=65536, exclude=True, repr=False)
    username: str = Field(default="", max_length=65536, exclude=True, repr=False)
    password: str = Field(default="", max_length=65536, exclude=True, repr=False)
    credential_refs: dict[str, str] = Field(
        default_factory=dict,
        max_length=32,
        description="Runtime secret references keyed by credential field name",
    )
    env_prefix: str = Field(
        default="",
        max_length=64,
        description="Env var prefix for auto-resolving credentials, e.g. 'PORTAINER'",
    )

    # Widget display
    fields: list[str] | None = Field(
        default=None,
        max_length=128,
        description="Specific fields to show (None = all available)",
    )
    refresh_interval: int = Field(
        default=30, ge=1, le=86400, description="Polling interval in seconds"
    )
    websocket: bool = Field(
        default=False, description="Use WebSocket for real-time updates if available"
    )

    # Layout
    column_span: int = Field(default=1, ge=1, le=4, description="Grid column span")
    row_span: int = Field(default=1, ge=1, le=100, description="Grid row span")
    visible: bool = Field(default=True, description="Whether the widget is shown")
    order: int = Field(default=0, description="Sort order within group")

    # Service link
    href: str = Field(
        default="",
        max_length=8192,
        description="URL to open when clicking the service card header",
    )
    target: str = Field(
        default="_blank", pattern=r"^_(?:blank|self)$", description="Link target"
    )

    @field_validator("credential_refs")
    @classmethod
    def _validate_credential_refs(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_key, raw_ref in value.items():
            key = str(raw_key).strip().lower()
            ref = str(raw_ref).strip()
            if not _CREDENTIAL_KEY_RE.fullmatch(key):
                raise ValueError("credential reference keys must be simple identifiers")
            if len(ref) > 2048 or not ref.startswith(_SECRET_REF_PREFIXES):
                raise ValueError(
                    "credential values must use a supported runtime secret reference"
                )
            normalized[key] = ref
        return normalized

    @field_validator("fields")
    @classmethod
    def _validate_fields(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not field or len(field) > 128 for field in value):
            raise ValueError("widget field names must contain 1..128 characters")
        return value


class ServiceGroup(BaseModel):
    """A named group of services displayed as a section.

    Mirrors Homepage's YAML group structure.
    """

    name: str = Field(min_length=1, max_length=256, description="Group name")
    services: list[ServiceConfig] = Field(default_factory=list, max_length=512)
    order: int = Field(default=0)
    collapsed: bool = Field(default=False)
    icon: str = Field(default="", max_length=512)


class DashboardLayout(BaseModel):
    """Full dashboard configuration — groups, settings, theme.

    Persisted to YAML at the XDG config path.
    """

    groups: list[ServiceGroup] = Field(default_factory=list, max_length=128)
    columns: int = Field(default=4, ge=1, le=12, description="Number of grid columns")
    theme: str = Field(
        default="system", description="Theme: system, dark, light, glass"
    )
    card_size: str = Field(
        default="medium", description="Card size: small, medium, large"
    )
    show_search: bool = Field(default=True)
    show_status_indicators: bool = Field(default=True)
    auto_refresh: bool = Field(default=True)
    refresh_interval: int = Field(
        default=30,
        ge=1,
        le=86400,
        description="Global default refresh interval in seconds",
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
