"""CONCEPT:ORCH-1.33 — Multi-CLI Agent Adapter Registry package."""

from __future__ import annotations

from .base import (
    AdapterDefinition,
    DetectedAdapter,
    ExecEvent,
    ExecEventType,
    PromptDelivery,
    StreamFormat,
)
from .registry import AdapterRegistry, get_adapter_registry

__all__ = [
    "AdapterDefinition",
    "DetectedAdapter",
    "ExecEvent",
    "ExecEventType",
    "PromptDelivery",
    "StreamFormat",
    "AdapterRegistry",
    "get_adapter_registry",
]
