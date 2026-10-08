"""Per-node harness selection.

An L3 node spec names its harness in a ``harness`` field
(:class:`~agent_utilities.models.execution_manifest.AgentSpec.harness`). A spec
without the field, or with an empty value, uses ``native``. An unknown name is
an error, never a silent fallback to another runtime.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.layers.harness_port import NATIVE_HARNESS, HarnessPort


class UnknownHarness(LookupError):
    """The node names a harness this process has not registered."""


def harness_name(spec: Any) -> str:
    """The harness a node spec selects (``native`` when unset)."""
    return str(getattr(spec, "harness", "") or NATIVE_HARNESS)


class HarnessRegistry:
    """Named :class:`HarnessPort` adapters; ``native`` is always present."""

    def __init__(self, native: HarnessPort | None = None) -> None:
        if native is None:
            from agent_utilities.layers.harness_native import NativeHarness

            native = NativeHarness()
        self._ports: dict[str, HarnessPort] = {NATIVE_HARNESS: native}

    def register(self, port: HarnessPort) -> None:
        """Add or replace one adapter under its own ``name``."""
        if not isinstance(port, HarnessPort):
            raise TypeError(f"{port!r} does not implement HarnessPort")
        self._ports[port.name] = port

    def names(self) -> list[str]:
        return sorted(self._ports)

    def get(self, name: str) -> HarnessPort:
        port = self._ports.get(name)
        if port is None:
            raise UnknownHarness(f"no harness registered as {name!r}")
        return port

    def select(self, spec: Any) -> HarnessPort:
        """The adapter for one L3 node spec."""
        return self.get(harness_name(spec))


_registry: HarnessRegistry | None = None


def harness_registry() -> HarnessRegistry:
    """The process-wide registry (created on first use)."""
    global _registry
    if _registry is None:
        _registry = HarnessRegistry()
    return _registry


__all__ = ["HarnessRegistry", "UnknownHarness", "harness_name", "harness_registry"]
