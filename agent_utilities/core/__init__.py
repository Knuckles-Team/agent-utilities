"""Core subsystem for agent-utilities.

CONCEPT:AU-OS.state.cognitive-scheduler-preemption — Cognitive Scheduler
CONCEPT:AU-OS.governance.wasm-micro-agent-sandbox — WASM Micro-Agent Sandbox
CONCEPT:AU-ORCH.sandbox.compiled-orchestration-kernel — Compiled Orchestration Kernel

This package provides:
- Cognitive scheduler (OS-5.2) — Priority-aware preemptive scheduling
- Sessions gateway — Durable agent session lifecycle management
- Workspace config — Project-level configuration resolution
- WASM agent runner (OS-5.4) — WebAssembly micro-agent sandbox with
  microsecond cold starts and linear memory isolation
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CognitiveScheduler": ("cognitive_scheduler", "CognitiveScheduler"),
    "WasmAgentRunner": ("wasm_runner", "WasmAgentRunner"),
    "ReleaseChannel": ("release_channel", "ReleaseChannel"),
    "ChannelRegistry": ("release_channel", "ChannelRegistry"),
    "active_channel": ("release_channel", "active_channel"),
    "set_active_channel": ("release_channel", "set_active_channel"),
    "channel_visible": ("release_channel", "channel_visible"),
    "component_visible": ("release_channel", "component_visible"),
    "get_component_channel": ("release_channel", "get_component_channel"),
    "release_channel": ("release_channel", "release_channel"),
}


def __getattr__(name: str) -> Any:
    """Load one explicitly exported core symbol without importing other runtimes."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = [
    "CognitiveScheduler",
    "WasmAgentRunner",
    # Release channels (OS-5.13)
    "ReleaseChannel",
    "ChannelRegistry",
    "active_channel",
    "set_active_channel",
    "channel_visible",
    "component_visible",
    "get_component_channel",
    "release_channel",
]
