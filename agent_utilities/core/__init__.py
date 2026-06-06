"""Core subsystem for agent-utilities.

CONCEPT:OS-5.2 — Cognitive Scheduler
CONCEPT:OS-5.4 — WASM Micro-Agent Sandbox
CONCEPT:ORCH-1.11 — Compiled Orchestration Kernel

This package provides:
- Cognitive scheduler (OS-5.2) — Priority-aware preemptive scheduling
- Sessions gateway — Durable agent session lifecycle management
- Workspace config — Project-level configuration resolution
- WASM agent runner (OS-5.4) — WebAssembly micro-agent sandbox with
  microsecond cold starts and linear memory isolation
"""

from .cognitive_scheduler import CognitiveScheduler
from .release_channel import (
    ChannelRegistry,
    ReleaseChannel,
    active_channel,
    channel_visible,
    component_visible,
    get_component_channel,
    release_channel,
    set_active_channel,
)
from .wasm_runner import WasmAgentRunner

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
