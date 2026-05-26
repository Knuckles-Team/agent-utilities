"""Core subsystem for agent-utilities.

CONCEPT:OS-5.2 — Cognitive Scheduler
CONCEPT:OS-5.4 — WASM Micro-Agent Sandbox
CONCEPT:ORCH-1.29 — Compiled Orchestration Kernel

This package provides:
- Cognitive scheduler (OS-5.2) — Priority-aware preemptive scheduling
- Sessions gateway — Durable agent session lifecycle management
- Workspace config — Project-level configuration resolution
- WASM agent runner (OS-5.4) — WebAssembly micro-agent sandbox with
  microsecond cold starts and linear memory isolation
"""

from .cognitive_scheduler import CognitiveScheduler
from .wasm_runner import WasmAgentRunner

__all__ = [
    "CognitiveScheduler",
    "WasmAgentRunner",
]
