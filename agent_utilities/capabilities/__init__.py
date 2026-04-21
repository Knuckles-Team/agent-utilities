#!/usr/bin/python
"""Capabilities package for agent-utilities.

This package provides operational reliability and multi-agent coordination
capabilities for Pydantic AI agents, many integrated with the knowledge graph.
"""

from .checkpointing import (
    Checkpoint,
    CheckpointMiddleware,
    CheckpointToolset,
    FileCheckpointStore,
    GraphCheckpointStore,
    InMemoryCheckpointStore,
    RewindRequested,
    fork_from_checkpoint,
)
from .context_warnings import ContextLimitWarner
from .eviction import ToolOutputEviction
from .hooks import Hook, HookEvent, HookInput, HookResult, HooksCapability
from .stuck_loop import StuckLoopDetection, StuckLoopError
from .teams import TeamCapability

__all__ = [
    "StuckLoopDetection",
    "StuckLoopError",
    "HooksCapability",
    "Hook",
    "HookEvent",
    "HookInput",
    "HookResult",
    "CheckpointMiddleware",
    "CheckpointToolset",
    "InMemoryCheckpointStore",
    "FileCheckpointStore",
    "GraphCheckpointStore",
    "Checkpoint",
    "RewindRequested",
    "fork_from_checkpoint",
    "ContextLimitWarner",
    "ToolOutputEviction",
    "TeamCapability",
]
