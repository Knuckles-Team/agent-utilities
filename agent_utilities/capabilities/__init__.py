#!/usr/bin/python
# coding: utf-8
"""Capabilities package for agent-utilities.

This package provides operational reliability and multi-agent coordination
capabilities for Pydantic AI agents, many integrated with the knowledge graph.
"""

from .stuck_loop import StuckLoopDetection, StuckLoopError
from .hooks import HooksCapability, Hook, HookEvent, HookInput, HookResult
from .checkpointing import (
    CheckpointMiddleware,
    CheckpointToolset,
    InMemoryCheckpointStore,
    FileCheckpointStore,
    GraphCheckpointStore,
    Checkpoint,
    RewindRequested,
    fork_from_checkpoint,
)
from .context_warnings import ContextLimitWarner
from .eviction import ToolOutputEviction
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
