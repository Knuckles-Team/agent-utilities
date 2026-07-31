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
from .composition import default_runtime_capabilities, merge_capabilities
from .content_guardrails import (
    PROMPT_INJECTION_BLOCK_PREFIX,
    SECRET_LEAK_BLOCK_MESSAGE,
    output_schema_guardrail,
    pii_redaction_guardrails,
    prompt_injection_guardrail,
    secret_leak_guardrail,
)
from .context_warnings import ContextLimitWarner
from .eviction import ToolOutputEviction
from .governed_dynamic_workflow import (
    ChildRunEvidence,
    DelegationStep,
    DynamicWorkflowUnavailableError,
    GovernedDynamicWorkflow,
    GovernedDynamicWorkflowResult,
    WorkflowResourceLimits,
    WorkflowScriptEvidence,
)
from .hooks import Hook, HookEvent, HookInput, HookResult, HooksCapability
from .kg_audit_sink import (
    AuditLog,
    AuditSink,
    KgAuditSink,
    RunAuditRecord,
    ToolCallRecord,
)
from .memento import MementoCompaction
from .output_repair import (
    RepairAttempt,
    StructuredOutputRepair,
    StructuredOutputRepairExhausted,
)
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
    "DelegationStep",
    "DynamicWorkflowUnavailableError",
    "GovernedDynamicWorkflow",
    "GovernedDynamicWorkflowResult",
    "WorkflowScriptEvidence",
    "ChildRunEvidence",
    "WorkflowResourceLimits",
    "MementoCompaction",
    "RepairAttempt",
    "StructuredOutputRepair",
    "StructuredOutputRepairExhausted",
    "TeamCapability",
    "AuditLog",
    "AuditSink",
    "KgAuditSink",
    "RunAuditRecord",
    "ToolCallRecord",
    "default_runtime_capabilities",
    "merge_capabilities",
    "PROMPT_INJECTION_BLOCK_PREFIX",
    "SECRET_LEAK_BLOCK_MESSAGE",
    "output_schema_guardrail",
    "pii_redaction_guardrails",
    "prompt_injection_guardrail",
    "secret_leak_guardrail",
]
