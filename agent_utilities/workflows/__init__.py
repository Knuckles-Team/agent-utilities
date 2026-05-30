"""Workflow Catalog & Runner — Externally Consumable Orchestration Flows.

CONCEPT:ORCH-1.24 — Workflow Lifecycle Management

Provides a unified system for defining, persisting, discovering, and
executing reusable agent workflows:

    ┌──────────────┐   load()     ┌─────────────────┐   register_in_kg()
    │  catalog.yaml │ ──────────► │ WorkflowCatalog  │ ────────────────►  KG
    └──────────────┘              └────────┬────────┘
                                           │ to_graph_plans()
                                  ┌────────▼────────┐
                                  │   GraphPlan[]    │
                                  └────────┬────────┘
                                           │
                                  ┌────────▼────────┐   run_agent()
                                  │ WorkflowRunner   │ ──────────────►  LLM
                                  └─────────────────┘

External consumers (other agents, UIs, CI) can:
    - Discover workflows via ``graph_orchestrate(action='list_workflows')``
    - Execute them via ``graph_orchestrate(action='execute_workflow')``
    - Export as JSON via ``graph_orchestrate(action='export_workflow')``
    - Create new ones via ``graph_orchestrate(action='compile_workflow')``
"""

from typing import Any

from agent_utilities.orchestration import WorkflowRunner

from .distillation_hook import WorkflowDistillationHook
from .skill_compiler import SkillCompiler

WorkflowResult = dict[str, Any]

__all__ = [
    "SkillCompiler",
    "WorkflowDistillationHook",
    "WorkflowRunner",
    "WorkflowResult",
]
