"""Workflow-context shielding strategy (Plan 03 Step 3).

Migrated from the former ``graph/workflow_context_router.py``. Routes context to
shielded, workflow-scoped boundaries instead of globally concatenating strings
into the prompt — reducing prompt bloat and leaking. Owns the ``ShieldedResult``
payload + ``WorkflowContextRouter`` behaviour.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ShieldedResult(BaseModel):
    """Payload encapsulating workflow-scoped context."""

    workflow_id: str
    allowed_namespaces: list[str] = Field(default_factory=list)
    ephemeral_data: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""

    def to_prompt_string(self) -> str:
        """Return a prompt-safe summary of the shielded context."""
        return self.summary or "No contextual summary available. Use ephemeral queries."


class WorkflowContextRouter:
    """Route context to shielded workflow boundaries rather than the global prompt."""

    name = "workflow_context"

    def __init__(self, engine: Any | None = None):
        self.engine = engine

    async def route_context(
        self,
        query: str,
        context_state: dict[str, Any] | None = None,
    ) -> ShieldedResult:
        """Return the task-relevant, budgeted context slice (CONCEPT:AU-KG.memory.tiered-memory-caching).

        ``context_state`` lets the caller scope retrieval to *this* task instead
        of dumping everything — "the N pieces that matter for the task in front
        of it". Recognised keys:

        * ``workflow_id`` / ``goal_id`` — identity to tag the shielded result.
        * ``required_caps`` — capabilities the task needs (biases search).
        * ``namespaces`` — explicit namespaces to allow (bypasses search).
        * ``top_k`` — how many context nodes (default 3).
        * ``token_budget`` — cap on retrieved context tokens.
        """
        state = context_state or {}
        workflow_id = (
            state.get("workflow_id")
            or state.get("goal_id")
            or f"wf-{uuid.uuid4().hex[:8]}"
        )
        top_k = int(state.get("top_k", 3))
        required_caps = state.get("required_caps") or []
        token_budget = state.get("token_budget")

        # Explicit namespaces short-circuit search (already task-scoped).
        explicit = state.get("namespaces")
        if explicit:
            return ShieldedResult(
                workflow_id=workflow_id,
                allowed_namespaces=list(explicit),
                summary="Task-scoped to caller-provided namespaces.",
            )

        namespaces: list[str] = []
        summary = ""
        if self.engine:
            try:
                # Bias the query with the task's required capabilities so we
                # retrieve what the task needs, not the whole graph.
                scoped_query = query
                if required_caps:
                    scoped_query = f"{query} :: caps={','.join(required_caps)}"
                results = self.engine.search_hybrid(scoped_query, top_k=top_k) or []
                if token_budget:
                    from ....knowledge_graph.retrieval.budget import fit_within

                    results = fit_within(
                        results,
                        token_budget,
                        text_of=lambda r: str(
                            r.get("content") or r.get("summary") or r.get("name") or r
                        ),
                    )
                if results:
                    summary = f"Task-scoped: {len(results)} context node(s)" + (
                        f", caps={','.join(required_caps)}" if required_caps else ""
                    )
                    namespaces = [
                        f"ephemeral-{r.get('id', 'unknown')}" for r in results
                    ]
            except Exception as e:
                logger.warning("WorkflowContextRouter engine lookup failed: %s", e)

        return ShieldedResult(
            workflow_id=workflow_id, allowed_namespaces=namespaces, summary=summary
        )
