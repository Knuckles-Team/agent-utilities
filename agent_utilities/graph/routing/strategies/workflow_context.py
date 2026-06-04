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

    async def route_context(self, query: str) -> ShieldedResult:
        """Determine relevant namespaces and return a bounded ``ShieldedResult``."""
        workflow_id = f"wf-{uuid.uuid4().hex[:8]}"
        namespaces: list[str] = []
        summary = ""

        if self.engine:
            try:
                results = self.engine.search_hybrid(query, top_k=3)
                if results:
                    summary = "Found relevant workflow contexts via KG search."
                    namespaces = [
                        f"ephemeral-{r.get('id', 'unknown')}" for r in results
                    ]
            except Exception as e:
                logger.warning("WorkflowContextRouter engine lookup failed: %s", e)

        return ShieldedResult(
            workflow_id=workflow_id, allowed_namespaces=namespaces, summary=summary
        )
