from pydantic import BaseModel, Field
from typing import Any
import logging

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
    """
    Routes context directly to shielded workflow boundaries instead of globally
    concatenating strings into the prompt.
    """

    def __init__(self, engine: Any | None = None):
        self.engine = engine

    async def route_context(self, query: str) -> ShieldedResult:
        """
        Dynamically determine relevant namespaces and return a ShieldedResult.
        """
        import uuid

        workflow_id = f"wf-{uuid.uuid4().hex[:8]}"
        namespaces = []
        summary = ""

        if self.engine:
            try:
                # E.g., lookup workflow contexts based on query
                results = self.engine.search_hybrid(query, top_k=3)
                if results:
                    summary = "Found relevant workflow contexts via KG search."
                    namespaces = [
                        f"ephemeral-{r.get('id', 'unknown')}" for r in results
                    ]
            except Exception as e:
                logger.warning(f"WorkflowContextRouter engine lookup failed: {e}")

        # In a real scenario, this might trigger the OWLBridge
        # For now, we return a ShieldedResult representing the bounded context.
        return ShieldedResult(
            workflow_id=workflow_id, allowed_namespaces=namespaces, summary=summary
        )
