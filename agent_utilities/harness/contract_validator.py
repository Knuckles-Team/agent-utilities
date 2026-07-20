"""Tool Contract Validator (OS-5.3).

CONCEPT: OS-5.3 Security & Guardrails / Contract Planning

Defines pre-conditions and post-conditions for execution nodes to guarantee state
correctness and safety boundaries.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class ToolContract(BaseModel):
    """Schema defining pre-conditions and post-conditions for an execution node or tool.

    CONCEPT: OS-5.3 Security & Guardrails / Contract Planning
    """

    node_id: str
    pre_condition: Callable[[dict[str, Any]], bool] | None = Field(
        default=None, description="Pre-execution context validator"
    )
    post_condition_schema: type[BaseModel] | None = Field(
        default=None, description="Post-execution output schema"
    )
    post_condition_verifier: Callable[[dict[str, Any]], bool] | None = Field(
        default=None, description="Post-execution custom verifier"
    )


class ContractValidator:
    """Verifies that execution nodes conform strictly to their tool contracts."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization in singleton
        if not hasattr(self, "_contracts"):
            self._contracts: dict[str, ToolContract] = {}

    @classmethod
    def instance(cls) -> "ContractValidator":
        """Get the singleton instance."""
        if not cls._instance:
            cls._instance = ContractValidator()
        return cls._instance

    def register_contract(self, contract: ToolContract):
        """Register a validation contract for a node_id."""
        self._contracts[contract.node_id] = contract

    def get_contract(self, node_id: str) -> ToolContract | None:
        """Retrieve the validation contract for a node_id."""
        return self._contracts.get(node_id)

    def validate_pre(self, node_id: str, context: dict[str, Any]) -> bool:
        """Validate pre-execution conditions against the current context."""
        contract = self.get_contract(node_id)
        if not contract or not contract.pre_condition:
            return True
        try:
            return contract.pre_condition(context)
        except Exception:
            return False

    def validate_post(self, node_id: str, output: dict[str, Any]) -> bool:
        """Validate post-execution outputs against schemas and verifier checks."""
        contract = self.get_contract(node_id)
        if not contract:
            return True

        if contract.post_condition_schema:
            try:
                contract.post_condition_schema.model_validate(output)
            except Exception:
                return False

        if contract.post_condition_verifier:
            try:
                return contract.post_condition_verifier(output)
            except Exception:
                return False

        return True
