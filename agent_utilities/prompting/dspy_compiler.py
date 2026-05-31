from __future__ import annotations

"""DSPy Compiler Module.

CONCEPT:AHE-3.1 — Mathematical Prompt Optimization
CONCEPT:KG-2.2 — DSPy Integration

This module provides the logic to dynamically translate agent-utilities
JSON prompt blueprints into `dspy.Signature` classes. This allows the AHE-3
evolution loop to optimize prompt prefixes and few-shot examples mathematically.
"""

import logging
from typing import Any

import dspy

logger = logging.getLogger(__name__)


def compile_json_to_signature(blueprint: dict[str, Any]) -> type[dspy.Signature]:
    """Dynamically compile a JSON blueprint into a DSPy Signature.

    Extracts 'instructions' and 'identity' to build the docstring for the
    Signature, which DSPy uses as the prefix during generation and optimization.

    Args:
        blueprint: The loaded JSON prompt dictionary.

    Returns:
        A dynamically generated subclass of dspy.Signature.
    """
    instructions = ""
    if "instructions" in blueprint:
        inst = blueprint["instructions"]
        if isinstance(inst, dict):
            instructions = "\n".join(str(v) for v in inst.values())
        else:
            instructions = str(inst)

    identity = ""
    if "identity" in blueprint:
        iden = blueprint["identity"]
        if isinstance(iden, dict):
            identity = "\n".join(f"{k}: {v}" for k, v in iden.items())
        else:
            identity = str(iden)

    # The docstring is critical in DSPy; it forms the instruction prefix.
    docstring = f"{identity}\n\n{instructions}".strip()
    if not docstring:
        # Fallback to description or role if missing
        docstring = blueprint.get("metadata", {}).get(
            "description", "Process the input and generate a response."
        )

    task_name = blueprint.get("task", "agent")
    class_name = f"Signature_{task_name.replace('-', '_')}"

    # For standard agent patterns, we supply context and task as inputs
    class GeneratedSignature(dspy.Signature):
        context: str = dspy.InputField(
            desc="The context, history, and workspace state."
        )
        task: str = dspy.InputField(desc="The user task or query to execute.")
        response: str = dspy.OutputField(
            desc="The agent's text response or structured action payload."
        )

    GeneratedSignature.__doc__ = docstring
    GeneratedSignature.__name__ = class_name
    GeneratedSignature.__qualname__ = class_name

    return GeneratedSignature


class AgentTaskModule(dspy.Module):
    """A standard DSPy module wrapper for agent tasks.

    This module encapsulates a signature and executes it using ChainOfThought
    or standard Predict, enabling clean compilation during the evolution loop.
    """

    def __init__(self, signature: type[dspy.Signature], use_cot: bool = False):
        super().__init__()
        self.signature = signature
        self.predictor: dspy.Predict | dspy.ChainOfThought
        if use_cot:
            self.predictor = dspy.ChainOfThought(self.signature)
        else:
            self.predictor = dspy.Predict(self.signature)

    def forward(self, context: str, task: str) -> dspy.Prediction:
        return self.predictor(context=context, task=task)
