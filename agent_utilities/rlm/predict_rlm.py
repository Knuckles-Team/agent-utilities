"""Predict-RLM: Structured, type-safe RLM executions using Pydantic signatures.

CONCEPT:ORCH-1.12 — Structured Predict-RLM Runtime

This module implements a native Pydantic signature system to replicate
the structured input/output contract of DSPy Signatures without adding
external dependencies.
"""

import ast
import inspect
from typing import Any

from pydantic import BaseModel, Field

from ..graph.state import GraphDeps
from .config import RLMConfig
from .repl import RLMEnvironment
from .schema import SchemaContract


def _validate_purity(source: str) -> None:
    """Validate that the tool function source code is pure.

    Rejects any code that uses global or nonlocal statements.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Global | ast.Nonlocal):
            raise ValueError("Tool function is not pure: uses global/nonlocal.")


def InputField(default: Any = ..., *, description: str = "", **kwargs) -> Any:
    """Helper to define an input field in an RLM Signature."""
    json_schema_extra = kwargs.setdefault("json_schema_extra", {})
    json_schema_extra["is_input"] = True
    return Field(default, description=description, **kwargs)


def OutputField(default: Any = ..., *, description: str = "", **kwargs) -> Any:
    """Helper to define an output field in an RLM Signature."""
    json_schema_extra = kwargs.setdefault("json_schema_extra", {})
    json_schema_extra["is_output"] = True
    return Field(default, description=description, **kwargs)


class PredictRLM:
    """Predict-RLM execution harness wrapping RLMEnvironment.

    Enforces Pydantic-based structured signatures and allows dynamic
    mounting of skills into the sandboxed REPL.
    """

    def __init__(
        self,
        signature: type[BaseModel],
        config: RLMConfig | None = None,
        graph_deps: GraphDeps | None = None,
    ):
        self.signature = signature
        self.config = config or RLMConfig()
        self.graph_deps = graph_deps
        self.skills: dict[str, Any] = {}

        # Inspect the signature to identify input and output fields
        self.inputs: list[str] = []
        self.outputs: list[str] = []

        for name, field in self.signature.model_fields.items():
            extra = getattr(field, "json_schema_extra", None) or {}
            is_input = False
            is_output = False
            if isinstance(extra, dict):
                is_input = bool(extra.get("is_input", False))
                is_output = bool(extra.get("is_output", False))

            if is_input:
                self.inputs.append(name)
            elif is_output:
                self.outputs.append(name)
            else:
                # If neither is marked, treat fields with default values as inputs,
                # and fields with no defaults as outputs by convention.
                if field.is_required():
                    self.outputs.append(name)
                else:
                    self.inputs.append(name)

    def mount_skill_unit(self, skill: Any) -> None:
        """Mount a composable :class:`~agent_utilities.rlm.skills.Skill` (CONCEPT:ORCH-1.28).

        Merges the skill's ``tools`` and ``modules`` (name→source) into the REPL skill set and
        appends its instructions to the task prompt. Raises on a name collision with already-mounted
        tools/modules so composition is explicit.
        """
        for nm, src in {**skill.modules, **skill.tools}.items():
            if nm in self.skills:
                raise ValueError(f"Skill unit conflict: {nm!r} already mounted")
            self.skills[nm] = src
        if getattr(skill, "instructions", ""):
            self._extra_instructions = (
                getattr(self, "_extra_instructions", "") + "\n\n" + skill.instructions
            )

    def mount_skill(self, name: str, skill_fn: Any):
        """Register a custom function or API client helper to be injected into the REPL."""
        import textwrap

        # Dedent so functions defined in an indented scope (inside a method or
        # class body) parse cleanly — inspect.getsource preserves leading
        # indentation, which would otherwise raise IndentationError in ast.parse.
        source = textwrap.dedent(inspect.getsource(skill_fn))
        _validate_purity(source)
        self.skills[name] = source

    def _generate_instruction_prompt(self, inputs: dict[str, Any]) -> str:
        """Construct the prompt guiding the RLM REPL to solve the task and assign outputs."""
        prompt = (
            f"TASK DESCRIPTION:\n"
            f"  {self.signature.__doc__ or 'No description provided.'}\n\n"
        )
        # CONCEPT:ORCH-1.28 — inject mounted composable-Skill instructions (the SOP) so they
        # actually reach the model, not just the skill's tool/module sources.
        extra = getattr(self, "_extra_instructions", "")
        if extra.strip():
            prompt += f"SKILL INSTRUCTIONS (standard operating procedure):\n{extra.strip()}\n\n"
        prompt += "INPUT VALUES PROVIDED:\n"
        for name in self.inputs:
            val = inputs.get(name, "Not provided")
            # If the input is massive, show metadata only in the prompt to prevent context pollution
            val_str = str(val)
            if len(val_str) > 1000:
                prompt += f"  - `{name}` (Massive field, length: {len(val_str)} chars). Access programmatically via `context['{name}']`.\n"
            else:
                prompt += f"  - `{name}`: {val_str}\n"

        prompt += "\nREQUIRED OUTPUT FIELDS TO POPULATE:\n"
        for name in self.outputs:
            field = self.signature.model_fields[name]
            prompt += f"  - `{name}`: {field.description or 'No description'}\n"
            # CONCEPT:ORCH-1.12 (per-field output schema) — show the exact JSON
            # Schema for each output field (not just its prose description) so
            # the model knows the precise shape.
            annotation = getattr(field, "annotation", None)
            if annotation is not None and annotation is not str:
                try:
                    contract = SchemaContract.from_spec(annotation)
                    prompt += f"    schema: {contract.json_schema_str}\n"
                except Exception:  # noqa: BLE001 - best-effort prompt hint
                    pass

        prompt += (
            "\nINSTRUCTIONS:\n"
            "  1. Analyze the inputs programmatically inside the Python REPL.\n"
            "  2. Compute values for each of the required output fields.\n"
            "  3. For EACH required output field, call `FINAL_VAR('field_name', value)` to record the result.\n"
            "  4. Ensure your output values conform to the types and constraints described.\n"
        )
        return prompt

    async def run(self, **inputs) -> BaseModel:
        """Execute the RLM REPL and return a validated signature model instance."""
        # Bundle inputs into a context dictionary
        context = {name: inputs.get(name) for name in self.inputs}

        # Inject mounted skills into context if needed, but wait, skills should go into globals
        env = RLMEnvironment(
            context=context,
            depth=0,
            config=self.config,
            graph_deps=self.graph_deps,
            signature=self.signature,
            inputs_keys=self.inputs,
            outputs_keys=self.outputs,
            tool_sources=self.skills,
        )

        prompt = self._generate_instruction_prompt(inputs)
        await env.run_full_rlm(prompt)

        # Gather final variables and validate with Pydantic
        output_data = {}
        for name in self.outputs:
            if name in env.vars:
                output_data[name] = env.vars[name]
            elif "__FINAL__" in env.vars and env.vars["__FINAL__"] == name:
                output_data[name] = env.vars[name]
            else:
                # Attempt to extract from globals/locals if written directly
                if name in env.globals_dict:
                    output_data[name] = env.globals_dict[name]

        # Combine inputs and outputs to construct the full signature model
        full_data = {**inputs, **output_data}
        return self.signature(**full_data)
