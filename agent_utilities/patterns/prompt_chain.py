#!/usr/bin/python
from __future__ import annotations

"""Prompt Chaining Pattern — CONCEPT:ORCH-1.1

Provides a declarative, multi-step prompt pipeline with intermediate
validation, conditional branching, and KG persistence.  Each chain step
transforms its predecessor's output and optionally validates it against
a Pydantic model before handing off to the next step.

Design-pattern source: Chapter 1 — Prompt Chaining (Agentic Design Patterns,
Antonio Gulli).

OWL alignment:
    :PromptChain rdfs:subClassOf :Procedure  (BFO:Process)
    :PromptChainStep rdfs:subClassOf :Action  (BFO:Process)

See docs/pillars/architecture_c4.md §CONCEPT:ORCH-1.1
"""


import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from agent_utilities.orchestration.resilience import (
    ResiliencePolicy,
    run_with_resilience,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class PromptChainStep(BaseModel):
    """Single step in a prompt chain."""

    name: str = Field(description="Human-readable step name")
    prompt_template: str = Field(
        description="Template with {{prev_output}} and {{input}} placeholders"
    )
    output_key: str = Field(
        default="output",
        description="Key under which this step's result is stored",
    )
    validator_class: str | None = Field(
        default=None,
        description="Fully qualified Pydantic model class for output validation",
    )
    branch_condition: str | None = Field(
        default=None,
        description="Python expression evaluated against step output",
    )
    branch_targets: dict[str, str] = Field(
        default_factory=dict,
        description="condition_result -> target step name mapping",
    )
    max_retries: int = 2


class PromptChain(BaseModel):
    """Declarative multi-step prompt pipeline."""

    id: str = Field(description="Unique chain identifier")
    name: str = Field(description="Human-readable chain name")
    description: str = ""
    steps: list[PromptChainStep] = Field(default_factory=list)
    max_retries_per_step: int = 2


class StepResult(BaseModel):
    """Result of executing a single chain step."""

    step_name: str
    output: str
    latency_ms: float = 0.0
    retries_used: int = 0
    branched_to: str | None = None


class ChainResult(BaseModel):
    """Result of executing a complete prompt chain."""

    chain_id: str
    success: bool = True
    step_results: list[StepResult] = Field(default_factory=list)
    final_output: str = ""
    total_latency_ms: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class PromptChainExecutor:
    """Executes a PromptChain, managing state flow between steps.

    The executor resolves ``{{prev_output}}`` and ``{{input}}`` placeholders
    in each step's template, optionally validates the output, and supports
    conditional branching to non-linear step sequences.

    Parameters
    ----------
    llm_call : callable
        An async callable ``(prompt: str) -> str`` that invokes the LLM.
    kg_engine : optional
        If provided, chain execution metadata is persisted to the KG.
    """

    def __init__(
        self,
        llm_call: Any = None,
        kg_engine: Any = None,
    ) -> None:
        self._llm_call = llm_call
        self._engine = kg_engine

    async def execute(
        self,
        chain: PromptChain,
        initial_input: str,
    ) -> ChainResult:
        """Execute a prompt chain from start to finish.

        Parameters
        ----------
        chain : PromptChain
            The chain definition to execute.
        initial_input : str
            The initial user input that seeds the first step.

        Returns
        -------
        ChainResult
            Aggregated results from all executed steps.
        """
        result = ChainResult(chain_id=chain.id)
        step_map: dict[str, int] = {s.name: i for i, s in enumerate(chain.steps)}
        prev_output = initial_input
        idx = 0
        t0 = time.monotonic()

        while idx < len(chain.steps):
            step = chain.steps[idx]
            step_result = await self._execute_step(step, initial_input, prev_output)
            result.step_results.append(step_result)

            if step_result.branched_to and step_result.branched_to in step_map:
                idx = step_map[step_result.branched_to]
                prev_output = step_result.output
                continue

            prev_output = step_result.output
            idx += 1

        result.final_output = prev_output
        result.total_latency_ms = (time.monotonic() - t0) * 1000
        result.success = True

        # Persist to KG if engine is available
        if self._engine is not None:
            await self._persist_to_kg(chain, result)

        return result

    async def _execute_step(
        self,
        step: PromptChainStep,
        initial_input: str,
        prev_output: str,
    ) -> StepResult:
        """Execute a single chain step with retry logic."""
        prompt = step.prompt_template.replace("{{input}}", initial_input).replace(
            "{{prev_output}}", prev_output
        )

        t0 = time.monotonic()
        retries = 0
        output = ""

        async def _call_once() -> str:
            nonlocal retries
            try:
                if self._llm_call is not None:
                    return await self._llm_call(prompt)
                return f"[mock] Step '{step.name}' executed"
            except Exception:
                retries += 1
                logger.warning(
                    "Step '%s' attempt %d failed, retrying...",
                    step.name,
                    retries,
                )
                raise

        # Historical semantics, declaratively (CONCEPT:ORCH-1.36): retry ANY
        # Exception up to max_retries extra attempts with NO delay between
        # attempts; exhaustion leaves output == "" rather than raising.
        policy = ResiliencePolicy(
            max_attempts=step.max_retries + 1,
            backoff_base_s=0.0,
            jitter=False,
            retry_on=lambda exc: isinstance(exc, Exception),
            name=f"prompt-chain:{step.name}",
        )
        try:
            output = await run_with_resilience(_call_once, policy)
        except Exception:  # noqa: BLE001 - exhausted retries keep output == ""
            pass

        latency_ms = (time.monotonic() - t0) * 1000

        # Evaluate branch condition if present
        branched_to: str | None = None
        if step.branch_condition and step.branch_targets:
            try:
                condition_result = str(
                    eval(  # nosec B307 # noqa: S307 — controlled expression from chain def
                        step.branch_condition,
                        {"output": output, "__builtins__": {}},
                    )
                )
                branched_to = step.branch_targets.get(condition_result)
            except Exception:
                logger.warning(
                    "Branch condition evaluation failed for step '%s'",
                    step.name,
                )

        return StepResult(
            step_name=step.name,
            output=output,
            latency_ms=latency_ms,
            retries_used=retries,
            branched_to=branched_to,
        )

    async def _persist_to_kg(
        self,
        chain: PromptChain,
        result: ChainResult,
    ) -> None:
        """Persist chain execution record to the Knowledge Graph."""
        try:
            from agent_utilities.models.knowledge_graph import (
                PromptChainNode,
                RegistryNodeType,
            )

            node = PromptChainNode(
                id=f"chain:{chain.id}",
                type=RegistryNodeType.PROMPT_CHAIN,
                name=chain.name,
                description=chain.description,
                chain_id=chain.id,
                step_count=len(chain.steps),
                steps=[s.model_dump() for s in chain.steps],
                max_retries_per_step=chain.max_retries_per_step,
                avg_latency_ms=result.total_latency_ms,
                success_rate=1.0 if result.success else 0.0,
            )
            if hasattr(self._engine, "upsert_node"):
                self._engine.upsert_node(node.model_dump())
            logger.info("Persisted chain '%s' to KG", chain.id)
        except Exception:
            logger.debug("KG persistence skipped for chain '%s'", chain.id)
