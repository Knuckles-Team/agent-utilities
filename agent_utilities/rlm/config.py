import os
from typing import Literal

from pydantic import BaseModel, Field

from ..base_utilities import to_boolean


class RLMConfig(BaseModel):
    """Configuration for Recursive Language Models (RLM).

    CONCEPT:ORCH-1.1 — Recursive Language Model Execution

    RLM provides a persistent Python REPL that enables agents to process
    arbitrarily long inputs through recursive, programmatic decomposition.
    Based on Zhang et al. (2025), the key insight is that long prompts
    should NOT be fed into the neural network directly but treated as
    part of the environment the LLM symbolically interacts with.

    Trigger Hierarchy (evaluated top-to-bottom, first match wins):
        1. ``enabled=True`` → Always use RLM (global override)
        2. ``trigger_on_large_output=True`` AND output > ``max_context_threshold``
        3. ``trigger_on_ahe_distillation=True`` AND traces > ``ahe_trace_threshold``
        4. ``trigger_on_kg_bulk_analysis=True`` AND nodes > ``kg_bulk_threshold``
        5. ``state.requires_long_horizon=True`` on the GraphState

    Attributes:
        enabled: Global RLM enable switch (env: ``ENABLE_RLM``).
        sub_llm_model_small: Model for recursive sub-calls at depth > 0.
        sub_llm_model_large: Model for the root-level (depth 0) reasoning.
        max_depth: Maximum recursion depth to prevent infinite loops.
        use_container: Run REPL in Docker/Podman sandbox (env: ``RLM_USE_CONTAINER``).
        max_context_threshold: Character threshold for auto-triggering on large outputs.
        trigger_on_large_output: Auto-invoke RLM when output exceeds threshold.
        trigger_on_ahe_distillation: Auto-invoke for AHE trace analysis (CONCEPT:AHE-3.0).
        trigger_on_kg_bulk_analysis: Auto-invoke for KG bulk queries.
        ahe_trace_threshold: Trace count that triggers AHE RLM routing.
        kg_bulk_threshold: Node count that triggers KG bulk RLM routing.
        metadata_only_root: Whitepaper-aligned: send only metadata to root LLM.
        async_enabled: Enable parallel asynchronous sub-calls.
        trajectory_storage: Where to persist RLM execution traces.
    """

    enabled: bool = Field(
        default_factory=lambda: to_boolean(os.getenv("ENABLE_RLM", "False")),
        description="Whether RLM is enabled globally (overrides all triggers).",
    )

    sub_llm_model_small: str = Field(
        default="openai:gpt-4o-mini",
        description="Default small model for cheap recursive sub-calls (used at depth > 0 and AHE sub-calls).",
    )

    sub_llm_model_large: str = Field(
        default="google:gemini-1.5-flash",
        description="Default large model for dense reasoning at root depth (depth 0).",
    )

    max_depth: int = Field(
        default=3,
        description="Maximum recursion depth for RLM.",
    )

    use_container: bool = Field(
        default_factory=lambda: to_boolean(os.getenv("RLM_USE_CONTAINER", "False")),
        description="Whether to run the REPL in a sandboxed container. If false, uses restricted local exec().",
    )

    use_wasm: bool = Field(
        default_factory=lambda: to_boolean(os.getenv("RLM_USE_WASM", "False")),
        description="Whether to run the REPL in a high-performance WASM sandbox (WasmAgentRunner). Overrides use_container.",
    )

    use_monty: bool = Field(
        default_factory=lambda: to_boolean(os.getenv("RLM_USE_MONTY", "False")),
        description="Force the monty sandbox (fast in-process isolation with host callbacks). Overrides use_wasm/use_container.",
    )

    sandbox: Literal["auto", "local", "monty", "wasm", "docker"] = Field(
        default_factory=lambda: os.getenv("RLM_SANDBOX", "auto"),  # type: ignore[arg-type]
        description=(
            "CONCEPT:ORCH-1.38 — sandbox selection. 'auto' engages the capability-driven "
            "router (cheapest capable backend, escalate on reject); any explicit value pins "
            "that one backend. The legacy use_monty/use_wasm/use_container booleans are "
            "honored as overrides for back-compat and map onto this field."
        ),
    )

    max_context_threshold: int = Field(
        default=50_000,
        description="Character threshold for tool outputs to auto-trigger RLM routing.",
    )

    # ── Semantic Trigger Conditions (CONCEPT:ORCH-1.1) ──

    trigger_on_large_output: bool = Field(
        default=True,
        description=(
            "Auto-invoke RLM when a tool/specialist output exceeds "
            "max_context_threshold. Does not require enabled=True."
        ),
    )

    trigger_on_ahe_distillation: bool = Field(
        default=True,
        description=(
            "Auto-invoke RLM during AHE trace distillation when trace "
            "count exceeds ahe_trace_threshold. Enables deep, programmatic "
            "analysis of large evidence corpora (CONCEPT:AHE-3.0)."
        ),
    )

    trigger_on_kg_bulk_analysis: bool = Field(
        default=True,
        description=(
            "Auto-invoke RLM for KG queries that return more nodes "
            "than kg_bulk_threshold. Enables bulk aggregation without "
            "polluting the root LLM context window."
        ),
    )

    ahe_trace_threshold: int = Field(
        default=500,
        description="Number of traces that triggers automatic RLM routing for AHE distillation.",
    )

    kg_bulk_threshold: int = Field(
        default=1000,
        description="Number of KG nodes that triggers automatic RLM routing for bulk analysis.",
    )

    # ── Whitepaper Alignment (Zhang et al. Algorithm 1) ──

    metadata_only_root: bool = Field(
        default=True,
        description=(
            "When True, the root LLM receives only metadata about the "
            "context (length, prefix, type, access instructions) rather "
            "than the full prompt. Aligns with Algorithm 1 of the RLM "
            "whitepaper — prevents context window pollution and forces "
            "the model to rely on symbolic variable access and sub-calls."
        ),
    )

    async_enabled: bool = Field(
        default=True,
        description="Enable parallel asynchronous sub-calls in RLM.",
    )

    max_turns: int = Field(
        default=5,
        ge=1,
        description="Max REPL turns in the full RLM loop before forcing a final answer.",
    )

    compaction_threshold: int = Field(
        default=200_000,
        description=(
            "Output size above which lossy Memento compaction (KG-2.20) is preferred "
            "when RLM (lossless) is not triggered — see select_long_context_strategy."
        ),
    )

    trajectory_storage: Literal["process_flow", "none"] = Field(
        default="process_flow",
        description="How to store RLM trajectories in the Knowledge Graph.",
    )

    def resolved_sandbox(self) -> str:
        """Collapse the ``sandbox`` field + legacy boolean overrides into one selection.

        CONCEPT:ORCH-1.38. Precedence (first wins): explicit ``sandbox`` != 'auto' →
        ``use_monty`` → ``use_wasm`` → ``use_container`` → 'auto'. Returns 'auto' (engage the
        router) or a concrete backend name (the router treats it as a forced pin).
        """
        if self.sandbox != "auto":
            return self.sandbox
        if self.use_monty:
            return "monty"
        if self.use_wasm:
            return "wasm"
        if self.use_container:
            return "docker"
        return "auto"

    def should_trigger(
        self,
        *,
        output_size: int = 0,
        trace_count: int = 0,
        kg_node_count: int = 0,
        requires_long_horizon: bool = False,
    ) -> bool:
        """Evaluate whether RLM should be invoked for a given context.

        Implements the trigger hierarchy documented in the class docstring.
        Call this instead of checking individual flags to ensure consistent
        routing decisions across the codebase.

        Args:
            output_size: Character count of the data to process.
            trace_count: Number of AHE traces in the current distillation.
            kg_node_count: Number of KG nodes in the query result.
            requires_long_horizon: Whether the graph state requests RLM.

        Returns:
            True if RLM should be invoked for this context.
        """
        if self.enabled:
            return True
        if requires_long_horizon:
            return True
        if self.trigger_on_large_output and output_size > self.max_context_threshold:
            return True
        if self.trigger_on_ahe_distillation and trace_count > self.ahe_trace_threshold:
            return True
        if self.trigger_on_kg_bulk_analysis and kg_node_count > self.kg_bulk_threshold:
            return True
        return False

    def select_long_context_strategy(
        self,
        *,
        output_size: int = 0,
        trace_count: int = 0,
        kg_node_count: int = 0,
        requires_long_horizon: bool = False,
    ) -> str:
        """Explicit lossless-vs-compaction decision for a long-context payload.

        CONCEPT:ORCH-1.12. Returns ``"rlm_lossless"`` when RLM should handle the
        context losslessly (recursive variable access, no information loss),
        ``"memento_compaction"`` when the payload is large but RLM isn't triggered
        (fall back to lossy KG-2.20 Memento compaction), else ``"none"``.
        """
        if self.should_trigger(
            output_size=output_size,
            trace_count=trace_count,
            kg_node_count=kg_node_count,
            requires_long_horizon=requires_long_horizon,
        ):
            return "rlm_lossless"
        if output_size > self.compaction_threshold:
            return "memento_compaction"
        return "none"
