import os
from typing import Literal

from pydantic import BaseModel, Field

from ..base_utilities import to_boolean


class RLMConfig(BaseModel):
    """Configuration for Recursive Language Models (RLM)."""

    enabled: bool = Field(
        default_factory=lambda: to_boolean(os.getenv("ENABLE_RLM", "False")),
        description="Whether RLM is enabled globally.",
    )

    sub_llm_model_small: str = Field(
        default="openai:gpt-4o-mini",
        description="Default small model for cheap recursive sub-calls.",
    )

    sub_llm_model_large: str = Field(
        default="google:gemini-1.5-flash",
        description="Default large model for dense reasoning sub-calls.",
    )

    max_depth: int = Field(
        default=3,
        description="Maximum recursion depth for RLM.",
    )

    use_container: bool = Field(
        default_factory=lambda: to_boolean(os.getenv("RLM_USE_CONTAINER", "False")),
        description="Whether to run the REPL in a sandboxed container. If false, uses restricted local exec().",
    )

    max_context_threshold: int = Field(
        default=50_000,
        description="Character threshold for tool outputs to trigger RLM routing.",
    )

    async_enabled: bool = Field(
        default=True,
        description="Enable parallel asynchronous sub-calls in RLM.",
    )

    trajectory_storage: Literal["process_flow", "none"] = Field(
        default="process_flow",
        description="How to store RLM trajectories in the Knowledge Graph.",
    )
