"""Model Registry.

Declarative configuration for the multi-model routing layer. A single
`ModelRegistry` can describe N LLM models (fast local LM Studio, cloud
frontier models, specialized reasoning models) with per-model cost rates,
routing tier, and capability tags.

Consumers:

- `agent_utilities.server` exposes the active registry via `GET /models`.
- `agent_webui.api_extensions` mirrors it at `GET /api/enhanced/models` so
  the web UI model picker can replace its hardcoded cost table.
- `agent_terminal_ui` `/model list | set | show` commands drive the
  per-turn model override via the `x-agent-model-id` request header.
- The graph orchestrator calls `pick_for_task()` when spawning specialists,
  so a `light` tier can be used for a cheap researcher and a `heavy` tier
  for a planner/synthesizer without any hardcoded model ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ModelTier = Literal["light", "medium", "heavy", "reasoning"]


class ModelCostRate(BaseModel):
    """USD cost per 1 million tokens.

    Zero values are legal and express a local / free-of-charge model
    (e.g. LM Studio running on localhost). Downstream UIs should render
    `$0.00` rather than `—` for a configured zero-cost model so users
    still see that token/tool counts are being tracked.
    """

    input: float = Field(ge=0.0, default=0.0)
    output: float = Field(ge=0.0, default=0.0)

    model_config = ConfigDict(extra="forbid")


class ModelDefinition(BaseModel):
    """One configured LLM model."""

    id: str = Field(description="Stable identifier (user-chosen).")
    name: str = Field(description="Human display name.")
    provider: str = Field(
        description=(
            "pydantic-ai provider string, e.g. 'openai', 'anthropic', "
            "'google-gla', 'ollama', or a custom label for local endpoints."
        ),
    )
    model_id: str = Field(
        description=(
            "The actual model identifier sent to the provider, e.g. "
            "'gpt-4o-mini', 'claude-3-5-haiku-20241022', "
            "'llama-3.2-3b-instruct'."
        ),
    )
    base_url: str | None = Field(
        default=None,
        description="Override base URL; useful for local servers.",
    )
    api_key_env: str | None = Field(
        default=None,
        description=(
            "Name of env var holding the API key. Null means no auth "
            "(e.g. local LM Studio)."
        ),
    )
    tier: ModelTier = Field(
        default="medium",
        description=(
            "Routing tier. 'light' for fast/cheap, 'heavy' for "
            "long-context, 'reasoning' for deep thinking."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Freeform capability tags, e.g. 'vision', 'code', 'tools'.",
    )
    cost: ModelCostRate = Field(
        default_factory=ModelCostRate,
        description="Per-1M-token cost. Use 0/0 for local/zero-cost models.",
    )
    context_window: int | None = Field(
        default=None, description="Max tokens the model accepts."
    )
    max_output_tokens: int | None = Field(
        default=None, description="Max tokens it will generate in one turn."
    )
    is_default: bool = False

    model_config = ConfigDict(extra="forbid", frozen=False)


class ModelRegistry(BaseModel):
    """In-memory model registry.

    The registry is intentionally a plain Pydantic model so it round-trips
    cleanly through JSON/YAML and through the HTTP boundary. All lookup and
    routing helpers are pure methods that never mutate state.
    """

    models: list[ModelDefinition] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    def get_default(self) -> ModelDefinition | None:
        """Return the model marked `is_default`, or fall back to the first.

        Returns:
            The default `ModelDefinition`, or ``None`` if the registry is empty.
        """
        for m in self.models:
            if m.is_default:
                return m
        return self.models[0] if self.models else None

    def get_by_id(self, model_id: str) -> ModelDefinition | None:
        """Return the model with matching id, or ``None`` if missing."""
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def list_by_tier(self, tier: ModelTier) -> list[ModelDefinition]:
        """Return all configured models in the given routing tier."""
        return [m for m in self.models if m.tier == tier]

    def pick_for_task(
        self,
        *,
        complexity: ModelTier = "medium",
        required_tags: list[str] | None = None,
    ) -> ModelDefinition:
        """Return the best-fit model for a task.

        Algorithm:
            1. Filter by ``required_tags`` (all tags must be present).
            2. Prefer an exact-tier match.
            3. Fallback order is tier-specific and biased toward capability:
               heavier tiers fall back to reasoning, lighter tiers fall back
               through medium/heavy.
            4. If tag filtering eliminates every candidate, re-try without
               tags before giving up.
            5. As a last resort, return the registry default.

        Args:
            complexity: Tier of the task being spawned.
            required_tags: Tags every candidate must carry (AND semantics).

        Returns:
            The selected `ModelDefinition`.

        Raises:
            ValueError: If the registry is empty.
        """
        required = required_tags or []
        tier_priority: dict[ModelTier, list[ModelTier]] = {
            "light": ["light", "medium", "heavy", "reasoning"],
            "medium": ["medium", "heavy", "light", "reasoning"],
            "heavy": ["heavy", "reasoning", "medium", "light"],
            "reasoning": ["reasoning", "heavy", "medium", "light"],
        }

        tagged = [m for m in self.models if all(t in m.tags for t in required)]

        for candidates in (tagged, self.models):
            if not candidates:
                continue
            for t in tier_priority[complexity]:
                for m in candidates:
                    if m.tier == t:
                        return m

        default = self.get_default()
        if default is None:
            raise ValueError("Model registry is empty; configure at least one model.")
        return default

    def add(self, model: ModelDefinition) -> None:
        """Append a model to the registry.

        Raises:
            ValueError: If another model with the same id already exists.
        """
        if self.get_by_id(model.id) is not None:
            raise ValueError(f"Duplicate model id: {model.id}")
        self.models.append(model)

    @classmethod
    def load_from_file(cls, path: str | Path) -> ModelRegistry:
        """Load a registry from a JSON or YAML file.

        The file extension picks the parser: ``.yaml`` / ``.yml`` use
        ``yaml.safe_load``; everything else is treated as JSON.

        Args:
            path: Path to the configuration file.

        Returns:
            A validated `ModelRegistry`.
        """
        p = Path(path)
        text = p.read_text()
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.model_validate(data)

    def to_api_payload(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot for API responses.

        Returns:
            ``{"models": [...], "default_id": "..."}``, mirroring the
            shape consumed by the terminal UI and web UI clients.
        """
        default = self.get_default()
        return {
            "models": [m.model_dump() for m in self.models],
            "default_id": default.id if default else None,
        }
