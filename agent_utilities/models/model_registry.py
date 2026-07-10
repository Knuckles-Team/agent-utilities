from __future__ import annotations

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
- The graph orchestrator calls `pick_for_task()` when spawning adaptive_agent_router,
  so a `light` tier can be used for a cheap researcher and a `heavy` tier
  for a planner/synthesizer without any hardcoded model ids.
"""


import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_utilities.agent.sampling_profile import DEFAULT_PROFILE, SamplingProfile

ModelTier = Literal["light", "medium", "heavy", "reasoning"]

# Ordered tier list for CONCEPT:AU-ORCH.routing.confidence-gated-routing-log confidence-gated routing helpers.
_TIER_ORDER: list[ModelTier] = ["light", "medium", "heavy", "reasoning"]

# CONCEPT:AU-ORCH.routing.conductor-per-step-model — Role-Specialized Model Routing.
# Functional roles a pipeline stage can request. Assimilated from Quarq Agent's
# three-specialized-model pattern (planner / generator / learner; agent-oss/agent.py:58-92),
# generalized to a role→(tier,tags) binding over the existing registry so any provider
# pool works and degrades gracefully via pick_for_task() instead of hardcoded model ids.
# CONCEPT:AU-ORCH.routing.conductor-per-step-model (+ORCH-1.12 RLM extension): planner/generator/learner/judge plus the RLM-GEPA
# roles — a cheap proxy executor + sub-LM optimized against a strong proposer (the AppWorld trick).
ModelRole = Literal[
    "planner",
    "generator",
    "learner",
    "judge",
    "rlm-executor",
    "rlm-proposer",
    "rlm-sublm",
]


class RoleSpec(BaseModel):
    """Tier + capability-tag query that a functional role binds to.

    CONCEPT:AU-ORCH.routing.conductor-per-step-model — resolved at runtime through :meth:`ModelRegistry.pick_for_task`,
    so a role degrades by tier when no exact-tag/tier model is configured.
    """

    tier: ModelTier = Field(default="medium")
    tags: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# Default role bindings mirroring Quarq's roles, expressed as portable tier+tag queries.
#   planner   → cheap/fast structured-plan generation (Quarq gpt-4o-mini HyDE planner)
#   generator → high-capability synthesis (Quarq gpt-4.1 generator)
#   learner   → high-capability fact extraction / targeted edits (Quarq gpt-4.1 learner)
#   judge     → deepest reasoning for binary evaluation (LongMemEval judge)
_DEFAULT_ROLE_ROUTING: dict[str, RoleSpec] = {
    "planner": RoleSpec(tier="light", tags=["plan", "json"]),
    "generator": RoleSpec(tier="heavy", tags=["synthesis"]),
    "learner": RoleSpec(tier="heavy", tags=["extraction"]),
    "judge": RoleSpec(tier="reasoning", tags=[]),
    # CONCEPT:AU-ORCH.routing.conductor-per-step-model RLM-GEPA roles: cheap executor/sub-LM run the skill; the strong proposer
    # reflects on traces and rewrites it. A skill optimized with a cheap executor still lifts a
    # strong one at eval — so this is the cost/quality Pareto knob for RLM-GEPA.
    "rlm-executor": RoleSpec(tier="light", tags=["code"]),
    "rlm-sublm": RoleSpec(tier="light", tags=[]),
    "rlm-proposer": RoleSpec(tier="reasoning", tags=["synthesis"]),
}


# CONCEPT:AU-ORCH.routing.sampling-profile-selection — curated per-task-class sampling profiles. Hand-tuned starting
# points: deterministic low-temp (+ tight top_k/min_p) for code/extraction/judge where
# we want one right answer; exploratory high-temp for generate/brainstorm where we want
# spread. The AHE-3.38 evolution loop refines these in place via promote_winner; they are
# module constants (one correct default each), never env flags (Configuration discipline).
_DEFAULT_TASK_PROFILES: dict[str, SamplingProfile] = {
    "code": SamplingProfile(
        task_class="code", temperature=0.1, top_p=0.9, top_k=20, min_p=0.0
    ),
    "extraction": SamplingProfile(
        task_class="extraction", temperature=0.0, top_p=0.8, top_k=20, min_p=0.0
    ),
    "judge": SamplingProfile(task_class="judge", temperature=0.0, top_p=1.0),
    "reasoning": SamplingProfile(task_class="reasoning", temperature=0.6, top_p=0.95),
    "plan": SamplingProfile(task_class="plan", temperature=0.4, top_p=0.9),
    "generate": SamplingProfile(task_class="generate", temperature=0.7, top_p=1.0),
    "brainstorm": SamplingProfile(
        task_class="brainstorm", temperature=1.0, top_p=1.0, presence_penalty=0.3
    ),
}

# How a functional role (ORCH-1.27) maps to a sampling task-class, so picking a model
# for a role also yields the matching profile.
_ROLE_TASK_CLASS: dict[str, str] = {
    "planner": "plan",
    "generator": "generate",
    "learner": "extraction",
    "judge": "judge",
    "rlm-executor": "code",
    "rlm-sublm": "code",
    "rlm-proposer": "reasoning",
}


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
    oauth2: dict[str, Any] | None = Field(
        default=None,
        description=(
            "OAuth2 client_credentials block (CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle) — "
            "machine-to-machine auth for enterprise OpenAI-compatible/Azure endpoints requiring a "
            "short-lived minted bearer instead of a static api_key_env. Mutually exclusive with "
            "api_key_env (validated below). Shape: "
            "agent_utilities.security.oauth_client_credentials.OAuth2ClientCredentialsConfig."
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

    @model_validator(mode="after")
    def _validate_auth_mode(self) -> ModelDefinition:
        """``api_key_env`` and ``oauth2`` are mutually exclusive (CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle)."""
        if self.api_key_env and self.oauth2:
            raise ValueError(
                f"ModelDefinition {self.id!r}: 'api_key_env' and 'oauth2' are mutually "
                "exclusive — configure exactly one authentication mode."
            )
        if self.oauth2:
            # Lazy import: avoids a core-models↔security import cycle at module-load time;
            # only fires when a definition actually carries an oauth2 block.
            from agent_utilities.security.oauth_client_credentials import (
                OAuth2ClientCredentialsConfig,
            )

            try:
                self.oauth2 = OAuth2ClientCredentialsConfig.model_validate(
                    self.oauth2
                ).model_dump()
            except Exception as exc:
                raise ValueError(
                    f"ModelDefinition {self.id!r}: invalid oauth2 block: {exc}"
                ) from exc
        return self


class ModelRegistry(BaseModel):
    """In-memory model registry.

    The registry is intentionally a plain Pydantic model so it round-trips
    cleanly through JSON/YAML and through the HTTP boundary. All lookup and
    routing helpers are pure methods that never mutate state.
    """

    models: list[ModelDefinition] = Field(default_factory=list)
    role_routing: dict[str, RoleSpec] = Field(
        default_factory=dict,
        description=(
            "CONCEPT:AU-ORCH.routing.conductor-per-step-model — optional role→(tier,tags) overrides. Empty keys "
            "fall back to the built-in default map. Round-trips through JSON/YAML."
        ),
    )
    task_class_profiles: dict[str, SamplingProfile] = Field(
        default_factory=dict,
        description=(
            "CONCEPT:AU-ORCH.routing.sampling-profile-selection — optional per-task-class sampling-profile overrides. "
            "Empty keys fall back to the curated built-in defaults; the AHE-3.38 loop "
            "writes learned profiles here. Round-trips through JSON/YAML."
        ),
    )

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

    # ── CONCEPT:AU-ORCH.routing.conductor-per-step-model role-specialized routing ───────────────────────────────

    def resolve_role(
        self,
        role: ModelRole | str,
        *,
        override: RoleSpec | None = None,
    ) -> RoleSpec:
        """Resolve a functional role to its tier+tags binding.

        Precedence (highest first): explicit ``override`` → this registry's
        ``role_routing`` → the built-in :data:`_DEFAULT_ROLE_ROUTING` →
        ``RoleSpec(tier="medium")`` for unknown roles.

        CONCEPT:AU-ORCH.routing.conductor-per-step-model.
        """
        if override is not None:
            return override
        if role in self.role_routing:
            return self.role_routing[role]
        return _DEFAULT_ROLE_ROUTING.get(role, RoleSpec(tier="medium"))

    def pick_for_role(
        self,
        role: ModelRole | str,
        *,
        override: RoleSpec | None = None,
    ) -> ModelDefinition:
        """Return the best-fit model for a functional role (planner/generator/learner/judge).

        Binds the role to a ``(tier, tags)`` query (see :meth:`resolve_role`) and
        delegates to :meth:`pick_for_task`, inheriting its tier-fallback semantics so a
        role never hard-fails on a sparse pool (unless the registry is empty).

        This is the agent-utilities answer to Quarq Agent's three hardcoded model
        clients (agent-oss/agent.py:58-92): portable across any configured provider pool.

        Args:
            role: Functional role to resolve.
            override: Optional per-call ``RoleSpec`` that wins over config/defaults.

        Returns:
            The selected ``ModelDefinition``.

        Raises:
            ValueError: If the registry is empty.
        """
        spec = self.resolve_role(role, override=override)
        return self.pick_for_task(complexity=spec.tier, required_tags=spec.tags)

    # ── CONCEPT:AU-ORCH.routing.sampling-profile-selection sampling-profile selection ─────────────────────────────

    def pick_profile_for_task(self, task_class: str) -> SamplingProfile:
        """Return the sampling profile for a task-class (CONCEPT:AU-ORCH.routing.sampling-profile-selection).

        Precedence: this registry's ``task_class_profiles`` (learned/operator
        overrides) → the curated :data:`_DEFAULT_TASK_PROFILES` built-ins →
        :data:`DEFAULT_PROFILE` for an unknown class. Never raises.
        """
        if task_class in self.task_class_profiles:
            return self.task_class_profiles[task_class]
        return _DEFAULT_TASK_PROFILES.get(task_class, DEFAULT_PROFILE)

    def pick_profile_for_role(self, role: ModelRole | str) -> SamplingProfile:
        """Return the sampling profile bound to a functional role (CONCEPT:AU-ORCH.routing.sampling-profile-selection).

        Maps the role to its sampling task-class (:data:`_ROLE_TASK_CLASS`) and
        delegates to :meth:`pick_profile_for_task`, so role-routed model selection
        and profile selection share one task-class key.
        """
        return self.pick_profile_for_task(_ROLE_TASK_CLASS.get(str(role), "default"))

    def set_task_profile(self, profile: SamplingProfile) -> None:
        """Install/replace the profile for ``profile.task_class`` (CONCEPT:AU-AHE.harness.evolvable-sampling-profiles).

        The single write seam the evolution loop's ``promote_winner`` and the
        ``ontology_sampling_profile`` ``set`` action use to publish a profile.
        """
        self.task_class_profiles[profile.task_class] = profile

    # ── CONCEPT:AU-ORCH.routing.confidence-gated-routing-log tier helpers ────────────────────────────────────────────

    @staticmethod
    def _tier_down(tier: ModelTier) -> ModelTier:
        """Return one tier below the given tier, clamped at 'light'.

        CONCEPT:AU-ORCH.routing.confidence-gated-routing-log — Confidence-Gated Router
        """
        idx = _TIER_ORDER.index(tier)
        return _TIER_ORDER[max(0, idx - 1)]

    @staticmethod
    def _tier_up(tier: ModelTier) -> ModelTier:
        """Return one tier above the given tier, clamped at 'reasoning'.

        CONCEPT:AU-ORCH.routing.confidence-gated-routing-log — Confidence-Gated Router
        """
        idx = _TIER_ORDER.index(tier)
        return _TIER_ORDER[min(len(_TIER_ORDER) - 1, idx + 1)]

    def pick_for_task_adaptive(
        self,
        *,
        complexity: ModelTier = "medium",
        confidence_signal: float = 0.5,
        routing_percentile: float = 50.0,
        required_tags: list[str] | None = None,
    ) -> ModelDefinition:
        """Confidence-gated adaptive model selection (CONCEPT:AU-ORCH.routing.confidence-gated-routing-log).

        Extends :meth:`pick_for_task` with a runtime confidence signal
        derived from specialist consensus or WorkspaceAttention scores.
        When confidence exceeds the routing threshold, the task is
        downgraded to a cheaper model tier.  When confidence is low,
        the task may be escalated to a more capable tier.

        This implements the core routing principle from Squeeze Evolve
        (Maheswaran et al., 2026): allocate model capability where it
        has the highest marginal utility.

        The routing threshold is ``routing_percentile / 100``.  Values
        above trigger a downgrade; values below ``1 - threshold``
        trigger an escalation.

        Args:
            complexity: Base tier of the task being spawned.
            confidence_signal: Normalised confidence ``[0, 1]`` from
                upstream scoring (e.g. WorkspaceAttention composite).
            routing_percentile: Threshold percentile ``[0, 100]``
                for the downgrade gate.  Default 50 gives a balanced
                split; lower values (e.g. 30) route more aggressively
                to cheap models.
            required_tags: Tags every candidate must carry (AND
                semantics).

        Returns:
            The selected ``ModelDefinition``.

        Raises:
            ValueError: If the registry is empty.
        """
        threshold = routing_percentile / 100.0
        effective_tier = complexity

        if confidence_signal > threshold:
            # High confidence → cheaper model is sufficient
            effective_tier = self._tier_down(complexity)
        elif confidence_signal < (1.0 - threshold):
            # Low confidence → escalate to more capable model
            effective_tier = self._tier_up(complexity)

        return self.pick_for_task(
            complexity=effective_tier, required_tags=required_tags
        )

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
            import yaml  # type: ignore

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


def inference_owl_ttl(registry: ModelRegistry | None = None) -> str:
    """Project the registry's models + sampling profiles to OWL turtle (CONCEPT:AU-KG.ontology.inference-profile-implementers).

    Emits each configured model as a ``kg:Model`` individual and each effective
    task-class profile (built-in defaults overlaid with the registry's learned
    overrides) as a ``kg:InferenceProfile`` individual carrying its knob values and
    a ``kg:tunedFor`` triple to its task-class — the OWL the reasoner extrapolates
    profiles across. Reuses the shared ``kg:`` namespace and the KG-2.96 edge names.
    """
    reg = registry or load_active_registry()
    kg = "http://knuckles.team/kg#"
    lines = [
        f"@prefix kg: <{kg}> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "",
    ]

    def _frag(text: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in text)

    effective: dict[str, SamplingProfile] = {**_DEFAULT_TASK_PROFILES}
    effective.update(reg.task_class_profiles)
    for task_class, profile in effective.items():
        pid = f"kg:profile_{_frag(task_class)}"
        lines.append(f"{pid} rdf:type kg:InferenceProfile ;")
        lines.append(f'    kg:taskClass "{task_class}" ;')
        for knob in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
        ):
            value = getattr(profile, knob)
            if value is not None:
                lines.append(f"    kg:{knob} {value} ;")
        lines.append(f"    kg:tunedFor kg:taskclass_{_frag(task_class)} .")
        lines.append("")

    for model in reg.models:
        mid = f"kg:model_{_frag(model.id)}"
        lines.append(f"{mid} rdf:type kg:Model ;")
        lines.append(f'    kg:modelId "{model.model_id}" ;')
        lines.append(f'    kg:tier "{model.tier}" .')
        lines.append("")

    return "\n".join(lines)


# Process-global active registry. Cached so a profile learned/set at runtime
# (AHE-3.38 promotion, the ontology_sampling_profile 'set' action) persists across
# calls and is seen by the router/factory (ORCH-1.58) within the process.
_ACTIVE_REGISTRY: ModelRegistry | None = None


def load_active_registry() -> ModelRegistry:
    """Return the process-global active registry (lazy-loaded, cached).

    CONCEPT:AU-ORCH.routing.sampling-profile-selection. Mirrors ``model_factory._resolve_role_model``'s load
    (``config.model_registry_path``), but always returns a usable ``ModelRegistry``
    so the curated/learned sampling profiles are available even in the zero-infra
    ``tiny`` profile where no registry file exists. The same object is returned on
    every call, so ``set_task_profile``/``evolve_profile`` writes are visible to the
    router and factory until the process restarts. Never raises.
    """
    global _ACTIVE_REGISTRY
    if _ACTIVE_REGISTRY is None:
        try:
            from agent_utilities.core.config import config

            cfg_path = getattr(config, "model_registry_path", None)
            if cfg_path and Path(cfg_path).is_file():
                _ACTIVE_REGISTRY = ModelRegistry.load_from_file(cfg_path)
        except Exception:  # noqa: BLE001 - registry load is best-effort
            _ACTIVE_REGISTRY = None
        if _ACTIVE_REGISTRY is None:
            _ACTIVE_REGISTRY = ModelRegistry()
    return _ACTIVE_REGISTRY


def reset_active_registry() -> None:
    """Drop the cached active registry (test isolation / config reload)."""
    global _ACTIVE_REGISTRY
    _ACTIVE_REGISTRY = None
