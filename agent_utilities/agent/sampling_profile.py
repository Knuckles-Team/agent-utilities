from __future__ import annotations

"""Task-aware sampling profiles — per-call inference-parameter selection.

CONCEPT:ORCH-1.57 — Task-aware per-call LLM sampling that selects and threads a SamplingProfile of temperature top_p top_k min_p repetition_penalty max_tokens and penalties per question by task-class and role

The router already picks *which model* answers a question; this picks *how to
sample from it*. A :class:`SamplingProfile` bundles the per-call inference knobs
(``temperature``, ``top_p``, ``max_tokens``, the OpenAI penalties) plus the
vLLM-only knobs (``top_k`` / ``min_p`` / ``repetition_penalty``) that ride in the
OpenAI-compatible ``extra_body`` — the exact mechanism already proven in
``knowledge_graph/extraction/fact_extractor.py``.

A profile is **selected** per task-class (deterministic low-temp for code /
extraction, exploratory high-temp for brainstorming), **threaded** as a per-call
``model_settings=`` override built *from* the agent's static base settings (so an
unset knob keeps its default — pydantic-ai replaces, it does not deep-merge), and
later **evolved** (CONCEPT:AHE-3.38) and **projected to OWL** (CONCEPT:KG-2.93/2.94).

The static defaults in :func:`agent_utilities.agent.factory.create_agent` are not
removed — they are the *base* this profile merges over, and :data:`DEFAULT_PROFILE`
(all-``None`` knobs) reproduces them exactly, guaranteeing zero behaviour change
when no specific profile is resolved.
"""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic_ai import ModelSettings

# vLLM-only sampling knobs are not first-class OpenAI fields; they travel in the
# request ``extra_body``. Keep this list in lockstep with what the vLLM
# OpenAI-compatible server accepts (and with fact_extractor's proven payload).
EXTRA_BODY_KNOBS: tuple[str, ...] = ("top_k", "min_p", "repetition_penalty")

# The task-class vocabulary the registry binds profiles to. Deliberately small —
# each maps to one curated default profile; evolution refines them in place.
TaskClass = Literal[
    "code",
    "extraction",
    "reasoning",
    "plan",
    "judge",
    "generate",
    "brainstorm",
    "default",
]


class SamplingProfile(BaseModel):
    """A per-call inference-parameter bundle bound to a task-class.

    Every knob is optional; ``None`` means *inherit the agent's base setting*.
    :meth:`to_model_settings` therefore only overrides the knobs a profile
    actually sets, so a sparse profile is a targeted nudge, not a full reset.

    CONCEPT:ORCH-1.57.
    """

    task_class: str = Field(
        default="default",
        description="The task class this profile is tuned for (see TaskClass).",
    )
    source: Literal["static", "role", "learned"] = Field(
        default="static",
        description=(
            "Provenance: 'static' = curated default, 'role' = role-bound, "
            "'learned' = promoted by the AHE-3.38 evolution loop."
        ),
    )

    # Standard OpenAI / pydantic-ai ModelSettings knobs.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    seed: int | None = Field(default=None)

    # vLLM-only knobs carried via extra_body.
    top_k: int | None = Field(default=None, ge=1)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)

    model_config = ConfigDict(extra="forbid")

    def to_model_settings(self, base: ModelSettings | dict[str, Any]) -> ModelSettings:
        """Build a per-call ``ModelSettings`` override from this profile over ``base``.

        ``base`` is the agent's static settings. Standard knobs this profile sets
        replace the base value; the vLLM-only knobs are **dict-merged** into
        ``extra_body`` so pre-existing keys (e.g. RLM's ``chat_template_kwargs``)
        survive. pydantic-ai applies a per-call ``model_settings`` by *replacing*
        the agent-level settings for that run, so the override must carry the full
        base — which is exactly why we copy ``base`` first.
        """
        merged: dict[str, Any] = dict(base)

        for knob in (
            "temperature",
            "top_p",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "seed",
        ):
            value = getattr(self, knob)
            if value is not None:
                merged[knob] = value

        extra: dict[str, Any] = dict(merged.get("extra_body") or {})
        for knob in EXTRA_BODY_KNOBS:
            value = getattr(self, knob)
            if value is not None:
                extra[knob] = value
        if extra:
            merged["extra_body"] = extra

        from pydantic_ai import ModelSettings as _MS

        return _MS(**merged)  # type: ignore[typeddict-item]


# Reproduces today's static behaviour: no knob set → ``to_model_settings`` returns
# the base unchanged. The merge floor for the No-Legacy single-representation rule.
DEFAULT_PROFILE = SamplingProfile(task_class="default", source="static")


# Keyword cues per task-class, ordered by specificity. The first class whose cues
# the prompt hits wins; absent a hit we fall back to "default". This is the single
# canonical classifier reused by the factory run-wrapper resolver and the router.
_TASK_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "extraction",
        (
            "extract",
            "parse",
            "pull out",
            "entities",
            "fields",
            "structured output",
            "json schema",
            "transcribe",
        ),
    ),
    (
        "code",
        (
            "code",
            "function",
            "implement",
            "bug",
            "refactor",
            "compile",
            "stack trace",
            "unit test",
            "python",
            "typescript",
            "rust",
        ),
    ),
    (
        "judge",
        (
            "judge",
            "evaluate",
            "grade",
            "score",
            "verdict",
            "is this correct",
            "rate the",
            "assess",
        ),
    ),
    ("plan", ("plan", "roadmap", "break down", "steps to", "design a", "architecture")),
    (
        "reasoning",
        (
            "prove",
            "derive",
            "reason",
            "why does",
            "explain how",
            "analyze",
            "think through",
        ),
    ),
    (
        "brainstorm",
        (
            "brainstorm",
            "ideas",
            "name ideas",
            "creative",
            "imagine",
            "come up with",
            "alternatives",
            "what could",
        ),
    ),
    (
        "generate",
        (
            "write",
            "draft",
            "compose",
            "generate",
            "summarize",
            "story",
            "blog",
            "email",
        ),
    ),
)


def classify_task(text: str | None) -> str:
    """Map free-text prompt to a task-class via keyword cues (CONCEPT:ORCH-1.57).

    Best-effort and order-sensitive: extraction/code/judge win over the broader
    generate/brainstorm classes. Returns ``"default"`` when nothing matches.
    """
    if not text:
        return "default"
    low = text.lower()
    for task_class, cues in _TASK_CUES:
        if any(cue in low for cue in cues):
            return task_class
    return "default"


def _prompt_text(user_prompt: Any) -> str:
    """Coerce a pydantic-ai ``user_prompt`` (str or content sequence) to text."""
    if user_prompt is None:
        return ""
    if isinstance(user_prompt, str):
        return user_prompt
    if isinstance(user_prompt, list | tuple):
        return " ".join(p for p in user_prompt if isinstance(p, str))
    return ""


def resolve_sampling_profile(
    task_text: str | None = None,
    *,
    role: str | None = None,
    task_class: str | None = None,
) -> SamplingProfile:
    """Resolve the profile for a prompt/role/task-class. Never raises.

    Loads the active :class:`~agent_utilities.models.model_registry.ModelRegistry`
    and asks it for the curated/learned profile, classifying the prompt via
    :func:`classify_task` when no explicit ``task_class``/``role`` is given.
    Degrades to :data:`DEFAULT_PROFILE` on any failure (missing registry, import
    error) so the live run path is never broken by profile resolution.
    """
    try:
        from agent_utilities.models.model_registry import load_active_registry

        registry = load_active_registry()
        if role is not None:
            return registry.pick_profile_for_role(role)
        cls = task_class or classify_task(task_text)
        return registry.pick_profile_for_task(cls)
    except Exception:  # noqa: BLE001 - resolution must never break a run
        return DEFAULT_PROFILE


def attach_profile_resolver(agent: Any, base_settings: Any) -> None:
    """Wrap an agent's run methods to thread a per-call task-aware profile.

    CONCEPT:ORCH-1.57 — the live seam. Each ``run``/``run_sync``/``run_stream``
    call, *unless the caller passes an explicit* ``model_settings``, classifies the
    user prompt, resolves the matching :class:`SamplingProfile`, and injects
    ``model_settings=profile.to_model_settings(base_settings)``. Built from the
    static ``base_settings`` so unset knobs keep the agent's defaults (pydantic-ai
    replaces per-call settings rather than deep-merging). Idempotent and best-effort:
    a resolution failure leaves the call untouched (the agent-level settings apply).
    """
    import functools

    base = dict(base_settings)

    def _make_wrapper(orig: Any) -> Any:
        @functools.wraps(orig)
        def wrapper(user_prompt: Any = None, *args: Any, **kwargs: Any) -> Any:
            if kwargs.get("model_settings") is None:
                try:
                    profile = resolve_sampling_profile(_prompt_text(user_prompt))
                    kwargs["model_settings"] = profile.to_model_settings(base)
                except Exception:  # noqa: BLE001 - never break the run
                    pass
            return orig(user_prompt, *args, **kwargs)

        wrapper._au_profile_wrapped = True  # type: ignore[attr-defined]
        return wrapper

    for method_name in ("run", "run_sync", "run_stream"):
        orig = getattr(agent, method_name, None)
        if orig is None or getattr(orig, "_au_profile_wrapped", False):
            continue
        setattr(agent, method_name, _make_wrapper(orig))

    agent._au_base_settings = base  # type: ignore[attr-defined]
