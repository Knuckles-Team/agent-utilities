from __future__ import annotations

"""Model profiles as graph resources (CONCEPT:AU-KG.ontology.model-profile-graph-resource).

Model definitions (:class:`~agent_utilities.models.model_registry.ModelDefinition`)
already live in code/config as the routing layer's source of truth. This module
projects a definition (plus any real observed statistics a caller supplies) into a
:class:`~agent_utilities.models.knowledge_graph.ModelProfileVersionNode` — a
content-addressed, graph-queryable resource, consistent with how this repo already
models skills/prompts/specs as :class:`~agent_utilities.models.knowledge_graph.ArtifactVersionNode`
subclasses (:class:`~agent_utilities.models.knowledge_graph.SpecVersionNode` is the
most recent precedent).

**Honesty contract**: :func:`build_model_profile` populates a field ONLY when a real
source exists (the ``ModelDefinition`` itself, or an ``observed`` statistic the
caller computed from real telemetry — e.g. a ``TraceDistiller``/``GenerationNode``
aggregate). Every field with no source is left ``None``/empty and recorded in
``unsourced_fields`` with a one-line reason — never a fabricated or
plausible-looking default. See ``ModelProfileVersionNode``'s docstring for the
full field list.
"""

import hashlib
from typing import Any

from agent_utilities.models.knowledge_graph import ModelProfileVersionNode
from agent_utilities.models.model_registry import ModelDefinition

# Reasons recorded for fields this repo has no data source for today. Kept as a
# module constant (not re-derived per call) so every profile's absence reasons are
# worded identically and can be grepped/audited in one place.
_NO_SOURCE_REASONS: dict[str, str] = {
    "prompt_cache_supported": "no prompt/KV-cache hit telemetry pipeline exists yet",
    "cache_hit_rate": "no prompt/KV-cache hit telemetry pipeline exists yet",
    "quality_by_domain": (
        "no per-skill/domain/topology eval aggregation is wired to the profile "
        "builder yet; pass observed_quality_by_domain once one exists"
    ),
    "prefill_latency_ms": (
        "GenerationNode records one scalar latency_ms, not a prefill/decode split "
        "or a distribution; pass observed_prefill_latency_ms once an aggregator exists"
    ),
    "decode_latency_ms": (
        "GenerationNode records one scalar latency_ms, not a prefill/decode split "
        "or a distribution; pass observed_decode_latency_ms once an aggregator exists"
    ),
    "availability_ratio": (
        "no trace-history aggregation over GenerationNode.error is wired yet; pass "
        "observed_availability_ratio once one exists"
    ),
    "error_rate": (
        "no trace-history aggregation over GenerationNode.error is wired yet; pass "
        "observed_error_rate once one exists"
    ),
    "throttle_rate": (
        "no throttle/429 telemetry is captured on GenerationNode today; pass "
        "observed_throttle_rate once one exists"
    ),
    "privacy_eligible": "no per-model privacy/residency declaration exists in config yet",
    "data_residency": "no per-model privacy/residency declaration exists in config yet",
    "cached_input_cost_per_million": "ModelCostRate carries only input/output; no cached-input cost field exists",
    "tool_cost_per_million": "ModelCostRate carries only input/output; no tool-cost field exists",
    "infra_cost_per_hour": "ModelCostRate carries only input/output; no infra-cost field exists",
    "quantization": "ModelDefinition carries no local-serving/hardware metadata yet",
    "serving_engine": "ModelDefinition carries no local-serving/hardware metadata yet",
    "accelerator": "ModelDefinition carries no local-serving/hardware metadata yet",
    "memory_gb": "ModelDefinition carries no local-serving/hardware metadata yet",
}

# Tags that, when present on a ModelDefinition, are a real (operator-configured)
# signal for a capability — absence means "unknown", never "false".
_MODALITY_TAGS = ("vision", "audio")
_TOOL_CALL_TAGS = ("tools", "tool_calls")
_STRUCTURED_OUTPUT_TAGS = ("json", "structured_output", "structured")
_REASONING_TAGS = ("reasoning",)


def profile_version_hash(definition: ModelDefinition) -> str:
    """Content hash over the identity+config fields that define one profile version.

    A new hash (and therefore a new persisted version) is produced whenever the
    provider, model id, tier, cost, or context/output limits change — the same
    content-addressed-lineage contract every other ``ArtifactVersionNode`` uses.
    """
    material = "|".join(
        [
            definition.provider,
            definition.model_id,
            definition.tier,
            f"{definition.cost.input:.6f}",
            f"{definition.cost.output:.6f}",
            str(definition.context_window),
            str(definition.max_output_tokens),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def model_profile_id(definition: ModelDefinition) -> str:
    """Content-addressed graph id: ``model_profile:<definition.id>:<version_hash>``."""
    return f"model_profile:{definition.id}:{profile_version_hash(definition)}"


def build_model_profile(
    definition: ModelDefinition,
    *,
    observed_quality_by_domain: dict[str, float] | None = None,
    observed_prefill_latency_ms: dict[str, float] | None = None,
    observed_decode_latency_ms: dict[str, float] | None = None,
    observed_availability_ratio: float | None = None,
    observed_error_rate: float | None = None,
    observed_throttle_rate: float | None = None,
    privacy_eligible: bool | None = None,
    data_residency: str | None = None,
) -> ModelProfileVersionNode:
    """Project a ``ModelDefinition`` into a ``ModelProfileVersionNode``.

    Fields sourced from ``definition`` are always populated (including a
    configured ``0.0`` cost, which this repo's ``ModelCostRate`` treats as a real
    "free/local" value, not an absence — see its docstring). Capability tags
    (``vision``/``tools``/``json``/``reasoning``) populate a field only when
    PRESENT — their absence means "unknown", so it is recorded in
    ``unsourced_fields`` rather than defaulted to ``False``. Every ``observed_*``
    kwarg is an explicit injection point for a real statistic a caller computed
    from telemetry (e.g. a trace-aggregation job); omitting it leaves the field
    unsourced with a standard reason (see ``_NO_SOURCE_REASONS``).
    """
    tags = set(definition.tags)
    unsourced: dict[str, str] = {}

    modalities = ["text"] + [t for t in _MODALITY_TAGS if t in tags]

    supports_tool_calls = True if tags & set(_TOOL_CALL_TAGS) else None
    if supports_tool_calls is None:
        unsourced["supports_tool_calls"] = (
            "no 'tools'/'tool_calls' tag configured; absence of the tag does not "
            "prove the model lacks tool-calling"
        )

    supports_structured_output = True if tags & set(_STRUCTURED_OUTPUT_TAGS) else None
    if supports_structured_output is None:
        unsourced["supports_structured_output"] = (
            "no 'json'/'structured_output' tag configured; absence of the tag "
            "does not prove the model lacks structured output"
        )

    reasoning_modes = ["reasoning"] if tags & set(_REASONING_TAGS) else []
    if not reasoning_modes:
        unsourced["reasoning_modes"] = (
            "no 'reasoning' tag configured; absence of the tag does not prove "
            "the model lacks a reasoning mode"
        )

    supported_effort_levels: list[str] = []
    if definition.reasoning_effort not in (None, "inherit"):
        supported_effort_levels = [definition.reasoning_effort]
    else:
        unsourced["supported_effort_levels"] = (
            "ModelDefinition.reasoning_effort is unset/'inherit'; only a pinned "
            "per-model override is a real signal, and none is configured"
        )

    for field_name, reason in _NO_SOURCE_REASONS.items():
        unsourced[field_name] = reason

    quality_by_domain = dict(observed_quality_by_domain or {})
    if quality_by_domain:
        unsourced.pop("quality_by_domain", None)
    prefill_latency_ms = dict(observed_prefill_latency_ms or {})
    if prefill_latency_ms:
        unsourced.pop("prefill_latency_ms", None)
    decode_latency_ms = dict(observed_decode_latency_ms or {})
    if decode_latency_ms:
        unsourced.pop("decode_latency_ms", None)
    if observed_availability_ratio is not None:
        unsourced.pop("availability_ratio", None)
    if observed_error_rate is not None:
        unsourced.pop("error_rate", None)
    if observed_throttle_rate is not None:
        unsourced.pop("throttle_rate", None)
    if privacy_eligible is not None:
        unsourced.pop("privacy_eligible", None)
    if data_residency is not None:
        unsourced.pop("data_residency", None)

    node_id = model_profile_id(definition)
    return ModelProfileVersionNode(
        id=node_id,
        name=definition.name or definition.id,
        artifact_kind="model_profile",
        artifact_id=definition.id,
        status="active",
        origin="model_registry",
        provider=definition.provider,
        model_id=definition.model_id,
        tier=definition.tier,
        modalities=modalities,
        context_window=definition.context_window,
        max_output_tokens=definition.max_output_tokens,
        supports_structured_output=supports_structured_output,
        supports_tool_calls=supports_tool_calls,
        reasoning_modes=reasoning_modes,
        supported_effort_levels=supported_effort_levels,
        quality_by_domain=quality_by_domain,
        prefill_latency_ms=prefill_latency_ms,
        decode_latency_ms=decode_latency_ms,
        availability_ratio=observed_availability_ratio,
        error_rate=observed_error_rate,
        throttle_rate=observed_throttle_rate,
        privacy_eligible=privacy_eligible,
        data_residency=data_residency,
        input_cost_per_million=definition.cost.input,
        output_cost_per_million=definition.cost.output,
        quantization=None,
        serving_engine=None,
        accelerator=None,
        memory_gb=None,
        unsourced_fields=unsourced,
    )


def persist_model_profile(engine: Any, node: ModelProfileVersionNode) -> None:
    """Upsert one ``ModelProfileVersionNode`` (mirrors ``skill_evolution._persist_skill_version``'s
    ``engine.add_node`` convention). Best-effort; never raises — profile persistence must not
    block model construction or routing."""
    if engine is None:
        return
    try:
        props = node.model_dump()
        props.pop("id", None)
        props["type"] = str(props.get("type", ""))
        engine.add_node(node.id, "model_profile", properties=props)
    except Exception:  # noqa: BLE001 - persistence is best-effort
        import logging

        logging.getLogger(__name__).debug(
            "persist_model_profile failed for %s", node.id, exc_info=True
        )


def sync_model_profiles(engine: Any, registry: Any) -> list[str]:
    """Upsert a ``ModelProfileVersionNode`` for every model in ``registry``.

    Bounded by construction — one profile per *configured* model, never per
    request. Returns the list of persisted node ids (empty list if ``engine`` is
    ``None`` or the registry has no models).
    """
    ids: list[str] = []
    if engine is None:
        return ids
    for definition in getattr(registry, "models", None) or []:
        node = build_model_profile(definition)
        persist_model_profile(engine, node)
        ids.append(node.id)
    return ids
