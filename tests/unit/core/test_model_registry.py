"""Unit tests for the multi-model registry (models/model_registry.py).

Covers:

- `ModelCostRate` / `ModelDefinition` / `ModelRegistry` serialization and
  round-tripping.
- `get_default`, `get_by_id`, `list_by_tier` lookups.
- Tier-priority fallbacks in `pick_for_task`.
- `required_tags` filtering (AND semantics) and graceful fallback when no
  tagged candidate matches.
- `add` duplicate-id guard.
- `load_from_file` for both JSON and YAML sources.
- `to_api_payload` contract used by the HTTP boundary.
- Local / zero-cost rendering (no `$` thrash on 0/0 rates).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agent_utilities.models.model_registry import (
    ModelCostRate,
    ModelDefinition,
    ModelRegistry,
)


@pytest.fixture
def sample_registry() -> ModelRegistry:
    return ModelRegistry(
        models=[
            ModelDefinition(
                id="local-fast",
                name="Local LM Studio",
                provider="openai",
                model_id="llama-3.2-3b-instruct",
                base_url="http://localhost:1234/v1",
                tier="light",
                is_default=True,
            ),
            ModelDefinition(
                id="cloud-mini",
                name="GPT-4o Mini",
                provider="openai",
                model_id="gpt-4o-mini",
                api_key_env="OPENAI_API_KEY",
                tier="medium",
                cost=ModelCostRate(input=0.15, output=0.6),
                tags=["code", "tools"],
            ),
            ModelDefinition(
                id="cloud-opus",
                name="Claude 3 Opus",
                provider="anthropic",
                model_id="claude-3-opus-20240229",
                api_key_env="ANTHROPIC_API_KEY",
                tier="heavy",
                cost=ModelCostRate(input=15, output=75),
                tags=["reasoning", "tools"],
            ),
            ModelDefinition(
                id="cloud-reasoning",
                name="o1 Preview",
                provider="openai",
                model_id="o1-preview",
                api_key_env="OPENAI_API_KEY",
                tier="reasoning",
                cost=ModelCostRate(input=15, output=60),
                tags=["reasoning"],
            ),
        ]
    )


def test_cost_rate_defaults_to_zero():
    rate = ModelCostRate()
    assert rate.input == 0.0
    assert rate.output == 0.0


def test_cost_rate_rejects_negative():
    with pytest.raises(ValidationError):
        ModelCostRate(input=-1.0, output=0.0)


def test_model_definition_requires_core_fields():
    with pytest.raises(ValidationError):
        ModelDefinition()  # type: ignore[call-arg]


def test_model_definition_forbids_extra_fields():
    with pytest.raises(ValidationError):
        ModelDefinition(
            id="m",
            name="m",
            provider="openai",
            model_id="gpt",
            foo="bar",  # type: ignore[call-arg]
        )


def test_registry_roundtrip_json(sample_registry: ModelRegistry):
    dumped = sample_registry.model_dump()
    restored = ModelRegistry.model_validate(dumped)
    assert restored == sample_registry


def test_get_default_prefers_is_default_flag(sample_registry: ModelRegistry):
    default = sample_registry.get_default()
    assert default is not None
    assert default.id == "local-fast"


def test_get_default_falls_back_to_first_when_none_flagged():
    reg = ModelRegistry(
        models=[
            ModelDefinition(
                id="a", name="A", provider="openai", model_id="a-model"
            ),
            ModelDefinition(
                id="b", name="B", provider="openai", model_id="b-model"
            ),
        ]
    )
    _default = reg.get_default()
    assert _default is not None
    assert _default.id == "a"


def test_get_default_empty_registry_returns_none():
    reg = ModelRegistry()
    assert reg.get_default() is None


def test_get_by_id(sample_registry: ModelRegistry):
    match = sample_registry.get_by_id("cloud-opus")
    assert match is not None
    assert match.tier == "heavy"
    assert sample_registry.get_by_id("missing") is None


def test_list_by_tier(sample_registry: ModelRegistry):
    heavy = sample_registry.list_by_tier("heavy")
    assert len(heavy) == 1
    assert heavy[0].id == "cloud-opus"

    light = sample_registry.list_by_tier("light")
    assert len(light) == 1
    assert light[0].id == "local-fast"

    assert sample_registry.list_by_tier("medium")[0].id == "cloud-mini"
    assert sample_registry.list_by_tier("reasoning")[0].id == "cloud-reasoning"


def test_pick_for_task_exact_tier(sample_registry: ModelRegistry):
    assert sample_registry.pick_for_task(complexity="light").id == "local-fast"
    assert sample_registry.pick_for_task(complexity="medium").id == "cloud-mini"
    assert sample_registry.pick_for_task(complexity="heavy").id == "cloud-opus"
    assert (
        sample_registry.pick_for_task(complexity="reasoning").id
        == "cloud-reasoning"
    )


def test_pick_for_task_heavy_prefers_reasoning_over_medium():
    reg = ModelRegistry(
        models=[
            ModelDefinition(
                id="mini",
                name="Mini",
                provider="openai",
                model_id="mini",
                tier="medium",
            ),
            ModelDefinition(
                id="o1",
                name="o1",
                provider="openai",
                model_id="o1",
                tier="reasoning",
            ),
        ]
    )
    assert reg.pick_for_task(complexity="heavy").id == "o1"


def test_pick_for_task_required_tags_filter(sample_registry: ModelRegistry):
    # Only two models carry 'tools'; for light tier they should fall back
    # through the tier ladder and select cloud-mini (medium/tools).
    match = sample_registry.pick_for_task(
        complexity="light", required_tags=["tools"]
    )
    assert match.id == "cloud-mini"


def test_pick_for_task_tag_miss_falls_back_to_tierwise_default(
    sample_registry: ModelRegistry,
):
    # 'vision' is not on any model. Spec says: after tag-filtering wipes
    # out every candidate, retry without tags rather than hard-fail.
    match = sample_registry.pick_for_task(
        complexity="heavy", required_tags=["vision"]
    )
    assert match.id == "cloud-opus"


def test_pick_for_task_empty_registry_raises():
    reg = ModelRegistry()
    with pytest.raises(ValueError, match="empty"):
        reg.pick_for_task(complexity="medium")


def test_add_appends_and_rejects_duplicates():
    reg = ModelRegistry()
    m1 = ModelDefinition(
        id="only", name="Only", provider="openai", model_id="foo"
    )
    reg.add(m1)
    assert reg.get_by_id("only") is m1
    with pytest.raises(ValueError, match="Duplicate model id"):
        reg.add(
            ModelDefinition(
                id="only", name="Clash", provider="openai", model_id="bar"
            )
        )


def test_to_api_payload_shape(sample_registry: ModelRegistry):
    payload = sample_registry.to_api_payload()
    assert set(payload) == {"models", "default_id"}
    assert payload["default_id"] == "local-fast"
    ids = [m["id"] for m in payload["models"]]
    assert ids == ["local-fast", "cloud-mini", "cloud-opus", "cloud-reasoning"]
    # Cost round-trips as a nested dict
    assert payload["models"][1]["cost"] == {"input": 0.15, "output": 0.6}


def test_to_api_payload_empty_registry_null_default():
    reg = ModelRegistry()
    payload = reg.to_api_payload()
    assert payload == {"models": [], "default_id": None}


def test_load_from_file_json(tmp_path: Path, sample_registry: ModelRegistry):
    p = tmp_path / "models.json"
    p.write_text(json.dumps(sample_registry.model_dump()))
    loaded = ModelRegistry.load_from_file(p)
    assert loaded == sample_registry


def test_load_from_file_yaml(tmp_path: Path, sample_registry: ModelRegistry):
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump(sample_registry.model_dump()))
    loaded = ModelRegistry.load_from_file(p)
    assert loaded == sample_registry
    # Sanity: default_id resolves through after YAML round-trip too.
    _loaded_default = loaded.get_default()
    assert _loaded_default is not None
    assert _loaded_default.id == "local-fast"


def test_load_from_file_accepts_string_path(
    tmp_path: Path, sample_registry: ModelRegistry
):
    p = tmp_path / "models.json"
    p.write_text(json.dumps(sample_registry.model_dump()))
    loaded = ModelRegistry.load_from_file(str(p))
    assert loaded == sample_registry


def test_local_zero_cost_model_serialises_as_zero(sample_registry: ModelRegistry):
    local = sample_registry.get_by_id("local-fast")
    assert local is not None
    payload = local.model_dump()
    assert payload["cost"] == {"input": 0.0, "output": 0.0}
    assert payload["api_key_env"] is None


def test_pick_for_task_returns_default_when_all_filters_fail():
    # Registry with no tags; pick_for_task should still succeed via the
    # tag-free retry path.
    reg = ModelRegistry(
        models=[
            ModelDefinition(
                id="only",
                name="Only",
                provider="openai",
                model_id="foo",
                tier="medium",
                is_default=True,
            )
        ]
    )
    match = reg.pick_for_task(
        complexity="reasoning", required_tags=["nothing"]
    )
    assert match.id == "only"
