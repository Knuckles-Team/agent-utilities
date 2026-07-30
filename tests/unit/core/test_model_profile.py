"""CONCEPT:AU-KG.ontology.model-profile-graph-resource — model profiles as graph resources.

Covers the honesty contract: every field ``build_model_profile`` cannot source stays
null/empty AND is recorded in ``unsourced_fields`` (never silently defaulted); fields
that ARE sourced (from the ``ModelDefinition`` or an ``observed_*`` kwarg) are absent
from ``unsourced_fields``. Also covers content-addressing and the persistence helper.
"""

from __future__ import annotations

import pytest

from agent_utilities.models.knowledge_graph import ModelProfileVersionNode
from agent_utilities.models.model_profile import (
    build_model_profile,
    model_profile_id,
    persist_model_profile,
    profile_version_hash,
    sync_model_profiles,
)
from agent_utilities.models.model_registry import (
    ModelCostRate,
    ModelDefinition,
    ModelRegistry,
)

pytestmark = pytest.mark.concept(id="AU-KG.ontology.model-profile-graph-resource")


@pytest.fixture
def cloud_definition() -> ModelDefinition:
    return ModelDefinition(
        id="cloud-mini",
        name="GPT-4o Mini",
        provider="openai",
        model_id="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        tier="medium",
        cost=ModelCostRate(input=0.15, output=0.6),
        context_window=128_000,
        max_output_tokens=16_384,
        tags=["code", "tools", "json"],
    )


def test_build_model_profile_sources_definition_fields(cloud_definition):
    node = build_model_profile(cloud_definition)
    assert isinstance(node, ModelProfileVersionNode)
    assert node.provider == "openai"
    assert node.model_id == "gpt-4o-mini"
    assert node.context_window == 128_000
    assert node.max_output_tokens == 16_384
    assert node.input_cost_per_million == 0.15
    assert node.output_cost_per_million == 0.6
    # Sourced from real config, so must NOT be recorded as unsourced.
    for f in (
        "provider",
        "model_id",
        "context_window",
        "max_output_tokens",
        "input_cost_per_million",
        "output_cost_per_million",
    ):
        assert f not in node.unsourced_fields


def test_build_model_profile_tags_are_a_real_positive_signal(cloud_definition):
    node = build_model_profile(cloud_definition)
    assert node.supports_tool_calls is True
    assert node.supports_structured_output is True
    assert "supports_tool_calls" not in node.unsourced_fields
    assert "supports_structured_output" not in node.unsourced_fields


def test_build_model_profile_absent_tag_is_unknown_not_false():
    definition = ModelDefinition(
        id="plain",
        name="Plain",
        provider="openai",
        model_id="plain-model",
        tier="light",
    )
    node = build_model_profile(definition)
    # No 'tools'/'json' tag configured -> unknown, never fabricated to False.
    assert node.supports_tool_calls is None
    assert node.supports_structured_output is None
    assert "supports_tool_calls" in node.unsourced_fields
    assert "supports_structured_output" in node.unsourced_fields


def test_build_model_profile_unsourced_fields_never_include_a_populated_field(
    cloud_definition,
):
    node = build_model_profile(cloud_definition)
    for field_name in node.unsourced_fields:
        value = getattr(node, field_name)
        assert value in (None, [], {}, "")


def test_build_model_profile_honors_observed_statistics(cloud_definition):
    node = build_model_profile(
        cloud_definition,
        observed_quality_by_domain={"code": 0.82},
        observed_availability_ratio=0.999,
    )
    assert node.quality_by_domain == {"code": 0.82}
    assert node.availability_ratio == 0.999
    assert "quality_by_domain" not in node.unsourced_fields
    assert "availability_ratio" not in node.unsourced_fields
    # Statistics NOT supplied stay unsourced.
    assert "error_rate" in node.unsourced_fields


def test_build_model_profile_local_serving_fields_always_unsourced_today(
    cloud_definition,
):
    node = build_model_profile(cloud_definition)
    for field_name in (
        "quantization",
        "serving_engine",
        "accelerator",
        "memory_gb",
        "prompt_cache_supported",
        "cache_hit_rate",
    ):
        assert getattr(node, field_name) is None
        assert field_name in node.unsourced_fields


def test_profile_version_hash_changes_with_cost(cloud_definition):
    h1 = profile_version_hash(cloud_definition)
    changed = cloud_definition.model_copy(
        update={"cost": ModelCostRate(input=0.30, output=1.2)}
    )
    h2 = profile_version_hash(changed)
    assert h1 != h2


def test_model_profile_id_is_content_addressed(cloud_definition):
    node1 = build_model_profile(cloud_definition)
    node2 = build_model_profile(cloud_definition)
    assert node1.id == node2.id == model_profile_id(cloud_definition)


class _FakeEngine:
    """Mirrors the REAL ``IntelligenceGraphEngine.add_node(node_id, node_type,
    properties=...)`` signature ``persist_model_profile`` calls (the same
    positional-node_type/properties-dict convention ``skill_evolution.py``'s
    ``_persist_skill_version`` uses) — a different duck-type than
    ``KGTraceBackend``'s ``backend`` facade (``add_node(id, **props)``)."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}

    def add_node(self, node_id: str, node_type: str = "", properties=None) -> None:
        self.nodes[node_id] = {"type": node_type, **(properties or {})}

    def get_node(self, node_id: str):
        return self.nodes.get(node_id)


def test_persist_model_profile_upserts_via_add_node(cloud_definition):
    engine = _FakeEngine()
    node = build_model_profile(cloud_definition)
    persist_model_profile(engine, node)
    assert node.id in engine.nodes
    assert engine.nodes[node.id]["provider"] == "openai"


def test_persist_model_profile_is_a_noop_without_an_engine(cloud_definition):
    node = build_model_profile(cloud_definition)
    persist_model_profile(None, node)  # must not raise


def test_sync_model_profiles_is_bounded_to_configured_models(cloud_definition):
    engine = _FakeEngine()
    registry = ModelRegistry(models=[cloud_definition])
    ids = sync_model_profiles(engine, registry)
    assert len(ids) == 1
    assert ids[0] in engine.nodes
