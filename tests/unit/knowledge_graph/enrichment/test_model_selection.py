"""KG-aware model selection for agent synthesis (CONCEPT:KG-2.10)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors.config_models import (
    extract as extract_models,
)
from agent_utilities.knowledge_graph.enrichment.orchestration import (
    AgentSpec,
    agent_to_batch,
    select_model,
)
from agent_utilities.knowledge_graph.enrichment.registry import get_source, write_batch

_MODELS = [
    {
        "id": "qwen/qwen3.5-9b",
        "provider": "openai",
        "intelligence_level": "normal",
        "can_route": False,
        "can_kg": True,
        "context_window": 65536,
    },
    {
        "id": "qwen-lite",
        "provider": "openai",
        "intelligence_level": "light",
        "can_route": True,
        "can_kg": False,
        "context_window": 32768,
    },
]


from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


def test_config_models_extractor_to_kg():
    batch = extract_models(
        {
            "data": {
                "chat_models": _MODELS,
                "embedding_models": [
                    {"id": "bge-m3", "provider": "openai", "context_window": 8192}
                ],
            }
        }
    )
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["model:qwen-lite"].props["can_route"] is True
    assert by_id["model:qwen-qwen3-5-9b"].props["can_kg"] is True
    assert by_id["model:bge-m3"].props["kind"] == "embedding"
    assert get_source("config_models") is not None


def test_select_model_light_vs_heavy():
    assert select_model(_MODELS, "light") == "qwen-lite"  # can_route
    assert select_model(_MODELS, "normal") == "qwen/qwen3.5-9b"  # can_kg
    assert select_model([], "light") == ""  # no models → fallback


def test_agent_spec_emits_uses_model_edge():
    a = AgentSpec(name="Router Bot", model="qwen-lite")
    backend = FakeBackend()
    write_batch(backend, agent_to_batch(a))
    assert ("agent:router-bot", "model:qwen-lite", "USES_MODEL") in backend.edges
