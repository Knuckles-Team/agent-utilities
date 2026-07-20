"""Configured model-class selection and privacy-safe run evidence."""

from __future__ import annotations

import pytest

from agent_utilities.core.config import ChatModelConfig, config
from agent_utilities.observability.trace_ontology import trace_id
from agent_utilities.orchestration import agent_runner
from agent_utilities.security.persistence_privacy import persistence_reference


def _models() -> list[ChatModelConfig]:
    return [
        ChatModelConfig(
            id="synthetic-economy",
            provider="openai",
            intelligence_level="light",
            api_key_ref="env://SYNTHETIC_MODEL_KEY",
        ),
        ChatModelConfig(
            id="synthetic-standard",
            provider="openai",
            intelligence_level="normal",
        ),
    ]


def test_model_class_selects_exact_configured_tier(monkeypatch) -> None:
    monkeypatch.setattr(config, "chat_models", _models())

    economy = agent_runner._configured_model_for_class("economy")
    standard = agent_runner._configured_model_for_class("standard")
    economy_cfg = agent_runner._build_execution_config(
        None,
        "synthetic-skill",
        {"type": "skill", "capabilities": [], "tools": []},
        recent_mementos=[],
        model_class="economy",
    )

    assert economy.id == "synthetic-economy"
    assert standard.id == "synthetic-standard"
    assert economy_cfg["agent_model"] == economy.id
    assert economy_cfg["router_model"] == economy.id
    assert economy_cfg["selected_model_class"] == "economy"
    assert economy_cfg["api_key_ref"] == "env://SYNTHETIC_MODEL_KEY"
    assert "api_key" not in economy_cfg


def test_model_class_has_no_cross_tier_fallback(monkeypatch) -> None:
    monkeypatch.setattr(config, "chat_models", [_models()[1]])

    with pytest.raises(RuntimeError, match="economy"):
        agent_runner._configured_model_for_class("economy")
    with pytest.raises(ValueError, match="economy or standard"):
        agent_runner._configured_model_for_class("legacy-tier")


class _Backend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, query, params):
        self.calls.append((query, params))
        return []


class _Engine:
    def __init__(self) -> None:
        self.backend = _Backend()
        self.nodes: dict[str, dict] = {}

    def add_node(self, node_id, label, properties=None):
        self.nodes[node_id] = {"label": label, **(properties or {})}

    def link_nodes(self, *_args, **_kwargs):
        return None


def test_trace_records_only_opaque_model_identity_and_skill_digest() -> None:
    engine = _Engine()
    model_name = "private-provider-model"
    model_ref = persistence_reference(
        "model", model_name, namespace="orchestration-run"
    )
    digest = "a" * 64

    run_id = "run:" + "d" * 32
    agent_runner._record_execution_trace(
        engine,
        run_id,
        "synthetic-skill",
        "synthetic task",
        status="completed",
        skill_used="synthetic-skill",
        skill_id="resource:skill:synthetic-skill",
        skill_instruction_digest=digest,
        model_ref=model_ref,
        model_class="economy",
    )

    trace = engine.nodes[trace_id(run_id)]
    assert trace["model_ref"] == model_ref
    assert trace["model_class"] == "economy"
    assert trace["skill_instruction_digest"] == digest
    assert model_name not in str(trace)
    uses_skill = [call for call in engine.backend.calls if "USES_SKILL" in call[0]]
    assert uses_skill[0][1]["rid"] == "resource:skill:synthetic-skill"


def test_trace_export_evidence_binds_exact_model_and_skill_body() -> None:
    model_ref = "pref_model_" + "a" * 64
    digest = "b" * 64

    evidence = agent_runner._trace_evidence_for_run(
        model_ref=model_ref,
        model_class="economy",
        skill_used="synthetic-skill",
        skill_instruction_digest=digest,
    )

    assert evidence == {
        "model_ref": model_ref,
        "model_class": "economy",
        "skill_ref": persistence_reference(
            "skill", "synthetic-skill", namespace="execution-trace"
        ),
        "skill_body_ref": persistence_reference(
            "skill_body", digest, namespace="skill-validation"
        ),
    }
