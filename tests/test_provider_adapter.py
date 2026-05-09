"""Tests for CONCEPT:ECO-4.5 — Provider Prompt Adaptation."""

import pytest
from agent_utilities.prompting.provider_adapter import (
    KGRuleBackend,
    ProviderPromptAdapter,
    ProviderRule,
    StaticRuleBackend,
)


@pytest.fixture
def mock_engine():
    import networkx as nx

    class _E:
        def __init__(self):
            self.graph = nx.MultiDiGraph()
            self.backend = None

    return _E()


@pytest.fixture
def static_adapter():
    return ProviderPromptAdapter(backend=StaticRuleBackend())


@pytest.fixture
def kg_adapter(mock_engine):
    return ProviderPromptAdapter(backend=KGRuleBackend(mock_engine))


class TestStaticBackend:
    def test_builtin_rules_exist(self):
        b = StaticRuleBackend()
        assert "openai" in b.list_providers()
        assert "anthropic" in b.list_providers()
        assert "google" in b.list_providers()

    def test_add_and_remove_rule(self):
        b = StaticRuleBackend()
        b.add_rule(
            ProviderRule(rule_id="t1", provider="openai", name="T1", priority=100)
        )
        assert any(r.rule_id == "t1" for r in b.get_rules("openai"))
        assert b.remove_rule("t1")
        assert not any(r.rule_id == "t1" for r in b.get_rules("openai"))

    def test_unknown_provider_empty(self):
        assert StaticRuleBackend().get_rules("unknown") == []


class TestKGBackend:
    def test_add_retrieve_remove(self, mock_engine):
        b = KGRuleBackend(mock_engine)
        b.add_rule(
            ProviderRule(rule_id="kg1", provider="openai", name="KG1", suffix="\nEnd.")
        )
        rules = b.get_rules("openai")
        assert len(rules) == 1 and rules[0].suffix == "\nEnd."
        assert b.remove_rule("kg1")
        assert len(b.get_rules("openai")) == 0


class TestPromptAdaptation:
    def test_openai_prefix(self, static_adapter):
        assert "specialized AI assistant" in static_adapter.adapt(
            "Write code.", "openai"
        )

    def test_anthropic_replacements(self, static_adapter):
        assert "Your task is to act as a" in static_adapter.adapt(
            "You are a code expert.", "anthropic"
        )

    def test_unknown_provider_passthrough(self, static_adapter):
        assert static_adapter.adapt("Hello.", "unknown") == "Hello."

    def test_conditional_rule_applies(self):
        b = StaticRuleBackend()
        b.add_rule(
            ProviderRule(
                rule_id="c1",
                provider="openai",
                name="C1",
                priority=95,
                suffix="\nCode only.",
                applicable_when={"task_type": "code"},
            )
        )
        a = ProviderPromptAdapter(backend=b)
        assert "Code only." in a.adapt("Test", "openai", context={"task_type": "code"})
        assert "Code only." not in a.adapt(
            "Test", "openai", context={"task_type": "chat"}
        )

    def test_register_unregister(self, static_adapter):
        static_adapter.register_rule(
            ProviderRule(rule_id="r1", provider="mistral", name="R1")
        )
        assert "mistral" in static_adapter.get_supported_providers()
        static_adapter.unregister_rule("r1")
        assert "mistral" not in static_adapter.get_supported_providers()


class TestAbstractedBackend:
    def test_default_is_static(self):
        assert isinstance(ProviderPromptAdapter().backend, StaticRuleBackend)

    def test_kg_backend_usable(self, mock_engine):
        a = ProviderPromptAdapter(backend=KGRuleBackend(mock_engine))
        assert isinstance(a.backend, KGRuleBackend)
