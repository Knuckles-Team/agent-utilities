"""CONCEPT:AU-ORCH.adapter.three-tier-credential-resolution — four-tier credential resolution
(secret_ref > env > file > none).

Covers the NEW highest-precedence tier (CONCEPT:AU-ORCH.adapter.openai-catalog-verification): a
provider whose AgentConfig carries a ``<provider>_api_key_ref`` runtime secret reference
(``env://``/``vault://``/``secret://``) resolves through it BEFORE the literal env var
or config-file tiers, so an OpenBao-backed key never has to be a literal in the process
environment.
"""

from __future__ import annotations

import pytest

from agent_utilities.core.credentials import CredentialResolver

pytestmark = pytest.mark.concept(id="AU-ORCH.adapter.three-tier-credential-resolution")


def test_resolve_falls_back_to_env_tier_when_no_ref_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-literal-env-value")
    creds = CredentialResolver(env={"OPENAI_API_KEY": "sk-literal-env-value"}).resolve(
        "openai"
    )
    assert creds.source == "env"
    assert creds.api_key == "sk-literal-env-value"


def test_resolve_prefers_secret_ref_over_env(monkeypatch):
    import agent_utilities.core.config as cfg_module

    monkeypatch.setattr(
        cfg_module.config,
        "openai_api_key_ref",
        "env://OPENAI_REF_TARGET",
        raising=False,
    )
    monkeypatch.setenv("OPENAI_REF_TARGET", "sk-from-ref")
    resolver = CredentialResolver(env={"OPENAI_API_KEY": "sk-literal-env-value"})
    creds = resolver.resolve("openai")
    assert creds.source == "secret_ref"
    assert creds.api_key == "sk-from-ref"


def test_resolve_falls_through_to_env_when_ref_unresolvable(monkeypatch):
    import agent_utilities.core.config as cfg_module

    monkeypatch.setattr(
        cfg_module.config,
        "openai_api_key_ref",
        "env://OPENAI_REF_NEVER_SET",
        raising=False,
    )
    monkeypatch.delenv("OPENAI_REF_NEVER_SET", raising=False)
    resolver = CredentialResolver(env={"OPENAI_API_KEY": "sk-literal-env-value"})
    creds = resolver.resolve("openai")
    assert creds.source == "env"
    assert creds.api_key == "sk-literal-env-value"


def test_resolve_ignores_ref_tier_for_a_provider_with_no_ref_field(monkeypatch):
    resolver = CredentialResolver(env={"ANTHROPIC_API_KEY": "sk-ant-value"})
    creds = resolver.resolve("anthropic")
    assert creds.source == "env"
    assert creds.api_key == "sk-ant-value"


def test_resolve_none_when_nothing_configured():
    resolver = CredentialResolver(env={}, config_path="/nonexistent/media-config.json")
    creds = resolver.resolve("openai")
    assert creds.source == "none"
    assert creds.api_key is None
