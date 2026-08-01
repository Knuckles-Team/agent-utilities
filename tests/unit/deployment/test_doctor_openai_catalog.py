"""``openai_catalog`` doctor check (CONCEPT:AU-ORCH.adapter.openai-catalog-verification).

Static (skip/fail/ok) and live-probed paths, with the live network call itself
mocked — the check must never report the resolved API key value.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_utilities.deployment import doctor as D
from agent_utilities.models.model_registry import ModelDefinition, ModelRegistry


@dataclass
class _Creds:
    provider: str
    api_key: str | None
    base_url: str | None = None
    source: str = "none"


def test_openai_catalog_skips_when_no_openai_models_configured(monkeypatch):
    import agent_utilities.models.model_registry as mr

    monkeypatch.setattr(mr, "_ACTIVE_REGISTRY", ModelRegistry(), raising=False)
    res = D._check_openai_catalog()
    assert res["status"] == "skip"
    assert res["data"]["configured_model_count"] == 0


def test_openai_catalog_fails_when_no_credential_available(monkeypatch):
    import agent_utilities.core.credentials as creds_module
    import agent_utilities.models.model_registry as mr

    registry = ModelRegistry(
        models=[
            ModelDefinition(
                id="cloud-mini",
                name="GPT-4o Mini",
                provider="openai",
                model_id="gpt-4o-mini",
                tier="medium",
            )
        ]
    )
    monkeypatch.setattr(mr, "_ACTIVE_REGISTRY", registry, raising=False)
    monkeypatch.setattr(
        creds_module.CredentialResolver,
        "resolve",
        lambda self, provider: _Creds(provider, api_key=None, source="none"),
    )
    res = D._check_openai_catalog()
    assert res["status"] == "fail"
    assert "OPENAI_API_KEY_REF" in res["remediation"]
    # Never leaks a credential value (there is none here, but assert the shape too).
    assert "api_key" not in res["data"]


def test_openai_catalog_static_ok_without_network_call(monkeypatch):
    import agent_utilities.core.credentials as creds_module
    import agent_utilities.models.model_registry as mr

    registry = ModelRegistry(
        models=[
            ModelDefinition(
                id="cloud-mini",
                name="GPT-4o Mini",
                provider="openai",
                model_id="gpt-4o-mini",
                tier="medium",
            )
        ]
    )
    monkeypatch.setattr(mr, "_ACTIVE_REGISTRY", registry, raising=False)
    monkeypatch.setattr(
        creds_module.CredentialResolver,
        "resolve",
        lambda self, provider: _Creds(
            provider, api_key="sk-should-never-appear-in-report", source="secret_ref"
        ),
    )
    res = D._check_openai_catalog(live=False)
    assert res["status"] == "ok"
    assert res["data"]["live_probed"] is False
    assert "sk-should-never-appear-in-report" not in str(res)


def test_openai_catalog_live_reports_unverified_models_without_leaking_key(
    monkeypatch,
):
    import agent_utilities.core.credentials as creds_module
    import agent_utilities.core.openai_catalog as catalog_module
    import agent_utilities.models.model_registry as mr
    from agent_utilities.core.openai_catalog import OpenAIModelVerification

    registry = ModelRegistry(
        models=[
            ModelDefinition(
                id="cloud-mini",
                name="GPT-4o Mini",
                provider="openai",
                model_id="gpt-4o-mini",
                tier="medium",
            ),
            ModelDefinition(
                id="cloud-typo",
                name="Typo'd model",
                provider="openai",
                model_id="gpt-4o-mno",
                tier="medium",
            ),
        ]
    )
    monkeypatch.setattr(mr, "_ACTIVE_REGISTRY", registry, raising=False)
    secret = "sk-should-never-appear-in-report"
    monkeypatch.setattr(
        creds_module.CredentialResolver,
        "resolve",
        lambda self, provider: _Creds(provider, api_key=secret, source="secret_ref"),
    )

    async def _fake_verify(model_id, *, api_key, base_url=None):
        assert api_key == secret  # the check really did pass the resolved key through
        return OpenAIModelVerification(
            model_id=model_id, exists=(model_id == "gpt-4o-mini")
        )

    monkeypatch.setattr(catalog_module, "verify_openai_model", _fake_verify)

    res = D._check_openai_catalog(live=True)
    assert res["status"] == "fail"
    assert res["data"]["verified_count"] == 1
    assert res["data"]["unverified_model_ids"] == ["gpt-4o-mno"]
    assert secret not in str(res)


def test_openai_catalog_never_raises_with_nothing_configured():
    res = D._check_openai_catalog()
    assert res["status"] in ("ok", "warn", "fail", "skip", "error")
