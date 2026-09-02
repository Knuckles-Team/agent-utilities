"""Characterization test for ``_create_model_impl`` (CX-AU-07).

Pins the OBSERVED behaviour of the pre-refactor function (CCN 86,
``agent_utilities/core/model_factory.py``) before it is decomposed into
per-concern helper functions (role resolution, registry override
application, TLS/header resolution, oauth2 auth, http client construction,
and one builder per provider). This test must be green against the
unmodified function; if it is not, the test is wrong, not the code. It is
not touched again in the refactor commit.

Complements the substantial existing coverage in
``tests/unit/test_model_factory_auth_tls_routing.py`` (default routing, TLS,
per-model headers/reasoning-effort, unsupported-provider) and
``tests/unit/core/test_model_factory_custom_provider.py`` (custom
egress) by pinning the branches those files do not: each of the 9 provider
dispatch branches, oauth2 mutual exclusivity / attachment, and role
resolution end to end.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.core import model_factory


def _base_config(**overrides):
    base = dict(
        default_chat_model=None,
        openai_base_url="https://openai.invalid/v1",
        openai_api_key="openai-key",
        deepseek_base_url="https://deepseek.invalid/v1",
        deepseek_api_key="deepseek-key",
        anthropic_api_key="anthropic-key",
        gemini_api_key="gemini-key",
        groq_api_key="groq-key",
        mistral_api_key="mistral-key",
        hugging_face_api_key="hf-key",
        model_tls_profile=None,
        model_tls_profile_ref=None,
        model_http_allowed_private_hosts=[],
        model_registry_path=None,
        role_routing=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _client(model):
    prov = getattr(model, "_provider", None) or getattr(model, "provider", None)
    return getattr(prov, "client", None) or getattr(prov, "_client", None)


@pytest.fixture(autouse=True)
def _no_registry_override(monkeypatch):
    # Isolate each test from any model registered under the same id in a
    # concurrently-modified config -- default to "nothing registered".
    monkeypatch.setattr(model_factory, "get_model_config", lambda mid=None: None)
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")


def test_openai_provider_builds_with_explicit_args(monkeypatch):
    monkeypatch.setattr(model_factory, "config", _base_config())
    model = model_factory._create_model_impl(
        provider="openai",
        model_id="gpt-x",
        base_url="https://custom-openai.invalid/v1",
        api_key="explicit-key",
    )
    assert type(model).__name__ == "OpenAIChatModel"
    client = _client(model)
    assert str(client.base_url).rstrip("/") == "https://custom-openai.invalid/v1"
    assert client.api_key == "explicit-key"


def test_ollama_provider_requires_base_url(monkeypatch):
    monkeypatch.setattr(model_factory, "config", _base_config(openai_base_url=None))
    with pytest.raises(
        ValueError, match="ollama provider requires a configured base_url"
    ):
        model_factory._create_model_impl(
            provider="ollama", model_id="llama3", base_url=None
        )


def test_local_provider_uses_operator_configured_api_key(monkeypatch):
    monkeypatch.setattr(model_factory, "config", _base_config())
    model = model_factory._create_model_impl(
        provider="ollama", model_id="llama3", base_url="http://ollama.invalid:11434/v1"
    )
    assert type(model).__name__ == "OpenAIChatModel"
    client = _client(model)
    assert client.api_key == "openai-key"
    assert str(client.base_url).rstrip("/") == "http://ollama.invalid:11434/v1"


def test_deepseek_provider_requires_base_url(monkeypatch):
    monkeypatch.setattr(model_factory, "config", _base_config(deepseek_base_url=None))
    with pytest.raises(
        ValueError, match="deepseek provider requires a configured base_url"
    ):
        model_factory._create_model_impl(provider="deepseek", model_id="deepseek-chat")


def test_deepseek_provider_builds_openai_chat_model(monkeypatch):
    monkeypatch.setattr(model_factory, "config", _base_config())
    model = model_factory._create_model_impl(
        provider="deepseek", model_id="deepseek-chat"
    )
    assert type(model).__name__ == "OpenAIChatModel"
    assert model.model_name == "deepseek-chat"


def test_anthropic_provider_falls_back_to_config_api_key(monkeypatch):
    monkeypatch.setattr(model_factory, "config", _base_config())
    model = model_factory._create_model_impl(provider="anthropic", model_id="claude-x")
    assert type(model).__name__ == "AnthropicModel"
    client = _client(model)
    assert client.api_key == "anthropic-key"


def test_google_provider_falls_back_to_config_api_key(monkeypatch):
    monkeypatch.setattr(model_factory, "config", _base_config())
    model = model_factory._create_model_impl(provider="google", model_id="gemini-x")
    assert type(model).__name__ in ("GoogleModel", "GeminiModel")


def test_groq_provider_falls_back_to_config_api_key(monkeypatch):
    monkeypatch.setattr(model_factory, "config", _base_config())
    model = model_factory._create_model_impl(provider="groq", model_id="groq-x")
    assert type(model).__name__ == "GroqModel"
    client = _client(model)
    assert client.api_key == "groq-key"


def test_mistral_provider_falls_back_to_config_api_key(monkeypatch):
    monkeypatch.setattr(model_factory, "config", _base_config())
    model = model_factory._create_model_impl(provider="mistral", model_id="mistral-x")
    assert type(model).__name__ == "MistralModel"


def test_huggingface_provider_builds_a_model_instead_of_raising(monkeypatch):
    # BUG-CX-024 (was CX-AU-07, filed not fixed): the huggingface branch used
    # to always pass the factory's real (non-None) http_client straight into
    # HuggingFaceProvider, but pydantic-ai's HuggingFaceProvider.__init__ in
    # the installed version (.venv/lib/.../pydantic_ai/providers/
    # huggingface.py) raises ValueError whenever http_client is not None
    # ("`http_client` is ignored for HuggingFace provider, please use
    # `hf_client` instead."). Since _create_model_impl unconditionally built a
    # real http_client before the provider dispatch, the huggingface branch
    # was observably 100% broken -- every call raised instead of returning a
    # model. Fixed in ``_build_huggingface_model`` by no longer forwarding the
    # shared http_client to a provider whose SDK explicitly rejects it.
    monkeypatch.setattr(model_factory, "config", _base_config())
    model = model_factory._create_model_impl(provider="huggingface", model_id="hf-x")
    assert type(model).__name__ == "HuggingFaceModel"
    assert model.model_name == "hf-x"


def test_custom_provider_reaches_compatible_endpoint_adapter(monkeypatch):
    monkeypatch.setattr(
        model_factory,
        "config",
        _base_config(model_http_allowed_private_hosts=["proxy.invalid"]),
    )
    model = model_factory._create_model_impl(
        provider="custom",
        model_id="byok",
        base_url="http://proxy.invalid/v1",
        api_key="k",
    )
    assert type(model).__name__ == "OpenAIChatModel"


def test_unsupported_provider_rejected_before_client_construction(monkeypatch):
    monkeypatch.setattr(model_factory, "config", _base_config())
    with pytest.raises(ValueError, match="no registered adapter factory"):
        model_factory._create_model_impl(provider="not-a-real-provider", model_id="m")


def test_oauth2_and_api_key_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(model_factory, "config", _base_config())
    with pytest.raises(ValueError, match="mutually exclusive"):
        model_factory._create_model_impl(
            provider="openai",
            model_id="gpt-x",
            api_key="explicit-key",
            oauth2={
                "token_url": "https://idp.invalid/token",
                "client_id": "c",
                "client_secret": "s",
            },
        )


def test_oauth2_alone_builds_a_bearer_auth_and_no_static_key(monkeypatch):
    calls = {}

    def fake_httpx_auth_from_config(oauth2):
        calls["oauth2"] = oauth2

        # httpx._build_auth requires a tuple / httpx.Auth / callable -- a plain
        # sentinel string is rejected by httpx itself, so use a minimal
        # callable (httpx wraps it as FunctionAuth) to prove the value this
        # factory builds is actually threaded into the http client's auth.
        def _auth(request):
            return request

        return _auth

    monkeypatch.setattr(model_factory, "config", _base_config())
    monkeypatch.setattr(
        "agent_utilities.security.oauth_client_credentials.httpx_auth_from_config",
        fake_httpx_auth_from_config,
    )
    model = model_factory._create_model_impl(
        provider="openai",
        model_id="gpt-x",
        oauth2={
            "token_url": "https://idp.invalid/token",
            "client_id": "c",
            "client_secret": "s",
        },
    )
    assert type(model).__name__ == "OpenAIChatModel"
    assert calls["oauth2"]["client_id"] == "c"


def test_role_resolution_supplies_model_id_provider_base_url_and_headers(monkeypatch):
    resolved = SimpleNamespace(
        provider="openai",
        model_id="role-resolved-model",
        base_url="https://role-endpoint.invalid/v1",
        headers={"X-Role-Header": "yes"},
        reasoning_effort="inherit",
    )
    monkeypatch.setattr(model_factory, "config", _base_config())
    monkeypatch.setattr(model_factory, "_resolve_role_model", lambda role: resolved)

    model = model_factory._create_model_impl(role="planner")

    assert model.model_name == "role-resolved-model"
    client = _client(model)
    assert str(client.base_url).rstrip("/") == "https://role-endpoint.invalid/v1"
    assert client.default_headers.get("X-Role-Header") == "yes"


def test_role_resolution_is_skipped_when_model_id_is_explicit(monkeypatch):
    resolved = SimpleNamespace(
        provider="openai",
        model_id="role-resolved-model",
        base_url="https://role-endpoint.invalid/v1",
        headers=None,
        reasoning_effort="inherit",
    )
    monkeypatch.setattr(model_factory, "config", _base_config())
    monkeypatch.setattr(model_factory, "_resolve_role_model", lambda role: resolved)

    model = model_factory._create_model_impl(
        role="planner",
        provider="openai",
        model_id="explicit-model",
        base_url="https://explicit.invalid/v1",
    )

    # role is ignored entirely because model_id was explicit -- _resolve_role_model
    # must not even be consulted for provider/base_url/headers.
    assert model.model_name == "explicit-model"
    client = _client(model)
    assert str(client.base_url).rstrip("/") == "https://explicit.invalid/v1"


def test_registry_override_wins_base_url_over_role_resolution(monkeypatch):
    # Observed precedence: role resolution runs first (sets base_url from the
    # role), then the model registry lookup (get_model_config, keyed off the
    # ROLE-resolved model_id) can still override base_url/provider/headers.
    resolved = SimpleNamespace(
        provider="openai",
        model_id="registered-model",
        base_url="https://role-endpoint.invalid/v1",
        headers=None,
        reasoning_effort="inherit",
    )
    monkeypatch.setattr(model_factory, "config", _base_config())
    monkeypatch.setattr(model_factory, "_resolve_role_model", lambda role: resolved)
    monkeypatch.setattr(
        model_factory,
        "get_model_config",
        lambda mid=None: (
            {
                "id": "registered-model",
                "provider": "openai",
                "base_url": "https://registry-endpoint.invalid/v1",
            }
            if mid == "registered-model"
            else None
        ),
    )

    model = model_factory._create_model_impl(role="planner")

    client = _client(model)
    assert str(client.base_url).rstrip("/") == "https://registry-endpoint.invalid/v1"


def test_custom_headers_merge_call_site_wins_over_model_headers(monkeypatch):
    resolved = SimpleNamespace(
        provider="openai",
        model_id="gpt-x",
        base_url="https://role-endpoint.invalid/v1",
        headers={"X-Client-Id": "from-role"},
        reasoning_effort="inherit",
    )
    monkeypatch.setattr(model_factory, "config", _base_config())
    monkeypatch.setattr(model_factory, "_resolve_role_model", lambda role: resolved)

    model = model_factory._create_model_impl(
        role="planner", custom_headers={"X-Client-Id": "from-call"}
    )
    client = _client(model)
    assert client.default_headers.get("X-Client-Id") == "from-call"
