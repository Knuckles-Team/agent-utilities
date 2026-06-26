#!/usr/bin/python
from __future__ import annotations

"""Model Factory Module.

CONCEPT:ORCH-1.2

This module provides a unified factory function to create and configure
different LLM providers (OpenAI, Anthropic, Google, Groq, Mistral, Ollama)
using pydantic-ai. It handles environment-based configuration, custom HTTP
clients, and SSL verification settings.
"""


import logging
import os
from typing import TYPE_CHECKING, Any

from agent_utilities.core.config import config, setting

if TYPE_CHECKING:
    pass


def _load_model_class(*candidates: tuple[str, str]) -> Any:
    """Resolve a pydantic-ai model class across versions (its import path moved between releases).

    Tries each ``(module, attribute)`` candidate in order via importlib and returns the first that
    resolves, else ``None``. Using importlib (rather than ``from ... import``) keeps the version
    fallbacks free of static ``attr-defined``/``no-redef`` noise while remaining behaviour-identical.
    """
    import importlib

    for module_path, attr in candidates:
        try:
            return getattr(importlib.import_module(module_path), attr)
        except (ImportError, AttributeError):
            continue
    return None


OpenAIChatModel = _load_model_class(
    ("pydantic_ai.models.openai", "OpenAIChatModel"),
    ("pydantic_ai.providers.openai", "OpenAIChatModel"),
)
GoogleModel = _load_model_class(
    ("pydantic_ai.models.google", "GoogleModel"),
    ("pydantic_ai.models.gemini", "GeminiModel"),
)
AnthropicModel = _load_model_class(
    ("pydantic_ai.models.anthropic", "AnthropicModel"),
    ("pydantic_ai.providers.anthropic", "AnthropicModel"),
)
GroqModel = _load_model_class(
    ("pydantic_ai.models.groq", "GroqModel"),
    ("pydantic_ai.providers.groq", "GroqModel"),
)
MistralModel = _load_model_class(
    ("pydantic_ai.models.mistral", "MistralModel"),
    ("pydantic_ai.providers.mistral", "MistralModel"),
)
HuggingFaceModel = _load_model_class(
    ("pydantic_ai.models.huggingface", "HuggingFaceModel"),
    ("pydantic_ai.providers.huggingface", "HuggingFaceModel"),
)


try:
    from openai import AsyncOpenAI
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:
    AsyncOpenAI = None  # type: ignore[misc, assignment]
    OpenAIProvider = None  # type: ignore[misc, assignment]

try:
    from anthropic import AsyncAnthropic
    from pydantic_ai.providers.anthropic import AnthropicProvider
except ImportError:
    AsyncAnthropic = None  # type: ignore[misc, assignment]
    AnthropicProvider = None  # type: ignore[misc, assignment]

try:
    from groq import AsyncGroq
    from pydantic_ai.providers.groq import GroqProvider
except ImportError:
    AsyncGroq = None  # type: ignore[misc, assignment]
    GroqProvider = None  # type: ignore[misc, assignment]


logger = logging.getLogger(__name__)


def get_model_config(model_id: str | None = None) -> dict | None:
    from agent_utilities.core.config import config

    _cfg = config
    _cfg.reload()

    for m in getattr(_cfg, "chat_models", []):
        if m.id == model_id:
            return m.model_dump()

    for m in getattr(_cfg, "embedding_models", []):
        if m.id == model_id:
            return m.model_dump()

    return None


def _resolve_role_model(role: str):
    """Resolve a functional role to a concrete model via the active registry.

    CONCEPT:ORCH-1.27. Loads the registry from ``config.model_registry_path`` (kept
    decoupled from the server layer on purpose) and calls ``pick_for_role``. Returns
    the selected ``ModelDefinition`` or ``None`` if no registry/match is available.
    Never raises — role resolution is best-effort and degrades to the caller's defaults.
    """
    try:
        from pathlib import Path

        from agent_utilities.models.model_registry import ModelRegistry

        cfg_path = getattr(config, "model_registry_path", None)
        if not cfg_path or not Path(cfg_path).is_file():
            return None
        registry = ModelRegistry.load_from_file(cfg_path)
        if not registry.models:
            return None
        # Merge AgentConfig.role_routing as a fallback when the registry file
        # carries no role override for this role (registry file wins on conflict).
        cfg_roles = getattr(config, "role_routing", None) or {}
        if role not in registry.role_routing and role in cfg_roles:
            from agent_utilities.models.model_registry import RoleSpec

            registry.role_routing[role] = RoleSpec.model_validate(cfg_roles[role])
        # CONCEPT:ORCH-1.79 — route adaptively from the learned per-role confidence
        # (cheaper/local when it keeps succeeding, escalate when it fails); fall back
        # to the static role pick when adaptive selection has nothing to say.
        from agent_utilities.core.model_router import pick_adaptive

        return pick_adaptive(registry, role) or registry.pick_for_role(role)
    except Exception as e:  # pragma: no cover - defensive
        logging.getLogger(__name__).debug("Role resolution failed for %r: %s", role, e)
        return None


def create_model(
    provider: str | None = None,
    model_id: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    custom_headers: dict | None = None,
    ssl_verify: bool = True,
    timeout: float = 300.0,
    role: str | None = None,
    reasoning_effort: str | None = "none",
):
    """Build a model and (when a KG trace sink is installed) wrap it so EVERY LLM call
    persists a GenerationNode with model/tokens/cost/latency — the always-on per-call
    observability chokepoint (CONCEPT:OS-5.68). The wrap is a no-op when no sink is wired
    (zero overhead, e.g. unit tests), so default behavior is unchanged.

    ``reasoning_effort`` controls thinking on a reasoning chat model (the standard
    ``qwen/qwen3.6-35b-a3b`` is one): it emits a long ``reasoning`` block and leaves
    ``content`` null until thinking finishes, so a utility call with a modest ``max_tokens``
    gets EMPTY content (``finish_reason=length``) — and a retry-on-empty path then blocks to
    the 300s router/verifier timeout. Default ``"none"`` turns thinking off so the model
    returns content directly (verified: only the top-level ``reasoning_effort`` request param
    works on this vLLM build; ``enable_thinking``/``chat_template_kwargs``/``/no_think`` do
    not). Pass an effort level (``"low"``/``"medium"``/``"high"``) or ``None`` per call to opt
    back into reasoning for genuinely hard tasks."""
    model = _create_model_impl(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        custom_headers=custom_headers,
        ssl_verify=ssl_verify,
        timeout=timeout,
        role=role,
        reasoning_effort=reasoning_effort,
    )
    try:
        from agent_utilities.harness.tracing import wrap_model_for_tracing

        return wrap_model_for_tracing(model)
    except Exception:  # pragma: no cover - never break model construction
        return model


def _openai_reasoning_settings(effort: str | None) -> Any | None:
    """Build OpenAI model settings that set ``reasoning_effort`` (or ``None`` for no override).

    Threads through ``extra_body`` (merged top-level into the request by the OpenAI client)
    rather than the typed ``openai_reasoning_effort`` field, because ``"none"`` is outside the
    OpenAI effort enum pydantic-ai validates but IS the value this vLLM build honors to
    suppress the reasoning block. Returns ``None`` when ``effort`` is ``None`` (caller keeps
    the model's own default) or when pydantic-ai's OpenAI settings type is unavailable."""
    if effort is None:
        return None
    try:
        from pydantic_ai.models.openai import OpenAIChatModelSettings
    except Exception:  # pragma: no cover - pydantic-ai shape changed
        return None
    return OpenAIChatModelSettings(extra_body={"reasoning_effort": effort})


def _create_model_impl(
    provider: str | None = None,
    model_id: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    custom_headers: dict | None = None,
    ssl_verify: bool = True,
    timeout: float = 300.0,
    role: str | None = None,
    reasoning_effort: str | None = "none",
):
    """Initialize a pydantic-ai Model instance.

    This factory handles the complexity of mapping standardized provider
    names to their respective pydantic-ai model classes and providers.
    It supports automatic environment variable resolution for API keys
    and base URLs.

    Args:
        provider: The model provider (openai, anthropic, google, groq,
                  mistral, huggingface, ollama).
        model_id: The specific model identifier (e.g., 'gpt-4o').
        base_url: Optional API endpoint override.
        api_key: Optional API key.
        custom_headers: Optional dictionary of HTTP headers for the LLM requests.
        ssl_verify: Whether to verify SSL certificates for requests.
        timeout: Request timeout in seconds.

    Returns:
        A configured pydantic_ai.models.Model instance.

    """
    if setting("AGENT_UTILITIES_TESTING") == "true":
        from pydantic_ai.models.test import TestModel

        return TestModel()

    # CONCEPT:ORCH-1.27 — resolve a functional role (planner/generator/learner/judge)
    # to a concrete model when an explicit model_id was not supplied. Explicit args win.
    if role is not None and model_id is None:
        _resolved = _resolve_role_model(role)
        if _resolved is not None:
            provider = provider or _resolved.provider
            model_id = _resolved.model_id
            if base_url is None:
                base_url = _resolved.base_url

    _model_id = model_id or "qwen/qwen3.6-35b-a3b"
    _provider = provider or "openai"
    # Default reasoning OFF (content-bearing, fast) for every OpenAI-compatible model built
    # here; opt back in per call with reasoning_effort=None / a level. See create_model docstring.
    _rsettings = _openai_reasoning_settings(reasoning_effort)

    # Check if this model is defined in models.json, and override settings if so
    model_info = get_model_config(_model_id)
    if model_info:
        if "provider" in model_info:
            _provider = model_info["provider"]
        # The model registry is the source of truth for WHERE a model is served:
        # a registered per-model base_url MUST win over a graph-level default that
        # the caller threads through for all roles. Otherwise a split deployment
        # (e.g. router 'qwen-lite' on vllm-lite.arpa vs KG model on vllm.arpa) collapses
        # onto one endpoint and the lite model 404s. Only honor a caller's base_url for
        # models the registry doesn't know about.
        if model_info.get("base_url"):
            base_url = model_info["base_url"]
        if "api_key" in model_info and api_key is None:
            api_key = model_info["api_key"]

    http_client = None
    if http_client is None:
        # httpx ships in the ``[mcp]`` extra, not base — import it lazily so this
        # core module (and everything that imports it, e.g. ``orchestration``)
        # stays importable in the lean serving/CI install where httpx is absent
        # (Dependency discipline). It is only needed once a model is actually built.
        import httpx

        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        timeout_obj = httpx.Timeout(timeout, connect=30.0)
        http_client = httpx.AsyncClient(
            verify=ssl_verify, timeout=timeout_obj, limits=limits
        )

    if _provider == "openai":
        target_base_url = base_url or config.openai_base_url
        target_api_key = api_key if api_key is not None else config.openai_api_key

        # Propagate to environment for downstream pydantic-ai inference
        if target_base_url:
            os.environ["OPENAI_BASE_URL"] = target_base_url
        if target_api_key:
            os.environ["OPENAI_API_KEY"] = target_api_key

        if AsyncOpenAI is not None and OpenAIProvider is not None:
            openai_client = AsyncOpenAI(
                api_key=target_api_key or "EMPTY",
                base_url=target_base_url,
                http_client=http_client,
                default_headers=custom_headers,
                timeout=timeout,
            )
            openai_provider = OpenAIProvider(openai_client=openai_client)
            return OpenAIChatModel(
                settings=_rsettings, model_name=_model_id, provider=openai_provider
            )

        return OpenAIChatModel(
            settings=_rsettings, model_name=_model_id, provider="openai"
        )

    elif _provider == "ollama":
        target_base_url = (
            base_url or config.openai_base_url or "http://localhost:11434/v1"
        )
        target_api_key = api_key or "ollama"

        if http_client and AsyncOpenAI is not None and OpenAIProvider is not None:
            openai_client = AsyncOpenAI(
                api_key=target_api_key,
                base_url=target_base_url,
                http_client=http_client,
                default_headers=custom_headers,
            )
            openai_provider = OpenAIProvider(openai_client=openai_client)
            return OpenAIChatModel(
                settings=_rsettings, model_name=_model_id, provider=openai_provider
            )

        os.environ["OPENAI_BASE_URL"] = target_base_url
        os.environ["OPENAI_API_KEY"] = target_api_key
        return OpenAIChatModel(
            settings=_rsettings, model_name=_model_id, provider="openai"
        )

    elif _provider == "deepseek":
        target_base_url = (
            base_url or config.deepseek_base_url or "https://api.deepseek.com"
        )
        target_api_key = api_key or config.deepseek_api_key

        try:
            from pydantic_ai.providers.deepseek import DeepSeekProvider

            if http_client and AsyncOpenAI is not None:
                openai_client = AsyncOpenAI(
                    api_key=target_api_key or "EMPTY",
                    base_url=target_base_url,
                    http_client=http_client,
                    default_headers=custom_headers,
                    timeout=timeout,
                )
                ds_provider = DeepSeekProvider(openai_client=openai_client)
                return OpenAIChatModel(
                    settings=_rsettings, model_name=_model_id, provider=ds_provider
                )
        except ImportError:
            pass

        # fallback to standard OpenAI driver
        if target_api_key:
            os.environ["OPENAI_API_KEY"] = target_api_key
        if target_base_url:
            os.environ["OPENAI_BASE_URL"] = target_base_url
        return OpenAIChatModel(
            settings=_rsettings, model_name=_model_id, provider="openai"
        )

    elif _provider == "anthropic":
        target_api_key = api_key or config.anthropic_api_key
        if target_api_key:
            os.environ["ANTHROPIC_API_KEY"] = target_api_key

        try:
            if (
                http_client
                and AsyncAnthropic is not None
                and AnthropicProvider is not None
            ):
                anthropic_client = AsyncAnthropic(
                    api_key=target_api_key,
                    http_client=http_client,
                )
                anthropic_provider = AnthropicProvider(
                    anthropic_client=anthropic_client
                )
                return AnthropicModel(model_name=_model_id, provider=anthropic_provider)
        except ImportError:
            pass

        return AnthropicModel(model_name=_model_id)

    elif _provider == "google":
        target_api_key = api_key or config.gemini_api_key
        if target_api_key:
            os.environ["GEMINI_API_KEY"] = target_api_key
        return GoogleModel(model_name=_model_id)

    elif _provider == "groq":
        target_api_key = api_key or config.groq_api_key
        if target_api_key:
            os.environ["GROQ_API_KEY"] = target_api_key

        if http_client and AsyncGroq is not None and GroqProvider is not None:
            groq_client = AsyncGroq(
                api_key=target_api_key,
                http_client=http_client,
            )
            groq_provider = GroqProvider(groq_client=groq_client)
            return GroqModel(model_name=_model_id, provider=groq_provider)

        return GroqModel(model_name=_model_id)

    elif _provider == "mistral":
        target_api_key = api_key or config.mistral_api_key
        if target_api_key:
            os.environ["MISTRAL_API_KEY"] = target_api_key

        return MistralModel(model_name=_model_id)

    elif _provider == "huggingface":
        target_api_key = api_key or config.hugging_face_api_key
        if target_api_key:
            os.environ["HUGGING_FACE_API_KEY"] = target_api_key
        return HuggingFaceModel(model_name=_model_id)

    elif _provider in ("custom", "proxy"):
        # CONCEPT:ORCH-1.34 — BYOK custom endpoint. The provider proxy emits OpenAI-compatible
        # canonical streams, so a custom endpoint is reached via an OpenAI-compatible client pointed at
        # the resolved base_url. Credentials resolve env > file > none; the base_url passes the
        # DNS-resolved SSRF egress guard before any client is constructed.
        from agent_utilities.core.credentials import CredentialResolver
        from agent_utilities.security.egress import validate_base_url_resolved

        creds = CredentialResolver().resolve("openai")
        target_base_url = base_url or creds.base_url or config.openai_base_url
        target_api_key = (
            api_key if api_key is not None else creds.api_key
        ) or config.openai_api_key
        if not target_base_url:
            raise ValueError(
                "custom/proxy provider requires a base_url (BYOK endpoint)"
            )
        decision = validate_base_url_resolved(target_base_url)
        if not decision.allowed:
            raise ValueError(
                f"custom/proxy base_url rejected by egress guard: {decision.reason}"
            )

        if AsyncOpenAI is not None and OpenAIProvider is not None:
            custom_client = AsyncOpenAI(
                api_key=target_api_key or "EMPTY",
                base_url=target_base_url,
                http_client=http_client,
                default_headers=custom_headers,
                timeout=timeout,
            )
            return OpenAIChatModel(
                settings=_rsettings,
                model_name=_model_id,
                provider=OpenAIProvider(openai_client=custom_client),
            )
        os.environ["OPENAI_BASE_URL"] = target_base_url
        if target_api_key:
            os.environ["OPENAI_API_KEY"] = target_api_key
        return OpenAIChatModel(
            settings=_rsettings, model_name=_model_id, provider="openai"
        )

    return OpenAIChatModel(settings=_rsettings, model_name=_model_id, provider="openai")
