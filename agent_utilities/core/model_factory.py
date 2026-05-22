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

import httpx

from agent_utilities.core.config import config

if TYPE_CHECKING:
    pass


try:
    from pydantic_ai.models.openai import OpenAIChatModel
except ImportError:
    try:
        from pydantic_ai.providers.openai import (
            OpenAIChatModel,  # type: ignore[no-redef]
        )
    except ImportError:
        OpenAIChatModel: Any = None  # type: ignore[no-redef]

try:
    from pydantic_ai.models.google import GoogleModel
except ImportError:
    try:
        from pydantic_ai.models.gemini import (
            GeminiModel as GoogleModel,  # type: ignore[no-redef]
        )
    except ImportError:
        GoogleModel: Any = None  # type: ignore[no-redef]

try:
    from pydantic_ai.models.anthropic import AnthropicModel
except ImportError:
    try:
        from pydantic_ai.providers.anthropic import (
            AnthropicModel,  # type: ignore[no-redef]
        )
    except ImportError:
        AnthropicModel: Any = None  # type: ignore[no-redef]

try:
    from pydantic_ai.models.groq import GroqModel
except ImportError:
    try:
        from pydantic_ai.providers.groq import GroqModel  # type: ignore[no-redef]
    except ImportError:
        GroqModel: Any = None  # type: ignore[no-redef]

try:
    from pydantic_ai.models.mistral import MistralModel
except ImportError:
    try:
        from pydantic_ai.providers.mistral import MistralModel  # type: ignore[no-redef]
    except ImportError:
        MistralModel: Any = None  # type: ignore[no-redef]

try:
    from pydantic_ai.models.huggingface import HuggingFaceModel
except ImportError:
    try:
        from pydantic_ai.providers.huggingface import (
            HuggingFaceModel,  # type: ignore[no-redef]
        )
    except ImportError:
        HuggingFaceModel: Any = None  # type: ignore[no-redef]


try:
    from openai import AsyncOpenAI
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:
    AsyncOpenAI = None
    OpenAIProvider = None

try:
    from anthropic import AsyncAnthropic
    from pydantic_ai.providers.anthropic import AnthropicProvider
except ImportError:
    AsyncAnthropic = None
    AnthropicProvider = None

try:
    from groq import AsyncGroq
    from pydantic_ai.providers.groq import GroqProvider
except ImportError:
    AsyncGroq = None
    GroqProvider = None


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


def create_model(
    provider: str | None = None,
    model_id: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    custom_headers: dict | None = None,
    ssl_verify: bool = True,
    timeout: float = 300.0,
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
    if os.environ.get("AGENT_UTILITIES_TESTING") == "true":
        from pydantic_ai.models.test import TestModel

        return TestModel()

    _model_id = model_id or "qwen/qwen3.5-9b"
    _provider = provider or "openai"

    # Check if this model is defined in models.json, and override settings if so
    model_info = get_model_config(_model_id)
    if model_info:
        if "provider" in model_info:
            _provider = model_info["provider"]
        if "base_url" in model_info and base_url is None:
            base_url = model_info["base_url"]
        if "api_key" in model_info and api_key is None:
            api_key = model_info["api_key"]

    http_client = None
    if http_client is None:
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

        if AsyncOpenAI and OpenAIProvider:
            client = AsyncOpenAI(
                api_key=target_api_key or "EMPTY",
                base_url=target_base_url,
                http_client=http_client,
                default_headers=custom_headers,
                timeout=timeout,
            )
            provider_instance = OpenAIProvider(openai_client=client)
            return OpenAIChatModel(model_name=_model_id, provider=provider_instance)

        return OpenAIChatModel(model_name=_model_id, provider="openai")

    elif _provider == "ollama":
        target_base_url = (
            base_url or config.openai_base_url or "http://localhost:11434/v1"
        )
        target_api_key = api_key or "ollama"

        if http_client and AsyncOpenAI and OpenAIProvider:
            client = AsyncOpenAI(
                api_key=target_api_key,
                base_url=target_base_url,
                http_client=http_client,
                default_headers=custom_headers,
            )
            provider_instance = OpenAIProvider(openai_client=client)
            return OpenAIChatModel(model_name=_model_id, provider=provider_instance)

        os.environ["OPENAI_BASE_URL"] = target_base_url
        os.environ["OPENAI_API_KEY"] = target_api_key
        return OpenAIChatModel(model_name=_model_id, provider="openai")

    elif _provider == "deepseek":
        target_base_url = (
            base_url or config.deepseek_base_url or "https://api.deepseek.com"
        )
        target_api_key = api_key or config.deepseek_api_key

        try:
            from pydantic_ai.providers.deepseek import DeepSeekProvider

            if http_client and AsyncOpenAI:
                client = AsyncOpenAI(
                    api_key=target_api_key or "EMPTY",
                    base_url=target_base_url,
                    http_client=http_client,
                    default_headers=custom_headers,
                    timeout=timeout,
                )
                provider_instance = DeepSeekProvider(openai_client=client)
                return OpenAIChatModel(model_name=_model_id, provider=provider_instance)
        except ImportError:
            pass

        # fallback to standard OpenAI driver
        if target_api_key:
            os.environ["OPENAI_API_KEY"] = target_api_key
        if target_base_url:
            os.environ["OPENAI_BASE_URL"] = target_base_url
        return OpenAIChatModel(model_name=_model_id, provider="openai")

    elif _provider == "anthropic":
        target_api_key = api_key or config.anthropic_api_key
        if target_api_key:
            os.environ["ANTHROPIC_API_KEY"] = target_api_key

        try:
            if http_client and AsyncAnthropic and AnthropicProvider:
                client = AsyncAnthropic(
                    api_key=target_api_key,
                    http_client=http_client,
                )
                provider_instance = AnthropicProvider(anthropic_client=client)
                return AnthropicModel(model_name=_model_id, provider=provider_instance)
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

        if http_client and AsyncGroq and GroqProvider:
            client = AsyncGroq(
                api_key=target_api_key,
                http_client=http_client,
            )
            provider_instance = GroqProvider(groq_client=client)
            return GroqModel(model_name=_model_id, provider=provider_instance)

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

    return OpenAIChatModel(model_name=_model_id, provider="openai")
