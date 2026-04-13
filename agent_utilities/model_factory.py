#!/usr/bin/python

from __future__ import annotations

import os
import logging
import httpx


from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


try:
    from pydantic_ai.models.openai import OpenAIChatModel
except ImportError:
    try:
        from pydantic_ai.providers.openai import OpenAIChatModel
    except ImportError:
        OpenAIChatModel = None

try:
    from pydantic_ai.models.gemini import GeminiModel as GoogleModel
except ImportError:
    try:
        from pydantic_ai.providers.gemini import GeminiModel as GoogleModel
    except ImportError:
        GoogleModel = None

try:
    from pydantic_ai.models.anthropic import AnthropicModel
except ImportError:
    try:
        from pydantic_ai.providers.anthropic import AnthropicModel
    except ImportError:
        AnthropicModel = None

try:
    from pydantic_ai.models.groq import GroqModel
except ImportError:
    try:
        from pydantic_ai.providers.groq import GroqModel
    except ImportError:
        GroqModel = None

try:
    from pydantic_ai.models.mistral import MistralModel
except ImportError:
    try:
        from pydantic_ai.providers.mistral import MistralModel
    except ImportError:
        MistralModel = None

try:
    from pydantic_ai.models.huggingface import HuggingFaceModel
except ImportError:
    try:
        from pydantic_ai.providers.huggingface import HuggingFaceModel
    except ImportError:
        HuggingFaceModel = None


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


def create_model(
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_headers: Optional[dict] = None,
    ssl_verify: bool = True,
    timeout: float = 300.0,
):
    """
    Create a Pydantic AI model with the specified provider and configuration.

    Args:
        provider: The model provider (openai, anthropic, google, groq, mistral, huggingface, ollama)
        model_id: The specific model ID to use
        base_url: Optional base URL for the API
        custom_headers: Optional dict of custom HTTP headers to send with every request to the LLM endpoint
        api_key: Optional API key
        ssl_verify: Whether to verify SSL certificates (default: True)

    Returns:
        A Pydantic AI Model instance
    """
    _provider = provider or os.environ.get("PROVIDER") or "openai"
    _model_id = model_id or os.environ.get("MODEL_ID") or "google/gemma-4-31b"

    http_client = None
    if http_client is None:
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        timeout_obj = httpx.Timeout(timeout, connect=30.0)
        http_client = httpx.AsyncClient(
            verify=ssl_verify, timeout=timeout_obj, limits=limits
        )

    if _provider == "openai":
        target_base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        target_api_key = (
            api_key
            if api_key is not None
            else (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        )

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
            base_url or os.environ.get("LLM_BASE_URL") or "http://localhost:11434/v1"
        )
        target_api_key = api_key or os.environ.get("LLM_API_KEY") or "ollama"

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

    elif _provider == "anthropic":
        target_api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
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
        target_api_key = (
            api_key or os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
        )
        if target_api_key:
            os.environ["GEMINI_API_KEY"] = target_api_key
        return GoogleModel(model_name=_model_id)

    elif _provider == "groq":
        target_api_key = (
            api_key or os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
        )
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
        target_api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("MISTRAL_API_KEY")
        )
        if target_api_key:
            os.environ["MISTRAL_API_KEY"] = target_api_key

        return MistralModel(model_name=_model_id)

    elif _provider == "huggingface":
        target_api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("HUGGING_FACE_API_KEY")
        )
        if target_api_key:
            os.environ["HUGGING_FACE_API_KEY"] = target_api_key
        return HuggingFaceModel(model_name=_model_id)

    return OpenAIChatModel(model_name=_model_id, provider="openai")
