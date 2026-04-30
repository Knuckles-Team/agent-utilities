#!/usr/bin/python
"""Embedding Utilities Module.

This module provides factory functions for initializing LlamaIndex-compatible
embedding models. It supports various providers including OpenAI, Ollama,
HuggingFace, and local models, with robust environment-based configuration.
"""

import os
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding


from agent_utilities.base_utilities import to_boolean

try:
    from llama_index.embeddings.ollama import OllamaEmbedding
except ImportError:
    OllamaEmbedding = None

try:
    from openai import AsyncOpenAI
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:
    AsyncOpenAI = None
    OpenAIProvider = None

try:
    from groq import AsyncGroq
    from pydantic_ai.providers.groq import GroqProvider
except ImportError:
    AsyncGroq = None
    GroqProvider = None

try:
    from mistralai import Mistral
    from pydantic_ai.providers.mistral import MistralProvider
except ImportError:
    Mistral = None
    MistralProvider = None

try:
    from anthropic import AsyncAnthropic
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
except ImportError:
    AnthropicModel = None
    AsyncAnthropic = None
    AnthropicProvider = None

__version__ = "0.2.40"


def create_embedding_model(
    provider: str | None = os.environ.get("EMBEDDING_PROVIDER", "openai").lower(),
    model: str | None = os.environ.get(
        "EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v2-moe"
    ),
    base_url: str | None = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1"),
    api_key: str | None = os.environ.get("LLM_API_KEY", None),
    ssl_verify: bool = to_boolean(string=os.environ.get("SSL_VERIFY", "true")),
    timeout: float = 300.0,
) -> "BaseEmbedding":
    """Initialize an embedding model based on provider and environment.

    Args:
        provider: Name of the embedding provider ('openai', 'ollama',
            'huggingface', 'local').
        model: Specific model identifier.
        base_url: Base URL for provider API requests.
        api_key: Optional API key for authentication.
        ssl_verify: Whether to verify SSL certificates.
        timeout: Request timeout in seconds.

    Returns:
        An initialized LlamaIndex BaseEmbedding instance.

    Raises:
        ImportError: If a requested provider's dependency is missing.
        ValueError: If an unsupported provider is specified.

    """
    from llama_index.embeddings.openai import OpenAIEmbedding

    http_client = None
    if not ssl_verify:
        http_client = httpx.AsyncClient(verify=False, timeout=timeout)  # nosec B501

    if provider == "mock" or os.environ.get("AGENT_UTILITIES_TESTING") == "true":
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.get_text_embedding.return_value = [1.0] + [0.0] * 1535
        return mock

    if provider == "openai":
        return OpenAIEmbedding(
            model_name=model,
            api_key=api_key,
            api_base=base_url,
            timeout=timeout,
            http_client=http_client,
        )

    elif provider == "huggingface":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        cache_folder = os.environ.get("HF_HOME")

        return HuggingFaceEmbedding(
            model_name=model,
            cache_folder=cache_folder,
            request_timeout=timeout,
        )

    elif provider == "ollama":
        if OllamaEmbedding is None:
            raise ImportError("llama-index-embeddings-ollama is not installed.")

        return OllamaEmbedding(
            model_name=model,
            base_url=base_url,
            timeout=timeout,
        )

    elif provider == "local":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=model)

    elif provider == "mock" or os.environ.get("AGENT_UTILITIES_TESTING") == "true":
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.get_text_embedding.return_value = [1.0] + [0.0] * 1535
        return mock

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
