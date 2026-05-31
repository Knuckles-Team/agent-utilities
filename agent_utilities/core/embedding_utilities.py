#!/usr/bin/python
"""Embedding Utilities Module.

CONCEPT:KG-2.3

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
from agent_utilities.core.config import config

try:
    from llama_index.embeddings.ollama import OllamaEmbedding
except ImportError:
    OllamaEmbedding = None


__version__ = "0.2.40"


def create_embedding_model(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
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

    # Resolve defaults from the model registry
    _embed_cfg = config.default_embedding_model
    _chat_cfg = config.default_chat_model
    provider_str = (
        provider
        or (_embed_cfg.provider if _embed_cfg else None)
        or (_chat_cfg.provider if _chat_cfg else None)
        or "openai"
    )
    provider_str = provider_str.lower()
    model_str = (
        model
        or (_embed_cfg.id if _embed_cfg else None)
        or "text-embedding-nomic-embed-text-v2-moe"
    )
    base_url_str = (
        base_url
        or (_embed_cfg.base_url if _embed_cfg else None)
        or (_chat_cfg.base_url if _chat_cfg else None)
        or "http://vllm-embed.arpa/v1"
    )
    api_key_str = (
        api_key
        or (_embed_cfg.api_key if _embed_cfg else None)
        or (_chat_cfg.api_key if _chat_cfg else None)
    )

    http_client: httpx.Client | None = None
    if not ssl_verify:
        http_client = httpx.Client(verify=False, timeout=timeout)  # nosec B501

    if provider_str == "mock" or os.environ.get("AGENT_UTILITIES_TESTING") == "true":
        from unittest.mock import MagicMock

        mock = MagicMock()
        dim = int(config.kg_embedding_dim or "768")
        mock.get_text_embedding.return_value = [1.0] + [0.0] * (dim - 1)
        return mock

    if provider_str == "openai":
        # Fallback for LM Studio / Local Testing
        if not api_key_str:
            api_key_str = config.openai_api_key or "Test-1234"

        import sys

        print(f"Creating OpenAIEmbedding with key={api_key_str}", file=sys.stderr)

        return OpenAIEmbedding(
            model_name=model_str,
            api_key=api_key_str,
            api_base=base_url_str,
            timeout=timeout,
            http_client=http_client,
        )

    elif provider_str == "huggingface":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        cache_folder = os.environ.get("HF_HOME")

        return HuggingFaceEmbedding(
            model_name=model_str,
            cache_folder=cache_folder,
            request_timeout=timeout,
        )

    elif provider_str == "ollama":
        if OllamaEmbedding is None:
            raise ImportError("llama-index-embeddings-ollama is not installed.")

        return OllamaEmbedding(
            model_name=model_str,
            base_url=base_url_str,
            timeout=timeout,
        )

    elif provider_str == "local":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=model_str)

    elif provider_str == "mock" or os.environ.get("AGENT_UTILITIES_TESTING") == "true":
        from unittest.mock import MagicMock

        mock = MagicMock()
        dim = int(config.kg_embedding_dim or "768")
        mock.get_text_embedding.return_value = [1.0] + [0.0] * (dim - 1)
        return mock

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
