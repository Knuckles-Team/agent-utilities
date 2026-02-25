#!/usr/bin/python
# coding: utf-8

import os
import httpx
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding
from .base_utilities import to_boolean

# from llama_index.core.embeddings import BaseEmbedding  # Optional
# from llama_index.embeddings.openai import OpenAIEmbedding  # Optional

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
    from pydantic_ai.models.anthropic import AnthropicModel
    from anthropic import AsyncAnthropic
    from pydantic_ai.providers.anthropic import AnthropicProvider
except ImportError:
    AnthropicModel = None
    AsyncAnthropic = None
    AnthropicProvider = None

__version__ = "0.1.11"


def create_embedding_model(
    provider: Optional[str] = os.environ.get("EMBEDDING_PROVIDER", "openai").lower(),
    model: Optional[str] = os.environ.get(
        "EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v2-moe"
    ),
    base_url: Optional[str] = os.environ.get(
        "LLM_BASE_URL", "http://localhost:1234/v1"
    ),
    api_key: Optional[str] = os.environ.get("LLM_API_KEY", None),
    ssl_verify: bool = to_boolean(string=os.environ.get("SSL_VERIFY", "true")),
    timeout: float = 300.0,
) -> "BaseEmbedding":
    """
    Get the embedding model based on parameters or environment variables.
    """
    from llama_index.embeddings.openai import OpenAIEmbedding

    http_client = None
    if not ssl_verify:
        http_client = httpx.AsyncClient(verify=False, timeout=timeout)

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

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
