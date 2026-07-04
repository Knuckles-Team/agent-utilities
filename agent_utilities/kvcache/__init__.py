"""Remote KV-cache connector for the epistemic-graph engine.

CONCEPT:AU-KG.backend.kvcache-vllm-connector — Python LMCache/vLLM connector for the epistemic-graph
KV-cache. The engine ships an HTTP KV-cache endpoint (CONCEPT:EG-KG.backend.is-configured-so-co) that
parallel vLLM / LMCache instances point at as a shared, content-addressed remote
backend so an identical KV page produced by two workers is stored **once** and a
cold worker can fetch a page a warm worker already computed.

This package is the Python half of that contract: a ``StorageBackend`` /
``RemoteBackend``-shaped connector (:class:`EpistemicGraphKVBackend`) that drives
the engine's ``/kv/<hash>`` HTTP surface, plus its typed configuration
(:class:`KvCacheConfig`). See
``epistemic-graph/docs/architecture/kvcache-remote-backend.md`` for the wire
contract and ``agent_utilities/kvcache/remote_backend.py`` for how to register the
backend with vLLM / LMCache.
"""

from __future__ import annotations

from agent_utilities.kvcache.config import KvCacheConfig
from agent_utilities.kvcache.l2_native_connector import EpistemicGraphL2Connector
from agent_utilities.kvcache.policy import (
    KVCacheDecision,
    KVCacheLayeringPolicy,
    fold_kv_hint,
)
from agent_utilities.kvcache.remote_backend import (
    EpistemicGraphKVBackend,
    KvCacheStats,
)

__all__ = [
    "EpistemicGraphKVBackend",
    "EpistemicGraphL2Connector",
    "KVCacheDecision",
    "KVCacheLayeringPolicy",
    "KvCacheConfig",
    "KvCacheStats",
    "fold_kv_hint",
]
