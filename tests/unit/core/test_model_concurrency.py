"""Tests for the per-model parallel-call concurrency controller (CONCEPT:KG-2.143)."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from agent_utilities.core.config import (
    ChatModelConfig,
    EmbeddingModelConfig,
    _total_model_capacity,
)
from agent_utilities.core.model_concurrency import (
    map_concurrent,
    map_concurrent_sync,
    reset_controllers,
    resolve_capacity,
)


@pytest.fixture(autouse=True)
def _isolate_controllers():
    reset_controllers()
    yield
    reset_controllers()


# --- capacity math ----------------------------------------------------------


def test_total_capacity_is_product():
    assert _total_model_capacity(1, 1) == 1
    assert _total_model_capacity(4, 8) == 32
    assert _total_model_capacity(3, 1) == 3
    assert _total_model_capacity(1, 5) == 5


def test_total_capacity_never_below_one():
    # Zero / negative / falsy collapse to safe sequential, never zero.
    assert _total_model_capacity(0, 0) == 1
    assert _total_model_capacity(0, 8) == 8  # 0 instances -> treated as 1
    assert _total_model_capacity(-3, 2) == 1


def test_chat_model_config_total_capacity():
    m = ChatModelConfig(
        id="m", provider="openai", parallel_instances=2, max_parallel_calls=10
    )
    assert m.total_capacity == 20
    default = ChatModelConfig(id="m", provider="openai")
    assert default.total_capacity == 1  # safe default


def test_embedding_model_config_total_capacity():
    e = EmbeddingModelConfig(
        id="bge-m3", provider="openai", parallel_instances=3, max_parallel_calls=16
    )
    assert e.total_capacity == 48
    assert EmbeddingModelConfig(id="x", provider="openai").total_capacity == 1


def test_config_model_capacity_resolution(monkeypatch):
    from agent_utilities.core import config as config_mod

    cfg = config_mod.config
    monkeypatch.setattr(
        cfg,
        "chat_models",
        [
            ChatModelConfig(
                id="chat-a",
                provider="openai",
                intelligence_level="normal",
                parallel_instances=2,
                max_parallel_calls=4,
            )
        ],
    )
    monkeypatch.setattr(
        cfg,
        "embedding_models",
        [
            EmbeddingModelConfig(
                id="bge-m3",
                provider="openai",
                parallel_instances=5,
                max_parallel_calls=10,
            )
        ],
    )
    assert cfg.model_capacity("chat-a") == 8
    assert cfg.model_capacity("chat") == 8  # role -> default chat
    assert cfg.model_capacity("bge-m3") == 50
    assert cfg.model_capacity("embedding") == 50
    assert cfg.embedding_capacity() == 50
    assert cfg.model_capacity("does-not-exist") == 1  # safe fallback


# --- helper: a fn that records max concurrent in-flight ----------------------


class _Tracker:
    def __init__(self, hold: float = 0.02):
        self.lock = threading.Lock()
        self.current = 0
        self.max_seen = 0
        self.hold = hold

    def __call__(self, x):
        with self.lock:
            self.current += 1
            self.max_seen = max(self.max_seen, self.current)
        time.sleep(self.hold)
        with self.lock:
            self.current -= 1
        return x * 10


# --- sync fan-out -----------------------------------------------------------


def test_sync_capacity_one_is_sequential():
    t = _Tracker()
    items = list(range(20))
    out = map_concurrent_sync(items, t, model=None, capacity=1)
    assert out == [i * 10 for i in items]  # order preserved
    assert t.max_seen == 1  # never more than 1 in flight


def test_sync_capacity_k_runs_up_to_k_concurrently():
    k = 5
    t = _Tracker()
    items = list(range(50))
    out = map_concurrent_sync(items, t, model="embedding", capacity=k)
    assert out == [i * 10 for i in items]  # order preserved despite concurrency
    # With 50 items and capacity 5, the pool should saturate to exactly k.
    assert t.max_seen == k


def test_sync_empty_and_single():
    assert map_concurrent_sync([], lambda x: x, capacity=8) == []
    assert map_concurrent_sync([7], lambda x: x + 1, capacity=8) == [8]


# --- async fan-out ----------------------------------------------------------


class _AsyncTracker:
    def __init__(self, hold: float = 0.02):
        self.current = 0
        self.max_seen = 0
        self.hold = hold

    async def __call__(self, x):
        self.current += 1
        self.max_seen = max(self.max_seen, self.current)
        await asyncio.sleep(self.hold)
        self.current -= 1
        return x * 10


def test_async_capacity_one_is_sequential():
    t = _AsyncTracker()
    items = list(range(15))
    out = asyncio.run(map_concurrent(items, t, capacity=1))
    assert out == [i * 10 for i in items]
    assert t.max_seen == 1


def test_async_capacity_k_runs_up_to_k():
    k = 6
    t = _AsyncTracker()
    items = list(range(40))
    out = asyncio.run(map_concurrent(items, t, model="embedding", capacity=k))
    assert out == [i * 10 for i in items]  # order preserved
    assert t.max_seen == k


def test_async_accepts_sync_fn_via_thread():
    t = _Tracker()
    items = list(range(30))
    out = asyncio.run(map_concurrent(items, t, capacity=4))
    assert out == [i * 10 for i in items]
    assert t.max_seen == 4


def test_async_empty():
    assert asyncio.run(map_concurrent([], lambda x: x, capacity=8)) == []


# --- resolve_capacity safety ------------------------------------------------


def test_resolve_capacity_unknown_defaults_to_one():
    assert resolve_capacity("totally-unknown-model-xyz") == 1


def test_resolve_capacity_reflects_config(monkeypatch):
    from agent_utilities.core import config as config_mod

    monkeypatch.setattr(
        config_mod.config,
        "embedding_models",
        [
            EmbeddingModelConfig(
                id="bge-m3",
                provider="openai",
                parallel_instances=2,
                max_parallel_calls=7,
            )
        ],
    )
    assert resolve_capacity("embedding") == 14
