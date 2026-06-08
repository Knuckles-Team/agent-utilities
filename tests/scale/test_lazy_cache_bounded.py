#!/usr/bin/python
"""Tests that the lazy config cache is a bounded LRU and cannot grow unbounded."""

import pytest

from agent_utilities.core.config import (
    _LAZY_CACHE,
    LAZY_CACHE_MAX_SIZE,
    BoundedLRUCache,
)


def test_module_cache_is_bounded_lru():
    assert isinstance(_LAZY_CACHE, BoundedLRUCache)
    assert _LAZY_CACHE.max_size == LAZY_CACHE_MAX_SIZE == 4096


def test_never_exceeds_cap():
    cache = BoundedLRUCache(max_size=8)
    for i in range(1000):
        cache[f"k{i}"] = i
        assert len(cache) <= 8
    assert len(cache) == 8


def test_evicts_least_recently_used():
    cache = BoundedLRUCache(max_size=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    # Touch "a" so it becomes most-recently-used; "b" is now LRU.
    assert cache["a"] == 1
    cache["d"] = 4  # should evict "b"
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache
    assert "d" in cache
    assert len(cache) == 3


def test_get_updates_recency_and_returns_default():
    cache = BoundedLRUCache(max_size=2)
    cache["x"] = 1
    cache["y"] = 2
    # get("x") promotes x; inserting z evicts y (LRU).
    assert cache.get("x") == 1
    cache["z"] = 3
    assert "y" not in cache
    assert cache.get("missing") is None
    assert cache.get("missing", "fallback") == "fallback"


def test_set_existing_key_updates_value_and_recency_without_growth():
    cache = BoundedLRUCache(max_size=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"] = 10  # overwrite, promotes a, no growth
    assert len(cache) == 2
    assert cache["a"] == 10
    cache["c"] = 3  # evicts b (LRU), not a
    assert "b" not in cache
    assert cache["a"] == 10
    assert cache["c"] == 3


def test_contains_does_not_mutate_recency():
    cache = BoundedLRUCache(max_size=2)
    cache["a"] = 1
    cache["b"] = 2
    assert "a" in cache  # membership must not promote
    cache["c"] = 3  # a is still LRU -> evicted
    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache


def test_invalid_max_size_rejected():
    with pytest.raises(ValueError):
        BoundedLRUCache(max_size=0)


def test_dict_like_interface_supports_existing_call_sites():
    # The real module uses: cache[k] = v, k in cache, cache[k], cache.get(k)
    cache = BoundedLRUCache(max_size=4096)
    cache["_config"] = object()
    assert "_config" in cache
    cache["DEFAULT_LLM_PROVIDER"] = "openai"
    # read-back pattern used inside _init_lazy_config
    assert cache["DEFAULT_LLM_PROVIDER"] == "openai"
    assert list(cache.keys()) == ["_config", "DEFAULT_LLM_PROVIDER"]
