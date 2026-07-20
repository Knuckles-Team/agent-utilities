"""Automatic embedder endpoint failover + capacity guard (CONCEPT:AU-KG.enrichment.each-call-resolves-active).

The embedding plane runs against a PRIMARY embedder in a dedicated synthetic
accelerator group and fails over to a FALLBACK in a shared synthetic accelerator
group when the primary's circuit breaker trips — routing back automatically on
recovery. These tests, with a fake config + a deterministic
``GPU_CONCURRENCY_BUDGETS`` (no live GPU/endpoint), lock the four guarantees:

(a) primary up → embeds resolve to the PRIMARY endpoint/key (dedicated budget);
(b) primary breaker OPEN → embeds auto-route to the FALLBACK under the shared
    JOINT budget — its ceiling governs (budget − the generator's reserved floor),
    NOT an unbounded fan-out that could OOM the shared box;
(c) primary recovers (cooldown elapsed) → embeds route BACK to the primary;
(d) the cached embedding client SWAPS endpoint on failover (no stale primary client).
"""

from __future__ import annotations

import pytest

from agent_utilities.core import config as config_mod
from agent_utilities.core import embedding_failover as ef
from agent_utilities.core import embedding_utilities as eu
from agent_utilities.core import model_concurrency as mc
from agent_utilities.core.config import ChatModelConfig, EmbeddingModelConfig
from agent_utilities.core.gpu_group_budget import group_allowed
from agent_utilities.core.model_circuit_breaker import get_circuit_breaker

# Captured before the unit suite's autouse hermetic-embeddings fixture stubs the
# factory, so test (d) can exercise the REAL create_embedding_model swap logic.
_REAL_CREATE_EMBEDDING_MODEL = eu.create_embedding_model

# Joint synthetic budget shared by the generator and fallback embedder, plus a
# dedicated synthetic budget for the primary embedder.
_SHARED_BUDGET = 20
_GENERATOR_FLOOR = 16
_PRIMARY_BUDGET = 16
_SHARED_GROUP = "accelerator-shared"
_PRIMARY_GROUP = "accelerator-primary"
_PRIMARY_ENDPOINT = "https://embedding-primary.example.test/v1"
_FALLBACK_ENDPOINT = "https://embedding-fallback.example.test/v1"


@pytest.fixture
def failover_config(monkeypatch):
    """Install dedicated primary and shared fallback synthetic endpoints."""
    monkeypatch.setenv("TEST_EMBEDDING_API_KEY", "synthetic-runtime-material")
    monkeypatch.setenv(
        "GPU_CONCURRENCY_BUDGETS",
        f'{{"{_SHARED_GROUP}": {_SHARED_BUDGET}, '
        f'"{_PRIMARY_GROUP}": {_PRIMARY_BUDGET}}}',
    )
    # The generator is a priority member of the shared group with a reserved floor.
    monkeypatch.setattr(
        config_mod.config,
        "chat_models",
        [
            ChatModelConfig(
                id="qwen3.6-27b",
                provider="openai",
                base_url="https://generation.example.test/v1",
                gpu_group=_SHARED_GROUP,
                parallel_instances=1,
                max_parallel_calls=_GENERATOR_FLOOR,
                max_concurrent_requests=_SHARED_BUDGET,
            )
        ],
    )
    # The embedder uses the dedicated group first, then the shared group.
    monkeypatch.setattr(
        config_mod.config,
        "embedding_models",
        [
            EmbeddingModelConfig(
                id="bge-m3",
                provider="openai",
                base_url=_PRIMARY_ENDPOINT,
                api_key_ref="env://TEST_EMBEDDING_API_KEY",
                gpu_group=_PRIMARY_GROUP,
                parallel_instances=4,
                max_concurrent_requests=16,
                fallback=EmbeddingModelConfig(
                    id="bge-m3",
                    provider="openai",
                    base_url=_FALLBACK_ENDPOINT,
                    api_key_ref="env://TEST_EMBEDDING_API_KEY",
                    gpu_group=_SHARED_GROUP,
                    parallel_instances=2,
                    max_concurrent_requests=8,
                ),
            )
        ],
    )
    mc.reset_controllers()
    yield
    mc.reset_controllers()


def _trip_primary() -> None:
    """Trip the PRIMARY embedder's breaker OPEN (a simulated 503 shed)."""
    get_circuit_breaker(ef.PRIMARY_KEY).record(ok=False, status=503)


# --- (a) primary up → dedicated primary endpoint and budget ----------------
def test_primary_up_routes_to_primary(failover_config):
    ep = ef.active_embedding_endpoint()
    assert ep.model_key == ef.PRIMARY_KEY
    assert ep.is_fallback is False
    assert ep.base_url == _PRIMARY_ENDPOINT
    assert config_mod.config.gpu_group(ef.PRIMARY_KEY) == _PRIMARY_GROUP


# --- (b) primary OPEN → fallback under the shared JOINT budget --------------
def test_failover_routes_to_fallback_under_shared_budget(failover_config):
    _trip_primary()
    assert get_circuit_breaker(ef.PRIMARY_KEY).is_tripped() is True

    ep = ef.active_embedding_endpoint()
    assert ep.model_key == ef.FALLBACK_KEY
    assert ep.is_fallback is True
    assert ep.base_url == _FALLBACK_ENDPOINT
    assert config_mod.config.gpu_group(ef.FALLBACK_KEY) == _SHARED_GROUP

    # Seed the capacity guard for the fallback (registers shared peers incl. the
    # generator's reserved floor) and assert the JOINT ceiling governs.
    resolved = mc.resolve_capacity(ef.FALLBACK_KEY)
    allowed = group_allowed(_SHARED_GROUP, ef.FALLBACK_KEY)
    assert allowed is not None, "the shared joint budget MUST govern the fallback"
    # The generator's floor is reserved off the top: the fallback can use at most
    # budget − generator_floor — NOT the whole box (the OOM-safety in fallback mode).
    assert allowed == _SHARED_BUDGET - _GENERATOR_FLOOR
    assert resolved <= allowed
    # Bounded, never unbounded: well under the fallback's own server ceiling (8) and
    # nowhere near MODEL_MAX_CONCURRENCY — the generator's slice is protected.
    assert resolved <= _SHARED_BUDGET - _GENERATOR_FLOOR


def test_failover_joint_cap_clamps_embed_fanout(failover_config):
    """The embed fan-out width is clamped to the shared fallback allowance."""
    from agent_utilities.knowledge_graph.enrichment import semantic as sem

    _trip_primary()
    ep = ef.active_embedding_endpoint()
    assert ep.model_key == ef.FALLBACK_KEY
    # A greedy request for 64-wide embedding is bounded to the shared allowance.
    capped = sem._joint_budget_cap(ef.FALLBACK_KEY, 64)
    assert capped <= _SHARED_BUDGET - _GENERATOR_FLOOR
    assert capped >= 1


# --- (c) primary recovers → routes back -------------------------------------
def test_recovery_routes_back_to_primary(failover_config, monkeypatch):
    _trip_primary()
    assert ef.active_embedding_endpoint().is_fallback is True

    # Simulate the breaker's backoff cooldown elapsing: once is_tripped() flips
    # False the router returns to the primary (the breaker's HALF_OPEN probe then
    # confirms recovery on the next real primary call).
    breaker = get_circuit_breaker(ef.PRIMARY_KEY)
    monkeypatch.setattr(breaker, "is_tripped", lambda: False)

    ep = ef.active_embedding_endpoint()
    assert ep.model_key == ef.PRIMARY_KEY
    assert ep.is_fallback is False
    assert ep.base_url == _PRIMARY_ENDPOINT


def test_status_counts_failover_and_recovery(failover_config, monkeypatch):
    ef.reset_embedding_failover()
    # prime → primary (first resolution is not a transition)
    ef.active_embedding_endpoint()
    _trip_primary()
    ef.active_embedding_endpoint()  # transition → fallback
    status = ef.embedding_endpoint_status()
    assert status["is_fallback"] is True
    assert status["fallback_configured"] is True
    assert status["failover_count"] == 1

    breaker = get_circuit_breaker(ef.PRIMARY_KEY)
    monkeypatch.setattr(breaker, "is_tripped", lambda: False)
    ef.active_embedding_endpoint()  # transition → primary
    status = ef.embedding_endpoint_status()
    assert status["is_fallback"] is False
    assert status["recovery_count"] == 1


# --- (d) the cached client swaps endpoint on failover -----------------------
class _DummyModel:
    embed_batch_size = 0

    def __init__(self, base_url: str | None):
        self.base_url = base_url

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_cached_client_swaps_endpoint_on_failover(failover_config, monkeypatch):
    """create_embedding_model() returns the ACTIVE endpoint's client; it swaps on
    failover and the cache holds no stale primary client."""
    builds: list[str | None] = []

    def _fake_build(**kw):
        builds.append(kw.get("base_url_str"))
        return _DummyModel(kw.get("base_url_str"))

    # Restore the REAL factory (the unit suite's autouse fixture stubs it) so we
    # exercise its actual endpoint-resolution + caching, with only the inner
    # construction faked out (no network).
    monkeypatch.setattr(eu, "create_embedding_model", _REAL_CREATE_EMBEDDING_MODEL)
    monkeypatch.setattr(eu, "_build_embedding_model", _fake_build)
    eu.clear_embedding_model_cache()

    # Primary up → the default embedder client targets the dedicated endpoint.
    m_primary = eu.create_embedding_model()
    assert m_primary.base_url == _PRIMARY_ENDPOINT

    # Primary down → the same default call yields the shared fallback client,
    # a DIFFERENT instance (no stale primary client served).
    _trip_primary()
    m_fallback = eu.create_embedding_model()
    assert m_fallback.base_url == _FALLBACK_ENDPOINT
    assert m_fallback is not m_primary
    assert _PRIMARY_ENDPOINT in builds
    assert _FALLBACK_ENDPOINT in builds

    eu.clear_embedding_model_cache()
