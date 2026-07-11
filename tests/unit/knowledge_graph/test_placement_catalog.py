"""Engine placement-catalog consumer tests (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw, DIST-P2-2b).

Covers the ONE entrypoint, ``resolve_placement``: the engine catalog is
consulted (not a pure static-hash decision), epoch caching avoids a re-call
within the TTL, a stale-epoch redirect triggers re-resolution + retry against
the new endpoint, and the static HRW ring is only ever the bootstrap/fallback
(catalog disabled, unreachable, or an older engine that doesn't advertise
one). No live engines: every catalog client is an injected fake.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import placement_catalog
from agent_utilities.knowledge_graph.core.placement_catalog import (
    PlacementResult,
    invalidate,
    resolve_placement,
    split_tenant_key,
)
from agent_utilities.knowledge_graph.core.shard_topology import shard_endpoint_for

pytestmark = pytest.mark.concept("AU-KG.sharding.tenant-partitioned-sharding-hrw")

THREE_SHARDS = ["tcp://shard-a:9100", "tcp://shard-b:9100", "tcp://shard-c:9100"]


class _FakeConfig:
    def __init__(self, *, enabled: bool = True, ttl_s: float = 5.0) -> None:
        self.placement_catalog_enabled = enabled
        self.placement_catalog_ttl_s = ttl_s
        # Insecure short-circuits `resolve_engine_auth` before it ever touches
        # `graph_service_auth_secret` / the persisted-secret file (hermetic).
        self.kg_engine_insecure = True


class _FakeCatalogClient:
    """A fake engine client exposing only the raw ``_send`` RPC seam."""

    def __init__(self, answer=None, error: Exception | None = None) -> None:
        self._answer = answer
        self._error = error
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def _send(self, method: str, params: dict) -> dict | None:
        self.calls.append((method, params))
        if self._error is not None:
            raise self._error
        return self._answer

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _clear_cache():
    invalidate(None)
    yield
    invalidate(None)


# ---------------------------------------------------------------------------
# split_tenant_key — must agree with the engine's raft::placement version
# ---------------------------------------------------------------------------


def test_split_tenant_key_parses_prefix():
    assert split_tenant_key("acme:ws1") == ("acme", "ws1")
    assert split_tenant_key("acme:ws:nested") == ("acme", "ws:nested")
    assert split_tenant_key("solo") == ("solo", "solo")
    assert split_tenant_key(":leading") == (":leading", ":leading")


# ---------------------------------------------------------------------------
# Zero-infra default: single endpoint never touches the catalog
# ---------------------------------------------------------------------------


def test_single_endpoint_is_identity_no_catalog_call():
    def _factory(endpoint):
        raise AssertionError("catalog must not be consulted for one endpoint")

    result = resolve_placement(
        "some_graph", ["unix:///tmp/x.sock"], client_factory=_factory
    )
    assert result == PlacementResult(
        endpoint="unix:///tmp/x.sock", epoch=0, source="hrw"
    )


# ---------------------------------------------------------------------------
# The catalog is actually consulted — not a pure static-hash decision
# ---------------------------------------------------------------------------


def test_placement_lookup_consults_engine_catalog(monkeypatch):
    hrw_pick = shard_endpoint_for("acme:ws1", THREE_SHARDS)
    other = next(e for e in THREE_SHARDS if e != hrw_pick)
    client = _FakeCatalogClient(
        answer={"explicit": True, "endpoint": other, "epoch": 7, "group": 42}
    )

    def _factory(endpoint):
        return client

    result = resolve_placement(
        "acme:ws1",
        THREE_SHARDS,
        config=_FakeConfig(),
        client_factory=_factory,
    )

    assert client.calls, "the catalog route RPC was never issued"
    method, params = client.calls[0]
    assert method == "PlacementRoute"
    assert params == {"tenant": "acme", "sub_key": "ws1", "client_epoch": 0}
    # The catalog's answer wins even though it disagrees with the static HRW
    # pick — proof this is not a pure hash decision.
    assert result.endpoint == other
    assert result.endpoint != hrw_pick
    assert result.epoch == 7
    assert result.source == "catalog"
    assert result.group == 42


def test_first_contact_endpoint_down_tries_next():
    hrw_pick = shard_endpoint_for("acme:ws2", THREE_SHARDS)
    winner = next(e for e in THREE_SHARDS if e != hrw_pick)
    down = _FakeCatalogClient(error=ConnectionError("refused"))
    up = _FakeCatalogClient(answer={"explicit": True, "endpoint": winner, "epoch": 1})
    clients = {hrw_pick: down}

    def _factory(endpoint):
        return clients.get(endpoint, up)

    result = resolve_placement(
        "acme:ws2", THREE_SHARDS, config=_FakeConfig(), client_factory=_factory
    )
    assert down.calls, "the down endpoint should still have been tried first"
    assert result.endpoint == winner
    assert result.source == "catalog"


# ---------------------------------------------------------------------------
# HRW fallback — catalog disabled, unreachable, or not advertised
# ---------------------------------------------------------------------------


def test_hrw_fallback_when_catalog_not_advertised():
    """Every contact endpoint errors (simulating an older engine with no
    PlacementRoute method) -> falls back to the static HRW ring."""
    expected = shard_endpoint_for("legacy_graph", THREE_SHARDS)
    client = _FakeCatalogClient(error=RuntimeError("unknown method 'PlacementRoute'"))

    def _factory(endpoint):
        return client

    result = resolve_placement(
        "legacy_graph", THREE_SHARDS, config=_FakeConfig(), client_factory=_factory
    )
    assert result == PlacementResult(endpoint=expected, epoch=0, source="hrw")


def test_hrw_fallback_on_explicit_no_placement_answer():
    """The catalog answers but has no explicit placement for this tenant ->
    HRW is authoritative for it (not an error, a definitive deferral)."""
    expected = shard_endpoint_for("unplaced_graph", THREE_SHARDS)
    client = _FakeCatalogClient(answer={"explicit": False})

    def _factory(endpoint):
        return client

    result = resolve_placement(
        "unplaced_graph", THREE_SHARDS, config=_FakeConfig(), client_factory=_factory
    )
    assert result == PlacementResult(endpoint=expected, epoch=0, source="hrw")


def test_catalog_disabled_config_skips_catalog_entirely():
    def _factory(endpoint):
        raise AssertionError("catalog must not be consulted when disabled")

    expected = shard_endpoint_for("acme:ws3", THREE_SHARDS)
    result = resolve_placement(
        "acme:ws3",
        THREE_SHARDS,
        config=_FakeConfig(enabled=False),
        client_factory=_factory,
    )
    assert result == PlacementResult(endpoint=expected, epoch=0, source="hrw")


def test_hermetic_testing_guard_skips_real_connect(monkeypatch):
    """Without an injected client_factory, the unit-suite testing flag must
    prevent any real network attempt (mirrors engine_resolver's own guard)."""

    def _boom(endpoint, auth_secret):
        raise AssertionError("must not dial a real socket under the test guard")

    monkeypatch.setattr(placement_catalog, "_default_connect", _boom)
    expected = shard_endpoint_for("acme:ws4", THREE_SHARDS)
    result = resolve_placement("acme:ws4", THREE_SHARDS, config=_FakeConfig())
    assert result == PlacementResult(endpoint=expected, epoch=0, source="hrw")


# ---------------------------------------------------------------------------
# Epoch caching — a second lookup within TTL does not re-call
# ---------------------------------------------------------------------------


def test_epoch_caching_second_lookup_within_ttl_no_recall():
    client = _FakeCatalogClient(
        answer={"explicit": True, "endpoint": THREE_SHARDS[0], "epoch": 3}
    )

    def _factory(endpoint):
        return client

    config = _FakeConfig(ttl_s=60.0)
    first = resolve_placement(
        "acme:ws5", THREE_SHARDS, config=config, client_factory=_factory
    )
    second = resolve_placement(
        "acme:ws5", THREE_SHARDS, config=config, client_factory=_factory
    )
    assert first == second
    assert len(client.calls) == 1, "second lookup within TTL must not re-call"


# ---------------------------------------------------------------------------
# Stale-epoch redirect — re-resolution + retry against the new endpoint
# ---------------------------------------------------------------------------


def test_stale_epoch_redirect_triggers_reresolution_and_retry():
    old_endpoint, new_endpoint = THREE_SHARDS[0], THREE_SHARDS[1]
    answers = [
        {"explicit": True, "endpoint": old_endpoint, "epoch": 1},
        {"explicit": True, "endpoint": new_endpoint, "epoch": 2},
    ]
    calls: list[tuple[str, dict]] = []

    class _SequencedClient:
        def _send(self, method, params):
            calls.append((method, params))
            return answers[len(calls) - 1]

    def _factory(endpoint):
        return _SequencedClient()

    config = _FakeConfig(ttl_s=60.0)
    first = resolve_placement(
        "acme:ws6", THREE_SHARDS, config=config, client_factory=_factory
    )
    assert first.endpoint == old_endpoint
    assert first.epoch == 1

    # A data request against `old_endpoint` came back rejected for a stale
    # epoch (the engine's fenced-cutover redirect) — the caller re-resolves.
    redirected = resolve_placement(
        "acme:ws6",
        THREE_SHARDS,
        config=config,
        client_factory=_factory,
        force_refresh=True,
    )
    assert redirected.endpoint == new_endpoint
    assert redirected.epoch == 2
    assert len(calls) == 2
    # The second call presented the previously-cached epoch, so the engine
    # could recognize this as a stale-epoch redirect rather than a cold ask.
    assert calls[1][1]["client_epoch"] == 1

    # And the cache now reflects the redirect target without another call.
    again = resolve_placement(
        "acme:ws6", THREE_SHARDS, config=config, client_factory=_factory
    )
    assert again.endpoint == new_endpoint
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# invalidate()
# ---------------------------------------------------------------------------


def test_invalidate_clears_only_named_graph():
    client_a = _FakeCatalogClient(
        answer={"explicit": True, "endpoint": THREE_SHARDS[0], "epoch": 1}
    )
    client_b = _FakeCatalogClient(
        answer={"explicit": True, "endpoint": THREE_SHARDS[1], "epoch": 1}
    )
    clients = {"acme:ws7": client_a, "beta:ws7": client_b}

    def _factory_for(graph):
        def _factory(endpoint):
            return clients[graph]

        return _factory

    config = _FakeConfig(ttl_s=60.0)
    resolve_placement(
        "acme:ws7", THREE_SHARDS, config=config, client_factory=_factory_for("acme:ws7")
    )
    resolve_placement(
        "beta:ws7", THREE_SHARDS, config=config, client_factory=_factory_for("beta:ws7")
    )
    assert len(client_a.calls) == 1
    assert len(client_b.calls) == 1

    invalidate("acme:ws7")

    # acme re-queries (cache dropped); beta stays cached.
    resolve_placement(
        "acme:ws7", THREE_SHARDS, config=config, client_factory=_factory_for("acme:ws7")
    )
    resolve_placement(
        "beta:ws7", THREE_SHARDS, config=config, client_factory=_factory_for("beta:ws7")
    )
    assert len(client_a.calls) == 2
    assert len(client_b.calls) == 1
