"""ARD federation relay tests (CONCEPT:ECO-4.97).

Offline: peer discovery and the local search are stubbed, so no network or KG is used.
"""

from __future__ import annotations

import pytest

from agent_utilities.ecosystem import ard_federation
from agent_utilities.ecosystem.ard_federation import ArdFederationRelay


def _local_result(name: str, score: float, domain: str = "local") -> dict:
    return {
        "id": f"mcp:{name}",
        "type": "application/mcp-server+json",
        "name": name,
        "score": score,
        "publisher": {"domain": domain},
    }


@pytest.fixture
def _relay(monkeypatch: pytest.MonkeyPatch) -> ArdFederationRelay:
    from agent_utilities.ecosystem import ard_registry

    monkeypatch.setattr(
        ard_registry,
        "ard_search",
        lambda *a, **k: {
            "results": [_local_result("a", 0.9)],
            "publisher": {"domain": "local"},
        },
    )
    monkeypatch.setattr(ard_federation, "origin", lambda: "local")
    return ArdFederationRelay()


def test_mode_none_is_local_only(_relay: ArdFederationRelay) -> None:
    out = _relay.federated_search("x", mode="none")
    assert out["federationMode"] == "none"
    assert [r["name"] for r in out["results"]] == ["a"]


def test_referrals_lists_peers(_relay: ArdFederationRelay, monkeypatch) -> None:
    monkeypatch.setattr(
        ArdFederationRelay,
        "list_registries",
        lambda self: [{"name": "hf", "url": "http://hf"}],
    )
    out = _relay.federated_search("x", mode="referrals")
    assert out["federationMode"] == "referrals"
    assert out["referrals"] == [{"name": "hf", "url": "http://hf"}]


def test_auto_merges_and_dedupes(_relay: ArdFederationRelay, monkeypatch) -> None:
    monkeypatch.setattr(
        ArdFederationRelay,
        "list_registries",
        lambda self: [{"name": "hf", "url": "http://hf"}],
    )
    # Peer returns a higher-scoring copy of "a" (same domain+id) + a unique "b".
    monkeypatch.setattr(
        ArdFederationRelay,
        "_post",
        lambda self, url, body: [
            _local_result("a", 0.95, "local"),
            _local_result("b", 0.5, "peer"),
        ],
    )
    out = _relay.federated_search("x", mode="auto", page_size=5)
    assert out["federationMode"] == "auto"
    names = sorted(r["name"] for r in out["results"])
    assert names == ["a", "b"]  # "a" de-duplicated across local + peer
    a = next(r for r in out["results"] if r["name"] == "a")
    assert a["score"] == 0.95  # the higher-scoring copy wins


def test_loop_break_when_origin_in_via(_relay: ArdFederationRelay, monkeypatch) -> None:
    called = {"fanout": False}

    def _no(self, *a, **k):  # pragma: no cover - must not run
        called["fanout"] = True
        return []

    monkeypatch.setattr(ArdFederationRelay, "_fanout", _no)
    out = _relay.federated_search("x", mode="auto", via=["local"])
    assert out["federationMode"] == "none"
    assert called["fanout"] is False
