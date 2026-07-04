"""Dedicated ingest-engine lifecycle (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw, Phase D).

Verifies the safe-by-default behavior: unset/unreachable ⇒ fall back to the query
engine (return None), only local unix endpoints are spawnable, and the health
check guards on socket existence so the client's silent fallback to the default
socket can't make a dead ingest engine look 'reachable'.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core import ingest_engine as ie


def test_unset_endpoint_returns_none():
    assert ie.ensure_ingest_engine(None, "secret") is None
    assert ie.ensure_ingest_engine("", "secret") is None


def test_non_local_endpoint_not_spawnable():
    # TCP shards are managed externally, never spawned as a local ingest engine.
    assert ie.ensure_ingest_engine("tcp://10.0.0.1:9100", "secret") is None


def test_socket_of_parses_unix_and_bare_paths():
    assert ie._socket_of("unix:///tmp/x.sock") == "/tmp/x.sock"
    assert ie._socket_of("/tmp/y.sock") == "/tmp/y.sock"
    assert ie._socket_of("tcp://h:1") is None


def test_reachable_false_for_missing_socket_no_fallback():
    # The client falls back to the default socket when the requested one is
    # absent; _reachable must guard on existence so a dead ingest socket never
    # reports the QUERY engine as reachable.
    assert ie._reachable("/tmp/definitely-not-a-real-socket-xyz.sock", None) is False


def test_reachable_endpoint_returns_without_spawn(monkeypatch):
    calls = {"reachable": 0, "spawned": 0}

    def fake_reachable(sock, secret):
        calls["reachable"] += 1
        return True

    def boom(*a, **k):
        calls["spawned"] += 1
        raise AssertionError("must not spawn when already reachable")

    monkeypatch.setattr(ie, "_reachable", fake_reachable)
    monkeypatch.setattr(ie.subprocess, "Popen", boom)
    ep = ie.ensure_ingest_engine("unix:///tmp/eg-ingest-test.sock", "secret")
    assert ep == "unix:///tmp/eg-ingest-test.sock"
    assert calls["spawned"] == 0
