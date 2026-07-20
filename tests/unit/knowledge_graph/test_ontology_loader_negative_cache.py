"""Bounded negative-cache coverage for remote ontology resolution."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from agent_utilities.knowledge_graph.core import ontology_loader
from agent_utilities.knowledge_graph.core.ontology_loader import OntologyLoader


class _Trust:
    def httpx_kwargs(self) -> dict[str, Any]:
        return {}

    def cleanup(self) -> None:
        return None


class _StatusClient:
    def __init__(self, status_code: int, calls: list[str]) -> None:
        self._status_code = status_code
        self._calls = calls

    def __enter__(self) -> _StatusClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get(self, url: str) -> httpx.Response:
        self._calls.append(url)
        request = httpx.Request("GET", url)
        return httpx.Response(self._status_code, request=request)


@pytest.fixture(autouse=True)
def _empty_remote_absence_cache() -> Iterator[None]:
    ontology_loader._clear_remote_absence_cache()
    yield
    ontology_loader._clear_remote_absence_cache()


def _install_status_client(
    monkeypatch: pytest.MonkeyPatch, *, status_code: int, calls: list[str]
) -> None:
    monkeypatch.setattr(
        "agent_utilities.core.transport_security.resolve_configured_tls_profile",
        lambda _profile: _Trust(),
    )
    monkeypatch.setattr(
        "agent_utilities.core.http_client.create_http_client",
        lambda **_kwargs: _StatusClient(status_code, calls),
    )


def test_registered_absent_federated_iri_never_becomes_remote_egress(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = OntologyLoader(cache_dir=tmp_path)
    fetches: list[str] = []
    monkeypatch.setattr(loader, "_federated_path_for", lambda _uri, _suffix: None)
    monkeypatch.setattr(
        loader, "_fetch_remote", lambda url: fetches.append(url) or "unexpected"
    )

    result = loader._fetch_ontology("http://knuckles.team/kg/ansible", tmp_path)

    assert result is None
    assert fetches == []


@pytest.mark.parametrize("status_code", [404, 410])
def test_remote_absence_is_negative_cached_across_loader_instances(
    tmp_path, monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    calls: list[str] = []
    _install_status_client(monkeypatch, status_code=status_code, calls=calls)
    url = "https://ontology.example.invalid/missing.ttl"

    assert OntologyLoader(cache_dir=tmp_path / "one")._fetch_remote(url) is None
    assert OntologyLoader(cache_dir=tmp_path / "two")._fetch_remote(url) is None

    assert calls == [url]


@pytest.mark.parametrize("status_code", [401, 403, 407])
def test_remote_authentication_failures_are_not_negative_cached(
    tmp_path, monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    calls: list[str] = []
    _install_status_client(monkeypatch, status_code=status_code, calls=calls)
    url = "https://ontology.example.invalid/protected.ttl"
    loader = OntologyLoader(cache_dir=tmp_path)

    assert loader._fetch_remote(url) is None
    assert loader._fetch_remote(url) is None

    assert calls == [url, url]


def test_remote_configuration_failure_is_not_negative_cached(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fail_configuration(_profile: str) -> _Trust:
        calls.append("resolve")
        raise ValueError("invalid TLS profile")

    monkeypatch.setattr(
        "agent_utilities.core.transport_security.resolve_configured_tls_profile",
        fail_configuration,
    )
    loader = OntologyLoader(cache_dir=tmp_path)
    url = "https://ontology.example.invalid/configuration.ttl"

    assert loader._fetch_remote(url) is None
    assert loader._fetch_remote(url) is None

    assert calls == ["resolve", "resolve"]


def test_remote_absence_cache_is_size_and_time_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(ontology_loader.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(ontology_loader, "_REMOTE_ABSENCE_CACHE_MAX_ENTRIES", 2)
    monkeypatch.setattr(ontology_loader, "_REMOTE_ABSENCE_CACHE_TTL_SECONDS", 5.0)

    ontology_loader._remember_remote_absence("https://example.invalid/one")
    ontology_loader._remember_remote_absence("https://example.invalid/two")
    ontology_loader._remember_remote_absence("https://example.invalid/three")

    assert len(ontology_loader._REMOTE_ABSENCE_CACHE) == 2
    assert all(
        len(key) == 64 and "example.invalid" not in key
        for key in ontology_loader._REMOTE_ABSENCE_CACHE
    )
    assert not ontology_loader._remote_absence_is_cached("https://example.invalid/one")
    assert ontology_loader._remote_absence_is_cached("https://example.invalid/two")
    assert ontology_loader._remote_absence_is_cached("https://example.invalid/three")

    now[0] += 5.0
    assert not ontology_loader._remote_absence_is_cached("https://example.invalid/two")
    assert not ontology_loader._remote_absence_is_cached(
        "https://example.invalid/three"
    )
