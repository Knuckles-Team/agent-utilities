"""Engine-authoritative placement routing; no caller-side fallback."""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.placement_catalog import (
    PlacementAuthorityError,
    PlacementResult,
    PlacementTopologyError,
    invalidate,
    resolve_placement,
    split_tenant_key,
)


class _Config:
    placement_catalog_ttl_s = 5.0

    def __init__(self, group_endpoints: dict[str, str] | None = None) -> None:
        self.graph_raft_group_endpoints = group_endpoints


class _Placement:
    def __init__(self, answer: Any, calls: list[int]) -> None:
        self._answer = answer
        self._calls = calls

    def route(self, tenant: str, sub_key: str, *, client_epoch: int = 0) -> Any:
        assert tenant
        assert sub_key
        self._calls.append(client_epoch)
        if isinstance(self._answer, Exception):
            raise self._answer
        answer = dict(self._answer)
        answer["tenant_ref"] = tenant
        answer["partition_ref"] = sub_key
        return answer


class _Client:
    def __init__(self, answer: Any, calls: list[int]) -> None:
        self.placement = _Placement(answer, calls)


def _answer(*, group: int = 4, epoch: int = 9, placed: bool = True) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "route_id": "route:opaque",
        "tenant_ref": "tenant:opaque",
        "partition_ref": "partition:opaque",
        "authoritative": True,
        "placed": placed,
        "group": group,
        "epoch": epoch,
        "fencing_token": group,
        "stale": False,
        "leader_ref": None,
    }


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    invalidate()


def test_split_tenant_key_matches_engine_contract() -> None:
    assert split_tenant_key("tenant:workspace") == ("tenant", "workspace")
    assert split_tenant_key("graph") == ("graph", "graph")
    assert split_tenant_key(":graph") == (":graph", ":graph")


def test_single_coordinator_maps_complete_placed_route() -> None:
    calls: list[int] = []
    result = resolve_placement(
        "tenant:workspace",
        ["tls://coordinator.invalid:9443"],
        config=_Config(),
        client_factory=lambda _endpoint: _Client(_answer(), calls),
    )
    assert result == PlacementResult(
        endpoint="tls://coordinator.invalid:9443",
        epoch=9,
        group=4,
        fencing_token=4,
        placed=True,
    )
    assert calls == [0]


def test_unplaced_single_node_route_is_still_authoritative() -> None:
    result = resolve_placement(
        "graph",
        ["unix://engine.sock"],
        config=_Config(),
        client_factory=lambda _endpoint: _Client(
            _answer(group=0, epoch=0, placed=False), []
        ),
    )
    assert result.placed is False
    assert result.group == 0
    assert result.epoch == 0
    assert result.fencing_token == 0


def test_multi_endpoint_route_requires_explicit_group_topology() -> None:
    contacts = ["tls://coordinator-a.invalid:9443", "tls://coordinator-b.invalid:9443"]
    with pytest.raises(PlacementTopologyError):
        resolve_placement(
            "tenant:workspace",
            contacts,
            config=_Config(),
            client_factory=lambda _endpoint: _Client(_answer(group=7), []),
        )

    result = resolve_placement(
        "tenant:workspace",
        contacts,
        config=_Config({"7": "tls://group-seven.invalid:9443"}),
        client_factory=lambda _endpoint: _Client(_answer(group=7), []),
    )
    assert result.endpoint == "tls://group-seven.invalid:9443"


def test_deployment_topology_owns_endpoint_mapping() -> None:
    result = resolve_placement(
        "tenant:workspace",
        ["tls://coordinator.invalid:9443"],
        config=_Config({"4": "tls://mapped.invalid:9443"}),
        client_factory=lambda _endpoint: _Client(_answer(), []),
    )
    assert result.endpoint == "tls://mapped.invalid:9443"


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"authoritative": False},
        _answer(epoch=0, placed=True),
        {**_answer(), "fencing_token": 99},
        {**_answer(), "placed": "true"},
        {**_answer(), "endpoint": "deployment-specific"},
    ],
)
def test_invalid_or_non_authoritative_answers_fail_closed(answer: Any) -> None:
    with pytest.raises(PlacementAuthorityError):
        resolve_placement(
            "tenant:workspace",
            ["tls://coordinator.invalid:9443"],
            config=_Config(),
            client_factory=lambda _endpoint: _Client(answer, []),
        )


def test_unreachable_contacts_fail_closed_instead_of_hashing() -> None:
    with pytest.raises(PlacementAuthorityError):
        resolve_placement(
            "tenant:workspace",
            ["tls://a.invalid:9443", "tls://b.invalid:9443"],
            config=_Config(),
            client_factory=lambda _endpoint: _Client(ConnectionError("down"), []),
        )


def test_raw_send_only_client_is_not_a_compatibility_reader() -> None:
    class _RawClient:
        def _send(self, *_args: Any, **_kwargs: Any) -> Any:
            return _answer()

    with pytest.raises(PlacementAuthorityError):
        resolve_placement(
            "tenant:workspace",
            ["tls://coordinator.invalid:9443"],
            config=_Config(),
            client_factory=lambda _endpoint: _RawClient(),
        )


def test_cache_and_force_refresh_carry_the_previous_epoch() -> None:
    calls: list[int] = []
    answers = iter([_answer(epoch=3), _answer(epoch=8)])

    def factory(_endpoint: str) -> _Client:
        return _Client(next(answers), calls)

    first = resolve_placement(
        "tenant:workspace",
        ["tls://coordinator.invalid:9443"],
        config=_Config(),
        client_factory=factory,
    )
    cached = resolve_placement(
        "tenant:workspace",
        ["tls://coordinator.invalid:9443"],
        config=_Config(),
        client_factory=factory,
    )
    refreshed = resolve_placement(
        "tenant:workspace",
        ["tls://coordinator.invalid:9443"],
        config=_Config(),
        force_refresh=True,
        client_factory=factory,
    )
    assert first.epoch == cached.epoch == 3
    assert refreshed.epoch == 8
    assert calls == [0, 3]


def test_first_failed_contact_uses_next_coordinator_without_guessing() -> None:
    contacts = ["tls://a.invalid:9443", "tls://b.invalid:9443"]

    def factory(endpoint: str) -> _Client:
        if endpoint == contacts[0]:
            return _Client(ConnectionError("down"), [])
        return _Client(_answer(group=2), [])

    result = resolve_placement(
        "tenant:workspace",
        contacts,
        config=_Config({"2": "tls://group-two.invalid:9443"}),
        client_factory=factory,
    )
    assert result.group == 2
    assert result.endpoint == "tls://group-two.invalid:9443"
