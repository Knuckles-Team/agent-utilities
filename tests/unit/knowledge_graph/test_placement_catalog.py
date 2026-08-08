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
    with pytest.raises(PlacementAuthorityError) as excinfo:
        resolve_placement(
            "tenant:workspace",
            ["tls://a.invalid:9443", "tls://b.invalid:9443"],
            config=_Config(),
            client_factory=lambda _endpoint: _Client(ConnectionError("down"), []),
        )
    assert isinstance(excinfo.value.__cause__, ConnectionError)


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


def test_last_contact_failure_is_the_chained_cause() -> None:
    """D-WD-4: the caller-visible error must chain the *real* per-contact
    cause (e.g. an engine ACCESS_DENIED), not just a bare failure count."""
    errors = iter([ConnectionError("first down"), PermissionError("ACCESS_DENIED")])

    def _factory(_endpoint: str) -> _Client:
        return _Client(next(errors), [])

    with pytest.raises(PlacementAuthorityError) as excinfo:
        resolve_placement(
            "tenant:workspace",
            ["tls://a.invalid:9443", "tls://b.invalid:9443"],
            config=_Config(),
            client_factory=_factory,
        )
    # The chained cause is the LAST contact tried, not the first -- callers
    # debugging "why did this fail" want the most recent attempt's reason.
    assert isinstance(excinfo.value.__cause__, PermissionError)
    assert "ACCESS_DENIED" in str(excinfo.value.__cause__)


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


# ── ADR-1 / W1.1 engine-authoritative endpoint discovery ────────────────────
# `reports/wave1/ADR-scale-trio.md` §ADR-1 decision 3: resolution order is
# (a) a static GRAPH_RAFT_GROUP_ENDPOINTS entry as an explicit OVERRIDE (wins
# when present), (b) PlacementRoute.endpoints (NEW -- no static config
# needed), (c) the single-contact fallback (unchanged, covered above).


def test_engine_endpoints_resolve_multi_contact_with_no_static_map() -> None:
    """The exact gap ADR-1 closes: >1 contact used to hard-require
    GRAPH_RAFT_GROUP_ENDPOINTS (`test_multi_endpoint_route_requires_explicit_
    group_topology` above); now the engine's own discovered endpoints suffice."""
    contacts = ["tls://coordinator-a.invalid:9443", "tls://coordinator-b.invalid:9443"]
    answer = {
        **_answer(group=7),
        "endpoints": ["tls://leader.invalid:9443", "tls://follower.invalid:9443"],
    }
    result = resolve_placement(
        "tenant:workspace",
        contacts,
        config=_Config(),  # no GRAPH_RAFT_GROUP_ENDPOINTS configured
        client_factory=lambda _endpoint: _Client(answer, []),
    )
    assert result.endpoint == "tls://leader.invalid:9443"


def test_static_override_wins_over_engine_endpoints_when_both_present() -> None:
    """An explicit operator override always wins when configured -- the
    escape hatch for a client that cannot reach the engine-discovered address
    (NAT / ingress-only network boundary)."""
    answer = {
        **_answer(group=7),
        "endpoints": ["tls://leader.invalid:9443"],
    }
    result = resolve_placement(
        "tenant:workspace",
        ["tls://seed.invalid:9443"],
        config=_Config({"7": "tls://operator-override.invalid:9443"}),
        client_factory=lambda _endpoint: _Client(answer, []),
    )
    assert result.endpoint == "tls://operator-override.invalid:9443"


def test_empty_engine_endpoints_fall_back_to_single_contact() -> None:
    """A single-node deployment (or a cluster with no self-reported member
    yet) answers empty `endpoints` -- the unchanged single-contact fallback
    still applies."""
    answer = {**_answer(), "endpoints": []}
    result = resolve_placement(
        "tenant:workspace",
        ["unix://engine.sock"],
        config=_Config(),
        client_factory=lambda _endpoint: _Client(answer, []),
    )
    assert result.endpoint == "unix://engine.sock"


def test_engine_endpoints_reflect_the_current_leader_after_a_refresh() -> None:
    """A `force_refresh` after a failover picks up the NEW leader-first
    endpoint list -- the client-side half of ADR-1's "kill leader -> client
    re-routes with zero config edits" acceptance criterion."""
    contacts = ["tls://a.invalid:9443", "tls://b.invalid:9443"]
    answers = iter(
        [
            {**_answer(group=1), "endpoints": ["tls://node-a.invalid:9443"]},
            {**_answer(group=1), "endpoints": ["tls://node-b.invalid:9443"]},
        ]
    )

    def factory(_endpoint: str) -> _Client:
        return _Client(next(answers), [])

    before = resolve_placement(
        "tenant:workspace", contacts, config=_Config(), client_factory=factory
    )
    after = resolve_placement(
        "tenant:workspace",
        contacts,
        config=_Config(),
        force_refresh=True,
        client_factory=factory,
    )
    assert before.endpoint == "tls://node-a.invalid:9443"
    assert after.endpoint == "tls://node-b.invalid:9443"


@pytest.mark.parametrize(
    "endpoints",
    [
        "tls://not-a-list.invalid:9443",
        [123],
        [""],
        [None],
    ],
)
def test_malformed_engine_endpoints_fail_closed(endpoints: Any) -> None:
    answer = {**_answer(), "endpoints": endpoints}
    with pytest.raises(PlacementAuthorityError):
        resolve_placement(
            "tenant:workspace",
            ["tls://coordinator.invalid:9443"],
            config=_Config(),
            client_factory=lambda _endpoint: _Client(answer, []),
        )


# ── Admin-capability broker fallback (register D-W6-ISO-1) ────────────────
#
# These exercise the ``client_factory is None`` real-identity path, so
# ``_request_authority``/``_default_connect``/``_broker_authority`` are
# monkeypatched directly rather than dialing anything real.

from agent_utilities.knowledge_graph.core import (  # noqa: E402
    placement_catalog as _pc,
)


def _admin_denied_error() -> RuntimeError:
    return RuntimeError(
        "ACCESS_DENIED: verified principal lacks admin capability required "
        "for 'admin:cluster-read'"
    )


def _scope_denied_error() -> RuntimeError:
    return RuntimeError(
        "ACCESS_DENIED: verified request context lacks required scope "
        "'admin:cluster-read'"
    )


def test_admin_capability_denied_matches_the_exact_engine_denial() -> None:
    assert _pc._admin_capability_denied(_admin_denied_error()) is True
    assert _pc._admin_capability_denied(_scope_denied_error()) is False
    assert _pc._admin_capability_denied(ConnectionError("unreachable")) is False
    assert _pc._admin_capability_denied(None) is False


def test_admin_capability_denied_walks_the_cause_chain() -> None:
    wrapped = RuntimeError("no configured engine returned an authoritative route")
    wrapped.__cause__ = _admin_denied_error()
    assert _pc._admin_capability_denied(wrapped) is True


def test_query_catalog_falls_back_to_broker_on_admin_capability_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller identity denied ONLY for missing engine admin capability is
    retried once under the broker identity; the broker's route is returned and
    the caller's own identity was tried first (never skipped)."""
    attempts: list[str] = []

    def fake_request_authority(config: Any) -> tuple[str, dict[str, Any]]:
        return "caller-secret", {"principal": "caller"}

    def fake_broker_authority(config: Any) -> tuple[str, dict[str, Any]] | None:
        return "broker-secret", {"principal": "broker"}

    def fake_default_connect(
        endpoint: str,
        auth_secret: str,
        config: Any,
        *,
        verified_context: dict[str, Any],
    ) -> Any:
        attempts.append(verified_context["principal"])
        if verified_context["principal"] == "caller":
            raise _admin_denied_error()
        return _Client({**_answer(), "endpoints": [endpoint]}, [])

    monkeypatch.setattr(_pc, "_request_authority", fake_request_authority)
    # The hermetic testing guard fails closed for the real client_factory=None
    # path under AGENT_UTILITIES_TESTING -- these tests exercise that exact
    # path with fully injected fakes, so opt back out for them specifically
    # (mirrors _hermetic_testing_guard's own documented client_factory escape
    # hatch, just applied via monkeypatch since client_factory itself must
    # stay None to reach the real _request_authority/_broker_authority code).
    monkeypatch.setattr(_pc, "_hermetic_testing_guard", lambda client_factory: False)
    monkeypatch.setattr(_pc, "_broker_authority", fake_broker_authority)
    monkeypatch.setattr(_pc, "_default_connect", fake_default_connect)

    result = _pc._query_catalog(
        "tenant",
        "workspace",
        ("tls://coordinator.invalid:9443",),
        _Config(),
        client_factory=None,
        client_epoch=0,
    )
    assert attempts == ["caller", "broker"]
    assert result.endpoint == "tls://coordinator.invalid:9443"


def test_query_catalog_never_brokers_a_plain_scope_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scope denial (the caller's JWT never carried kg:admin at all) must
    never be retried under the broker -- only the specific admin-CAPABILITY
    denial (a verified kg:admin caller with no engine-side registration) may."""
    broker_calls: list[None] = []

    def fake_request_authority(config: Any) -> tuple[str, dict[str, Any]]:
        return "caller-secret", {"principal": "caller"}

    def fake_broker_authority(config: Any) -> tuple[str, dict[str, Any]] | None:
        broker_calls.append(None)
        return "broker-secret", {"principal": "broker"}

    def fake_default_connect(
        endpoint: str,
        auth_secret: str,
        config: Any,
        *,
        verified_context: dict[str, Any],
    ) -> Any:
        raise _scope_denied_error()

    monkeypatch.setattr(_pc, "_request_authority", fake_request_authority)
    # The hermetic testing guard fails closed for the real client_factory=None
    # path under AGENT_UTILITIES_TESTING -- these tests exercise that exact
    # path with fully injected fakes, so opt back out for them specifically
    # (mirrors _hermetic_testing_guard's own documented client_factory escape
    # hatch, just applied via monkeypatch since client_factory itself must
    # stay None to reach the real _request_authority/_broker_authority code).
    monkeypatch.setattr(_pc, "_hermetic_testing_guard", lambda client_factory: False)
    monkeypatch.setattr(_pc, "_broker_authority", fake_broker_authority)
    monkeypatch.setattr(_pc, "_default_connect", fake_default_connect)

    with pytest.raises(PlacementAuthorityError):
        _pc._query_catalog(
            "tenant",
            "workspace",
            ("tls://coordinator.invalid:9443",),
            _Config(),
            client_factory=None,
            client_epoch=0,
        )
    assert broker_calls == []


def test_query_catalog_reraises_original_denial_when_broker_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request_authority(config: Any) -> tuple[str, dict[str, Any]]:
        return "caller-secret", {"principal": "caller"}

    def fake_default_connect(
        endpoint: str,
        auth_secret: str,
        config: Any,
        *,
        verified_context: dict[str, Any],
    ) -> Any:
        raise _admin_denied_error()

    monkeypatch.setattr(_pc, "_request_authority", fake_request_authority)
    # The hermetic testing guard fails closed for the real client_factory=None
    # path under AGENT_UTILITIES_TESTING -- these tests exercise that exact
    # path with fully injected fakes, so opt back out for them specifically
    # (mirrors _hermetic_testing_guard's own documented client_factory escape
    # hatch, just applied via monkeypatch since client_factory itself must
    # stay None to reach the real _request_authority/_broker_authority code).
    monkeypatch.setattr(_pc, "_hermetic_testing_guard", lambda client_factory: False)
    monkeypatch.setattr(_pc, "_broker_authority", lambda config: None)
    monkeypatch.setattr(_pc, "_default_connect", fake_default_connect)

    with pytest.raises(PlacementAuthorityError):
        _pc._query_catalog(
            "tenant",
            "workspace",
            ("tls://coordinator.invalid:9443",),
            _Config(),
            client_factory=None,
            client_epoch=0,
        )


def test_query_catalog_reraises_original_denial_when_broker_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request_authority(config: Any) -> tuple[str, dict[str, Any]]:
        return "caller-secret", {"principal": "caller"}

    def fake_broker_authority(config: Any) -> tuple[str, dict[str, Any]] | None:
        return "broker-secret", {"principal": "broker"}

    def fake_default_connect(
        endpoint: str,
        auth_secret: str,
        config: Any,
        *,
        verified_context: dict[str, Any],
    ) -> Any:
        raise _admin_denied_error()

    monkeypatch.setattr(_pc, "_request_authority", fake_request_authority)
    # The hermetic testing guard fails closed for the real client_factory=None
    # path under AGENT_UTILITIES_TESTING -- these tests exercise that exact
    # path with fully injected fakes, so opt back out for them specifically
    # (mirrors _hermetic_testing_guard's own documented client_factory escape
    # hatch, just applied via monkeypatch since client_factory itself must
    # stay None to reach the real _request_authority/_broker_authority code).
    monkeypatch.setattr(_pc, "_hermetic_testing_guard", lambda client_factory: False)
    monkeypatch.setattr(_pc, "_broker_authority", fake_broker_authority)
    monkeypatch.setattr(_pc, "_default_connect", fake_default_connect)

    with pytest.raises(PlacementAuthorityError):
        _pc._query_catalog(
            "tenant",
            "workspace",
            ("tls://coordinator.invalid:9443",),
            _Config(),
            client_factory=None,
            client_epoch=0,
        )


def test_query_catalog_client_factory_path_never_consults_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The injected-``client_factory`` seam (every OTHER test in this file)
    must stay byte-for-byte unaffected: it never resolves real identity and
    must never consult the broker, even on an admin-capability-shaped error."""
    broker_calls: list[None] = []
    monkeypatch.setattr(
        _pc,
        "_broker_authority",
        lambda config: broker_calls.append(None) or None,
    )

    def factory(_endpoint: str) -> _Client:
        return _Client(_admin_denied_error(), [])

    with pytest.raises(PlacementAuthorityError):
        resolve_placement(
            "tenant:workspace",
            ["tls://coordinator.invalid:9443"],
            config=_Config(),
            client_factory=factory,
        )
    assert broker_calls == []
