"""Per-agent on-behalf-of identity — acceptance tests (W2.1 / ADR-4).

Covers the six ADR decisions and the ``off | warn | on`` rollout matrix end to end:

* chain visible in provenance (RunTrace stamp) + wire-valid against the real eg client;
* ceiling-violation denied (apply_tool_scope, fail-closed);
* expired-token spawn dies at its next lease renewal (work_item.heartbeat gate);
* PID-fallback removal proven by a startup-failure test (run_token fail-closed);
* the off/warn/on behavior matrix for every consumer.

The RFC 8693 exchange is mocked at the ``delegated_auth`` seam (the machinery's own test
pattern) — no live IdP is required.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent_utilities.security import delegation as deleg
from agent_utilities.security import run_token as rt


@pytest.fixture(autouse=True)
def _reset_run_token_ephemeral() -> None:
    """The run-token ephemeral secret is a process global; reset it per test."""
    rt._EPHEMERAL = None
    yield
    rt._EPHEMERAL = None


@pytest.fixture
def _secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AGENT_UTILITIES_TOKEN_SECRET", "unit-test-signing-secret-0123456789"
    )


def _mode(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("ENABLE_DELEGATED_IDENTITY", value)


# ---------------------------------------------------------------------------
# Rollout mode resolution — default warn, unrecognized degrades to warn
# ---------------------------------------------------------------------------


def test_default_mode_is_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENABLE_DELEGATED_IDENTITY", raising=False)
    assert deleg.delegation_mode() is deleg.DelegationMode.WARN
    assert deleg.delegation_enabled() is True


@pytest.mark.parametrize(
    "value,expected",
    [
        ("off", deleg.DelegationMode.OFF),
        ("warn", deleg.DelegationMode.WARN),
        ("on", deleg.DelegationMode.ON),
        ("ON", deleg.DelegationMode.ON),
        ("garbage", deleg.DelegationMode.WARN),
    ],
)
def test_mode_matrix(monkeypatch: pytest.MonkeyPatch, value: str, expected) -> None:
    _mode(monkeypatch, value)
    assert deleg.delegation_mode() is expected


def test_off_mode_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _mode(monkeypatch, "off")
    assert deleg.delegation_enabled() is False


# ---------------------------------------------------------------------------
# Decision 3 — run-token: PID fallback REMOVED, fail-closed when delegation on
# ---------------------------------------------------------------------------


def test_run_token_secret_is_not_pid_derived(monkeypatch: pytest.MonkeyPatch) -> None:
    """The forgeable PID-derived fallback secret is gone (random per-process instead)."""
    import hashlib
    import os

    _mode(monkeypatch, "off")
    monkeypatch.delenv("AGENT_UTILITIES_TOKEN_SECRET", raising=False)
    pid_derived = hashlib.sha256(f"au-run-token:{os.getpid()}".encode()).digest()
    assert rt._secret() != pid_derived


def test_run_token_fail_closed_on_when_secret_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ACCEPTANCE: startup-failure test — on + no secret must fail closed."""
    _mode(monkeypatch, "on")
    monkeypatch.delenv("AGENT_UTILITIES_TOKEN_SECRET", raising=False)
    with pytest.raises(rt.RunTokenSecretUnavailable):
        rt.mint_token("run:x")
    with pytest.raises(rt.RunTokenSecretUnavailable):
        rt.require_token_secret()


def test_run_token_on_with_secret_succeeds(
    monkeypatch: pytest.MonkeyPatch, _secret: None
) -> None:
    _mode(monkeypatch, "on")
    rt.require_token_secret()  # does not raise
    token = rt.mint_token("run:y", ttl_seconds=60)
    decoded = rt.validate_token(token)
    assert decoded.run_id == "run:y"


@pytest.mark.parametrize("mode", ["off", "warn"])
def test_run_token_ephemeral_when_not_enforced(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """off/warn tolerate an absent secret (ephemeral) — warn never fails startup."""
    _mode(monkeypatch, mode)
    monkeypatch.delenv("AGENT_UTILITIES_TOKEN_SECRET", raising=False)
    rt.require_token_secret()  # no-op, never raises
    token = rt.mint_token("run:z", endpoints=("search",), ttl_seconds=30)
    assert rt.validate_token(token, endpoint="search").run_id == "run:z"


def test_mint_spawn_run_token_scopes_and_caps_ttl(
    monkeypatch: pytest.MonkeyPatch, _secret: None
) -> None:
    import time

    _mode(monkeypatch, "on")
    issued = time.time()
    token, expires_at = deleg.mint_spawn_run_token(
        "run:s",
        principal="user:alice",
        allowed_tools=["search_nodes", "add_node"],
        ttl_seconds=10_000_000,  # asks for way too long
    )
    decoded = rt.decode_token(token)
    assert set(decoded.endpoints) == {"search_nodes", "add_node"}
    # TTL is capped at the module bound, not the (absurd) requested value.
    assert decoded.expires_at - issued <= deleg.DEFAULT_RUN_TOKEN_TTL_S + 5
    assert expires_at == pytest.approx(decoded.expires_at, abs=1.0)


# ---------------------------------------------------------------------------
# Decision 1 — RFC 8693 exchange (mocked at the delegated_auth seam)
# ---------------------------------------------------------------------------

_IDP_CFG = {
    "token_endpoint": "https://keycloak.example/realms/homelab/protocol/openid-connect/token",
    "oidc_client_id": "graph-os",
    "oidc_client_secret": "s3cret",
    "audience": "agent-services",
}


def test_exchange_appends_acting_agent() -> None:
    import agent_utilities.mcp.delegated_auth as da

    with patch.object(
        da, "_post_token", return_value={"access_token": "DELEGATED"}
    ) as m:
        out = da.exchange_token_for_agent(
            "agent:researcher:run-1", subject_token="CALLER.JWT", config=_IDP_CFG
        )
    assert out == "DELEGATED"
    sent = m.call_args[0][1]
    assert sent["grant_type"] == "urn:ietf:params:oauth:grant-type:token-exchange"
    assert sent["subject_token"] == "CALLER.JWT"
    assert sent["act_as"] == "agent:researcher:run-1"


def test_exchange_requires_a_caller_token() -> None:
    import agent_utilities.mcp.delegated_auth as da

    with pytest.raises(ValueError):
        da.exchange_token_for_agent("agent:x", subject_token=None, config=_IDP_CFG)


# ---------------------------------------------------------------------------
# Decision 2 — the envelope delegation chain (wire-valid against the real client)
# ---------------------------------------------------------------------------


def test_chain_build_is_principal_first_agent_last() -> None:
    d = deleg.build_spawn_delegation(
        agent_name="researcher",
        run_id="run-1",
        principal="user:alice",
        mode=deleg.DelegationMode.ON,
    )
    assert d.chain[0] == "user:alice"
    assert d.chain[-1] == d.agent_instance_id == "agent:researcher:run-1"
    assert len(d.chain) >= 2


def test_nested_spawn_extends_the_same_chain() -> None:
    parent = deleg.build_spawn_delegation(
        agent_name="researcher",
        run_id="run-1",
        principal="user:alice",
        mode=deleg.DelegationMode.ON,
    )
    child = deleg.build_spawn_delegation(
        agent_name="sub",
        run_id="run-2",
        principal="user:alice",
        parent_chain=parent.chain,
        mode=deleg.DelegationMode.ON,
    )
    assert child.chain[0] == "user:alice"
    assert parent.agent_instance_id in child.chain
    assert child.chain[-1] == "agent:sub:run-2"


@pytest.mark.parametrize(
    "mode,emits_chain",
    [
        (deleg.DelegationMode.ON, True),
        (deleg.DelegationMode.WARN, False),
        (deleg.DelegationMode.OFF, False),
    ],
)
def test_envelope_chain_only_in_on_mode(mode, emits_chain: bool) -> None:
    from agent_utilities.knowledge_graph.core.session import GraphSession

    d = deleg.build_spawn_delegation(
        agent_name="researcher", run_id="run-1", principal="user:alice", mode=mode
    )
    context = {"principal": "user:alice", "agent_id": "user:alice", "delegation": []}
    with deleg.use_delegation(d):
        GraphSession._apply_spawn_delegation(context, "user:alice")
    if emits_chain:
        assert context["agent_id"] == "agent:researcher:run-1"
        assert context["delegation"] == list(d.chain)
    else:
        assert context["agent_id"] == "user:alice"
        assert context["delegation"] == []


def test_emitted_chain_passes_engine_wire_validation() -> None:
    """The emitted envelope must satisfy the real engine client's validation."""
    epistemic_graph_client = pytest.importorskip("epistemic_graph.client")
    from agent_utilities.knowledge_graph.core.session import GraphSession

    d = deleg.build_spawn_delegation(
        agent_name="researcher",
        run_id="run-1",
        principal="user:alice",
        mode=deleg.DelegationMode.ON,
    )
    envelope = {
        "principal": "user:alice",
        "tenant": "homelab",
        "audience": "epistemic-graph",
        "agent_id": "user:alice",
        "roles": [],
        "scopes": [],
        "policy_version": "p1",
        "delegation": [],
    }
    with deleg.use_delegation(d):
        GraphSession._apply_spawn_delegation(envelope, "user:alice")
    # Raises ValueError if the chain is malformed — proves wire-correctness.
    epistemic_graph_client.validate_request_context(envelope)


def test_spawn_cannot_forge_a_foreign_principal() -> None:
    from agent_utilities.knowledge_graph.core.session import GraphSession

    d = deleg.build_spawn_delegation(
        agent_name="researcher",
        run_id="run-1",
        principal="user:alice",
        mode=deleg.DelegationMode.ON,
    )
    context = {"principal": "user:eve", "agent_id": "user:eve", "delegation": []}
    with deleg.use_delegation(d):
        GraphSession._apply_spawn_delegation(context, "user:eve")
    assert context["delegation"] == []  # principal mismatch → stays legacy


# ---------------------------------------------------------------------------
# Decision 5 / W2.1-1 — the oidc_token forward (SpawnDelegation -> envelope)
# ---------------------------------------------------------------------------


def test_apply_spawn_delegation_forwards_oidc_token_when_enforced() -> None:
    from agent_utilities.knowledge_graph.core.session import GraphSession

    d = deleg.build_spawn_delegation(
        agent_name="researcher",
        run_id="run-1",
        principal="user:alice",
        mode=deleg.DelegationMode.ON,
        oidc_token="eyJhbGciOiJSUzI1NiJ9.fixture.sig",  # sanitizer:ignore — synthetic fixture JWT, not a real token
    )
    context = {"principal": "user:alice", "agent_id": "user:alice", "delegation": []}
    with deleg.use_delegation(d):
        GraphSession._apply_spawn_delegation(context, "user:alice")
    assert context["oidc_token"] == "eyJhbGciOiJSUzI1NiJ9.fixture.sig"


def test_apply_spawn_delegation_omits_oidc_token_when_delegation_carries_none() -> None:
    """No `SpawnDelegation.oidc_token` set -> no `oidc_token` key at all (not a
    ``None``), so a delegation predating decision 5 forwards byte-for-byte
    as before."""
    from agent_utilities.knowledge_graph.core.session import GraphSession

    d = deleg.build_spawn_delegation(
        agent_name="researcher",
        run_id="run-1",
        principal="user:alice",
        mode=deleg.DelegationMode.ON,
    )
    assert d.oidc_token is None
    context = {"principal": "user:alice", "agent_id": "user:alice", "delegation": []}
    with deleg.use_delegation(d):
        GraphSession._apply_spawn_delegation(context, "user:alice")
    assert "oidc_token" not in context


@pytest.mark.parametrize(
    "mode",
    [deleg.DelegationMode.WARN, deleg.DelegationMode.OFF],
)
def test_apply_spawn_delegation_never_forwards_oidc_token_unless_enforced(mode) -> None:
    """`oidc_token` is a delegation-authority claim, exactly like `agent_id`/
    `delegation` above -- it must never leak onto the wire outside `on` mode."""
    from agent_utilities.knowledge_graph.core.session import GraphSession

    d = deleg.build_spawn_delegation(
        agent_name="researcher",
        run_id="run-1",
        principal="user:alice",
        mode=mode,
        oidc_token="eyJhbGciOiJSUzI1NiJ9.fixture.sig",  # sanitizer:ignore — synthetic fixture JWT, not a real token
    )
    context = {"principal": "user:alice", "agent_id": "user:alice", "delegation": []}
    with deleg.use_delegation(d):
        GraphSession._apply_spawn_delegation(context, "user:alice")
    assert "oidc_token" not in context


def test_emitted_oidc_token_passes_engine_wire_validation() -> None:
    """The oidc_token-carrying envelope must satisfy the real engine client's
    validation, mirroring `test_emitted_chain_passes_engine_wire_validation`
    above for decision 5's new claim.

    Cross-repo staging note (W2.1-1): this asserts against WHATEVER
    ``epistemic_graph`` package is installed in this environment. Until the
    eg-side ``RequestContextClaims.oidc_token`` claim ships and is installed
    here, the currently-installed client legitimately rejects it as an
    unsupported claim — skip (not fail) in that window rather than blocking
    unrelated au work ahead of the eg/au integration merge.
    """
    epistemic_graph_client = pytest.importorskip("epistemic_graph.client")
    if (
        "oidc_token"
        not in getattr(
            epistemic_graph_client, "RequestContextClaims", object
        ).__annotations__
    ):
        pytest.skip(
            "installed epistemic_graph.client predates the oidc_token claim "
            "(W2.1-1) — will activate once the eg/au integration merge lands"
        )
    from agent_utilities.knowledge_graph.core.session import GraphSession

    d = deleg.build_spawn_delegation(
        agent_name="researcher",
        run_id="run-1",
        principal="user:alice",
        mode=deleg.DelegationMode.ON,
        oidc_token="eyJhbGciOiJSUzI1NiJ9.fixture.sig",  # sanitizer:ignore — synthetic fixture JWT, not a real token
    )
    envelope = {
        "principal": "user:alice",
        "tenant": "homelab",
        "audience": "epistemic-graph",
        "agent_id": "user:alice",
        "roles": [],
        "scopes": [],
        "policy_version": "p1",
        "delegation": [],
    }
    with deleg.use_delegation(d):
        GraphSession._apply_spawn_delegation(envelope, "user:alice")
    assert envelope["oidc_token"] == "eyJhbGciOiJSUzI1NiJ9.fixture.sig"
    # Raises ValueError if the claim is rejected — proves wire-correctness.
    validated = epistemic_graph_client.validate_request_context(envelope)
    assert validated["oidc_token"] == "eyJhbGciOiJSUzI1NiJ9.fixture.sig"


# ---------------------------------------------------------------------------
# Decision 4 — ceiling intersection (a spawn can never exceed its principal)
# ---------------------------------------------------------------------------


class _State:
    def __init__(self, allowed, ceiling) -> None:
        self.invoker_allowed_tools = allowed
        self.invoker_capability_ceiling = ceiling


class _Tool:
    def __init__(self, name: str) -> None:
        self.__name__ = name


def test_ceiling_narrows_in_on_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_utilities.graph.executor import apply_tool_scope

    _mode(monkeypatch, "on")
    state = _State(["a", "b", "c"], ["a", "b"])
    scoped, _ = apply_tool_scope(state, [_Tool("a"), _Tool("b"), _Tool("c")], [])
    assert sorted(t.__name__ for t in scoped) == ["a", "b"]


def test_ceiling_violation_denied_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """ACCEPTANCE: ceiling-violation denied — a full out-of-ceiling request must NOT open up."""
    from agent_utilities.graph.executor import apply_tool_scope

    _mode(monkeypatch, "on")
    state = _State(["forbidden_tool"], ["kg:read"])
    with pytest.raises(RuntimeError, match="ceiling denied EVERY"):
        apply_tool_scope(state, [_Tool("forbidden_tool")], [])


def test_warn_mode_observes_but_does_not_narrow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.graph.executor import apply_tool_scope

    _mode(monkeypatch, "warn")
    state = _State(["a", "b", "c"], ["a"])  # b,c exceed the ceiling
    scoped, _ = apply_tool_scope(state, [_Tool("a"), _Tool("b"), _Tool("c")], [])
    assert sorted(t.__name__ for t in scoped) == ["a", "b", "c"]  # unchanged


def test_unrestricted_principal_is_not_narrowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mode(monkeypatch, "on")
    decision = deleg.enforce_ceiling(
        ["a", "b"], ["kg:admin"], mode=deleg.DelegationMode.ON
    )
    assert decision.effective == ("a", "b")
    assert decision.denied == ()


def test_no_ceiling_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_utilities.graph.executor import _intersect_principal_ceiling

    _mode(monkeypatch, "on")
    state = _State(["a", "b"], None)
    assert _intersect_principal_ceiling(state, ["a", "b"], [], []) == ["a", "b"]


# ---------------------------------------------------------------------------
# Decision 6 — revocation: expired spawn dies at its next lease renewal
# ---------------------------------------------------------------------------


def test_is_delegation_live_false_when_run_token_expired(
    monkeypatch: pytest.MonkeyPatch, _secret: None
) -> None:
    _mode(monkeypatch, "on")
    expired = rt.mint_token("run:e", ttl_seconds=-5)
    d = deleg.build_spawn_delegation(
        agent_name="a",
        run_id="run:e",
        principal="user:bob",
        run_token=expired,
        mode=deleg.DelegationMode.ON,
    )
    assert deleg.is_delegation_live(d) is False


def test_is_delegation_live_true_when_fresh(
    monkeypatch: pytest.MonkeyPatch, _secret: None
) -> None:
    _mode(monkeypatch, "on")
    fresh = rt.mint_token("run:f", ttl_seconds=300)
    d = deleg.build_spawn_delegation(
        agent_name="a",
        run_id="run:f",
        principal="user:bob",
        run_token=fresh,
        mode=deleg.DelegationMode.ON,
    )
    assert deleg.is_delegation_live(d) is True


def test_heartbeat_denied_on_expiry_in_on_mode(
    monkeypatch: pytest.MonkeyPatch, _secret: None
) -> None:
    """ACCEPTANCE: expired-token spawn dies at next renewal — on mode fails the renewal."""
    from agent_utilities.orchestration.work_item import _delegation_still_live

    _mode(monkeypatch, "on")
    d = deleg.build_spawn_delegation(
        agent_name="a",
        run_id="run:h",
        principal="user:bob",
        run_token=rt.mint_token("run:h", ttl_seconds=-5),
        mode=deleg.DelegationMode.ON,
    )
    with deleg.use_delegation(d):
        assert _delegation_still_live() is False


def test_heartbeat_renews_in_warn_mode_despite_expiry(
    monkeypatch: pytest.MonkeyPatch, _secret: None
) -> None:
    from agent_utilities.orchestration.work_item import _delegation_still_live

    _mode(monkeypatch, "warn")
    d = deleg.build_spawn_delegation(
        agent_name="a",
        run_id="run:h",
        principal="user:bob",
        run_token=rt.mint_token("run:h", ttl_seconds=-5),
        mode=deleg.DelegationMode.WARN,
    )
    with deleg.use_delegation(d):
        assert _delegation_still_live() is True  # logs, but renews (legacy)


def test_heartbeat_live_without_delegation() -> None:
    from agent_utilities.orchestration.work_item import _delegation_still_live

    assert _delegation_still_live() is True


# ---------------------------------------------------------------------------
# Decision 6 — provenance: chain stamped on the RunTrace (principal opaque)
# ---------------------------------------------------------------------------


def test_run_trace_stamps_the_delegation_chain() -> None:
    """ACCEPTANCE: chain visible end-to-end in provenance."""
    from agent_utilities.orchestration.agent_runner import _stamp_run_identity

    d = deleg.build_spawn_delegation(
        agent_name="researcher",
        run_id="run-1",
        principal="user:alice",
        mode=deleg.DelegationMode.ON,
    )
    props: dict = {}
    _stamp_run_identity(props, delegation=d)
    assert props["delegation_agent_instance"] == "agent:researcher:run-1"
    assert props["delegation_mode"] == "on"
    chain = props["delegation_chain"]
    assert chain[-1] == "agent:researcher:run-1"  # agent id verbatim
    assert chain[0] != "user:alice"  # ultimate principal referenced opaquely


# ---------------------------------------------------------------------------
# End-to-end integration — the run_agent spawn-delegation builder
# ---------------------------------------------------------------------------


def test_prepare_spawn_delegation_off_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.orchestration.agent_runner import _prepare_spawn_delegation

    _mode(monkeypatch, "off")
    config: dict = {}
    assert _prepare_spawn_delegation("researcher", "run-1", config) is None
    assert "invoker_capability_ceiling" not in config


@pytest.mark.parametrize("mode", ["warn", "on"])
def test_prepare_spawn_delegation_builds_pipeline(
    monkeypatch: pytest.MonkeyPatch, _secret: None, mode: str
) -> None:
    from agent_utilities.orchestration.agent_runner import _prepare_spawn_delegation

    _mode(monkeypatch, mode)
    config = {"invoker_allowed_tools": ["search_nodes"]}
    d = _prepare_spawn_delegation("researcher", "run-1", config)
    assert d is not None
    assert d.mode.value == mode
    assert d.agent_instance_id == "agent:researcher:run-1"
    assert d.chain[0] == d.principal and d.chain[-1] == d.agent_instance_id
    assert d.run_token  # minted (endpoint scope derived from allowed_tools)
    decoded = rt.decode_token(d.run_token)
    assert "search_nodes" in decoded.endpoints
