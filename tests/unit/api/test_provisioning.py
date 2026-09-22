"""Fail-closed contracts for AU's GraphOS provisioning authority."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import re
from typing import Any, cast

import pytest
from epistemic_graph.generated.connector_pack import (
    AgentLibraryMutationContext,
    McpCatalogSnapshotBinding,
)

from agent_utilities.api.provisioning import (
    PackImportAuthorityResolver,
    ProvisioningAuthorityError,
    pack_import_authority,
)
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    ScopeError,
    use_session,
)
from agent_utilities.orchestration.action_policy import (
    ActionDecision,
    ActionRequest,
    PolicyDisposition,
    PolicyReceipt,
)
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext


def _session(
    *, scopes: frozenset[str] = frozenset({"agent:pack-control"})
) -> GraphSession:
    actor = ActorContext(
        actor_id="service:graph-os",
        actor_type=ActorType.AUTOMATED_SERVICE,
        tenant_id="tenant-a",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant="tenant-a",
        scopes=scopes,
        graph="tenant-a",
        audience="graph-os",
        policy_version="policy-a",
    )


class _Policy:
    def __init__(self, *, allow: bool) -> None:
        self.allow = allow
        self.requests: list[ActionRequest] = []

    def decide(self, request: ActionRequest) -> ActionDecision:
        self.requests.append(request)
        receipt = PolicyReceipt(
            receipt_id="action_decision:test",
            request_digest=request.digest(),
            disposition=(
                PolicyDisposition.APPROVE if self.allow else PolicyDisposition.DENY
            ),
            policy_origin="test",
        )
        return ActionDecision(
            decision="allow" if self.allow else "deny",
            tier="auto" if self.allow else "forbidden",
            request=request,
            receipt=receipt,
        )


class _AllowWithoutReceipt:
    def decide(self, request: ActionRequest) -> ActionDecision:
        return ActionDecision(decision="allow", tier="auto", request=request)


def _install_policy(monkeypatch: pytest.MonkeyPatch, policy: Any) -> None:
    monkeypatch.setattr(
        "agent_utilities.api.provisioning.get_action_policy",
        lambda _engine: policy,
    )


def _binding() -> Any:
    digest = "ab" * 32
    return McpCatalogSnapshotBinding(
        authorization_scope_digest=digest,
        catalog_generation=3,
        child_connection_generation=4,
        configuration_revision=7,
        snapshot_digest=digest,
    )


def _resolve_pack_import(
    resolver: PackImportAuthorityResolver,
) -> tuple[McpCatalogSnapshotBinding, AgentLibraryMutationContext]:
    async def await_resolver() -> tuple[
        McpCatalogSnapshotBinding, AgentLibraryMutationContext
    ]:
        return await resolver("mcp-main")

    return asyncio.run(await_resolver())


def _assert_policy_refuses_before_binding_load(
    monkeypatch: pytest.MonkeyPatch, policy: Any
) -> None:
    session = _session()
    binding_loaded = False
    _install_policy(monkeypatch, policy)

    def binding() -> Any:
        nonlocal binding_loaded
        binding_loaded = True
        return _binding()

    with use_session(session):
        resolver = pack_import_authority(
            object(),
            session,
            catalog_binding=binding,
            serving_principal=lambda: "principal:sha256:" + ("ab" * 32),
        )
        with pytest.raises(ProvisioningAuthorityError, match="not authorized"):
            _resolve_pack_import(resolver)
    assert not binding_loaded


def test_policy_issued_resolver_returns_generated_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    owner = "principal:sha256:" + ("cd" * 32)
    session = _session()
    policy = _Policy(allow=True)
    _install_policy(monkeypatch, policy)
    with use_session(session):
        resolver = pack_import_authority(
            object(),
            session,
            catalog_binding=lambda: binding,
            serving_principal=lambda: owner,
        )
        actual_binding, context = _resolve_pack_import(resolver)

    assert actual_binding is binding
    assert context.principal == owner
    assert context.tenant_id == "tenant-a"
    assert context.purpose_id == "connector-pack:import"
    assert context.actor_scope == context.caller_principal
    assert context.caller_principal == (
        "principal:sha256:" + hashlib.sha256(b"service:graph-os").hexdigest()
    )
    assert re.fullmatch(r"[0-9a-f]{64}", context.attempt_nonce)
    assert context.policy_decision_id == "action_decision:test"
    assert policy.requests[0].kind == "connector_pack_import"
    assert policy.requests[0].target == "mcp-main"


def test_resolver_requires_pack_control_scope_before_providers() -> None:
    session = _session(scopes=frozenset())
    called = False

    def binding() -> Any:
        nonlocal called
        called = True
        return _binding()

    with use_session(session), pytest.raises(ScopeError):
        pack_import_authority(
            object(),
            session,
            catalog_binding=binding,
            serving_principal=lambda: "principal:sha256:" + ("ab" * 32),
        )
    assert not called


def test_resolver_does_not_accept_a_caller_supplied_policy() -> None:
    assert "policy" not in inspect.signature(pack_import_authority).parameters


def test_resolver_refuses_without_effect_authorizing_policy_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_policy_refuses_before_binding_load(monkeypatch, _Policy(allow=False))


def test_resolver_refuses_an_allow_decision_without_a_policy_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_policy_refuses_before_binding_load(monkeypatch, _AllowWithoutReceipt())


def test_resolver_refuses_malformed_authoritative_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    _install_policy(monkeypatch, _Policy(allow=True))
    with use_session(session):
        resolver = pack_import_authority(
            object(),
            session,
            catalog_binding=lambda: _binding(),
            serving_principal=lambda: "service:graph-os",
        )
        with pytest.raises(ProvisioningAuthorityError, match="opaque"):
            _resolve_pack_import(resolver)


def test_resolver_refuses_a_non_generated_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    _install_policy(monkeypatch, _Policy(allow=True))
    with use_session(session):
        resolver = pack_import_authority(
            object(),
            session,
            catalog_binding=lambda: cast(McpCatalogSnapshotBinding, object()),
            serving_principal=lambda: "principal:sha256:" + ("ab" * 32),
        )
        with pytest.raises(ProvisioningAuthorityError, match="generated"):
            _resolve_pack_import(resolver)
