"""Actor-only authorization seams reject expired bearer authority."""

from __future__ import annotations

import time

import pytest

from agent_utilities.knowledge_graph.actions import (
    ActionExecutor,
    ActionRegistry,
    OntologyAction,
)
from agent_utilities.knowledge_graph.core import secured_reads, tenant_sharing
from agent_utilities.knowledge_graph.ontology import permissioning
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import (
    ActorContext,
    CredentialExpiredError,
    CredentialLease,
)
from agent_utilities.security.entitlements import identity_scoped_resources
from agent_utilities.security.permissions_kernel import PermissionsKernel
from agent_utilities.usage.authorization import is_usage_admin


def _expired_actor(*roles: str, lease: CredentialLease | None = None) -> ActorContext:
    return ActorContext(
        actor_id="principal:test",
        actor_type=ActorType.AI_AGENT,
        roles=roles,
        tenant_id="tenant-test",
        authenticated=True,
        credential_expires_at=int(time.time()) - 1,
        credential_lease=lease,
    )


@pytest.mark.parametrize(
    "authorize",
    [
        lambda actor: identity_scoped_resources(
            "k8s", ["prod"], actor=actor
        ),
        lambda actor: secured_reads.scope("MATCH (n) RETURN n", actor),
        lambda actor: permissioning.redact_object({"id": "node"}, actor),
        lambda actor: tenant_sharing.visibility_predicate(actor),
        is_usage_admin,
    ],
    ids=["entitlements", "secured-reads", "permissioning", "tenant-sharing", "usage"],
)
def test_explicit_actor_authorization_rejects_expired_credential(authorize) -> None:
    with pytest.raises(CredentialExpiredError):
        authorize(_expired_actor("admin", "k8s:*"))


def test_actor_authorization_observes_renewable_lease_expiry() -> None:
    lease = CredentialLease(int(time.time()) + 60)
    actor = _expired_actor("admin", lease=lease)
    assert is_usage_admin(actor) is True

    lease.renew(int(time.time()) - 1)

    with pytest.raises(CredentialExpiredError):
        is_usage_admin(actor)


def test_ontology_action_handler_does_not_run_for_expired_actor() -> None:
    invoked = False

    def handler(_params):
        nonlocal invoked
        invoked = True

    registry = ActionRegistry()
    registry.register(
        OntologyAction(
            name="test.read",
            verb="read",
            required_capability="kg_read",
            acts_on=["concept"],
        ),
        handler,
    )
    executor = ActionExecutor(
        registry,
        kernel=PermissionsKernel(
            signing_key="test-signing-authority-material-32b"
        ),
        persist=False,
    )

    with pytest.raises(CredentialExpiredError):
        executor.execute("test.read", _expired_actor("kg_read"), {})
    assert invoked is False


def test_ontology_action_rejects_unverified_role_claims() -> None:
    invoked = False

    def handler(_params):
        nonlocal invoked
        invoked = True

    registry = ActionRegistry()
    registry.register(
        OntologyAction(
            name="test.read",
            verb="read",
            required_capability="admin",
            acts_on=["concept"],
        ),
        handler,
    )
    executor = ActionExecutor(
        registry,
        kernel=PermissionsKernel(
            signing_key="test-signing-authority-material-32b"
        ),
        persist=False,
    )

    with pytest.raises(PermissionError, match="verified tenant"):
        executor.execute(
            "test.read",
            ActorContext(actor_id="caller", roles=("admin",)),
            {},
        )
    assert invoked is False
