#!/usr/bin/python
from __future__ import annotations

"""Tests for CONCEPT:AU-OS.config.secrets-authentication — Permissions Kernel."""


import time
from types import SimpleNamespace

import pytest

from agent_utilities.security.permissions_kernel import (
    AgentIdentity,
    AgentPolicy,
    AgentRole,
    AuthDecision,
    PermissionBootstrapError,
    PermissionPolicyError,
    PermissionsKernel,
    resolve_permission_context,
)

TEST_SIGNING_KEY = "test-signing-authority-material-32b"


@pytest.fixture
def kernel() -> PermissionsKernel:
    """Create a kernel with default policies and a fixed signing key."""
    return PermissionsKernel(signing_key=TEST_SIGNING_KEY)


def test_permissions_kernel_requires_explicit_strong_authority() -> None:
    with pytest.raises(TypeError):
        PermissionsKernel()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="32 bytes"):
        PermissionsKernel(signing_key="too-short")


class TestAgentIdentity:
    """Test AgentIdentity model."""

    def test_defaults(self) -> None:
        identity = AgentIdentity(agent_id="test-agent")
        assert identity.role == AgentRole.SPECIALIST
        assert identity.capabilities == []
        assert identity.issued_at > 0

    def test_payload_string(self) -> None:
        import json

        identity = AgentIdentity(
            agent_id="agent-a",
            role=AgentRole.ADMIN,
            capabilities=["code", "deploy"],
            issued_at=1000.0,
        )
        payload = json.loads(identity.payload_string())
        assert payload == {
            "agent_id": "agent-a",
            "capabilities": ["code", "deploy"],
            "expires_at": 0.0,
            "issued_at": 1000.0,
            "role": "admin",
        }


class TestAgentPolicy:
    """Test AgentPolicy model."""

    def test_defaults(self) -> None:
        policy = AgentPolicy(role=AgentRole.SPECIALIST)
        assert policy.allowed_tools == ["*"]
        assert policy.denied_tools == []
        assert policy.max_token_quota == 100_000


class TestIdentityLifecycle:
    """Test identity issuance and verification."""

    def test_issue_identity(self, kernel: PermissionsKernel) -> None:
        identity = kernel.issue_identity("agent-a", AgentRole.SPECIALIST)
        assert identity.agent_id == "agent-a"
        assert identity.role == AgentRole.SPECIALIST
        assert identity.signature != ""

    def test_verify_valid_identity(self, kernel: PermissionsKernel) -> None:
        identity = kernel.issue_identity("agent-a", AgentRole.SPECIALIST)
        assert kernel.verify_identity(identity) is True

    def test_verify_tampered_identity(self, kernel: PermissionsKernel) -> None:
        identity = kernel.issue_identity("agent-a", AgentRole.SPECIALIST)
        # Tamper with the role
        identity.role = AgentRole.ADMIN
        assert kernel.verify_identity(identity) is False

    def test_verify_expired_identity(self, kernel: PermissionsKernel) -> None:
        identity = kernel.issue_identity(
            "agent-a", AgentRole.SPECIALIST, ttl_seconds=0.001
        )
        time.sleep(0.01)  # Wait for expiry
        assert kernel.verify_identity(identity) is False

    def test_verify_tampered_expiry(self, kernel: PermissionsKernel) -> None:
        identity = kernel.issue_identity("agent-a", AgentRole.SPECIALIST)
        identity.expires_at = time.time() + 3_600
        assert kernel.verify_identity(identity) is False

    def test_get_identity(self, kernel: PermissionsKernel) -> None:
        kernel.issue_identity("agent-a", AgentRole.OPERATOR)
        retrieved = kernel.get_identity("agent-a")
        assert retrieved is not None
        assert retrieved.role == AgentRole.OPERATOR

    def test_get_missing_identity(self, kernel: PermissionsKernel) -> None:
        assert kernel.get_identity("nonexistent") is None


class TestAuthorization:
    """Test authorization decisions."""

    def test_admin_allows_everything(self, kernel: PermissionsKernel) -> None:
        identity = kernel.issue_identity("admin-agent", AgentRole.ADMIN)
        assert (
            kernel.authorize_tool(identity, "delete_everything") == AuthDecision.ALLOW
        )
        assert kernel.authorize_tool(identity, "reboot_server") == AuthDecision.ALLOW
        assert kernel.authorize_tool(identity, "read_file") == AuthDecision.ALLOW

    def test_operator_requires_approval_for_destructive(
        self, kernel: PermissionsKernel
    ) -> None:
        identity = kernel.issue_identity("ops-agent", AgentRole.OPERATOR)
        assert kernel.authorize_tool(identity, "read_file") == AuthDecision.ALLOW
        assert (
            kernel.authorize_tool(identity, "delete_user")
            == AuthDecision.REQUIRE_APPROVAL
        )

    def test_specialist_denied_os_operations(self, kernel: PermissionsKernel) -> None:
        identity = kernel.issue_identity("spec-agent", AgentRole.SPECIALIST)
        assert kernel.authorize_tool(identity, "reboot_server") == AuthDecision.DENY
        assert kernel.authorize_tool(identity, "read_file") == AuthDecision.ALLOW

    def test_specialist_requires_approval_for_destructive(
        self, kernel: PermissionsKernel
    ) -> None:
        identity = kernel.issue_identity("spec-agent", AgentRole.SPECIALIST)
        assert (
            kernel.authorize_tool(identity, "delete_item")
            == AuthDecision.REQUIRE_APPROVAL
        )

    def test_sandbox_read_only(self, kernel: PermissionsKernel) -> None:
        identity = kernel.issue_identity("sandbox-agent", AgentRole.SANDBOX)
        assert kernel.authorize_tool(identity, "read_file") == AuthDecision.ALLOW
        assert kernel.authorize_tool(identity, "list_items") == AuthDecision.ALLOW
        assert kernel.authorize_tool(identity, "delete_file") == AuthDecision.DENY

    def test_guest_denied_everything(self, kernel: PermissionsKernel) -> None:
        identity = kernel.issue_identity("guest-agent", AgentRole.GUEST)
        assert kernel.authorize_tool(identity, "read_file") == AuthDecision.DENY
        assert kernel.authorize_tool(identity, "list_items") == AuthDecision.DENY

    def test_invalid_identity_denied(self, kernel: PermissionsKernel) -> None:
        identity = kernel.issue_identity("agent-a", AgentRole.ADMIN)
        identity.signature = "tampered"
        assert kernel.authorize_tool(identity, "read_file") == AuthDecision.DENY

    def test_non_empty_capabilities_are_additional_closed_world_grants(
        self, kernel: PermissionsKernel
    ) -> None:
        identity = kernel.issue_identity(
            "scoped-agent",
            AgentRole.SPECIALIST,
            capabilities=["read_*"],
        )

        assert kernel.authorize_tool(identity, "read_file") == AuthDecision.ALLOW
        assert kernel.authorize_tool(identity, "list_items") == AuthDecision.DENY

    def test_semantic_required_capability_can_satisfy_identity_scope(
        self, kernel: PermissionsKernel
    ) -> None:
        identity = kernel.issue_identity(
            "scoped-agent",
            AgentRole.SPECIALIST,
            capabilities=["kg.read"],
        )

        assert (
            kernel.authorize_tool(
                identity,
                "knowledge_query",
                required_capability="kg.read",
            )
            == AuthDecision.ALLOW
        )

    def test_capability_grant_never_overrides_role_deny(
        self, kernel: PermissionsKernel
    ) -> None:
        identity = kernel.issue_identity(
            "scoped-agent",
            AgentRole.SPECIALIST,
            capabilities=["reboot_*"],
        )

        assert kernel.authorize_tool(identity, "reboot_server") == AuthDecision.DENY


class TestPolicyManagement:
    """Test policy loading and management."""

    def test_default_policies_loaded(self, kernel: PermissionsKernel) -> None:
        policies = kernel.get_policies()
        assert len(policies) == 5  # admin, operator, specialist, sandbox, guest
        roles = {p.role for p in policies}
        assert AgentRole.ADMIN in roles
        assert AgentRole.GUEST in roles

    def test_load_policies_from_file(self, tmp_path) -> None:
        import json

        policy_data = {
            "policies": [
                {
                    "role": "admin",
                    "allowed_tools": ["*"],
                    "denied_tools": [],
                    "require_approval_for": [],
                    "max_token_quota": 999999,
                    "description": "Custom admin",
                }
            ]
        }
        path = tmp_path / "agent_policies.json"
        path.write_text(json.dumps(policy_data))

        kernel = PermissionsKernel(
            signing_key=TEST_SIGNING_KEY, policies_path=str(path)
        )
        policies = kernel.get_policies()
        assert len(policies) == 1
        assert policies[0].max_token_quota == 999999

    def test_missing_configured_policy_fails_closed(self, tmp_path) -> None:
        with pytest.raises(PermissionPolicyError, match="unavailable"):
            PermissionsKernel(
                signing_key=TEST_SIGNING_KEY,
                policies_path=str(tmp_path / "missing.json"),
            )

    def test_malformed_policy_clears_existing_policy_set(
        self, kernel, tmp_path
    ) -> None:
        path = tmp_path / "agent_policies.json"
        path.write_text('{"policies": [{"role": "admin"}]}')

        with pytest.raises(PermissionPolicyError, match="invalid"):
            kernel.load_policies(str(path))

        assert kernel.get_policies() == []

    def test_token_quota_for_role(self, kernel: PermissionsKernel) -> None:
        assert kernel.get_token_quota_for_role(AgentRole.ADMIN) == 500_000
        assert kernel.get_token_quota_for_role(AgentRole.GUEST) == 10_000


class TestPatternMatching:
    """Test glob pattern matching."""

    def test_wildcard(self) -> None:
        assert PermissionsKernel._matches_patterns("anything", ["*"]) is True

    def test_prefix_glob(self) -> None:
        assert PermissionsKernel._matches_patterns("delete_user", ["delete_*"]) is True
        assert PermissionsKernel._matches_patterns("read_user", ["delete_*"]) is False

    def test_contains_glob(self) -> None:
        assert (
            PermissionsKernel._matches_patterns("do_delete_now", ["*delete*"]) is True
        )

    def test_case_insensitive(self) -> None:
        assert PermissionsKernel._matches_patterns("delete_user", ["*DELETE*"]) is True

    def test_no_patterns(self) -> None:
        assert PermissionsKernel._matches_patterns("anything", []) is False


class TestPermissionContextBootstrap:
    def test_resolves_reference_and_returns_verified_pair(self) -> None:
        cfg = SimpleNamespace(
            permissions_signing_key_ref="env://PERMISSION_TEST_KEY",
            agent_policies_path=None,
        )
        context = resolve_permission_context(
            cfg,
            secret_resolver=lambda _reference: TEST_SIGNING_KEY,
        )

        assert context is not None
        assert context.kernel.verify_identity(context.identity)
        assert context.identity.agent_id.startswith("agent:")
        assert "graph-runtime" not in context.identity.agent_id

    def test_bootstrap_identity_is_stable_opaque_and_capability_bound(self) -> None:
        cfg = SimpleNamespace(
            permissions_signing_key_ref="env://PERMISSION_TEST_KEY",
            agent_policies_path=None,
        )

        first = resolve_permission_context(
            cfg,
            agent_subject="configured-display-name",
            capabilities=("read_*",),
            secret_resolver=lambda _reference: TEST_SIGNING_KEY,
        )
        second = resolve_permission_context(
            cfg,
            agent_subject="configured-display-name",
            capabilities=("read_*",),
            secret_resolver=lambda _reference: TEST_SIGNING_KEY,
        )
        other = resolve_permission_context(
            cfg,
            agent_subject="different-display-name",
            capabilities=("read_*",),
            secret_resolver=lambda _reference: TEST_SIGNING_KEY,
        )

        assert first is not None and second is not None and other is not None
        assert first.identity.agent_id == second.identity.agent_id
        assert first.identity.agent_id != other.identity.agent_id
        assert "configured-display-name" not in first.identity.agent_id
        assert first.identity.capabilities == ["read_*"]

    def test_missing_reference_fails_without_process_authority(self) -> None:
        cfg = SimpleNamespace(
            permissions_signing_key_ref=None,
            agent_policies_path=None,
        )

        with pytest.raises(PermissionBootstrapError, match="reference is required"):
            resolve_permission_context(cfg)

    def test_injected_context_must_verify(self, kernel: PermissionsKernel) -> None:
        other = PermissionsKernel(signing_key="different-signing-authority-key-32b")
        identity = other.issue_identity("graph-runtime")

        with pytest.raises(PermissionBootstrapError, match="invalid"):
            resolve_permission_context(
                SimpleNamespace(),
                permissions_kernel=kernel,
                agent_identity=identity,
            )
