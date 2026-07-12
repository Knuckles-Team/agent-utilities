"""Identity-scoped resource auto-load (CONCEPT:AU-OS.identity.identity-scoped-resource-autoload).

The caller's normalized capabilities decide which backend resources auto-load —
proven end-to-end from an Okta groups claim and a Keycloak roles claim resolving
to the same entitled kube contexts.
"""

from __future__ import annotations

import pytest

from agent_utilities.security.brain_context import ActorContext, use_actor
from agent_utilities.security.entitlements import (
    entitled_resources,
    grants_all_in_namespace,
    identity_scoped_resources,
    is_entitled,
)
from agent_utilities.security.identity import base_capabilities, normalize_identity


class TestEntitledResources:
    @pytest.mark.concept("CONCEPT:AU-OS.identity.identity-scoped-resource-autoload")
    def test_namespaced_caps_select_subset_of_available(self):
        caps = ["k8s:prod", "k8s:staging"]
        assert entitled_resources(caps, "k8s", ["prod", "staging", "dev"]) == (
            "prod",
            "staging",
        )

    @pytest.mark.concept("CONCEPT:AU-OS.identity.identity-scoped-resource-autoload")
    def test_bare_group_name_matches_resource_zero_config(self):
        # An Okta group / Keycloak role literally named after the context.
        caps = ["prod"]
        assert entitled_resources(caps, "k8s", ["prod", "dev"]) == ("prod",)

    @pytest.mark.concept("CONCEPT:AU-OS.identity.identity-scoped-resource-autoload")
    def test_namespace_wildcard_grants_all_available(self):
        assert entitled_resources(["k8s:*"], "k8s", ["a", "b"]) == ("a", "b")

    @pytest.mark.concept("CONCEPT:AU-OS.identity.identity-scoped-resource-autoload")
    def test_admin_super_cap_grants_all(self):
        assert entitled_resources(["admin"], "k8s", ["a", "b"]) == ("a", "b")

    @pytest.mark.concept("CONCEPT:AU-OS.identity.identity-scoped-resource-autoload")
    def test_fail_closed_when_no_match(self):
        assert entitled_resources(["other:x"], "k8s", ["a", "b"]) == ()

    @pytest.mark.concept("CONCEPT:AU-OS.identity.identity-scoped-resource-autoload")
    def test_no_catalog_returns_named_resources(self):
        assert entitled_resources(["k8s:prod", "ssh:r820"], "k8s") == ("prod",)

    @pytest.mark.concept("CONCEPT:AU-OS.identity.identity-scoped-resource-autoload")
    def test_is_entitled(self):
        assert is_entitled(["k8s:prod"], "k8s", "prod")
        assert not is_entitled(["k8s:prod"], "k8s", "dev")
        assert is_entitled(["admin"], "k8s", "anything")

    @pytest.mark.concept("CONCEPT:AU-OS.identity.identity-scoped-resource-autoload")
    def test_grants_all_in_namespace(self):
        assert grants_all_in_namespace(["k8s:all"], "k8s")
        assert grants_all_in_namespace(["system"], "k8s")
        assert not grants_all_in_namespace(["k8s:prod"], "k8s")


class TestOktaKeycloakInterchangeableAutoload:
    @pytest.mark.concept("CONCEPT:AU-OS.identity.identity-scoped-resource-autoload")
    def test_okta_groups_and_keycloak_roles_autoload_same_contexts(self):
        """The end-to-end contract: same kube contexts auto-load regardless of IdP."""
        available = ["prod", "staging", "dev"]
        okta = base_capabilities(
            normalize_identity(
                {"sub": "u", "iss": "https://x.okta.com", "groups": ["k8s:prod", "k8s:staging"]}
            )
        )
        keycloak = base_capabilities(
            normalize_identity(
                {
                    "sub": "u",
                    "iss": "https://kc/realms/x",
                    "realm_access": {"roles": ["k8s:prod", "k8s:staging"]},
                }
            )
        )
        assert entitled_resources(okta, "k8s", available) == entitled_resources(
            keycloak, "k8s", available
        )
        assert entitled_resources(okta, "k8s", available) == ("prod", "staging")


class TestIdentityScopedResources:
    @pytest.mark.concept("CONCEPT:AU-OS.identity.identity-scoped-resource-autoload")
    def test_authenticated_caller_scopes_to_entitled(self):
        actor = ActorContext(
            actor_id="user:ada",
            roles=("k8s:prod",),
            authenticated=True,
        )
        with use_actor(actor):
            assert identity_scoped_resources("k8s", ["prod", "staging", "dev"]) == (
                "prod",
            )

    @pytest.mark.concept("CONCEPT:AU-OS.identity.identity-scoped-resource-autoload")
    def test_unauthenticated_system_actor_sees_all_backcompat(self):
        # Ambient SYSTEM_ACTOR (admin/system) → all resources: today's behaviour
        # until a real identity with specific groups scopes it down.
        assert identity_scoped_resources("k8s", ["prod", "dev"]) == ("prod", "dev")
