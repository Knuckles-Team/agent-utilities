"""IdP-agnostic identity normalization (CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance).

Proves Okta groups and Keycloak roles are first-class and INTERCHANGEABLE: a
caller granted access via an Okta ``groups`` claim and a caller granted the same
access via a Keycloak ``realm_access.roles`` claim resolve to the same base
capability set, so downstream authorization behaves identically regardless of
provider.
"""

from __future__ import annotations

import pytest

from agent_utilities.security.identity import (
    NormalizedIdentity,
    base_capabilities,
    detect_provider,
    normalize_identity,
)


class TestNormalizeIdentity:
    @pytest.mark.concept("CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance")
    def test_okta_groups_and_keycloak_roles_yield_same_capabilities(self):
        """The interchangeability contract: same access, different provider."""
        okta = normalize_identity(
            {
                "sub": "00u1ada",
                "iss": "https://acme.okta.com/oauth2/default",
                "email": "ada@acme.com",
                "groups": ["kg-admin", "engineering"],
                "tenant_id": "acme",
            }
        )
        keycloak = normalize_identity(
            {
                "sub": "user:ada",
                "iss": "https://kc.acme.com/realms/acme",
                "email": "ada@acme.com",
                "realm_access": {"roles": ["kg-admin", "engineering"]},
                "tid": "acme",
            }
        )
        # Same base capabilities → identical downstream authorization.
        assert base_capabilities(okta) == base_capabilities(keycloak)
        assert set(base_capabilities(okta)) == {"kg-admin", "engineering"}
        # Provider is detected but is informational only.
        assert okta.provider == "okta"
        assert keycloak.provider == "keycloak"
        # Groups are retained distinctly (Okta side) for k8s impersonation.
        assert okta.groups == ("kg-admin", "engineering")

    @pytest.mark.concept("CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance")
    def test_keycloak_client_roles_and_scopes_fold_into_capabilities(self):
        ident = normalize_identity(
            {
                "sub": "svc:ci",
                "realm_access": {"roles": ["realm-role"]},
                "resource_access": {"kg": {"roles": ["client-role"]}},
                "scope": "kg:read kg:write",
            }
        )
        caps = set(base_capabilities(ident))
        assert {"realm-role", "client-role", "kg:read", "kg:write"} <= caps

    @pytest.mark.concept("CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance")
    def test_group_map_translates_opaque_okta_group_ids(self):
        """An opaque Okta group id maps to the same capability a KC role grants."""
        ident = normalize_identity({"sub": "u", "groups": ["0oa9xyz"]})
        mapped = base_capabilities(ident, {"0oa9xyz": ["kg-admin"]})
        assert "kg-admin" in mapped
        # Unmapped groups fall back to their own name (never silently dropped).
        ident2 = normalize_identity({"sub": "u", "groups": ["engineering"]})
        assert "engineering" in base_capabilities(ident2, {"other": ["x"]})

    @pytest.mark.concept("CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance")
    def test_keycloak_group_paths_are_normalized(self):
        ident = normalize_identity({"sub": "u", "groups": ["/engineering/kg-admin"]})
        assert ident.groups == ("engineering/kg-admin",)

    @pytest.mark.concept("CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance")
    def test_order_preserving_dedup(self):
        ident = normalize_identity({"sub": "u", "roles": ["b", "a", "b"]})
        assert ident.roles == ("b", "a")

    @pytest.mark.concept("CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance")
    def test_empty_claims_are_safe(self):
        ident = normalize_identity({})
        assert ident == NormalizedIdentity(subject="jwt")
        assert base_capabilities(ident) == ()

    @pytest.mark.concept("CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance")
    def test_group_dict_entries_reduced_to_name(self):
        ident = normalize_identity({"sub": "u", "groups": [{"name": "kg-admin"}]})
        assert ident.groups == ("kg-admin",)

    @pytest.mark.concept("CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance")
    def test_base_is_a_superset_that_downstream_only_narrows(self):
        """The base is a ceiling: an intersect with a downstream allow-list
        never exceeds it (the inherit-then-restrict invariant)."""
        ident = normalize_identity({"sub": "u", "roles": ["a", "b", "c"]})
        base = set(base_capabilities(ident))
        downstream_allow = {"b", "c", "escalate"}
        effective = base & downstream_allow
        assert effective == {"b", "c"}
        assert "escalate" not in effective  # cannot escalate beyond the base

    def test_detect_provider(self):
        assert detect_provider({"iss": "https://x.okta.com/..."}) == "okta"
        assert detect_provider({"realm_access": {"roles": []}}) == "keycloak"
        assert detect_provider({"iss": "https://idp/"}) == "oidc"
