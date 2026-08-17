"""Shape contract for the ONE live verified-identity carrier (GOC-15,
CONCEPT:AU-OS.identity.verified-carrier-contract).

``CARRIER_CLAIM_FIELDS``/``OPTIONAL_CARRIER_CLAIM_FIELDS``/``validate_carrier_claims``
pin the exact key set ``GraphSession.engine_verified_context()`` emits and
``crates/eg-types/src/acl.rs::RequestContextClaims`` deserializes. This proves
the fail-closed behavior against known-bad shapes (missing required field,
unrecognized field, empty required field, wrong-typed list field) — not just
that a good carrier validates.
"""

from __future__ import annotations

import pytest

from agent_utilities.security.request_identity import (
    CARRIER_CLAIM_FIELDS,
    OPTIONAL_CARRIER_CLAIM_FIELDS,
    validate_carrier_claims,
)


def _good_claims(**overrides: object) -> dict[str, object]:
    claims: dict[str, object] = {
        "principal": "principal:opaque-123",
        "tenant": "tenant-a",
        "audience": "agent-services",
        "agent_id": "principal:opaque-123",
        "roles": ["kg:read"],
        "scopes": ["kg:read"],
        "delegation": [],
        "policy_version": "policy-1",
    }
    claims.update(overrides)
    return claims


def test_the_two_field_sets_are_disjoint_and_match_the_wire_contract():
    """Pins the exact 8 required + 3 optional keys documented in
    docs/architecture/verified-identity-carrier-contract.md — a change here
    is a wire-contract change, not a refactor."""
    assert CARRIER_CLAIM_FIELDS == {
        "principal",
        "tenant",
        "audience",
        "agent_id",
        "roles",
        "scopes",
        "delegation",
        "policy_version",
    }
    assert OPTIONAL_CARRIER_CLAIM_FIELDS == {"node", "priority", "oidc_token"}
    assert CARRIER_CLAIM_FIELDS.isdisjoint(OPTIONAL_CARRIER_CLAIM_FIELDS)


def test_a_well_formed_carrier_validates():
    validate_carrier_claims(_good_claims())


def test_a_well_formed_carrier_with_every_optional_claim_validates():
    validate_carrier_claims(
        _good_claims(
            node="node-1", priority="interactive", oidc_token="exchanged-token"
        )
    )


@pytest.mark.parametrize("field", sorted(CARRIER_CLAIM_FIELDS))
def test_a_missing_required_field_is_rejected(field):
    claims = _good_claims()
    del claims[field]
    with pytest.raises(ValueError, match="missing required field"):
        validate_carrier_claims(claims)


@pytest.mark.parametrize(
    "field", ["principal", "tenant", "audience", "agent_id", "policy_version"]
)
def test_an_empty_scalar_field_is_rejected(field):
    with pytest.raises(ValueError, match="non-empty string"):
        validate_carrier_claims(_good_claims(**{field: ""}))


@pytest.mark.parametrize(
    "field", ["principal", "tenant", "audience", "agent_id", "policy_version"]
)
def test_a_whitespace_only_scalar_field_is_rejected(field):
    with pytest.raises(ValueError, match="non-empty string"):
        validate_carrier_claims(_good_claims(**{field: "   "}))


@pytest.mark.parametrize("field", ["roles", "scopes", "delegation"])
def test_a_non_list_list_field_is_rejected(field):
    with pytest.raises(ValueError, match="list of strings"):
        validate_carrier_claims(_good_claims(**{field: "kg:read"}))


@pytest.mark.parametrize("field", ["roles", "scopes", "delegation"])
def test_a_list_field_with_a_non_string_item_is_rejected(field):
    with pytest.raises(ValueError, match="list of strings"):
        validate_carrier_claims(_good_claims(**{field: ["kg:read", 7]}))


def test_an_unrecognized_field_is_rejected():
    """A claims dict carrying a key `RequestContextClaims`'s
    `#[serde(deny_unknown_fields)]` would reject on the wire must already be
    rejected here — a caller-controlled extra key (e.g. a smuggled
    `display_name` or `path`) is exactly what the engine-side struct refuses."""
    with pytest.raises(ValueError, match="unrecognized field"):
        validate_carrier_claims(_good_claims(display_name="Alice"))


def test_a_non_dict_is_rejected():
    with pytest.raises(ValueError, match="must be a dict"):
        validate_carrier_claims(["principal:opaque-123"])  # type: ignore[arg-type]


def test_a_cross_tenant_substitution_is_still_shape_valid_by_itself():
    """`validate_carrier_claims` checks SHAPE only — it is not the tenant/
    audience verification boundary (that is `GraphSession.__post_init__` on
    the AU side and `server::auth::bind_verified_identity`/
    `authenticated_iceberg_bearer` on the EG side, both already proven
    elsewhere: `tests/unit/test_graph_session.py`'s cross-tenant
    parametrization and EG's `mod iceberg_bearer_carrier` negative matrix).
    A shape-only validator that ALSO rejected on cross-tenant content would
    be silently duplicating those checks with different logic — this test
    documents the boundary rather than assuming it away."""
    validate_carrier_claims(_good_claims(tenant="tenant-attacker"))
