"""Fail-closed Cypher scoping-variable resolution (CONCEPT:AU-KG.backend.company-brain-write-guard).

Proves the fix for the landmine documented in ``cypher_scoping.py``: both
``TenancyManager.scope_cypher_query`` and ``tenant_sharing.apply_visibility``
used to hardcode the literal Cypher variable ``n`` when injecting a
tenant/visibility predicate, so a query binding its node under a different
variable name (``MATCH (x:Entity) RETURN x``) got a predicate referencing an
unbound ``n`` — silently wrong, and on a lenient backend, silently zero rows
instead of a raised error. These tests exercise the pure-Python scoping logic
directly (no compiled epistemic-graph backend required).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import tenant_sharing as ts
from agent_utilities.knowledge_graph.core.company_brain import TenancyManager
from agent_utilities.knowledge_graph.core.cypher_scoping import (
    UnscopableQueryError,
    first_bound_node_variable,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


def _user(actor_id="alice", tenant="acme", roles=()):
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.HUMAN,
        roles=tuple(roles),
        tenant_id=tenant,
        authenticated=True,
    )


# --- first_bound_node_variable ---------------------------------------------


def test_first_bound_node_variable_simple():
    assert first_bound_node_variable("MATCH (n:Entity) RETURN n") == "n"


def test_first_bound_node_variable_non_n_name():
    assert first_bound_node_variable("MATCH (x:Entity) RETURN x") == "x"


def test_first_bound_node_variable_no_label():
    assert first_bound_node_variable("MATCH (doc) RETURN doc") == "doc"


def test_first_bound_node_variable_multi_hop_takes_first():
    q = "MATCH (person:Person)-[:WORKS_AT]->(org:Org) RETURN person, org"
    assert first_bound_node_variable(q) == "person"


def test_first_bound_node_variable_raises_when_unscopable():
    with pytest.raises(UnscopableQueryError):
        first_bound_node_variable("CALL db.labels() YIELD label RETURN label")


# --- TenancyManager.scope_cypher_query --------------------------------------


def test_scope_cypher_query_uses_actual_variable_not_hardcoded_n():
    tm = TenancyManager()
    out = tm.scope_cypher_query("MATCH (x:Entity) RETURN x", tenant_id="acme")
    # The historical bug injected `n.tenant_id = ...` here, referencing a
    # variable this query never binds.
    assert "x.tenant_id = 'acme'" in out
    assert "n.tenant_id" not in out


def test_scope_cypher_query_still_supports_n():
    tm = TenancyManager()
    out = tm.scope_cypher_query("MATCH (n:Entity) RETURN n", tenant_id="acme")
    assert "n.tenant_id = 'acme'" in out


def test_scope_cypher_query_fails_closed_when_unscopable():
    """A query with a RETURN/WHERE to inject into but no derivable MATCH
    variable must raise — never silently return an unscoped or
    wrong-scoped query that a lenient backend would evaluate to zero rows."""
    tm = TenancyManager()
    with pytest.raises(UnscopableQueryError):
        tm.scope_cypher_query("CALL db.labels() YIELD label RETURN label", tenant_id="acme")


def test_scope_cypher_query_noop_without_where_or_return():
    tm = TenancyManager()
    q = "CREATE (n:Entity {name: 'x'})"
    assert tm.scope_cypher_query(q, tenant_id="acme") == q


def test_scope_cypher_query_noop_without_tenant_id():
    tm = TenancyManager()
    q = "MATCH (x:Entity) RETURN x"
    assert tm.scope_cypher_query(q, tenant_id="") == q


# --- tenant_sharing.apply_visibility -----------------------------------------


def test_apply_visibility_uses_actual_variable_not_hardcoded_n():
    out = ts.apply_visibility("MATCH (x:Entity) WHERE x.y = 1 RETURN x", _user("alice"))
    assert "x._owner_id = 'alice'" in out
    assert "n._owner_id" not in out


def test_apply_visibility_fails_closed_when_unscopable():
    with pytest.raises(UnscopableQueryError):
        ts.apply_visibility(
            "CALL db.labels() YIELD label RETURN label", _user("alice")
        )


def test_apply_visibility_privileged_bypass_skips_variable_derivation():
    # A privileged actor gets no predicate at all — an unscopable query must
    # not be refused on their behalf for a restriction that would never apply.
    q = "CALL db.labels() YIELD label RETURN label"
    assert ts.apply_visibility(q, _user("root", roles=("kg:admin",))) == q


def test_apply_visibility_explicit_var_override_still_works():
    out = ts.apply_visibility(
        "MATCH (x:Entity) RETURN x", _user("alice"), var="x"
    )
    assert "x._owner_id = 'alice'" in out
