"""Structural tests for the one-writer relational authority contract."""

from __future__ import annotations

from copy import deepcopy

import pytest

from agent_utilities.governance.relational_authority import (
    AuthorityMapError,
    declared_schemas,
    load_authority_map,
    validate_authority_map,
    validation_errors,
)


def test_durable_map_matches_all_declared_base_schemas():
    validate_authority_map(schemas=declared_schemas())


def test_missing_domain_fails_closed():
    document = deepcopy(load_authority_map())
    document["domains"] = [
        domain for domain in document["domains"] if domain["name"] != "state_store"
    ]

    errors = validation_errors(document, schemas=declared_schemas())

    assert any("missing authority domain: state_store" in error for error in errors)
    with pytest.raises(AuthorityMapError):
        validate_authority_map(document, schemas=declared_schemas())


def test_missing_read_model_fails_closed():
    document = deepcopy(load_authority_map())
    document["read_models"] = [
        model for model in document["read_models"] if model["name"] != "fleet_topology"
    ]

    errors = validation_errors(document, schemas=declared_schemas())

    assert "missing read model: fleet_topology" in errors
    with pytest.raises(AuthorityMapError):
        validate_authority_map(document, schemas=declared_schemas())


def test_unknown_read_model_fails_closed():
    document = deepcopy(load_authority_map())
    document["read_models"].append(
        {
            "name": "unapproved_projection",
            "owner_domain": "engine_fleet_catalog",
            "source_tables": ["mcp_servers"],
            "fields": [],
            "write_forbidden": True,
        }
    )

    errors = validation_errors(document, schemas=declared_schemas())

    assert "unknown read model: unapproved_projection" in errors
    with pytest.raises(AuthorityMapError):
        validate_authority_map(document, schemas=declared_schemas())


def test_discovery_tables_require_subject_and_grant_binding_fields():
    document = deepcopy(load_authority_map())
    table = next(
        table
        for domain in document["domains"]
        if domain["name"] == "engine_fleet_catalog"
        for table in domain["tables"]
        if table["name"] == "mcp_tools"
    )
    table["authoritative_fields"].remove("discovery_grant_digest")

    errors = validation_errors(document, schemas=declared_schemas())

    assert any("missing discovery binding field" in error for error in errors)
    with pytest.raises(AuthorityMapError):
        validate_authority_map(document, schemas=declared_schemas())


def test_duplicate_and_conflicting_field_authority_fails_closed():
    document = deepcopy(load_authority_map())
    table = next(
        table
        for domain in document["domains"]
        if domain["name"] == "engine_fleet_catalog"
        for table in domain["tables"]
        if table["name"] == "mcp_tools"
    )
    table["authoritative_fields"].append("id")
    table["derived_fields"].append("name")

    errors = validation_errors(document, schemas=declared_schemas())

    assert any("duplicate authoritative fields" in error for error in errors)
    assert any("conflicting field roles" in error for error in errors)


def test_schema_drift_fails_closed_even_when_the_map_is_well_formed():
    document = deepcopy(load_authority_map())
    schemas = deepcopy(declared_schemas())
    schemas["state_store"]["turns"] = frozenset(
        set(schemas["state_store"]["turns"]) | {"unexpected_owner"}
    )

    errors = validation_errors(document, schemas=schemas)

    assert any("schema drift: state_store.turns" in error for error in errors)
    with pytest.raises(AuthorityMapError):
        validate_authority_map(document, schemas=schemas)


def test_a_table_cannot_omit_a_peer_from_the_dual_write_prohibition():
    document = deepcopy(load_authority_map())
    table = document["domains"][0]["tables"][0]
    table["prohibited_dual_write_domains"].pop()

    errors = validation_errors(document, schemas=declared_schemas())

    assert any(
        "does not prohibit every other write domain" in error for error in errors
    )


def test_conflicting_authority_owner_fails_closed():
    document = deepcopy(load_authority_map())
    document["domains"][1]["authority"] = document["domains"][0]["authority"]

    errors = validation_errors(document, schemas=declared_schemas())

    assert any(
        "authority conflicts with the owning domain" in error for error in errors
    )
