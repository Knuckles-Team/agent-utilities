"""IDM-05: AU's session-scope allowlist is GENERATED from the EG scope registry.

A scope EG registers reaches a GraphSession; a scope it does not register never
does; and the generated module is exactly what the installed EG registry
renders to -- there is no hand-maintained list to drift.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest

from agent_utilities.security import scope_registry
from agent_utilities.security.request_identity import (
    actor_from_claims,
    mint_graph_session,
)

ROOT = Path(__file__).resolve().parents[3]


def _generator():
    spec = importlib.util.spec_from_file_location(
        "gen_scope_registry", ROOT / "scripts" / "gen_scope_registry.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _mint(scope: str):
    config = MagicMock()
    config.kg_auth_token_ref = None
    config.identity_group_capability_map = {}
    actor = actor_from_claims(
        {
            "sub": "principal:verified",
            "scope": scope,
            "tenant_id": "tenant-a",
            "exp": int(time.time()) + 300,
        }
    )
    with mock.patch("agent_utilities.core.config.config", config):
        return mint_graph_session(actor)


def test_the_generated_module_is_the_installed_eg_registry():
    generator = _generator()
    rendered = generator.render(generator.installed_registry_bytes())
    assert (ROOT / "agent_utilities" / "security" / "scope_registry.py").read_text(
        encoding="utf-8"
    ) == rendered


def test_registered_scopes_reach_the_session_and_nothing_else_does():
    session = _mint(
        "rbac:approve-elevation capacity:throttle finance:alerts fleet:events "
        "admin:cluster-read unrelated:claim admin"
    )
    assert session.scopes == frozenset(
        {
            "rbac:approve-elevation",
            "capacity:throttle",
            "finance:alerts",
            "fleet:events",
            "admin:cluster-read",
        }
    )


def test_the_kg_hierarchy_still_expands():
    assert _mint("kg:admin").scopes == frozenset({"kg:read", "kg:write", "kg:admin"})


def test_the_session_allowlist_is_exactly_the_registry():
    assert frozenset(scope_registry.SCOPE_CLASSES) == scope_registry.SESSION_SCOPES
    assert scope_registry.SCOPE_CLASSES["rbac:approve-elevation"] == "approver"
    assert (
        scope_registry.APPROVER_GROUPS["rbac:approve-elevation"]
        == "elevation-approvers"
    )


def test_the_generator_refuses_an_unknown_scope_class():
    bad = json.dumps({"scopes": [{"scope": "x:y", "class": "superuser"}]}).encode()
    with pytest.raises(ValueError, match="unknown scope classes"):
        _generator().render(bad)


def test_the_broker_and_self_service_identity_scopes_reach_the_session():
    """graph-os runs admin and self-service identity ops under the caller's own
    session and its broker under identity:authenticate: none may be filtered."""
    scopes = "identity:authenticate identity:admin identity:self identity:read identity:provision"
    assert _mint(scopes).scopes == frozenset(scopes.split())
