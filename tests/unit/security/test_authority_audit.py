"""AU-SEC requirement 009: no learned score grants or extends authorization.

A static review of every authorization-decision path in agent-utilities --
minting a session, approving/revoking elevation, evolving a guardrail
profile, and activating a schema repair -- confirms none of them reference a
learned probability, similarity score, or other model-derived confidence
value. Every grant traces to an explicit policy or approval record instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.security.authority_audit import (
    AUDITED_DECISION_PATHS,
    AuthorityAuditError,
    learned_score_violations,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_no_learned_score_grants_or_extends_authority() -> None:
    violations = learned_score_violations(REPO_ROOT)
    assert violations == [], violations


def test_every_audited_decision_path_resolves() -> None:
    # Every entry must still exist; a renamed/removed function would
    # silently drop coverage from the audit instead of failing it.
    for rel_path, name in AUDITED_DECISION_PATHS:
        from agent_utilities.security.authority_audit import decision_source

        source = decision_source(REPO_ROOT, rel_path, name)
        assert source


def test_audit_covers_every_known_decision_module() -> None:
    covered = {rel_path for rel_path, _ in AUDITED_DECISION_PATHS}
    assert "agent_utilities/security/request_identity.py" in covered
    assert "agent_utilities/security/elevation.py" in covered
    assert "agent_utilities/security/guardrail_evolution.py" in covered
    assert "agent_utilities/security/guardrail_profile.py" in covered
    assert "agent_utilities/knowledge_graph/schema_drift/activation.py" in covered
    assert "agent_utilities/knowledge_graph/schema_drift/gate.py" in covered
    assert "agent_utilities/knowledge_graph/schema_drift/policy.py" in covered


def test_missing_decision_path_raises() -> None:
    with pytest.raises(AuthorityAuditError):
        learned_score_violations(
            REPO_ROOT,
            paths=(("agent_utilities/security/elevation.py", "DoesNotExist"),),
        )
