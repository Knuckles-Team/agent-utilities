"""Static audit: no learned score grants or extends security authority (AU-SEC-R009).

No learned probability, similarity score, or other model-derived confidence
value may itself grant or extend authorization in agent-utilities. Every
authorization decision must trace to an explicit policy or approval record
instead. This module inspects the source of the functions/classes that
actually decide authorization -- mint a session, approve/revoke elevation,
evolve a guardrail profile, or activate a schema repair -- and fails if any
of them reference a learned-score-shaped identifier.

This is deliberately a source-level audit, not a runtime guard: the absence
of these tokens from a decision path is the evidence the requirement asks
for, and the audited list below is itself reviewed whenever a new
authorization decision path is added.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: Identifier substrings that denote a learned/model-derived confidence value.
#: These are legitimate inputs to detection and ranking, but a decision that
#: *grants or extends authority* must never read one.
LEARNED_SCORE_TOKENS: tuple[str, ...] = (
    "confidence",
    "probability",
    "likelihood",
    "similarity_score",
    "model_score",
    "ml_score",
    "learned_score",
)

#: (path relative to the repo root, function/class name) for every code path
#: in agent-utilities that decides whether authorization is granted, renewed,
#: or extended. Add to this list whenever a new decision path is introduced;
#: ``test_authority_audit_covers_known_decision_paths`` guards against
#: accidental removal.
AUDITED_DECISION_PATHS: tuple[tuple[str, str], ...] = (
    ("agent_utilities/security/request_identity.py", "mint_graph_session"),
    ("agent_utilities/security/request_identity.py", "apply_served_security_profile"),
    ("agent_utilities/security/request_identity.py", "_mint_local_process_authority"),
    ("agent_utilities/security/elevation.py", "ElevationService"),
    ("agent_utilities/security/guardrail_evolution.py", "GuardrailEvolution"),
    ("agent_utilities/security/guardrail_profile.py", "plan_move"),
    ("agent_utilities/knowledge_graph/schema_drift/activation.py", "activate_approved"),
    ("agent_utilities/knowledge_graph/schema_drift/gate.py", "run_gate"),
    ("agent_utilities/knowledge_graph/schema_drift/policy.py", "ContractEvolutionPolicy"),
)


class AuthorityAuditError(RuntimeError):
    """A decision path could not be located or inspected."""


def _find_node(tree: ast.Module, name: str) -> ast.AST:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ):
            return node
    raise AuthorityAuditError(f"{name!r} not found")


def decision_source(repo_root: Path, rel_path: str, name: str) -> str:
    """Return the source text of one audited decision path."""
    full_path = repo_root / rel_path
    try:
        tree = ast.parse(full_path.read_text(), filename=str(full_path))
    except OSError as exc:
        raise AuthorityAuditError(f"cannot read {rel_path}: {exc}") from exc
    try:
        node = _find_node(tree, name)
    except AuthorityAuditError as exc:
        raise AuthorityAuditError(f"{rel_path}: {exc}") from exc
    return ast.unparse(node)


def learned_score_violations(
    repo_root: Path,
    paths: tuple[tuple[str, str], ...] = AUDITED_DECISION_PATHS,
) -> list[str]:
    """Return one description per audited decision path that references a
    learned-score token. An empty list is the passing state."""
    violations: list[str] = []
    for rel_path, name in paths:
        source = decision_source(repo_root, rel_path, name).lower()
        for token in LEARNED_SCORE_TOKENS:
            if token in source:
                violations.append(f"{rel_path}:{name} references {token!r}")
    return violations
