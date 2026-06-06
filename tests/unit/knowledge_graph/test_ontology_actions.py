#!/usr/bin/python
"""CONCEPT:KG-2.25 — Ontology Action System tests.

Covers the governed verb layer end-to-end: permission grant/deny, parameter
validation, registry duplicate-rejection + lookup, SHACL accept/reject of action
definitions, OWL reasoned eligibility (``mayBeInvokedBy`` property chain), and a
live-path test that the default registry is populated and runs through the
executor. All tests pass offline — the KG backend is optional and skipped
cleanly when no engine is reachable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.actions import (
    DEFAULT_EXECUTOR,
    DEFAULT_REGISTRY,
    ActionEffect,
    ActionExecutor,
    ActionParameter,
    ActionRegistry,
    ActionStatus,
    OntologyAction,
)
from agent_utilities.knowledge_graph.actions import executor as executor_mod
from agent_utilities.security.permissions_kernel import (
    AgentRole,
    PermissionsKernel,
)

KG_DIR = Path(__file__).resolve().parents[3] / "agent_utilities" / "knowledge_graph"


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def registry() -> ActionRegistry:
    reg = ActionRegistry()
    reg.register(
        OntologyAction(
            name="demo.read",
            verb="read",
            description="A safe demo read.",
            parameters=[ActionParameter(name="key", type="string", required=True)],
            acts_on=["concept"],
            required_capability="kg_read",
            produces_effect=ActionEffect.READ,
        ),
        handler=lambda params: f"read:{params['key']}",
    )
    return reg


@pytest.fixture
def kernel() -> PermissionsKernel:
    return PermissionsKernel()


@pytest.fixture
def executor(registry: ActionRegistry, kernel: PermissionsKernel) -> ActionExecutor:
    # persist=False keeps the unit path hermetic; persistence is tested separately.
    return ActionExecutor(registry, kernel=kernel, persist=False)


# ── registry ────────────────────────────────────────────────────────────────


def test_registry_rejects_duplicates(registry: ActionRegistry) -> None:
    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            OntologyAction(
                name="demo.read",
                verb="read",
                required_capability="kg_read",
                acts_on=["concept"],
            ),
            handler=lambda p: None,
        )


def test_registry_lookup_by_type(registry: ActionRegistry) -> None:
    assert [a.name for a in registry.actions_for_type("concept")] == ["demo.read"]
    assert registry.actions_for_type("CONCEPT")  # case-insensitive
    assert registry.actions_for_type("nonexistent") == []
    assert registry.get("demo.read") is not None
    assert registry.get("missing") is None


# ── permission grant / deny ─────────────────────────────────────────────────


def test_permission_grant_executes_and_audits(
    executor: ActionExecutor, kernel: PermissionsKernel
) -> None:
    actor = kernel.issue_identity(
        "agent:reader", role=AgentRole.SPECIALIST, capabilities=["kg_read"]
    )
    inv = executor.execute("demo.read", actor, {"key": "alpha"})
    assert inv.status == ActionStatus.SUCCESS
    assert inv.result_summary == "read:alpha"
    # Audited: an AuditLog entry exists referencing this invocation.
    assert inv.audit_ref
    records = executor.audit.query(action="ontology_action.invoke")
    assert any(r.id == inv.audit_ref for r in records)
    assert records[0].details["status"] == "success"


def test_permission_deny_blocks_and_audits(
    executor: ActionExecutor, kernel: PermissionsKernel
) -> None:
    # Sandbox actor without the required capability is denied.
    actor = kernel.issue_identity(
        "agent:guest", role=AgentRole.SANDBOX, capabilities=[]
    )
    inv = executor.execute("demo.read", actor, {"key": "alpha"})
    assert inv.status == ActionStatus.DENIED
    assert "kg_read" in inv.result_summary
    # Denial is audited; handler never ran (no result summary side effect).
    assert inv.audit_ref
    denied = [
        r
        for r in executor.audit.query(action="ontology_action.invoke")
        if r.details.get("status") == "denied"
    ]
    assert denied


def test_param_validation_rejects_bad_input(
    executor: ActionExecutor, kernel: PermissionsKernel
) -> None:
    actor = kernel.issue_identity(
        "agent:reader", role=AgentRole.SPECIALIST, capabilities=["kg_read"]
    )
    # Missing required 'key'.
    inv = executor.execute("demo.read", actor, {})
    assert inv.status == ActionStatus.ERROR
    assert "missing required parameter 'key'" in inv.error
    # Unknown extra param.
    inv2 = executor.execute("demo.read", actor, {"key": "x", "bogus": 1})
    assert inv2.status == ActionStatus.ERROR
    assert "unknown parameter 'bogus'" in inv2.error


def test_unknown_action_is_errored_and_audited(
    executor: ActionExecutor, kernel: PermissionsKernel
) -> None:
    actor = kernel.issue_identity("agent:x", capabilities=["kg_read"])
    inv = executor.execute("does.not.exist", actor, {})
    assert inv.status == ActionStatus.ERROR
    assert "unknown action" in inv.error


# ── persistence (lazy / optional backend) ───────────────────────────────────


class _FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, query: str, params: dict) -> list:
        self.calls.append((query, params))
        return []


def test_persistence_writes_node_and_edges(
    monkeypatch, registry: ActionRegistry, kernel: PermissionsKernel
) -> None:
    store = _FakeStore()

    class _FakeKG:
        store = None

    fake = _FakeKG()
    fake.store = store  # type: ignore[assignment]
    monkeypatch.setattr(executor_mod, "_persistence_facade", lambda: fake)

    ex = ActionExecutor(registry, kernel=kernel, persist=True)
    actor = kernel.issue_identity("agent:reader", capabilities=["kg_read"])
    inv = ex.execute("demo.read", actor, {"key": "k"}, target_id="concept:topic")
    assert inv.persisted is True
    queries = " ".join(q for q, _ in store.calls)
    assert "action_invocation" in queries
    assert "INVOKED_BY" in queries
    assert "ACTS_ON" in queries


def test_persistence_skips_cleanly_offline(
    monkeypatch, registry: ActionRegistry, kernel: PermissionsKernel
) -> None:
    # No backend reachable → facade returns None → persistence is a no-op.
    monkeypatch.setattr(executor_mod, "_persistence_facade", lambda: None)
    ex = ActionExecutor(registry, kernel=kernel, persist=True)
    actor = kernel.issue_identity("agent:reader", capabilities=["kg_read"])
    inv = ex.execute("demo.read", actor, {"key": "k"})
    assert inv.status == ActionStatus.SUCCESS
    assert inv.persisted is False


# ── live path: default registry is populated and runs ───────────────────────


def test_default_registry_is_populated() -> None:
    names = {a.name for a in DEFAULT_REGISTRY.list_actions()}
    assert {"kg.search", "finance.forensic_screen"} <= names
    assert len(DEFAULT_REGISTRY) >= 2


def test_default_executor_runs_builtin_live_path() -> None:
    # Exercise the *module-level* default executor end-to-end. kg.search degrades
    # to [] when no backend exists, but the governed path (authorize → validate →
    # handle → audit) must complete with SUCCESS.
    DEFAULT_EXECUTOR.persist = False
    actor = DEFAULT_EXECUTOR.kernel.issue_identity(
        "agent:live", role=AgentRole.SPECIALIST, capabilities=["kg_read"]
    )
    inv = DEFAULT_EXECUTOR.execute(
        "kg.search", actor, {"cypher": "MATCH (n) RETURN n LIMIT 1"}
    )
    assert inv.status == ActionStatus.SUCCESS
    assert inv.audit_ref


def test_default_executor_denies_without_capability() -> None:
    DEFAULT_EXECUTOR.persist = False
    actor = DEFAULT_EXECUTOR.kernel.issue_identity(
        "agent:nocap", role=AgentRole.SANDBOX, capabilities=[]
    )
    inv = DEFAULT_EXECUTOR.execute("kg.search", actor, {"cypher": "MATCH (n) RETURN n"})
    assert inv.status == ActionStatus.DENIED


# ── SHACL: valid action def accepted, invalid rejected ──────────────────────


def _build_action_graph(*, with_required: bool):
    """Build a tiny RDF graph with one OntologyAction individual."""
    rdflib = pytest.importorskip("rdflib")
    from rdflib import RDF, Literal, Namespace

    KG = Namespace("http://knuckles.team/kg#")
    g = rdflib.Graph()
    a = KG["action:test.screen"]
    g.add((a, RDF.type, KG.OntologyAction))
    g.add((a, KG.name, Literal("test.screen")))
    g.add((a, KG.acts_on, Literal("financial_instrument")))
    if with_required:
        g.add((a, KG.required_capability, Literal("finance_screen")))
    return g


def test_shacl_accepts_valid_action_def() -> None:
    pytest.importorskip("pyshacl")
    from agent_utilities.knowledge_graph.core.shacl_validator import SHACLValidator

    shapes = KG_DIR / "shapes" / "governance.shapes.ttl"
    g = _build_action_graph(with_required=True)
    report = SHACLValidator().validate(g, shapes)
    assert report["conforms"], report["results_text"]


def test_shacl_rejects_invalid_action_def() -> None:
    pytest.importorskip("pyshacl")
    from agent_utilities.knowledge_graph.core.shacl_validator import SHACLValidator

    shapes = KG_DIR / "shapes" / "governance.shapes.ttl"
    # Missing required_capability → must be quarantined by the gate.
    g = _build_action_graph(with_required=False)
    report = SHACLValidator().validate(g, shapes)
    assert not report["conforms"]
    assert any(
        "required_capability" in (v.get("message") or "") for v in report["violations"]
    )


# ── OWL: reasoned eligibility (the "for free" payoff) ───────────────────────


def test_owl_reasoned_eligibility_may_be_invoked_by() -> None:
    """An Agent that providesCapability X may invoke an Action requiringCapability X.

    Demonstrates the OWL-substrate dividend: the ``mayBeInvokedBy`` property
    chain ``( :requiresCapability :providedBy )`` infers action eligibility from
    the same Capability/Tool pattern that powers tool swappability (KG-2.7).
    """
    rdflib = pytest.importorskip("rdflib")
    owlrl = pytest.importorskip("owlrl")
    from rdflib import RDF, Namespace

    KG = Namespace("http://knuckles.team/kg#")
    g = rdflib.Graph()
    g.parse(str(KG_DIR / "ontology_action.ttl"), format="turtle")
    g.parse(str(KG_DIR / "ontology_capability.ttl"), format="turtle")

    action = KG["action:finance.forensic_screen"]
    agent = KG["agent:fin"]
    cap = KG["FinanceScreenCapability"]
    g.add((action, RDF.type, KG.OntologyAction))
    g.add((action, KG.requiresCapability, cap))
    g.add((agent, RDF.type, KG.Agent))
    g.add((agent, KG.providesCapability, cap))

    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)

    # The reasoner infers eligibility without any hand-wired edge.
    assert (action, KG.mayBeInvokedBy, agent) in g
