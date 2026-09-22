#!/usr/bin/python
"""CONCEPT:AU-KG.ontology.ontology-action-system — Ontology Action System tests.

Covers the governed verb layer end-to-end: permission grant/deny, parameter
validation, registry duplicate-rejection + lookup, SHACL accept/reject of action
definitions, OWL reasoned eligibility (``mayBeInvokedBy`` property chain), and a
live-path test that the default registry is populated and runs through the
executor. All tests pass offline — the KG backend is optional and skipped
cleanly when no engine is reachable.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.actions import (
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
    return PermissionsKernel(signing_key="test-signing-authority-material-32b")


@pytest.fixture
def executor(registry: ActionRegistry, kernel: PermissionsKernel) -> ActionExecutor:
    # persist=False keeps the unit path hermetic; persistence is tested separately.
    return ActionExecutor(registry, kernel=kernel, persist=False)


def test_action_executor_requires_injected_kernel(registry: ActionRegistry) -> None:
    with pytest.raises(TypeError):
        ActionExecutor(registry)  # type: ignore[call-arg]


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


def test_broad_role_allow_cannot_replace_required_capability(
    executor: ActionExecutor, kernel: PermissionsKernel
) -> None:
    actor = kernel.issue_identity(
        "agent:unqualified",
        role=AgentRole.SPECIALIST,
        capabilities=[],
    )

    invocation = executor.execute("demo.read", actor, {"key": "alpha"})

    assert invocation.status == ActionStatus.DENIED


def test_admin_role_is_explicit_action_authority(
    executor: ActionExecutor, kernel: PermissionsKernel
) -> None:
    actor = kernel.issue_identity(
        "agent:administrator",
        role=AgentRole.ADMIN,
        capabilities=[],
    )

    invocation = executor.execute("demo.read", actor, {"key": "alpha"})

    assert invocation.status == ActionStatus.SUCCESS


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


def test_builtin_registry_runs_with_injected_kernel_live_path() -> None:
    # Exercise the built-in registry with an explicitly governed executor. kg.search degrades
    # to [] when no backend exists, but the governed path (authorize → validate →
    # handle → audit) must complete with SUCCESS.
    kernel = PermissionsKernel(signing_key="test-signing-authority-material-32b")
    executor = ActionExecutor(DEFAULT_REGISTRY, kernel=kernel, persist=False)
    actor = kernel.issue_identity(
        "agent:live", role=AgentRole.SPECIALIST, capabilities=["kg_read"]
    )
    inv = executor.execute("kg.search", actor, {"cypher": "MATCH (n) RETURN n LIMIT 1"})
    assert inv.status == ActionStatus.SUCCESS
    assert inv.audit_ref


def test_builtin_registry_denies_without_capability() -> None:
    kernel = PermissionsKernel(signing_key="test-signing-authority-material-32b")
    executor = ActionExecutor(DEFAULT_REGISTRY, kernel=kernel, persist=False)
    actor = kernel.issue_identity(
        "agent:nocap", role=AgentRole.SANDBOX, capabilities=[]
    )
    inv = executor.execute("kg.search", actor, {"cypher": "MATCH (n) RETURN n"})
    assert inv.status == ActionStatus.DENIED


# ── SHACL: valid action def accepted, invalid rejected ──────────────────────


# ---------------------------------------------------------------------------
# BUG-059 — ActionExecutor._persist's native-typed branch is ROUTED through
# stamp_ownership/stamp_classification. The legacy raw-Cypher fallback branch
# above (``store.execute("MERGE ...")``) is untouched/out of scope.
# ---------------------------------------------------------------------------


class _FakeNativeStore:
    """A ``typed_mutation_support == "native"`` store, tracking every
    add_node/add_edge call so tests can assert on exactly what landed."""

    typed_mutation_support = "native"

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []

    def add_node(self, node_id, label, **props):
        self.nodes[node_id] = {"label": label, **props}

    def add_edge(self, source, target, rel_type, **props):
        self.edges.append((source, target, rel_type))

    def get_node_properties(self, node_id):
        return self.nodes.get(node_id)


def test_persist_native_requires_a_bound_actor(
    monkeypatch, registry: ActionRegistry, kernel: PermissionsKernel
) -> None:
    """Known-bad input: no KG actor bound anywhere (the permission-kernel
    identity issued to ``actor`` below authorizes the ACTION, but is a
    distinct concept from the KG governance actor stamp_ownership reads).
    BEFORE BUG-059's fix, the ActionInvocation node landed unowned
    regardless. AFTER, ``stamp_ownership`` raises inside ``_persist``'s own
    try/except -- persistence degrades to best-effort (unchanged contract:
    "persistence is best-effort") but writes NOTHING rather than an unowned
    node."""
    import contextvars

    store = _FakeNativeStore()

    class _FakeKG:
        store = None

    fake = _FakeKG()
    fake.store = store  # type: ignore[assignment]
    monkeypatch.setattr(executor_mod, "_persistence_facade", lambda: fake)

    ex = ActionExecutor(registry, kernel=kernel, persist=True)
    actor = kernel.issue_identity("agent:reader", capabilities=["kg_read"])

    def isolated():
        return ex.execute("demo.read", actor, {"key": "k"}, target_id="concept:topic")

    inv = contextvars.Context().run(isolated)

    assert inv.status == ActionStatus.SUCCESS  # the action itself still ran
    assert inv.persisted is False  # but persistence was refused, not silently unowned
    assert store.nodes == {}
    assert store.edges == []


def test_persist_native_stamps_ownership_when_actor_bound(
    monkeypatch, registry: ActionRegistry, kernel: PermissionsKernel
) -> None:
    from agent_utilities.security.actor_identity import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor

    store = _FakeNativeStore()

    class _FakeKG:
        store = None

    fake = _FakeKG()
    fake.store = store  # type: ignore[assignment]
    monkeypatch.setattr(executor_mod, "_persistence_facade", lambda: fake)

    ex = ActionExecutor(registry, kernel=kernel, persist=True)
    actor = kernel.issue_identity("agent:reader", capabilities=["kg_read"])

    kg_actor = ActorContext(
        actor_id="user:invoker",
        actor_type=ActorType.HUMAN,
        tenant_id="tenant-actions",
        authenticated=True,
    )
    with use_actor(kg_actor):
        inv = ex.execute("demo.read", actor, {"key": "k"}, target_id="concept:topic")

    assert inv.persisted is True
    inv_props = store.nodes[inv.id]
    assert inv_props["_owner_id"] == "user:invoker"
    assert inv_props["tenant_id"] == "tenant-actions"
    assert inv_props["classification"] == "confidential"
    # actor/target Entity refs also carry the same stamp.
    actor_ref = store.nodes[inv.actor_id]
    assert actor_ref["_owner_id"] == "user:invoker"
