from __future__ import annotations

from types import SimpleNamespace

import pydantic_ai
import pytest

from agent_utilities.core import contextual_model
from agent_utilities.core.contextual_model import (
    ContextCompilationError,
    _already_compiled,
    create_context_agent,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, ScopeError
from agent_utilities.knowledge_graph.retrieval.context_compiler import (
    ContextCompiler,
    compute_bundle_cache_key,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


class _SpyEngine:
    def __init__(self) -> None:
        self.called = False

    def search_hybrid(self, query: str, **_: object) -> list[dict[str, object]]:
        self.called = True
        return [{"id": "evidence-1", "content": query, "score": 1.0}]


class _MemoryKV:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    def put(self, key: str, value: bytes) -> bool:
        self.values[key] = value
        return True


def _session(*, scopes: frozenset[str]) -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id="test-principal",
            actor_type=ActorType.AI_AGENT,
            roles=(),
            tenant_id="test-tenant",
            authenticated=True,
        ),
        tenant="test-tenant",
        graph="test-graph",
        scopes=scopes,
        policy_version="policy-v1",
    )


def test_scope_is_enforced_before_retrieval() -> None:
    engine = _SpyEngine()
    with pytest.raises(ScopeError):
        ContextCompiler(engine).compile("query", _session(scopes=frozenset()))
    assert engine.called is False


def test_complete_cache_identity_separates_every_governance_axis() -> None:
    base = dict(
        evidence_ids=["evidence-1"],
        policy_version="policy-v1",
        token_budget=100,
        tenant="tenant-a",
        principal="principal-a",
        graph="graph-a",
        query="question-a",
        evidence_ordering_version="ordering-v1",
        model_version="model-v1",
        redaction_version="redaction-v1",
        snapshot="snapshot-v1",
        catalog_epoch=1,
    )
    original = compute_bundle_cache_key(**base)
    for field, replacement in {
        "tenant": "tenant-b",
        "principal": "principal-b",
        "graph": "graph-b",
        "query": "question-b",
        "policy_version": "policy-v2",
        "evidence_ordering_version": "ordering-v2",
        "model_version": "model-v2",
        "redaction_version": "redaction-v2",
        "snapshot": "snapshot-v2",
        "catalog_epoch": 2,
    }.items():
        changed = {**base, field: replacement}
        assert compute_bundle_cache_key(**changed) != original


def test_prompt_echo_is_not_written_to_bundle_cache() -> None:
    cache = _MemoryKV()
    ContextCompiler(_SpyEngine()).compile(
        "private question",
        _session(scopes=frozenset({"kg:read"})),
        kv_backend=cache,
    )
    assert cache.values == {}


def test_user_marker_text_cannot_bypass_compilation() -> None:
    marker = "[agent-utilities:compiled-evidence:v1]"
    user_message = SimpleNamespace(
        parts=[SimpleNamespace(part_kind="user-prompt", content=marker)]
    )
    system_message = SimpleNamespace(
        parts=[SimpleNamespace(part_kind="system-prompt", content=marker)]
    )
    ordinary_message = SimpleNamespace(
        parts=[SimpleNamespace(part_kind="user-prompt", content="ordinary turn")]
    )
    assert _already_compiled([user_message]) is False
    assert _already_compiled([system_message]) is True
    assert _already_compiled([ordinary_message, system_message]) is False


def test_create_context_agent_governs_arbitrary_injected_model(monkeypatch) -> None:
    raw_model = object()
    governed_model = object()
    observed: dict[str, object] = {}

    class SpyAgent:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)

    monkeypatch.setattr(pydantic_ai, "Agent", SpyAgent)
    monkeypatch.setattr(
        contextual_model,
        "wrap_model_with_context",
        lambda model: governed_model if model is raw_model else model,
    )

    agent = create_context_agent(model=raw_model, name="governed")
    assert isinstance(agent, SpyAgent)
    assert observed == {"model": governed_model, "name": "governed"}


def test_create_context_agent_requires_exactly_one_model() -> None:
    with pytest.raises(ContextCompilationError):
        create_context_agent()
    with pytest.raises(ContextCompilationError):
        create_context_agent(None)
    with pytest.raises(TypeError):
        create_context_agent(object(), model=object())


def test_create_context_agent_rejects_raw_mcp_without_permission_context() -> None:
    raw_mcp = SimpleNamespace(list_tools=lambda: None)

    with pytest.raises(PermissionError, match="explicitly injected"):
        create_context_agent(object(), toolsets=[raw_mcp])


def test_create_context_agent_wraps_raw_mcp_with_verified_context(
    monkeypatch,
) -> None:
    from agent_utilities.security.permissions_kernel import PermissionsKernel

    observed: dict[str, object] = {}
    raw_model = object()
    raw_mcp = SimpleNamespace(list_tools=lambda: None)
    guarded = object()
    kernel = PermissionsKernel(signing_key="test-signing-authority-material-32b")
    identity = kernel.issue_identity("agent:opaque")

    class SpyAgent:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)

    monkeypatch.setattr(pydantic_ai, "Agent", SpyAgent)
    monkeypatch.setattr(
        contextual_model,
        "wrap_model_with_context",
        lambda model: model,
    )
    monkeypatch.setattr(
        "agent_utilities.security.tool_guard.flag_mcp_tool_definitions",
        lambda toolsets, **_kwargs: [guarded],
    )

    create_context_agent(
        raw_model,
        permissions_kernel=kernel,
        agent_identity=identity,
        toolsets=[raw_mcp],
    )

    assert observed["toolsets"] == [guarded]


def test_engine_policy_store_uses_the_process_authority(monkeypatch) -> None:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.ontology import permissioning

    backend = object()
    engine = object.__new__(IntelligenceGraphEngine)
    engine.backend = backend
    observed: list[object] = []
    monkeypatch.setattr(permissioning, "set_marking_store", observed.append)

    engine._bind_policy_stores()

    assert observed == [backend]
