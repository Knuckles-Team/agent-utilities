"""Focused contracts for mandatory session and one-client graph ownership."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.facade import KnowledgeGraph
from agent_utilities.security.brain_context import ActorContext, ActorType

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_ROOT = _PROJECT_ROOT / "agent_utilities"


def _source_files() -> list[Path]:
    return sorted(_SOURCE_ROOT.rglob("*.py"))


def _verified_session(*scopes: str) -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id="opaque-subject",
            actor_type=ActorType.AUTOMATED_SERVICE,
            tenant_id="tenant-a",
            authenticated=True,
        ),
        tenant="tenant-a",
        scopes=frozenset(scopes),
        graph="tenant-a",
    )


def _process_engine(*, backend: object, compute: object | None = None):
    return SimpleNamespace(
        backend=backend,
        graph_compute=compute if compute is not None else object(),
        _bind_policy_stores=lambda: None,
    )


@pytest.fixture(autouse=True)
def _restore_process_engine():
    previous = IntelligenceGraphEngine.get_active()
    IntelligenceGraphEngine.set_active(None)
    yield
    IntelligenceGraphEngine.set_active(previous)


def test_process_engine_factory_executes_once():
    created: list[object] = []

    def factory():
        engine = SimpleNamespace(backend=object())
        created.append(engine)
        return engine

    first = IntelligenceGraphEngine.get_or_create(factory=factory)
    second = IntelligenceGraphEngine.get_or_create(factory=factory)

    assert first is second
    assert created == [first]


def test_process_rejects_direct_duplicate_engine_construction():
    IntelligenceGraphEngine.set_active(_process_engine(backend=object()))

    with pytest.raises(RuntimeError, match="get_or_create"):
        IntelligenceGraphEngine(backend=object())


def test_engine_write_does_not_double_apply_authority_compute():
    authority = Mock()
    engine = object.__new__(IntelligenceGraphEngine)
    engine.backend = SimpleNamespace(graph=authority)
    engine.graph_compute = authority
    engine._compute_is_authority = True
    engine.active_schema_pack = None
    engine._normalize_label = lambda label: label
    engine._upsert_node = Mock(return_value=[{"id": "node-opaque"}])

    with use_session(_verified_session("kg:write")):
        result = engine.add_node("node-opaque", "Record", {"value": 1})

    engine._upsert_node.assert_called_once()
    authority.add_node.assert_not_called()
    assert result == [{"id": "node-opaque"}]


@pytest.mark.asyncio
async def test_session_bound_view_returns_process_owned_backend():
    from agent_utilities.graph.client import get_process_graph_backend

    backend = object()
    IntelligenceGraphEngine.set_active(_process_engine(backend=backend))
    with use_session(_verified_session("kg:read")):
        assert await get_process_graph_backend() is backend


def test_facade_lazy_layers_bind_only_to_process_engine_authority() -> None:
    backend = object()
    compute = object()
    IntelligenceGraphEngine.set_active(
        _process_engine(backend=backend, compute=compute)
    )

    facade = KnowledgeGraph()

    assert facade.store is backend
    assert facade.compute is compute


def test_facade_lazy_layers_fail_closed_without_process_engine() -> None:
    facade = KnowledgeGraph()

    with pytest.raises(RuntimeError, match="active process-owned graph engine"):
        _ = facade.store
    with pytest.raises(RuntimeError, match="active process-owned graph engine"):
        _ = facade.compute

    # A failed first access must not cache an absent authority. Once GraphOS
    # establishes the process engine, the same facade binds to that authority.
    backend = object()
    compute = object()
    IntelligenceGraphEngine.set_active(
        _process_engine(backend=backend, compute=compute)
    )
    assert facade.store is backend
    assert facade.compute is compute


def test_direct_engine_connect_is_limited_to_transport_bootstrap() -> None:
    """Feature modules must use process views, never open another transport."""
    allowed = {
        Path("knowledge_graph/core/graph_compute.py"),
        Path("knowledge_graph/core/placement_catalog.py"),
    }
    offenders: list[str] = []
    pattern = re.compile(r"(?:Sync|Async)EpistemicGraphClient\.connect\s*\(")
    for path in _source_files():
        if not pattern.search(path.read_text(encoding="utf-8")):
            continue
        relative = path.relative_to(_SOURCE_ROOT)
        if relative not in allowed:
            offenders.append(relative.as_posix())
    assert offenders == []


def test_feature_modules_do_not_construct_graph_compute_directly() -> None:
    """The class constructor is private-by-convention; callers use get_or_create."""
    offenders: list[str] = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "GraphComputeEngine":
                offenders.append(
                    f"{path.relative_to(_SOURCE_ROOT).as_posix()}:{node.lineno}"
                )
    assert offenders == []


def test_feature_modules_do_not_construct_operational_engine_directly() -> None:
    """Only the GraphOS singleton factory may invoke the engine constructor."""
    allowed = {Path("mcp/kg_server.py")}
    offenders: list[str] = []
    for path in _source_files():
        relative = path.relative_to(_SOURCE_ROOT)
        if relative in allowed or relative == Path("knowledge_graph/core/engine.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "IntelligenceGraphEngine":
                    offenders.append(f"{relative.as_posix()}:{node.lineno}")
    assert offenders == []


def test_named_graph_view_has_no_implicit_provisioning_write() -> None:
    source = inspect.getsource(
        __import__(
            "agent_utilities.knowledge_graph.core.graph_compute",
            fromlist=["GraphComputeEngine"],
        ).GraphComputeEngine.for_graph
    )
    assert "tenants.create" not in source


def test_no_agentlease_writer_remains() -> None:
    """AgentLease is absent from the current writable model."""
    offenders: list[str] = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr != "add_node" or len(node.args) < 2:
                    continue
                label = node.args[1]
                if isinstance(label, ast.Constant) and label.value == "AgentLease":
                    offenders.append(
                        f"{path.relative_to(_SOURCE_ROOT).as_posix()}:{node.lineno}"
                    )
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            normalized = " ".join(node.value.upper().split())
            if "AGENTLEASE" in normalized and (
                "CREATE " in normalized or "MERGE " in normalized
            ):
                offenders.append(
                    f"{path.relative_to(_SOURCE_ROOT).as_posix()}:{node.lineno}"
                )
    assert offenders == []


def test_workitem_has_no_parallel_claim_selector_module() -> None:
    assert not (_SOURCE_ROOT / "orchestration" / "engine_claim.py").exists()
    source = (_SOURCE_ROOT / "orchestration" / "agent_dispatch_worker.py").read_text(
        encoding="utf-8"
    )
    assert "claim_agent_task" not in source
    assert "execute_agent_task" not in source


def test_orchestrator_and_ingestion_tasks_have_distinct_native_queues() -> None:
    source = (_SOURCE_ROOT / "orchestration" / "work_item.py").read_text(
        encoding="utf-8"
    )
    assert 'queue="orchestrator_task"' in source
    assert 'kind="ingest_task"' in source


def test_served_tool_dispatch_requires_minted_graph_session() -> None:
    source = (_SOURCE_ROOT / "mcp" / "kg_server.py").read_text(encoding="utf-8")
    assert "with verified_tool_session_scope()" in source
    assert "session.engine_verified_context()" in source
    assert "_actor_from_kwargs" not in source


def test_dedicated_ingest_engine_seam_is_absent() -> None:
    assert not (_SOURCE_ROOT / "knowledge_graph" / "core" / "ingest_engine.py").exists()
