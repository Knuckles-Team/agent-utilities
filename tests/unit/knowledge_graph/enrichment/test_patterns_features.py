"""Design-pattern detection + feature clustering (CONCEPT:KG-2.8 Phase 2)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.features import (
    cluster_features,
    resolve_call_edges,
)
from agent_utilities.knowledge_graph.enrichment.models import CodeEntity
from agent_utilities.knowledge_graph.enrichment.patterns import detect_patterns


def _cls(name, bases=None, methods=None, decorators=None, is_abstract=False):
    return CodeEntity(
        id=f"code:m.py::{name}",
        name=name,
        qualname=name,
        kind="class",
        file_path="m.py",
        line=1,
        ast_hash="h",
        bases=bases or [],
        methods=methods or [],
        decorators=decorators or [],
        is_abstract=is_abstract,
    )


def _fn(name, calls=None, decorators=None):
    return CodeEntity(
        id=f"code:m.py::{name}",
        name=name,
        qualname=name,
        kind="function",
        file_path="m.py",
        line=1,
        ast_hash="h",
        calls=calls or [],
        decorators=decorators or [],
    )


def test_detect_abc_strategy_contextmanager_datamodel():
    assert "AbstractBaseClass" in detect_patterns(
        _cls("Base", bases=["ABC"], is_abstract=True)
    )
    assert "Strategy" in detect_patterns(_cls("FastBackend", bases=["Backend"]))
    assert "ContextManager" in detect_patterns(
        _cls("Conn", methods=["__enter__", "__exit__"])
    )
    assert "DataModel" in detect_patterns(_cls("Cfg", bases=["BaseModel"]))
    assert "Repository" in detect_patterns(_cls("UserRepository"))


def test_detect_function_styles():
    assert "Property" in detect_patterns(_fn("x", decorators=["property"]))
    assert "Factory" in detect_patterns(_fn("create_widget"))
    assert "Memoized" in detect_patterns(_fn("y", decorators=["lru_cache"]))


def test_resolve_call_edges_builds_call_graph():
    a = _fn("orchestrate", calls=["plan", "execute"])
    b = _fn("plan", calls=["execute"])
    c = _fn("execute")
    edges = resolve_call_edges([a, b, c])
    pairs = {(e.source.split("::")[1], e.target.split("::")[1]) for e in edges}
    assert ("orchestrate", "plan") in pairs
    assert ("orchestrate", "execute") in pairs
    assert ("plan", "execute") in pairs
    assert all(e.rel_type == "CALLS" for e in edges)


def test_cluster_features_uses_injected_community_detection():
    code = [_fn(n) for n in ("a", "b", "c", "d", "e")]

    def fake_community(node_ids, edges):
        # one community of 3, one of 2 (below min_size)
        return [node_ids[:3], node_ids[3:]]

    feats = cluster_features(code, fake_community, min_size=3)
    assert len(feats) == 1
    assert feats[0].size == 3
    assert len(feats[0].member_ids) == 3
