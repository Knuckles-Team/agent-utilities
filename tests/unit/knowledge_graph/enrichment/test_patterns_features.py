"""Design-pattern detection + feature clustering (CONCEPT:EG-KG.storage.nonblocking-checkpoint Phase 2)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.features import (
    cluster_features,
    make_community_fn,
    resolve_call_edges,
)
from agent_utilities.knowledge_graph.enrichment.models import CodeEntity
from agent_utilities.knowledge_graph.enrichment.patterns import detect_patterns


class _RecordingCompute:
    """Stand-in GraphComputeEngine that records calls (no real engine)."""

    def __init__(self):
        self.nodes: list[str] = []
        self.edges: list[tuple[str, str]] = []

    def add_node(self, nid, props):
        self.nodes.append(nid)

    def add_edge(self, src, tgt, props):
        self.edges.append((src, tgt))

    def community_detection(self, resolution):
        return [["a", "b"]]


def test_make_community_fn_loads_then_detects():
    # No bulk op on the stand-in → per-element fallback loads the full call graph
    # (every node + edge) before community detection runs.
    gc = _RecordingCompute()
    fn = make_community_fn(gc)
    result = fn(["a", "b", "c"], [("a", "b"), ("b", "c")])
    assert gc.nodes == ["a", "b", "c"]
    assert gc.edges == [("a", "b"), ("b", "c")]
    assert result == [["a", "b"]]


class _BulkCompute:
    """Stand-in with a bulk op (like the real engine) recording batch_update ops."""

    def __init__(self):
        self.bulk_calls: list[list[dict]] = []
        self.per_node = 0

    def bulk_mutate(self, ops):
        self.bulk_calls.append(ops)

    def add_node(self, nid, props):
        self.per_node += 1

    def add_edge(self, src, tgt, props):
        self.per_node += 1

    def community_detection(self, resolution):
        return [["a", "b"]]


def test_make_community_fn_uses_bulk_load_nodes_before_edges():
    gc = _BulkCompute()
    fn = make_community_fn(gc)
    result = fn(["a", "b", "c"], [("a", "b"), ("b", "c")])
    assert result == [["a", "b"]]
    assert gc.per_node == 0  # NOT the per-element path
    flat = [op for call in gc.bulk_calls for op in call]
    # All node ops precede all edge ops (so endpoints exist).
    kinds = [op["op"] for op in flat]
    assert kinds == ["add_node", "add_node", "add_node", "add_edge", "add_edge"]
    assert flat[0] == {"op": "add_node", "id": "a", "properties": {"type": "Code"}}
    assert flat[3] == {
        "op": "add_edge",
        "source": "a",
        "target": "b",
        "properties": {"type": "CALLS"},
    }


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


def test_resolve_call_edges_caps_ambiguous_fanout():
    # An ambiguous callee (more than _MAX_CALL_FANOUT same-named targets, like Java
    # toString) is dropped — it would N×M-explode the edge set without signal.
    from agent_utilities.knowledge_graph.enrichment.features import (
        _MAX_CALL_FANOUT,
        resolve_call_edges,
    )

    n = _MAX_CALL_FANOUT + 5
    # n functions all named "toString" + one caller that calls both toString and "uniq".
    toStrings = [
        _fn("toString") for _ in range(n)
    ]  # share name → same id, so vary the id
    for i, t in enumerate(toStrings):
        t.id = f"code:m.py::toString#{i}"
    uniq = _fn("uniq")
    caller = _fn("caller", calls=["toString", "uniq"])
    edges = resolve_call_edges([*toStrings, uniq, caller])
    targets = {e.target for e in edges if e.source == caller.id}
    # The ambiguous "toString" fan-out (n > cap) is skipped; the precise "uniq" stays.
    assert uniq.id in targets
    assert not any("toString" in t for t in targets)


def test_cluster_features_uses_injected_community_detection():
    code = [_fn(n) for n in ("a", "b", "c", "d", "e")]

    def fake_community(node_ids, edges):
        # one community of 3, one of 2 (below min_size)
        return [node_ids[:3], node_ids[3:]]

    feats = cluster_features(code, fake_community, min_size=3)
    assert len(feats) == 1
    assert feats[0].size == 3
    assert len(feats[0].member_ids) == 3


def test_cluster_features_honors_precomputed_call_edges(monkeypatch):
    # The ingest pipeline resolves CALLS edges once and passes them in so the
    # fan-out resolution isn't recomputed. Assert the provided edges are used and
    # resolve_call_edges is NOT called again.
    import agent_utilities.knowledge_graph.enrichment.features as feat_mod

    called = {"n": 0}
    real = feat_mod.resolve_call_edges

    def spy(code):
        called["n"] += 1
        return real(code)

    monkeypatch.setattr(feat_mod, "resolve_call_edges", spy)

    code = [_fn(n) for n in ("a", "b", "c", "d")]
    a, b = code[0], code[1]
    provided = [
        feat_mod.EnrichmentEdge(source=a.id, target=b.id, rel_type="CALLS"),
    ]

    seen_edges = {}

    def fake_community(node_ids, edges):
        seen_edges["e"] = edges
        return [node_ids]

    feat_mod.cluster_features(code, fake_community, min_size=3, call_edges=provided)
    assert called["n"] == 0, "resolve_call_edges recomputed despite provided call_edges"
    assert seen_edges["e"] == [(a.id, b.id)], (
        "community_fn did not get the provided edges"
    )
