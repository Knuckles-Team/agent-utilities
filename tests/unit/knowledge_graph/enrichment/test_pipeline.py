"""EnrichmentPipeline: in-process, hash-incremental, backend-agnostic writes."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.pipeline import EnrichmentPipeline


class FakeBackend:
    """Captures GraphBackend writes without a daemon."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def add_edge(self, source, target, **props):
        self.edges.append((source, target, props.get("rel_type")))


def _parse_fn_factory():
    """Return a parse_fn that emits a mock-heavy test + covered app fn per file."""
    def parse_fn(file_path, source):
        if file_path.endswith("app.py"):
            return {"nodes": [{
                "node_id": "symbol:compute", "node_type": "SYMBOL",
                "properties": {"symbol_type": "Function", "name": "compute",
                               "line": "1", "ast_hash": "a", "file_path": file_path,
                               "is_test": "false"}}]}
        return {"nodes": [{
            "node_id": "symbol:test_x", "node_type": "SYMBOL",
            "properties": {"symbol_type": "Function", "name": "test_x", "line": "1",
                           "ast_hash": "t", "file_path": file_path, "is_test": "true",
                           "assert_count": "0", "mock_count": "4", "fixture_count": "1",
                           "marks": "", "is_skipped": "false", "calls": "compute"}}]}
    return parse_fn


def test_pipeline_writes_typed_nodes_edges_and_flags_needs_work(tmp_path):
    (tmp_path / "app.py").write_text("def compute():\n    return 1\n")
    (tmp_path / "test_x.py").write_text("def test_x():\n    pass\n")
    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory())

    summary = pipe.enrich(tmp_path)

    assert summary.tests == 1 and summary.code == 1
    assert summary.tests_needing_work == 1
    assert summary.covers_edges == 1
    # Test node carries metrics + needs_work + issues evidence (id is file::name).
    tnode = next(n for n in backend.nodes.values() if n.get("type") == "Test")
    assert tnode["name"] == "test_x"
    assert tnode["needs_work"] is True
    assert "MockHeavyTest" in tnode["issues"]
    # COVERS edge test -> code (one edge, COVERS)
    assert len(backend.edges) == 1
    src, tgt, rel = backend.edges[0]
    assert rel == "COVERS" and src.startswith("test:") and tgt.endswith("::compute")


def test_pipeline_enriches_patterns_features_and_cards(tmp_path):
    (tmp_path / "svc.py").write_text("class X: pass\n")
    backend = FakeBackend()

    def parse_fn(file_path, source):
        # Two app functions calling each other + an ABC class → patterns/features.
        return {"nodes": [
            {"node_id": "s1", "node_type": "SYMBOL", "properties": {
                "symbol_type": "Function", "name": "orchestrate", "line": "1",
                "ast_hash": "a1", "file_path": file_path, "is_test": "false",
                "calls": "plan,execute"}},
            {"node_id": "s2", "node_type": "SYMBOL", "properties": {
                "symbol_type": "Function", "name": "plan", "line": "5",
                "ast_hash": "a2", "file_path": file_path, "is_test": "false",
                "calls": "execute"}},
            {"node_id": "s3", "node_type": "SYMBOL", "properties": {
                "symbol_type": "Function", "name": "execute", "line": "9",
                "ast_hash": "a3", "file_path": file_path, "is_test": "false",
                "calls": ""}},
            {"node_id": "s4", "node_type": "SYMBOL", "properties": {
                "symbol_type": "Class", "name": "BaseStrategy", "line": "13",
                "ast_hash": "a4", "file_path": file_path, "is_abstract": "true",
                "bases": "ABC", "methods": "run", "decorators": ""}},
        ]}

    def fake_llm(prompt):
        return '{"summary": "does a thing", "responsibilities": ["r1"]}'

    def fake_community(node_ids, edges):
        return [[i for i in node_ids if i.endswith(("orchestrate", "plan", "execute"))]]

    pipe = EnrichmentPipeline(
        backend, parse_fn, llm_fn=fake_llm, community_fn=fake_community,
        min_feature_size=3,
    )
    summary = pipe.enrich(tmp_path)

    assert summary.code == 4
    assert summary.calls_edges == 3            # orchestrate->plan, orchestrate->execute, plan->execute
    assert summary.patterns_tagged >= 1        # BaseStrategy -> AbstractBaseClass/Strategy
    assert summary.cards_generated == 4
    assert summary.features == 1
    # ABC class carries pattern tags + a card summary
    abc_node = next(n for n in backend.nodes.values() if n.get("name") == "BaseStrategy")
    assert "AbstractBaseClass" in abc_node["patterns"]
    assert abc_node["summary"] == "does a thing"
    # Feature node + PART_OF_FEATURE edges written
    assert any(n.get("type") == "Feature" for n in backend.nodes.values())
    assert any(rel == "PART_OF_FEATURE" for _, _, rel in backend.edges)


def test_pipeline_is_hash_incremental(tmp_path):
    f = tmp_path / "test_x.py"
    f.write_text("def test_x():\n    pass\n")
    backend = FakeBackend()
    seen: dict[str, str] = {}
    pipe = EnrichmentPipeline(backend, _parse_fn_factory(), hash_seen=seen)

    first = pipe.enrich(tmp_path)
    assert first.files_parsed == 1 and first.files_skipped_unchanged == 0

    # Unchanged content → skipped on the second run.
    second = pipe.enrich(tmp_path)
    assert second.files_parsed == 0 and second.files_skipped_unchanged == 1

    # Changed content → re-parsed.
    f.write_text("def test_x():\n    assert True\n")
    third = pipe.enrich(tmp_path)
    assert third.files_parsed == 1
