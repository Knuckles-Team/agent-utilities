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
            return {
                "nodes": [
                    {
                        "node_id": "symbol:compute",
                        "node_type": "SYMBOL",
                        "properties": {
                            "symbol_type": "Function",
                            "name": "compute",
                            "line": "1",
                            "ast_hash": "a",
                            "file_path": file_path,
                            "is_test": "false",
                        },
                    }
                ]
            }
        return {
            "nodes": [
                {
                    "node_id": "symbol:test_x",
                    "node_type": "SYMBOL",
                    "properties": {
                        "symbol_type": "Function",
                        "name": "test_x",
                        "line": "1",
                        "ast_hash": "t",
                        "file_path": file_path,
                        "is_test": "true",
                        "assert_count": "0",
                        "mock_count": "4",
                        "fixture_count": "1",
                        "marks": "",
                        "is_skipped": "false",
                        "calls": "compute",
                    },
                }
            ]
        }

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
        return {
            "nodes": [
                {
                    "node_id": "s1",
                    "node_type": "SYMBOL",
                    "properties": {
                        "symbol_type": "Function",
                        "name": "orchestrate",
                        "line": "1",
                        "ast_hash": "a1",
                        "file_path": file_path,
                        "is_test": "false",
                        "calls": "plan,execute",
                    },
                },
                {
                    "node_id": "s2",
                    "node_type": "SYMBOL",
                    "properties": {
                        "symbol_type": "Function",
                        "name": "plan",
                        "line": "5",
                        "ast_hash": "a2",
                        "file_path": file_path,
                        "is_test": "false",
                        "calls": "execute",
                    },
                },
                {
                    "node_id": "s3",
                    "node_type": "SYMBOL",
                    "properties": {
                        "symbol_type": "Function",
                        "name": "execute",
                        "line": "9",
                        "ast_hash": "a3",
                        "file_path": file_path,
                        "is_test": "false",
                        "calls": "",
                    },
                },
                {
                    "node_id": "s4",
                    "node_type": "SYMBOL",
                    "properties": {
                        "symbol_type": "Class",
                        "name": "BaseStrategy",
                        "line": "13",
                        "ast_hash": "a4",
                        "file_path": file_path,
                        "is_abstract": "true",
                        "bases": "ABC",
                        "methods": "run",
                        "decorators": "",
                    },
                },
            ]
        }

    def fake_llm(prompt):
        card = '{"summary": "does a thing", "responsibilities": ["r1"]}'
        # Multi-symbol batch prompts ask for a JSON array (CONCEPT:KG-2.8, #2).
        if "JSON array" in prompt:
            return "[" + ", ".join([card] * 16) + "]"
        return card

    def fake_community(node_ids, edges):
        return [[i for i in node_ids if i.endswith(("orchestrate", "plan", "execute"))]]

    pipe = EnrichmentPipeline(
        backend,
        parse_fn,
        llm_fn=fake_llm,
        community_fn=fake_community,
        min_feature_size=3,
    )
    summary = pipe.enrich(tmp_path)

    assert summary.code == 4
    assert (
        summary.calls_edges == 3
    )  # orchestrate->plan, orchestrate->execute, plan->execute
    assert summary.patterns_tagged >= 1  # BaseStrategy -> AbstractBaseClass/Strategy
    assert summary.cards_generated == 4
    assert summary.features == 1
    # ABC class carries pattern tags + a card summary
    abc_node = next(
        n for n in backend.nodes.values() if n.get("name") == "BaseStrategy"
    )
    assert "AbstractBaseClass" in abc_node["patterns"]
    assert abc_node["summary"] == "does a thing"
    # Feature node + PART_OF_FEATURE edges written
    assert any(n.get("type") == "Feature" for n in backend.nodes.values())
    assert any(rel == "PART_OF_FEATURE" for _, _, rel in backend.edges)


def _feature_parse_fn(file_path, source):
    return {
        "nodes": [
            {
                "node_id": "s1",
                "node_type": "SYMBOL",
                "properties": {
                    "symbol_type": "Function",
                    "name": "orchestrate",
                    "line": "1",
                    "ast_hash": "a1",
                    "file_path": file_path,
                    "is_test": "false",
                    "calls": "plan,execute",
                },
            },
            {
                "node_id": "s2",
                "node_type": "SYMBOL",
                "properties": {
                    "symbol_type": "Function",
                    "name": "plan",
                    "line": "5",
                    "ast_hash": "a2",
                    "file_path": file_path,
                    "is_test": "false",
                    "calls": "execute",
                },
            },
            {
                "node_id": "s3",
                "node_type": "SYMBOL",
                "properties": {
                    "symbol_type": "Function",
                    "name": "execute",
                    "line": "9",
                    "ast_hash": "a3",
                    "file_path": file_path,
                    "is_test": "false",
                    "calls": "",
                },
            },
        ]
    }


def _community_all(node_ids, edges):
    return [list(node_ids)]


def test_pipeline_mints_capabilities_and_realizes_edges(tmp_path):
    (tmp_path / "svc.py").write_text("def orchestrate(): pass\n")
    backend = FakeBackend()
    pushed = []

    def _wb(nodes):
        pushed.extend(nodes)
        return _Result(len(nodes))

    pipe = EnrichmentPipeline(
        backend,
        _feature_parse_fn,
        community_fn=_community_all,
        min_feature_size=3,
        mint_capabilities=True,
        writeback_fn=_wb,
    )
    summary = pipe.enrich(tmp_path)

    assert summary.features == 1
    assert summary.capabilities_minted == 1
    assert summary.realizes_edges == 1
    assert summary.capabilities_pushed == 1
    # A provisional BusinessCapability node + a REALIZES edge were written.
    assert any(
        n.get("type") == "BusinessCapability" and n.get("provisional") is True
        for n in backend.nodes.values()
    )
    assert any(rel == "REALIZES" for _, _, rel in backend.edges)
    assert len(pushed) == 1


def test_pipeline_matches_existing_capability_no_mint(tmp_path):
    (tmp_path / "svc.py").write_text("def orchestrate(): pass\n")
    backend = FakeBackend()

    # Provide an existing capability whose name overlaps the feature members.
    caps = [
        {"id": "capability:ORCH", "name": "orchestrate plan execute", "summary": ""}
    ]
    pipe = EnrichmentPipeline(
        backend,
        _feature_parse_fn,
        community_fn=_community_all,
        min_feature_size=3,
        mint_capabilities=False,
        capability_provider=lambda: caps,
    )
    summary = pipe.enrich(tmp_path)

    assert summary.capabilities_minted == 0
    assert summary.realizes_edges == 1
    assert "REALIZES" in {rel for _, _, rel in backend.edges}
    assert any(t == "capability:ORCH" for _, t, _ in backend.edges)


class _Result:
    def __init__(self, n):
        self.archi_pushed = n
        self.leanix_pushed = 0


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


# ── CONCEPT:KG-2.100 — single-round-trip Rust resolver path ──────────────────


class PropBackend:
    """Like FakeBackend but keeps full edge props (strategy/confidence)."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def add_edge(self, source, target, **props):
        rel = props.pop("rel_type", None)
        self.edges.append((source, target, rel, props))


def _sym(node_id, name, file_path, sym_type="Function"):
    return {
        "node_id": node_id,
        "node_type": "SYMBOL",
        "properties": {
            "symbol_type": sym_type,
            "name": name,
            "line": "1",
            "ast_hash": node_id,
            "file_path": file_path,
            "is_test": "false",
        },
    }


def test_pipeline_index_resolver_writes_resolved_calls_and_struct_edges(tmp_path):
    """When the engine advertises IndexRepository, the pipeline uses ONE resolver
    round-trip: symbols + already-bound CALLS/INHERITS edges (with strategy/
    confidence), bypassing the Python name-only resolver."""
    (tmp_path / "app.py").write_text(
        "def caller():\n    return helper()\n\ndef helper():\n    return 1\n"
    )
    (tmp_path / "model.py").write_text(
        "class Base:\n    pass\n\nclass Child(Base):\n    pass\n"
    )
    app = str(tmp_path / "app.py")
    model = str(tmp_path / "model.py")

    def index_fn(_files):
        return {
            "nodes": [
                _sym("symbol:caller", "caller", app),
                _sym("symbol:helper", "helper", app),
                _sym("symbol:Base", "Base", model, sym_type="Class"),
                _sym("symbol:Child", "Child", model, sym_type="Class"),
            ],
            "edges": [
                {
                    "source": "symbol:caller",
                    "target": "symbol:helper",
                    "edge_type": "calls",
                    "properties": {
                        "name": "helper",
                        "strategy": "same_file",
                        "confidence": "0.90",
                    },
                },
                {
                    "source": "symbol:Child",
                    "target": "symbol:Base",
                    "edge_type": "inherits",
                    "properties": {"name": "Base"},
                },
                {
                    "source": "symbol:caller",
                    "target": "symbol:helper",
                    "edge_type": "similar_to",
                    "properties": {"score": "0.75"},
                },
            ],
            "calls_resolved": 1,
            "inherits_edges": 1,
            "similar_edges": 1,
        }

    backend = PropBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory(), index_fn=index_fn)
    summary = pipe.enrich(tmp_path)

    assert summary.code == 4  # caller, helper, Base, Child
    assert summary.calls_edges == 1
    assert summary.inherits_edges == 1
    assert summary.similar_edges == 1
    calls = [e for e in backend.edges if e[2] == "CALLS"]
    # Resolved (not name-only): the strategy/confidence props are persisted.
    assert calls and calls[0][3].get("strategy") == "same_file"
    assert calls[0][3].get("confidence") == "0.90"
    assert any(e[2] == "INHERITS" for e in backend.edges)
    # Model-free similarity edge persisted with its score (CONCEPT:KG-2.101).
    sim = [e for e in backend.edges if e[2] == "SIMILAR_TO"]
    assert sim and sim[0][3].get("score") == "0.75"


def test_pipeline_extracts_routes_from_decorators(tmp_path):
    """A handler with a route decorator yields a Route node + SERVES edge
    (CONCEPT:KG-2.102)."""
    (tmp_path / "app.py").write_text("def list_users():\n    return []\n")

    def parse_fn(file_path, source):
        return {
            "nodes": [
                {
                    "node_id": "symbol:list_users",
                    "node_type": "SYMBOL",
                    "properties": {
                        "symbol_type": "Function",
                        "name": "list_users",
                        "line": "1",
                        "ast_hash": "h",
                        "file_path": file_path,
                        "is_test": "false",
                        "decorators": 'app.route("/users", methods=["GET"])',
                    },
                }
            ]
        }

    backend = PropBackend()
    pipe = EnrichmentPipeline(backend, parse_fn)
    summary = pipe.enrich(tmp_path)

    assert summary.routes == 1
    assert summary.serves_edges == 1
    assert any(
        n.get("type") == "Route" and n.get("path") == "/users"
        for n in backend.nodes.values()
    )
    assert any(e[2] == "SERVES" and e[1] == "route:GET:/users" for e in backend.edges)


def test_pipeline_extracts_iac_resources(tmp_path):
    """A Dockerfile alongside the code yields a Resource node (CONCEPT:KG-2.103)."""
    (tmp_path / "app.py").write_text("def compute():\n    return 1\n")
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\nEXPOSE 8080\n")

    backend = PropBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory())
    summary = pipe.enrich(tmp_path)

    assert summary.resources == 1
    res = next(n for n in backend.nodes.values() if n.get("type") == "Resource")
    assert res["kind"] == "container_image"
    assert res["base_image"] == "python:3.12-slim"


def test_pipeline_falls_back_to_name_resolution_when_index_fn_errors(tmp_path):
    """An index_fn that fails must degrade to the parse + name-only path."""
    (tmp_path / "app.py").write_text("def compute():\n    return 1\n")
    (tmp_path / "test_x.py").write_text("def test_x():\n    pass\n")

    def boom(_files):
        raise RuntimeError("engine without resolver")

    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory(), index_fn=boom)
    summary = pipe.enrich(tmp_path)

    assert summary.code == 1 and summary.tests == 1
    assert summary.covers_edges == 1  # name-only COVERS still resolved
