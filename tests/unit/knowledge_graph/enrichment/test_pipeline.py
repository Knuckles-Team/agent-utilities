"""EnrichmentPipeline: in-process, hash-incremental, backend-agnostic writes."""

from __future__ import annotations

import contextvars

import pytest

from agent_utilities.knowledge_graph.enrichment.pipeline import EnrichmentPipeline


class FakeBackend:
    """Captures GraphBackend writes without a daemon."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def add_edge(self, source, target, **props):
        self.edges.append((source, target, props.get("relationship")))


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
    tnode = next(n for n in backend.nodes.values() if n.get("node_type") == "Test")
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
        # Multi-symbol batch prompts ask for a JSON array (CONCEPT:EG-KG.storage.nonblocking-checkpoint, #2).
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
    assert any(n.get("node_type") == "Feature" for n in backend.nodes.values())
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
        n.get("node_type") == "BusinessCapability" and n.get("provisional") is True
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


# ── CONCEPT:EG-KG.compute.type-scope-resolved-call — single-round-trip Rust resolver path ──────────────────


class PropBackend:
    """Like FakeBackend but keeps full edge props (strategy/confidence)."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def add_edge(self, source, target, **props):
        rel = props.pop("relationship", None)
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
            "files_parsed": 2,
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
    # Model-free similarity edge persisted with its score (CONCEPT:EG-KG.compute.model-free-similar-code).
    sim = [e for e in backend.edges if e[2] == "SIMILAR_TO"]
    assert sim and sim[0][3].get("score") == "0.75"


def test_pipeline_extracts_routes_from_decorators(tmp_path):
    """A handler with a route decorator yields a Route node + SERVES edge
    (CONCEPT:AU-KG.compute.http-route-graph)."""
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
        n.get("node_type") == "Route" and n.get("path") == "/users"
        for n in backend.nodes.values()
    )
    assert any(e[2] == "SERVES" and e[1] == "route:GET:/users" for e in backend.edges)


def test_pipeline_extracts_iac_resources(tmp_path):
    """A Dockerfile alongside the code yields a Resource node (CONCEPT:AU-KG.enrichment.read-them-here-so)."""
    (tmp_path / "app.py").write_text("def compute():\n    return 1\n")
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\nEXPOSE 8080\n")

    backend = PropBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory())
    summary = pipe.enrich(tmp_path)

    assert summary.resources == 1
    res = next(n for n in backend.nodes.values() if n.get("node_type") == "Resource")
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


# ---------------------------------------------------------------------------
# BUG-059 — enrich_documents used to bypass governance; enrich_files did not.
# ---------------------------------------------------------------------------
#
# ``enrich_documents`` wrote every Document/Concept node straight through
# ``self.backend`` (the real backend), never through the governed
# ``_BatchedBackend`` wrapper ``enrich_files`` already swaps in for its own
# write section. So a Document/Concept node from ``enrich_documents`` never
# reached ``stamp_ownership``/``stamp_classification``, regardless of actor
# state — a distinct defect class from "no actor bound" (BUG-055) and
# "caller not audited" (BUG-056): the governance chokepoint was never
# entered at all. Fixed by making ``enrich_documents`` swap in
# ``_BatchedBackend`` exactly like ``enrich_files`` already does.


def _noop_llm_fn(_prompt: str) -> str:
    """An llm_fn that yields no concepts (degrades cleanly, per extract_concepts)."""
    return ""


def test_enrich_documents_now_requires_a_bound_actor_like_enrich_files(tmp_path):
    """Known-bad input: no actor bound anywhere (a fresh, isolated context —
    the same technique BUG-033/BUG-056's regression tests use to simulate a
    genuinely actor-free background caller). BEFORE this fix, this call
    silently wrote an unowned Document node through the raw backend and
    returned normally. AFTER, it raises — the same refusal enrich_files'
    governed _BatchedBackend.add_node already produces."""
    from agent_utilities.security.brain_context import IdentityRequiredError

    (tmp_path / "doc.md").write_text("# Title\n\nSome document content.\n")
    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory(), llm_fn=_noop_llm_fn)

    def isolated():
        with pytest.raises(IdentityRequiredError):
            pipe.enrich_documents([tmp_path / "doc.md"])

    contextvars.Context().run(isolated)
    # And the write never landed — refused before any partial state persisted.
    assert backend.nodes == {}


def test_enrich_documents_stamps_ownership_when_actor_is_bound(tmp_path):
    """With a real actor bound, the write succeeds and the Document node
    carries the SAME governance stamp enrich_files' nodes carry — proving
    enrich_documents now enters the identical chokepoint as its sibling."""
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor

    (tmp_path / "doc.md").write_text("# Title\n\nSome document content.\n")
    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory(), llm_fn=_noop_llm_fn)

    actor = ActorContext(
        actor_id="user:doc-writer",
        actor_type=ActorType.HUMAN,
        tenant_id="tenant-docs",
        authenticated=True,
    )
    with use_actor(actor):
        concepts, edges, summary = pipe.enrich_documents([tmp_path / "doc.md"])

    assert summary.documents == 1
    (doc_props,) = [
        p for p in backend.nodes.values() if p.get("node_type") == "Document"
    ]
    assert doc_props["_owner_id"] == "user:doc-writer"
    assert doc_props["tenant_id"] == "tenant-docs"
    assert doc_props["_shared_scope"] == "private"
    assert doc_props["classification"] == "confidential"
    # self.backend is restored to the real backend after the call (no leaked
    # _BatchedBackend swap across calls).
    assert pipe.backend is backend


# ── U-23 — exact parser acknowledgement and watermark authority ───────────
#
# ``files_parsed`` used to be a bare incremented ``int`` with no per-input
# acknowledgement: a partial/malformed/unknown-identity native response was
# accepted, only the returned files were hashed, and a caller (the ingestion
# engine's git-HEAD watermark) could advance past files that were never
# actually verified. A file with zero SYMBOL nodes (a genuinely empty file)
# was ALSO silently dropped from the result set — indistinguishable from a
# file the engine failed to acknowledge at all.
#
# Every case below proves the invariant INDIVIDUALLY (not one aggregate
# test): the whole defect was that distinct failure modes were being
# collapsed into a single "succeeded" signal. ``IncompleteParse`` is the one
# raised on every rejection; each test proves BOTH that it raises AND that
# NOTHING was persisted to ``pipe._hash_seen`` for the batch (the "no
# watermark advance on error" guarantee — the ingestion engine only persists
# a repository's HEAD/per-file hashes after ``enrich_files``/``enrich``
# returns without raising).

from agent_utilities.knowledge_graph.enrichment.pipeline import IncompleteParse


def _index_sym(file_path: str, name: str = "fn") -> dict:
    return {
        "node_id": f"symbol:{file_path}:{name}",
        "node_type": "SYMBOL",
        "properties": {
            "symbol_type": "Function",
            "name": name,
            "line": "1",
            "ast_hash": "h",
            "file_path": file_path,
            "is_test": "false",
        },
    }


def test_full_success_response_acknowledges_every_file_and_advances_hash_seen(
    tmp_path,
):
    """PASS case: every requested file has a SYMBOL node, files_parsed matches
    — all files land in hash_seen (the delta-skip state the ingestion engine
    persists as the watermark basis)."""
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("def fn():\n    pass\n")
    b.write_text("def fn2():\n    pass\n")

    def index_fn(_files):
        return {
            "nodes": [_index_sym(str(a)), _index_sym(str(b), "fn2")],
            "edges": [],
            "files_parsed": 2,
        }

    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory(), index_fn=index_fn)
    summary = pipe.enrich(tmp_path)

    assert summary.files_parsed == 2
    assert pipe._hash_seen[str(a)] and pipe._hash_seen[str(b)]


def test_legitimate_empty_file_is_recorded_verified_not_dropped(tmp_path):
    """A file with content but ZERO symbols must still be acknowledged: its
    hash lands in hash_seen exactly like a non-empty file, distinguishing
    "parsed, found nothing" from "never acknowledged"."""
    empty = tmp_path / "empty.py"
    nonempty = tmp_path / "app.py"
    empty.write_text("# just a comment, no symbols\n")
    nonempty.write_text("def fn():\n    pass\n")

    def index_fn(_files):
        return {
            # Only app.py gets a SYMBOL node -- empty.py is requested but
            # genuinely has nothing to report.
            "nodes": [_index_sym(str(nonempty))],
            "edges": [],
            "files_parsed": 2,
        }

    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory(), index_fn=index_fn)
    summary = pipe.enrich(tmp_path)

    assert summary.files_parsed == 2
    assert str(empty) in pipe._hash_seen, (
        "a verified-empty file must still be acknowledged in hash_seen, "
        "never silently dropped"
    )
    assert str(nonempty) in pipe._hash_seen


def test_mixed_success_response_records_every_file_exactly_once(tmp_path):
    """A batch with both a symbol-bearing file and a verified-empty file is a
    single successful acknowledgement set -- both land in hash_seen, neither
    twice, and no IncompleteParse is raised."""
    busy = tmp_path / "busy.py"
    quiet = tmp_path / "quiet.py"
    busy.write_text("def fn():\n    pass\n")
    quiet.write_text("PLACEHOLDER = 1\n")  # no functions/classes -> no SYMBOL

    def index_fn(_files):
        return {
            "nodes": [_index_sym(str(busy))],
            "edges": [],
            "files_parsed": 2,
        }

    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory(), index_fn=index_fn)
    summary = pipe.enrich(tmp_path)

    assert summary.files_parsed == 2
    assert list(pipe._hash_seen) == sorted(pipe._hash_seen)  # sanity: no dupes
    assert len(pipe._hash_seen) == 2


def test_partial_response_is_rejected_and_advances_nothing(tmp_path):
    """KNOWN-BAD: files_parsed undercounts the request (a truncated/partial
    native response). Must raise and leave hash_seen untouched for the WHOLE
    batch -- not just the file that went missing."""
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("def fn():\n    pass\n")
    b.write_text("def fn2():\n    pass\n")

    def index_fn(_files):
        return {
            "nodes": [_index_sym(str(a))],
            "edges": [],
            "files_parsed": 1,  # only acknowledges 1 of the 2 requested inputs
        }

    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory(), index_fn=index_fn)

    with pytest.raises(IncompleteParse):
        pipe.enrich(tmp_path)
    assert pipe._hash_seen == {}


def test_unknown_identity_response_is_rejected_and_advances_nothing(tmp_path):
    """KNOWN-BAD: the response's SYMBOL node names a file that was never in
    the request set. Must raise, never silently accept an out-of-scope file."""
    a = tmp_path / "a.py"
    a.write_text("def fn():\n    pass\n")

    def index_fn(_files):
        return {
            "nodes": [_index_sym("/etc/not-requested.py")],
            "edges": [],
            "files_parsed": 1,
        }

    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory(), index_fn=index_fn)

    with pytest.raises(IncompleteParse):
        pipe.enrich(tmp_path)
    assert pipe._hash_seen == {}


def test_malformed_files_parsed_is_rejected(tmp_path):
    """KNOWN-BAD: ``files_parsed`` is not a usable integer (a malformed wire
    response). Must raise, never coerce/ignore it."""
    a = tmp_path / "a.py"
    a.write_text("def fn():\n    pass\n")

    def index_fn(_files):
        return {
            "nodes": [_index_sym(str(a))],
            "edges": [],
            "files_parsed": "not-a-number",
        }

    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory(), index_fn=index_fn)

    with pytest.raises(IncompleteParse):
        pipe.enrich(tmp_path)
    assert pipe._hash_seen == {}


def test_duplicate_identity_in_request_is_rejected(tmp_path):
    """KNOWN-BAD: the SAME logical file is submitted twice in one batch (a
    caller-side duplicate-identity defect). Must raise before the native
    engine is even called -- no ambiguous double acknowledgement is possible."""
    a = tmp_path / "a.py"
    a.write_text("def fn():\n    pass\n")
    calls = []

    def index_fn(files):
        calls.append(files)
        return {"nodes": [_index_sym(str(a))], "edges": [], "files_parsed": 1}

    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory(), index_fn=index_fn)

    with pytest.raises(IncompleteParse):
        pipe.enrich_files([a, a], source_root=tmp_path)
    assert not calls, "the native engine must never be called on a duplicate-identity request"
    assert pipe._hash_seen == {}


def test_symlink_escape_is_rejected(tmp_path):
    """KNOWN-BAD: a symlink inside the source root resolves to a file OUTSIDE
    it. Must raise before the escaped file's content is ever read/parsed."""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret = outside_dir / "secret.py"
    secret.write_text("def leaked():\n    pass\n")

    repo = tmp_path / "repo"
    repo.mkdir()
    link = repo / "vendored.py"
    link.symlink_to(secret)

    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory())

    with pytest.raises(IncompleteParse):
        pipe.enrich_files([link], source_root=repo)
    assert pipe._hash_seen == {}
    assert backend.nodes == {}


def test_unreadable_file_is_rejected_not_silently_skipped(tmp_path):
    """KNOWN-BAD: a file that cannot be read (e.g. deleted mid-walk, a
    permission race). Must raise the whole batch rather than silently
    proceeding as if that file did not exist."""
    a = tmp_path / "a.py"
    missing = tmp_path / "missing.py"
    a.write_text("def fn():\n    pass\n")
    # Never created -- read_text() raises OSError.

    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory())

    with pytest.raises(IncompleteParse):
        pipe.enrich_files([a, missing])
    assert pipe._hash_seen == {}


def test_parser_exception_in_fallback_path_is_rejected_not_silently_empty(
    tmp_path,
):
    """KNOWN-BAD: the per-file parse fallback (no index_fn) hits a genuine
    parse exception for one file. Must raise for the whole batch, not
    degrade that file into an indistinguishable empty-but-successful result
    -- and must not record ANY file's hash for this batch, including a
    sibling file that parsed fine."""
    good = tmp_path / "good.py"
    bad = tmp_path / "bad.py"
    good.write_text("def fn():\n    pass\n")
    bad.write_text("def other():\n    pass\n")

    def parse_fn(file_path, _source):
        if file_path.endswith("bad.py"):
            raise RuntimeError("native parse crashed")
        return {
            "nodes": [
                {
                    "node_id": "symbol:good",
                    "node_type": "SYMBOL",
                    "properties": {
                        "symbol_type": "Function",
                        "name": "fn",
                        "line": "1",
                        "ast_hash": "h",
                        "file_path": file_path,
                        "is_test": "false",
                    },
                }
            ]
        }

    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, parse_fn)  # no index_fn -> per-file path

    with pytest.raises(IncompleteParse):
        pipe.enrich(tmp_path)
    assert pipe._hash_seen == {}


def test_batch_parse_fn_partial_response_is_rejected(tmp_path):
    """KNOWN-BAD: a batch_parse_fn response with fewer entries than requested
    files (mirrors the index_fn partial-response case for the OTHER
    fallback path). Must raise, not silently pad missing slots as empty."""
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("def fn():\n    pass\n")
    b.write_text("def fn2():\n    pass\n")

    def batch_parse_fn(files):
        return [{"nodes": []}]  # only 1 result for 2 requested files

    backend = FakeBackend()
    pipe = EnrichmentPipeline(
        backend, _parse_fn_factory(), batch_parse_fn=batch_parse_fn
    )

    with pytest.raises(IncompleteParse):
        pipe.enrich(tmp_path)
    assert pipe._hash_seen == {}


def test_recorded_hash_is_always_locally_computed_never_trusted_from_wire(
    tmp_path,
):
    """A "hash mismatch" attack is structurally impossible here: the recorded
    content_hash always comes from the caller's own local sha256 of the bytes
    it sent, never from anything the native response echoes back (the wire
    result carries no per-file hash field at all). Proven by a response with
    no hash-like content and confirming the recorded hash still matches the
    independently-computed local sha256."""
    import hashlib

    a = tmp_path / "a.py"
    source_text = "def fn():\n    pass\n"
    a.write_text(source_text)
    expected_hash = hashlib.sha256(
        source_text.encode("utf-8", "surrogatepass")
    ).hexdigest()

    def index_fn(_files):
        return {"nodes": [_index_sym(str(a))], "edges": [], "files_parsed": 1}

    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory(), index_fn=index_fn)
    pipe.enrich(tmp_path)

    assert pipe._hash_seen[str(a)] == expected_hash


def test_native_call_failure_still_degrades_safely_to_verified_fallback(
    tmp_path,
):
    """Regression: when the index_fn RPC itself fails (engine unreachable/
    unsupported -- not a malformed response), the existing safe degrade to
    the per-file fallback still applies and still succeeds and records
    hashes -- only a TRUSTED-but-wrong response (IncompleteParse) must abort,
    never a failed call."""
    f = tmp_path / "app.py"
    f.write_text("def compute():\n    pass\n")

    def boom(_files):
        raise RuntimeError("engine without resolver")

    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory(), index_fn=boom)
    summary = pipe.enrich(tmp_path)

    assert summary.files_parsed == 1
    assert str(f) in pipe._hash_seen


def test_explicit_only_files_subset_is_reflected_in_hash_seen_not_whole_repo(
    tmp_path,
):
    """A caller enriching an explicit file SUBSET (mirrors the ingestion
    engine's ``only_files``/git-delta callers) only records hashes for the
    files it was actually given -- the pipeline itself never has a notion of
    "the whole repo", so it cannot silently claim coverage beyond its input.
    (The whole-repository HEAD watermark decision belongs to the caller --
    ``ingestion/engine.py``'s ``_run_codebase_structural`` already gates that
    on ``not explicit`` and on this call not raising.)"""
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("def fn():\n    pass\n")
    b.write_text("def fn2():\n    pass\n")

    backend = FakeBackend()
    pipe = EnrichmentPipeline(backend, _parse_fn_factory())

    summary = pipe.enrich_files([a], source_root=tmp_path)

    assert summary.files_parsed == 1
    assert str(a) in pipe._hash_seen
    assert str(b) not in pipe._hash_seen


def test_idempotent_replay_reproduces_identical_hash_seen(tmp_path):
    """Replaying the exact same acknowledged batch is a no-op the second
    time (unchanged-content skip) and never re-raises or double-records."""
    a = tmp_path / "a.py"
    a.write_text("def fn():\n    pass\n")

    def index_fn(_files):
        return {"nodes": [_index_sym(str(a))], "edges": [], "files_parsed": 1}

    backend = FakeBackend()
    seen: dict[str, str] = {}
    pipe = EnrichmentPipeline(
        backend, _parse_fn_factory(), index_fn=index_fn, hash_seen=seen
    )

    first = pipe.enrich(tmp_path)
    assert first.files_parsed == 1
    hash_after_first = dict(pipe._hash_seen)

    second = pipe.enrich(tmp_path)
    assert second.files_parsed == 0
    assert second.files_skipped_unchanged == 1
    assert dict(pipe._hash_seen) == hash_after_first
