"""Live-path proof for the ``CodeSymbol`` evidence-locus producer
(CONCEPT:AU-KG.identity.evidence-spine-convergence, Evidence seam completion).

The seam audit (``reports/seam-closure-audit-2026-07-22.md``) declined an
unconditional per-symbol write: the AST symbol mapper
(``enrichment/extractors/code_test.py``) fires on EVERY symbol across the
whole continuously reingested fleet, with no ``claim_id`` — a volume/shape
mismatch against the audio/image/table producers. This test proves the
bounded remedy: :func:`register_claim_materialization` — THE one shared
seam every real, floor-cleared ``:Claim`` from every finding family already
passes through — now also through-writes a ``CodeSymbol`` evidence locus
whenever a claim's own ``source_ids`` genuinely cite a resolvable, engine-
stored code symbol. A claim with no code-symbol source id writes nothing;
one that DOES cite one gets a real, claim-linked evidence write, sourced from
the exact line of the real file on disk — never fabricated.

Offline throughout: the engine backend and the media store are both faked;
no real epistemic-graph round-trip. The source "file" is a real temp file so
the read-back snippet is genuine, not synthesized.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.memory import native_ingest
from agent_utilities.knowledge_graph.research.candidate_insight import (
    register_claim_materialization,
)
from agent_utilities.models.knowledge_graph import ClaimNode, RegistryNodeType


class _StubBackend:
    def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
        self._nodes = nodes

    def get_node_properties(self, node_id: str) -> dict[str, Any] | None:
        return self._nodes.get(node_id)


class _StubEngine:
    def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
        self.backend = _StubBackend(nodes)
        self.edges: list[tuple[str, str, dict]] = []
        self.materialized: list[str] = []

    def add_edge(self, source: str, target: str, **props) -> None:
        self.edges.append((source, target, props))

    def register_materialization(self, derived_id: str) -> None:
        self.materialized.append(derived_id)


class _FakeStore:
    def __init__(self) -> None:
        self.symbol_calls: list[tuple[bytes, dict]] = []

    def store_code_symbol_evidence(self, data: bytes, **kwargs):
        self.symbol_calls.append((data, kwargs))
        return object()


def _claim(source_ids: list[str]) -> ClaimNode:
    return ClaimNode(
        id="claim:test:1",
        type=RegistryNodeType.CLAIM,
        name="test claim",
        claim_text="a claim citing real evidence",
        confidence=0.9,
        claim_type="finding",
        source_ids=source_ids,
        is_verified=False,
    )


def test_claim_citing_a_code_symbol_writes_evidence(tmp_path, monkeypatch):
    src = tmp_path / "widget.py"
    src.write_text("def foo():\n    return 1\n\n\ndef bar():\n    return 2\n")
    code_id = f"code:{src}::bar"

    engine = _StubEngine({code_id: {"file_path": str(src), "line": 5, "name": "bar"}})

    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    claim = _claim([code_id, "concept:AU-KG.something"])
    errors: list[str] = []
    register_claim_materialization(engine, claim, errors, context="test_ctx")

    assert errors == []
    # Provenance edges written for EVERY source id (unchanged behavior).
    assert {t for _s, t, _p in engine.edges} == {code_id, "concept:AU-KG.something"}
    assert engine.materialized == [claim.id]

    # Exactly ONE evidence write — only for the source id that resolved to a
    # real code symbol; the concept id is not a code symbol and wrote nothing.
    assert len(store.symbol_calls) == 1
    data, kw = store.symbol_calls[0]
    assert data == b"def bar():"  # the REAL text of line 5
    assert kw["file_path"] == str(src)
    assert kw["symbol"] == "bar"
    assert kw["start_line"] == 5
    assert kw["end_line"] == 5
    assert kw["claim_id"] == claim.id
    assert kw["source"] == "test_ctx"


def test_claim_with_no_code_symbol_source_writes_nothing(monkeypatch):
    engine = _StubEngine({})
    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    claim = _claim(["concept:AU-KG.something", "capability:foo"])
    errors: list[str] = []
    register_claim_materialization(engine, claim, errors, context="test_ctx")

    assert errors == []
    assert store.symbol_calls == []


def test_unresolvable_code_id_skips_silently(monkeypatch):
    """A ``code:`` id the backend has no properties for (e.g. a node that never
    made it into the graph) is a clean skip, never a raised error."""
    engine = _StubEngine({})  # backend knows nothing
    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    claim = _claim(["code:/nowhere.py::ghost"])
    errors: list[str] = []
    register_claim_materialization(engine, claim, errors, context="test_ctx")

    assert errors == []
    assert store.symbol_calls == []


def test_code_symbol_with_unreadable_file_records_error_but_does_not_raise(
    monkeypatch,
):
    code_id = "code:/definitely/not/a/real/path.py::ghost"
    engine = _StubEngine(
        {
            code_id: {
                "file_path": "/definitely/not/a/real/path.py",
                "line": 1,
                "name": "ghost",
            }
        }
    )
    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    claim = _claim([code_id])
    errors: list[str] = []
    register_claim_materialization(engine, claim, errors, context="test_ctx")

    assert store.symbol_calls == []
    assert any("code_symbol_read" in e for e in errors)


def test_test_prefixed_source_id_also_resolves(tmp_path, monkeypatch):
    """``test:<file>::<name>`` (the TestEntity id convention) resolves too,
    not just ``code:``."""
    src = tmp_path / "test_widget.py"
    src.write_text("def test_bar():\n    assert True\n")
    test_id = f"test:{src}::test_bar"

    engine = _StubEngine(
        {test_id: {"file_path": str(src), "line": 1, "name": "test_bar"}}
    )
    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    claim = _claim([test_id])
    errors: list[str] = []
    register_claim_materialization(engine, claim, errors, context="test_ctx")

    assert len(store.symbol_calls) == 1
    data, kw = store.symbol_calls[0]
    assert data == b"def test_bar():"
    assert kw["symbol"] == "test_bar"
