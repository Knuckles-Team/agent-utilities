"""Live-path proof for the ``TraceSpan`` evidence-locus producer
(CONCEPT:AU-KG.identity.evidence-spine-convergence, Evidence seam completion).

The seam audit (``reports/seam-closure-audit-2026-07-22.md``) declined an
unconditional write off ``KGTraceBackend.record_event`` — it fires on EVERY
span/generation in the harness, a genuinely hot path with no ``claim_id``.
This test proves the bounded remedy: :func:`register_claim_materialization`
(the SAME shared seam :func:`_persist_code_symbol_evidence` uses) now also
through-writes a ``TraceSpan`` evidence locus whenever a claim's own
``source_ids`` genuinely cite a resolvable, engine-stored span/generation
node — the real shape an ops-causal root-cause finding produces (its causal
path runs from an ingested Trace/Generation through agent/tool/model/
service/deploy). A claim citing neither a code symbol nor a trace span
writes nothing; one citing BOTH kinds in the same ``source_ids`` writes both
loci from the one shared pass.

Offline throughout: the engine backend and the media store are both faked.
"""

from __future__ import annotations

import json
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

    def add_edge(self, source: str, target: str, rel_type: str = "", **props) -> None:
        self.edges.append((source, target, {"rel_type": rel_type, **props}))

    def register_materialization(self, derived_id: str) -> None:
        self.materialized.append(derived_id)


class _FakeStore:
    def __init__(self) -> None:
        self.span_calls: list[tuple[bytes, dict]] = []

    def store_trace_span_evidence(self, data: bytes, **kwargs):
        self.span_calls.append((data, kwargs))
        return object()


def _claim(source_ids: list[str]) -> ClaimNode:
    return ClaimNode(
        id="claim:opscausal:1",
        type=RegistryNodeType.CLAIM,
        name="root cause finding",
        claim_text="span_xyz is a probable root cause",
        confidence=0.8,
        claim_type="finding",
        source_ids=source_ids,
        is_verified=False,
    )


def test_claim_citing_a_span_writes_trace_span_evidence(monkeypatch):
    span_id = "span_abc123"
    span_props = {
        "node_type": "span",
        "trace_id": "trace_root_1",
        "name": "call_tool",
        "span_kind": "tool",
        "latency_ms": 42.5,
        "error": None,
    }
    engine = _StubEngine({span_id: span_props})

    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    claim = _claim([span_id, "capability:foo"])
    errors: list[str] = []
    register_claim_materialization(engine, claim, errors, context="ops_causal")

    assert errors == []
    assert len(store.span_calls) == 1
    data, kw = store.span_calls[0]
    assert data == json.dumps(span_props, sort_keys=True, default=str).encode("utf-8")
    assert kw["trace_id"] == "trace_root_1"
    assert kw["span_id"] == span_id
    assert kw["claim_id"] == claim.id
    assert kw["source"] == "ops_causal"


def test_claim_citing_a_generation_writes_trace_span_evidence(monkeypatch):
    gen_id = "gen_xyz789"
    gen_props = {
        "node_type": "generation",
        "trace_id": "trace_root_2",
        "model": "gpt-x",
        "input_tokens": 100,
        "output_tokens": 50,
    }
    engine = _StubEngine({gen_id: gen_props})
    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    claim = _claim([gen_id])
    errors: list[str] = []
    register_claim_materialization(engine, claim, errors, context="ops_causal")

    assert len(store.span_calls) == 1
    _data, kw = store.span_calls[0]
    assert kw["trace_id"] == "trace_root_2"
    assert kw["span_id"] == gen_id


def test_claim_with_no_trace_span_source_writes_nothing(monkeypatch):
    engine = _StubEngine({"other_node": {"node_type": "capability"}})
    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    claim = _claim(["other_node"])
    errors: list[str] = []
    register_claim_materialization(engine, claim, errors, context="ops_causal")

    assert errors == []
    assert store.span_calls == []


def test_claim_citing_both_code_symbol_and_trace_span_writes_both_loci(
    tmp_path, monkeypatch
):
    """One claim, mixed provenance — both new producers fire independently
    from the SAME shared pass, neither interferes with the other."""
    src = tmp_path / "m.py"
    src.write_text("def f():\n    pass\n")
    code_id = f"code:{src}::f"
    span_id = "span_mixed"

    engine = _StubEngine(
        {
            code_id: {"file_path": str(src), "line": 1, "name": "f"},
            span_id: {"node_type": "span", "trace_id": "trace_mixed"},
        }
    )

    from agent_utilities.knowledge_graph.memory.media_store import (  # noqa: F401
        MediaStore,
    )

    class _FakeCombinedStore:
        def __init__(self) -> None:
            self.symbol_calls: list[tuple] = []
            self.span_calls: list[tuple] = []

        def store_code_symbol_evidence(self, data: bytes, **kwargs):
            self.symbol_calls.append((data, kwargs))

        def store_trace_span_evidence(self, data: bytes, **kwargs):
            self.span_calls.append((data, kwargs))

    combined_store = _FakeCombinedStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: combined_store)

    claim = _claim([code_id, span_id])
    errors: list[str] = []
    register_claim_materialization(engine, claim, errors, context="mixed")

    assert errors == []
    assert len(combined_store.symbol_calls) == 1
    assert len(combined_store.span_calls) == 1
    assert combined_store.symbol_calls[0][1]["claim_id"] == claim.id
    assert combined_store.span_calls[0][1]["claim_id"] == claim.id
