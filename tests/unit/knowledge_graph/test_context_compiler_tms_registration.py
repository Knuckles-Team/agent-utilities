#!/usr/bin/python
"""ContextCompiler bundle materialization registration (W3.2 TMS live-wiring).

CONCEPT:EG-KG.epistemic.truth-maintenance. Proves the third of the three AU-side
derived-artifact producers the audit named
(``reports/surpass-6mo/01-eg-epistemic-core.md`` item 2, "AU side never
registers ITS derived artifacts"): a compiled bundle registers a live
TruthMaintenance materialization ONLY once it is actually persisted beyond the
request (a successful KV-cache store) — an ephemeral, uncached bundle is never
registered, since there is nothing durable to invalidate.

Deliberately placed under ``tests/unit/`` (not ``tests/retrieval/``, where the
compiler's OTHER tests live) — ``tests/retrieval/`` is outside both
``pytest.ini``'s ``testpaths`` and the pre-commit ``pytest`` hook's
``tests/unit`` target, so nothing there is ever actually run; every one of its
four ``test_context_compiler*.py`` files also independently fails offline
(a hardcoded custom ``GraphSession`` fighting the real ambient-session
authority the root ``tests/conftest.py`` establishes) — a pre-existing,
unrelated gap flagged for a follow-up, not fixed here.

``ContextCompiler.compile``'s full pipeline also runs a policy-enforcement
pass (``ontology.permissioning.enforce``) that normally resolves a live
marking store; the wiring test below monkeypatches it to an identity
pass-through (offline, no live engine needed — the same constraint the
``tests/retrieval/`` files above run into) so the REAL ``compile()`` entry
point runs end-to-end. The remaining tests exercise
:meth:`ContextCompiler._register_bundle_materialization` directly — the exact
seam ``compile`` invokes right after a successful KV-cache store — to cover
its degrade-soft/no-citations edge cases without that extra setup.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.retrieval import context_compiler as cc
from agent_utilities.knowledge_graph.retrieval.context_compiler import (
    Citation,
    ContextBundle,
    ContextCompiler,
)


class FakeGraphEngine:
    """Records ``add_node`` calls — nothing else on ``ContextCompiler`` needs
    an engine for this focused test of the registration seam alone."""

    def __init__(self) -> None:
        self.added_nodes: list[tuple[str, str, dict[str, Any]]] = []

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.added_nodes.append((node_id, node_type, dict(properties or {})))


def _bundle(citation_ids: list[str]) -> ContextBundle:
    return ContextBundle(
        query="test query",
        citations=[Citation(node_id=nid, kind="claim") for nid in citation_ids],
    )


def test_registers_a_materialization_after_a_cache_store():
    engine = FakeGraphEngine()
    compiler = ContextCompiler(engine)
    bundle = _bundle(["claim:a", "claim:b"])

    compiler._register_bundle_materialization(bundle, "cache-key-1")

    assert len(engine.added_nodes) == 1
    node_id, node_type, props = engine.added_nodes[0]
    assert node_id == "context_bundle:cache-key-1"
    assert node_type == "ContextBundleMaterialization"
    assert props["cache_key"] == "cache-key-1"
    # invalidation_deps is exactly the bundle's own cited evidence ids — the
    # engine's Channel-1 provenance property `eg_epistemic::register_from_
    # provenance` reads directly off a freshly-written node — never a
    # fabricated or unrelated dependency set.
    assert set(props["invalidation_deps"]) == {"claim:a", "claim:b"}


def test_a_bundle_with_no_citations_is_never_registered():
    """Nothing to depend on -- the engine's own Channel-1 contract treats an
    empty ``invalidation_deps`` as "not a materialization"; skip the write
    entirely rather than send a meaningless empty-deps node."""
    engine = FakeGraphEngine()
    compiler = ContextCompiler(engine)

    compiler._register_bundle_materialization(_bundle([]), "cache-key-2")

    assert engine.added_nodes == []


def test_registration_degrades_soft_when_engine_has_no_add_node():
    """A bare retrieval-only engine (no graph write surface) must never raise —
    registration is a best-effort audit overlay, never load-bearing."""

    class _RetrieveOnlyEngine:
        def retrieve_hybrid(self, query, context_window=10, **kwargs):
            return []

    compiler = ContextCompiler(_RetrieveOnlyEngine())

    compiler._register_bundle_materialization(_bundle(["claim:a"]), "cache-key-3")
    # No exception raised is the assertion; nothing else to observe.


def test_registration_degrades_soft_when_add_node_raises():
    class _RaisingEngine:
        def add_node(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("kg unreachable")

    compiler = ContextCompiler(_RaisingEngine())

    compiler._register_bundle_materialization(_bundle(["claim:a"]), "cache-key-4")
    # Best-effort: the failure is swallowed (logged), never raised into compile().


class _FakeCandidateRetriever:
    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self._candidates = candidates

    def retrieve_hybrid(self, query: str, context_window: int = 10, **kwargs: Any):
        return list(self._candidates)[:context_window]


class _FakeKVBackend:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.put_calls = 0

    def get(self, key: str) -> bytes | None:
        return self.store.get(key)

    def put(self, key: str, value: bytes) -> bool:
        self.put_calls += 1
        self.store[key] = value
        return True


def test_compile_calls_registration_only_after_a_successful_store(monkeypatch):
    """Wire-First: drive the REAL ``compile()`` entry point end-to-end (not
    just the helper in isolation) and prove it invokes
    ``_register_bundle_materialization`` exactly once a store actually lands.

    ``enforce`` is monkeypatched to an identity pass-through — the ONE piece
    of ``compile()`` that needs a live policy/marking store this offline
    sandbox has no deployed engine for (see module docstring); everything
    else, including the KV-cache store and the registration seam under test,
    runs for real.
    """
    monkeypatch.setattr(cc, "enforce", lambda candidates, actor, mask=True: candidates)

    engine = _FakeCandidateRetriever(
        [
            {
                "id": "claim:a",
                "type": "Claim",
                "name": "Claim A",
                "description": "a distinctive premise about widgets",
                "score": 0.9,
            }
        ]
    )
    # Give the retrieval double an `add_node` too, so it also serves as the
    # registration target `compile()` writes through.
    added_nodes: list[tuple[str, str, dict[str, Any]]] = []
    engine.add_node = lambda node_id, node_type, properties=None: added_nodes.append(
        (node_id, node_type, dict(properties or {}))
    )

    calls: list[tuple[ContextBundle, str]] = []
    original = ContextCompiler._register_bundle_materialization

    def _spy(self, bundle, cache_key):
        calls.append((bundle, cache_key))
        return original(self, bundle, cache_key)

    monkeypatch.setattr(ContextCompiler, "_register_bundle_materialization", _spy)

    compiler = ContextCompiler(engine)
    kv = _FakeKVBackend()
    bundle = compiler.compile(
        "a distinctive nonrepeating query about widget provenance",
        top_k=1,
        candidate_pool=1,
        kv_backend=kv,
    )

    assert kv.put_calls == 1
    assert len(calls) == 1
    assert calls[0] == (bundle, bundle.cache_key)
    assert len(added_nodes) == 1
    node_id, node_type, props = added_nodes[0]
    assert node_id == f"context_bundle:{bundle.cache_key}"
    assert node_type == "ContextBundleMaterialization"
    assert props["invalidation_deps"] == ["claim:a"]
