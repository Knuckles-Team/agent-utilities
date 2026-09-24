"""An EG-free stand-in for ``ontology_integrity.canonical_ttl_hash`` (EH-471).

Connector-manifest gate tests that build their OWN synthetic manifests only need
the gate to recompute the SAME digest they pinned; the byte-compatibility of
EG's real canonical digest with the pinned bundled-manifest hashes is proven in
EG (``OntologyInspect`` pins the arr manifest's digest) and by the engine-backed
bundled-manifest gate test. The stand-in hashes the compiled document's text and
counts its statements; it parses nothing.
"""

from __future__ import annotations

import hashlib
from typing import Any


def _stub_canonical_ttl_hash(ttl: str) -> tuple[str, int]:
    digest = hashlib.sha256(b"test-stub\0" + ttl.encode("utf-8")).hexdigest()
    return digest, ttl.count(" .\n")


def install(monkeypatch: Any) -> None:
    """Route every gate/script hash through the stand-in for this test."""
    from agent_utilities.knowledge_graph.ontology import ontology_integrity

    monkeypatch.setattr(
        ontology_integrity, "canonical_ttl_hash", _stub_canonical_ttl_hash
    )


def use_engine_for_canonical_hash(monkeypatch: Any, engine_graph: Any) -> None:
    """Route the real EG ``OntologyInspect`` hash through the test's tenant graph."""
    from agent_utilities.knowledge_graph.core import graph_compute

    monkeypatch.setattr(
        graph_compute.GraphComputeEngine,
        "get_or_create",
        lambda *_a, **_k: engine_graph,
    )
