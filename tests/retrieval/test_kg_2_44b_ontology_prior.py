"""Ontology-prior-guided retrieval ranking (CONCEPT:KG-2.44b).

``CapabilityIndex.designate`` re-projects the flat cosine neighbourhood through the
ontology type structure (the structured-prior analogue of arXiv:2606.09828's
depth-guided back-projection): the dominant ontology type among the strongest cosine
hits is boosted, so a type-coherent neighbourhood survives even when a different-type
candidate interleaves on raw cosine. With the prior disabled the ranking is exactly
pure cosine (parity).
"""

from __future__ import annotations

import numpy as np
import pytest

from agent_utilities.knowledge_graph.retrieval.capability_index import CapabilityIndex

pytestmark = pytest.mark.concept("KG-2.44b")

DIM = 8


def _index_with_interleaved_types() -> tuple[CapabilityIndex, list[str]]:
    """Three 'Document' + two 'Widget' candidates, with one Widget interleaved
    between the Documents on cosine to the query."""
    e = np.eye(DIM, dtype=np.float32)

    def vec(tilt_dim: int, tilt: float) -> np.ndarray:
        v = e[0] + tilt * e[tilt_dim]
        return v / np.linalg.norm(v)

    spec = [
        ("doc-1", "Document", 1, 0.10),
        ("wid-1", "Widget", 2, 0.12),  # cosine-interleaved between doc-1 and doc-2
        ("doc-2", "Document", 3, 0.14),
        ("doc-3", "Document", 4, 0.16),
        ("wid-2", "Widget", 5, 0.40),
    ]
    idx = CapabilityIndex(dim=DIM, prefer_backend="numpy")
    for cid, ctype, td, tilt in spec:
        idx.add(cid, vec(td, tilt), capabilities=["answer"], node_type=ctype)
    return idx, e[0].tolist()


def test_prior_recovers_type_coherent_topk():
    idx, query = _index_with_interleaved_types()
    flat = [
        d.id for d in idx.designate(query, k=3, reward_weight=0.0, prior_weight=0.0)
    ]
    withprior = [d.id for d in idx.designate(query, k=3)]  # prior on by default
    # Flat cosine drags the interleaved Widget into the top-3.
    assert "wid-1" in flat
    # The ontology prior recovers an all-Document (modal-type-coherent) top-3.
    assert withprior == ["doc-1", "doc-2", "doc-3"]


def test_prior_off_is_pure_cosine_parity():
    idx, query = _index_with_interleaved_types()
    flat = [
        d.id for d in idx.designate(query, k=5, reward_weight=0.0, prior_weight=0.0)
    ]
    # Pure cosine order is strictly by tilt magnitude (smaller tilt = higher cosine).
    assert flat == ["doc-1", "wid-1", "doc-2", "doc-3", "wid-2"]


def test_neutral_when_no_types_stored():
    """No node types ⇒ no structured signal ⇒ identical to pure cosine (parity)."""
    e = np.eye(DIM, dtype=np.float32)
    idx = CapabilityIndex(dim=DIM, prefer_backend="numpy")
    idx.add("a", (e[0] + 0.1 * e[1]), capabilities=["x"])  # no node_type
    idx.add("b", (e[0] + 0.2 * e[2]), capabilities=["x"])
    default = [d.id for d in idx.designate(e[0].tolist(), k=2)]
    flat = [
        d.id
        for d in idx.designate(e[0].tolist(), k=2, reward_weight=0.0, prior_weight=0.0)
    ]
    assert default == flat == ["a", "b"]


def test_explicit_prior_overrides_default():
    """An injected (e.g. subsumption-aware) prior is honoured over the default."""
    idx, query = _index_with_interleaved_types()
    # Force the Widgets to win via an explicit alignment map.
    boost = {"wid-1": 1.0, "wid-2": 1.0}
    ranked = [
        d.id
        for d in idx.designate(
            query, k=2, reward_weight=0.0, ontology_prior=boost, prior_weight=1.0
        )
    ]
    assert "wid-1" in ranked


def test_prior_provenance_recorded():
    idx, query = _index_with_interleaved_types()
    des = idx.designate(query, k=3)
    top = des[0]
    assert "ontology_prior" in top.provenance
    assert top.provenance.get("ontology_type") == "Document"
