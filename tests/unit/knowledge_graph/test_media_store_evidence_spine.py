"""Seam 2 — AU `AssetOccurrence` -> EG evidence-graph through-write
(CONCEPT:AU-KG.identity.evidence-spine-convergence, EG-X1).

Unit-level (fake engine client, no live engine — mirrors the ``_FakeClient``
pattern in ``test_media_store_identity.py``) proof of the AU HALF of the
cross-repo round trip: :meth:`MediaStore.store_document_page_evidence` writes
the EXACT node/edge property shape epistemic-graph's
``eg_epistemic::BeliefGraph::from_graph_view`` decodes and
``eg_epistemic::evidence_citations``/``Method::ExplainEvidence`` resolves.

The engine-side half of this same round trip — that shape ACTUALLY decoding
and resolving through the real engine code — is proven independently in
``crates/eg-epistemic/tests/x1_au_occurrence_chain.rs`` (epistemic-graph repo),
which mirrors these exact literal values. Together the two prove "AU's
occurrence is now citable through the ONE EG evidence spine" without this
suite needing a live `evidence-graph`-featured engine (an opt-in, non-default
Cargo feature — see that crate's `Cargo.toml`) and without AU building a
second citation resolver.
"""

from __future__ import annotations

import hashlib

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.knowledge_graph.memory.media_store import MediaStore
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext

pytestmark = [pytest.mark.concept("AU-KG.identity.evidence-spine-convergence")]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _FakeBlob:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.refcounts: dict[str, int] = {}

    def store(self, data: bytes) -> str:
        digest = _digest(data)
        self.data[digest] = data
        return digest

    def incref(self, digest: str) -> int:
        self.refcounts[digest] = self.refcounts.get(digest, 0) + 1
        return self.refcounts[digest]

    def unref(self, digest: str) -> int:
        self.refcounts[digest] = max(0, self.refcounts.get(digest, 0) - 1)
        return self.refcounts[digest]

    def fetch(self, digest: str) -> bytes:
        return self.data[digest]


class _FakeTxn:
    """Writes land immediately (not staged-until-commit) — sufficient to assert
    on the property/edge SHAPE this test cares about, same posture as the
    identity-suite's fake."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.blob_refs: list[tuple[str, str]] = []
        self.embeddings: dict[str, list[float]] = {}
        self._n = 0

    def begin(self, graph: str | None = None) -> str:
        self._n += 1
        return f"txn{self._n}"

    def add_node(self, txn, node_id, props):
        self.nodes[node_id] = dict(props)

    def blob_ref(self, txn, node_id, digest):
        self.blob_refs.append((node_id, digest))
        return True

    def add_embedding(self, txn, node_id, embedding):
        self.embeddings[node_id] = list(embedding)
        return True

    def commit(self, txn) -> bool:
        return True


class _FakeNodes:
    """Fake ``client.nodes`` — backed by the SAME dict the txn writes into
    (read-after-write across ``store_media``'s txn commit and the plain
    ``nodes.add`` calls Seam 2 adds), plus a real ``add`` (the identity suite's
    fake has none — Seam 2 is the first caller to need it)."""

    def __init__(self, backing: dict[str, dict]) -> None:
        self._backing = backing

    def has(self, node_id: str) -> bool:
        return node_id in self._backing

    def properties(self, node_id: str) -> dict | None:
        return self._backing.get(node_id)

    def add(self, node_id: str, properties: dict | None = None) -> None:
        self._backing[node_id] = dict(properties or {})


class _FakeEdges:
    def __init__(self) -> None:
        self.edges: list[tuple[str, str, dict]] = []

    def add(self, source, dest, props):
        self.edges.append((source, dest, dict(props or {})))


class _FakeClient:
    def __init__(self) -> None:
        self.blob = _FakeBlob()
        self.txn = _FakeTxn()
        self.nodes = _FakeNodes(self.txn.nodes)
        self.edges = _FakeEdges()


class _FakeCompute:
    def __init__(self, client: _FakeClient | None = None) -> None:
        self._client = client or _FakeClient()
        self.graph_name = "__commons__"


def _session(tenant: str = "acme") -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id="user:1", actor_type=ActorType.HUMAN, tenant_id=tenant
        ),
        tenant=tenant,
    )


PAGE_BYTES = b"%PDF-1.4 page-4-bytes" + bytes(range(64))


def test_store_document_page_evidence_writes_the_full_identity_chain():
    """SourceObject -> AssetOccurrence -> Blob -> Evidence(PageBox) round-trips as
    the SAME node/edge shape `eg_epistemic::BeliefGraph::from_graph_view` decodes.
    """
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))

    result = store.store_document_page_evidence(
        PAGE_BYTES,
        document_id="doc-quarterly-report",
        page=4,
        x=72.0,
        y=120.5,
        width=400.0,
        height=18.0,
        session=_session(),
    )

    assert result is not None
    assert result.source_object_id == "sourceobject:doc-quarterly-report"
    assert result.occurrence_id.startswith("occurrence:")
    assert result.blob_id.startswith("blob:")
    assert result.evidence_id.startswith("evidence:")
    assert result.claim_id is None

    # -- :SourceObject ---------------------------------------------------
    source_props = client.nodes.properties(result.source_object_id)
    assert source_props is not None
    assert source_props["type"] == "SourceObject"
    assert source_props["document_id"] == "doc-quarterly-report"

    # -- :Evidence — the exact shape `BeliefGraph::from_graph_view` decodes --
    ev_props = client.nodes.properties(result.evidence_id)
    assert ev_props is not None
    assert ev_props["type"] == "Evidence"
    assert ev_props["confidence"] == 1.0
    assert ev_props["occurrence_id"] == result.occurrence_id
    assert ev_props["blob_ref"] == result.blob_id
    assert ev_props["evidence_span"] == {
        "PageBox": {
            "document_id": "doc-quarterly-report",
            "page": 4,
            "x": 72.0,
            "y": 120.5,
            "width": 400.0,
            "height": 18.0,
        }
    }

    # -- :AssetOccurrence -> :Blob (unchanged AU-P1-4 write) --------------
    occ_props = client.nodes.properties(result.occurrence_id)
    assert occ_props is not None
    assert occ_props["type"] == "AssetOccurrence"
    assert occ_props["blob_id"] == result.blob_id

    # -- structural (non-epistemic) edges: `{"type": ...}`, never
    # `relationship_type` — so `BeliefGraph` correctly ignores them. ------
    assert (
        result.source_object_id,
        result.occurrence_id,
        {"type": "hasOccurrence"},
    ) in client.edges.edges
    assert (
        result.evidence_id,
        result.occurrence_id,
        {"type": "extractedFrom"},
    ) in client.edges.edges
    assert (
        result.occurrence_id,
        result.blob_id,
        {"type": "hasBlob"},
    ) in client.edges.edges

    # No claim given -> no SUPPORTS edge written.
    assert not any(
        props.get("relationship_type") == "SUPPORTS"
        for _s, _t, props in client.edges.edges
    )


def test_store_document_page_evidence_links_supports_edge_when_claim_given():
    """When ``claim_id`` is given, the SAME `relationship_type: "SUPPORTS"`
    convention `eg_epistemic`'s own claim materialization
    (`src/server/handlers/mining.rs::materialize_claim`) writes is used — no
    engine-side change needed for `evidence_citations`'s support-walk to see it.
    """
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))

    result = store.store_document_page_evidence(
        PAGE_BYTES,
        document_id="doc-quarterly-report",
        page=4,
        x=72.0,
        y=120.5,
        width=400.0,
        height=18.0,
        claim_id="claim:revenue-q3",
        session=_session(),
    )
    assert result is not None
    assert result.claim_id == "claim:revenue-q3"
    assert (
        result.evidence_id,
        "claim:revenue-q3",
        {"relationship_type": "SUPPORTS"},
    ) in client.edges.edges


def test_repeat_calls_for_the_same_document_reuse_one_source_object():
    """Two pages of the SAME document share one `:SourceObject` (upserted once,
    per the method's docstring) but mint distinct evidence/occurrence nodes —
    mirrors AU-P1-4's own "identity vs occurrence" separation one level up."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))

    r1 = store.store_document_page_evidence(
        PAGE_BYTES,
        document_id="doc-multi-page",
        page=1,
        x=0.0,
        y=0.0,
        width=10.0,
        height=10.0,
        session=_session(),
    )
    r2 = store.store_document_page_evidence(
        PAGE_BYTES + b"-page2",
        document_id="doc-multi-page",
        page=2,
        x=0.0,
        y=0.0,
        width=10.0,
        height=10.0,
        session=_session(),
    )
    assert r1 is not None and r2 is not None
    assert r1.source_object_id == r2.source_object_id == "sourceobject:doc-multi-page"
    assert r1.occurrence_id != r2.occurrence_id
    assert r1.evidence_id != r2.evidence_id

    # Only ONE `:SourceObject` node was ever staged (the second call saw it
    # already present via `nodes.has` and did not re-add it).
    source_nodes = [n for n in client.txn.nodes if n.startswith("sourceobject:")]
    assert source_nodes == [r1.source_object_id]


# --------------------------------------------------------------------------- #
# The other ten loci (Seam 2 completion) — one parametrized proof that every
# ``store_<locus>_evidence`` method writes the SAME chain shape as the shipped
# `PageBox` seam above, just with a different externally-tagged `evidence_span`
# variant. See ``docs/architecture/evidence_spine_convergence.md``.
# --------------------------------------------------------------------------- #

LOCUS_BYTES = b"locus-evidence-bytes" + bytes(range(32))

LOCUS_CASES: list[tuple[str, dict, str, dict]] = [
    (
        "store_document_span_evidence",
        {"document_id": "doc-span-1", "start": 10, "end": 42},
        "doc-span-1",
        {"DocumentSpan": {"document_id": "doc-span-1", "start": 10, "end": 42}},
    ),
    (
        "store_table_cell_evidence",
        {
            "table_id": "table-1",
            "row_start": 1,
            "row_end": 3,
            "col_start": 0,
            "col_end": 2,
        },
        "table-1",
        {
            "TableCellRange": {
                "table_id": "table-1",
                "row_start": 1,
                "row_end": 3,
                "col_start": 0,
                "col_end": 2,
            }
        },
    ),
    (
        "store_image_region_evidence",
        {"image_id": "img-1", "x": 1.0, "y": 2.0, "width": 30.0, "height": 40.0},
        "img-1",
        {
            "ImageRegion": {
                "image_id": "img-1",
                "x": 1.0,
                "y": 2.0,
                "width": 30.0,
                "height": 40.0,
            }
        },
    ),
    (
        "store_audio_segment_evidence",
        {"audio_id": "audio-1", "start_ms": 1000, "end_ms": 4500},
        "audio-1",
        {"AudioSegment": {"audio_id": "audio-1", "start_ms": 1000, "end_ms": 4500}},
    ),
    (
        "store_video_shot_evidence",
        {"video_id": "vid-1", "start_ms": 2000, "end_ms": 5000},
        "vid-1",
        {"VideoShot": {"video_id": "vid-1", "start_ms": 2000, "end_ms": 5000}},
    ),
    (
        "store_video_frame_range_evidence",
        {"video_id": "vid-1", "start_frame": 48, "end_frame": 96},
        "vid-1",
        {
            "VideoFrameRange": {
                "video_id": "vid-1",
                "start_frame": 48,
                "end_frame": 96,
            }
        },
    ),
    (
        "store_metric_window_evidence",
        {"metric": "cpu.load", "start_ms": 0, "end_ms": 60000},
        "cpu.load",
        {"MetricWindow": {"metric": "cpu.load", "start_ms": 0, "end_ms": 60000}},
    ),
    (
        "store_row_version_evidence",
        {"table": "orders", "row_id": "42", "version": 7},
        "orders:42",
        {"RowVersion": {"table": "orders", "row_id": "42", "version": 7}},
    ),
    (
        "store_code_symbol_evidence",
        {
            "file_path": "agent_utilities/foo.py",
            "symbol": "_fence_still_valid",
            "start_line": 210,
            "end_line": 245,
        },
        "agent_utilities/foo.py",
        {
            "CodeSymbol": {
                "file_path": "agent_utilities/foo.py",
                "symbol": "_fence_still_valid",
                "start_line": 210,
                "end_line": 245,
            }
        },
    ),
    (
        "store_trace_span_evidence",
        {"trace_id": "trace-1", "span_id": "span-1"},
        "trace-1",
        {"TraceSpan": {"trace_id": "trace-1", "span_id": "span-1"}},
    ),
]


@pytest.mark.parametrize(
    "method_name,kwargs,about_id,expected_span",
    LOCUS_CASES,
    ids=[c[0] for c in LOCUS_CASES],
)
def test_store_locus_evidence_writes_the_full_identity_chain(
    method_name, kwargs, about_id, expected_span
):
    """Every non-PageBox locus method writes the exact same
    `:SourceObject -> :AssetOccurrence -> :Blob -> :Evidence` chain shape
    `eg_epistemic::BeliefGraph::from_graph_view` decodes, keyed off its own
    externally-tagged `evidence_span` variant."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    method = getattr(store, method_name)

    result = method(LOCUS_BYTES, session=_session(), **kwargs)

    assert result is not None
    assert result.source_object_id == f"sourceobject:{about_id}"
    assert result.occurrence_id.startswith("occurrence:")
    assert result.blob_id.startswith("blob:")
    assert result.evidence_id.startswith("evidence:")
    assert result.claim_id is None

    source_props = client.nodes.properties(result.source_object_id)
    assert source_props is not None
    assert source_props["type"] == "SourceObject"
    assert source_props["object_id"] == about_id

    ev_props = client.nodes.properties(result.evidence_id)
    assert ev_props is not None
    assert ev_props["type"] == "Evidence"
    assert ev_props["confidence"] == 1.0
    assert ev_props["occurrence_id"] == result.occurrence_id
    assert ev_props["blob_ref"] == result.blob_id
    assert ev_props["evidence_span"] == expected_span

    occ_props = client.nodes.properties(result.occurrence_id)
    assert occ_props is not None
    assert occ_props["type"] == "AssetOccurrence"
    assert occ_props["blob_id"] == result.blob_id

    assert (
        result.source_object_id,
        result.occurrence_id,
        {"type": "hasOccurrence"},
    ) in client.edges.edges
    assert (
        result.evidence_id,
        result.occurrence_id,
        {"type": "extractedFrom"},
    ) in client.edges.edges
    assert (
        result.occurrence_id,
        result.blob_id,
        {"type": "hasBlob"},
    ) in client.edges.edges
    assert not any(
        props.get("relationship_type") == "SUPPORTS"
        for _s, _t, props in client.edges.edges
    )


@pytest.mark.parametrize(
    "method_name,kwargs,about_id,expected_span",
    LOCUS_CASES,
    ids=[c[0] for c in LOCUS_CASES],
)
def test_store_locus_evidence_links_supports_edge_when_claim_given(
    method_name, kwargs, about_id, expected_span
):
    """Same `relationship_type: "SUPPORTS"` convention as the PageBox seam, for
    every other locus kind."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    method = getattr(store, method_name)

    result = method(
        LOCUS_BYTES, claim_id="claim:evidence-1", session=_session(), **kwargs
    )

    assert result is not None
    assert result.claim_id == "claim:evidence-1"
    assert (
        result.evidence_id,
        "claim:evidence-1",
        {"relationship_type": "SUPPORTS"},
    ) in client.edges.edges


def test_returns_none_and_never_raises_on_underlying_store_media_failure():
    """`store_media` failing (e.g. blob store error) propagates as `None`, never
    a raised exception — matching every other write in this module."""

    class _FailingBlob(_FakeBlob):
        def store(self, data: bytes) -> str:  # noqa: D102
            raise RuntimeError("simulated blob store failure")

    client = _FakeClient()
    client.blob = _FailingBlob()
    store = MediaStore(_FakeCompute(client))

    result = store.store_document_page_evidence(
        PAGE_BYTES,
        document_id="doc-x",
        page=1,
        x=0.0,
        y=0.0,
        width=1.0,
        height=1.0,
        session=_session(),
    )
    assert result is None
