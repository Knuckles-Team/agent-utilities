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
            actor_id="user:1",
            actor_type=ActorType.HUMAN,
            tenant_id=tenant,
            authenticated=True,
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
    assert source_props["node_type"] == "SourceObject"
    assert source_props["document_id"] == "doc-quarterly-report"

    # -- :Evidence — the exact shape `BeliefGraph::from_graph_view` decodes --
    ev_props = client.nodes.properties(result.evidence_id)
    assert ev_props is not None
    assert ev_props["node_type"] == "Evidence"
    assert ev_props["confidence"] == 1.0
    assert ev_props["occurrence_id"] == result.occurrence_id
    assert ev_props["blob_ref"] == result.blob_id
    # The EXACT `eg_modality::EvidenceLocus` wire shape
    # `crates/eg-epistemic/tests/x1_au_occurrence_chain.rs` proves the engine
    # decodes off `evidence_locus` (NOT the former, now-retired `evidence_span`
    # externally-tagged shape) — same literal page/x/y/width/height as that
    # Rust test, so the two are directly diffable against each other.
    assert ev_props["evidence_locus"] == {
        "id": f"eg:locus:{result.evidence_id.split(':', 1)[1]}",
        "subject": {
            "kind": "occurrence",
            "id": f"eg:occurrence:{result.occurrence_id.split(':', 1)[1]}",
        },
        "address": {
            "kind": "page_region",
            "page": 4,
            "x": 72.0,
            "y": 120.5,
            "width": 400.0,
            "height": 18.0,
        },
        "policy_ref": f"eg:policy:{result.evidence_id.split(':', 1)[1]}",
        "derivation_ref": f"eg:derivation:{result.evidence_id.split(':', 1)[1]}",
    }

    # -- :AssetOccurrence -> :Blob (unchanged AU-P1-4 write) --------------
    occ_props = client.nodes.properties(result.occurrence_id)
    assert occ_props is not None
    assert occ_props["node_type"] == "AssetOccurrence"
    assert occ_props["blob_id"] == result.blob_id

    # -- structural (non-epistemic) edges: canonical ``relationship`` property
    # (same key as every edge, epistemic or not — see mining.rs's own
    # ``supports_edge``), but a VALUE outside ``classify_relationship``'s
    # SUPPORTS/CONTRADICTS/ATTACKS whitelist, so ``BeliefGraph`` correctly
    # ignores them. ------------------------------------------------------
    assert (
        result.source_object_id,
        result.occurrence_id,
        {"relationship": "hasOccurrence"},
    ) in client.edges.edges
    assert (
        result.evidence_id,
        result.occurrence_id,
        {"relationship": "extractedFrom"},
    ) in client.edges.edges
    assert (
        result.occurrence_id,
        result.blob_id,
        {"relationship": "hasBlob"},
    ) in client.edges.edges

    # No claim given -> no SUPPORTS edge written.
    assert not any(
        props.get("relationship") == "SUPPORTS" for _s, _t, props in client.edges.edges
    )


def test_store_document_page_evidence_links_supports_edge_when_claim_given():
    """When ``claim_id`` is given, the SAME `relationship: "SUPPORTS"`
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
        {"relationship": "SUPPORTS"},
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
# `_governed_locus` itself — hardcoded expected literals (never built by
# calling `_governed_locus` again), so a regression in the helper cannot pass
# by comparing its output to itself. Pinned to the SAME literal values
# `crates/eg-epistemic/tests/x1_au_occurrence_chain.rs` uses for its
# `page_region` case, so the two are directly diffable.
# --------------------------------------------------------------------------- #


def test_governed_locus_matches_the_engine_contract():
    """`MediaStore._governed_locus` must produce the EXACT
    `eg_modality::EvidenceLocus` wire shape the engine's
    `BeliefGraph::from_graph_view`/`evidence_citations`/`Method::ExplainEvidence`
    decode off an `:Evidence` node's `evidence_locus` property — proven
    engine-side by `crates/eg-epistemic/tests/
    x1_au_occurrence_chain.rs::externally_authored_locus_uses_the_universal_contract`.
    """
    token = "a" * 32
    occ_token = "b" * 32
    locus = MediaStore._governed_locus(
        "page_region",
        {"page": 4, "x": 72.0, "y": 120.5, "width": 400.0, "height": 18.0},
        evidence_id=f"evidence:{token}",
        occurrence_id=f"occurrence:{occ_token}",
    )
    assert locus == {
        "id": f"eg:locus:{token}",
        "subject": {"kind": "occurrence", "id": f"eg:occurrence:{occ_token}"},
        "address": {
            "kind": "page_region",
            "page": 4,
            "x": 72.0,
            "y": 120.5,
            "width": 400.0,
            "height": 18.0,
        },
        "policy_ref": f"eg:policy:{token}",
        "derivation_ref": f"eg:derivation:{token}",
    }


def test_opaque_hex_is_deterministic_lowercase_hex():
    """`_opaque_hex` must always be usable as an `eg_modality::OpaqueRef` token:
    lowercase hex only, and stable across calls with the same inputs (so
    re-ingesting the same symbol/row/span yields the SAME opaque reference)."""
    token = MediaStore._opaque_hex("agent_utilities/foo.py", "_fence_still_valid")
    assert token == MediaStore._opaque_hex(
        "agent_utilities/foo.py", "_fence_still_valid"
    )
    assert len(token) == 64
    assert all(c in "0123456789abcdef" for c in token)
    # Different inputs -> different tokens (no accidental collision from a
    # naive concatenation without a separator).
    assert token != MediaStore._opaque_hex("agent_utilities/foo.py_fence_still_valid")


# --------------------------------------------------------------------------- #
# The other ten loci (Seam 2 completion) — one parametrized proof that every
# ``store_<locus>_evidence`` method writes the SAME chain shape as the shipped
# `PageRegion` seam above, through the governed `evidence_locus` property,
# keyed off its own `EvidenceAddress` kind + fields. See
# ``docs/architecture/evidence_spine_convergence.md``.
# --------------------------------------------------------------------------- #

LOCUS_BYTES = b"locus-evidence-bytes" + bytes(range(32))

LOCUS_CASES: list[tuple[str, dict, str, str, dict]] = [
    (
        "store_document_span_evidence",
        {"document_id": "doc-span-1", "start": 10, "end": 42},
        "doc-span-1",
        "character_range",
        {"start": 10, "end": 42},
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
        "table_cell_range",
        {"row_start": 1, "row_end": 3, "col_start": 0, "col_end": 2},
    ),
    (
        "store_image_region_evidence",
        {"image_id": "img-1", "x": 1.0, "y": 2.0, "width": 30.0, "height": 40.0},
        "img-1",
        "image_region",
        {"x": 1.0, "y": 2.0, "width": 30.0, "height": 40.0},
    ),
    (
        "store_audio_segment_evidence",
        {"audio_id": "audio-1", "start_ms": 1000, "end_ms": 4500},
        "audio-1",
        "audio_range",
        {"start_ms": 1000, "end_ms": 4500},
    ),
    (
        "store_video_shot_evidence",
        {"video_id": "vid-1", "start_ms": 2000, "end_ms": 5000},
        "vid-1",
        "video_time_range",
        {"start_ms": 2000, "end_ms": 5000},
    ),
    (
        "store_video_frame_range_evidence",
        {"video_id": "vid-1", "start_frame": 48, "end_frame": 96},
        "vid-1",
        "frame_range",
        {"start_frame": 48, "end_frame": 96},
    ),
    (
        "store_metric_window_evidence",
        {"metric": "cpu.load", "start_ms": 0, "end_ms": 60000},
        "cpu.load",
        "metric_window",
        {"start_ms": 0, "end_ms": 60000},
    ),
    (
        "store_row_version_evidence",
        {"table": "orders", "row_id": "42", "version": 7},
        "orders:42",
        "row_version",
        {
            "row_ref": f"eg:row:{MediaStore._opaque_hex('orders', '42')}",
            "version": 7,
        },
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
        "code_symbol",
        {
            "revision_ref": f"eg:revision:{MediaStore._opaque_hex('agent_utilities/foo.py')}",
            "symbol_ref": f"eg:symbol:{MediaStore._opaque_hex('agent_utilities/foo.py', '_fence_still_valid')}",
            "start_line": 210,
            "end_line": 245,
        },
    ),
    (
        "store_trace_span_evidence",
        {"trace_id": "trace-1", "span_id": "span-1"},
        "trace-1",
        "trace_span",
        {
            "trace_ref": f"eg:trace:{MediaStore._opaque_hex('trace-1')}",
            "span_ref": f"eg:span:{MediaStore._opaque_hex('trace-1', 'span-1')}",
        },
    ),
]


@pytest.mark.parametrize(
    "method_name,kwargs,about_id,locus_kind,address",
    LOCUS_CASES,
    ids=[c[0] for c in LOCUS_CASES],
)
def test_store_locus_evidence_writes_the_full_identity_chain(
    method_name, kwargs, about_id, locus_kind, address
):
    """Every non-PageRegion locus method writes the exact same
    `:SourceObject -> :AssetOccurrence -> :Blob -> :Evidence` chain shape
    `eg_epistemic::BeliefGraph::from_graph_view` decodes, keyed off its own
    governed `EvidenceLocus` (`_governed_locus`'s own contract is pinned
    independently above — this proves each WRAPPER method feeds it the right
    `locus_kind`/`address`)."""
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
    assert source_props["node_type"] == "SourceObject"
    assert source_props["object_id"] == about_id

    ev_props = client.nodes.properties(result.evidence_id)
    assert ev_props is not None
    assert ev_props["node_type"] == "Evidence"
    assert ev_props["confidence"] == 1.0
    assert ev_props["occurrence_id"] == result.occurrence_id
    assert ev_props["blob_ref"] == result.blob_id
    assert "evidence_span" not in ev_props
    expected_locus = MediaStore._governed_locus(
        locus_kind,
        address,
        evidence_id=result.evidence_id,
        occurrence_id=result.occurrence_id,
    )
    assert ev_props["evidence_locus"] == expected_locus
    # Structurally valid per the engine's own governed-identity rules (every
    # opaque reference is `eg:<namespace>:<16-128 lowercase hex chars>`) —
    # exercises the same shape `eg_modality::EvidenceLocus::validate()` checks.
    assert ev_props["evidence_locus"]["address"]["kind"] == locus_kind

    occ_props = client.nodes.properties(result.occurrence_id)
    assert occ_props is not None
    assert occ_props["node_type"] == "AssetOccurrence"
    assert occ_props["blob_id"] == result.blob_id

    assert (
        result.source_object_id,
        result.occurrence_id,
        {"relationship": "hasOccurrence"},
    ) in client.edges.edges
    assert (
        result.evidence_id,
        result.occurrence_id,
        {"relationship": "extractedFrom"},
    ) in client.edges.edges
    assert (
        result.occurrence_id,
        result.blob_id,
        {"relationship": "hasBlob"},
    ) in client.edges.edges
    assert not any(
        props.get("relationship") == "SUPPORTS" for _s, _t, props in client.edges.edges
    )


@pytest.mark.parametrize(
    "method_name,kwargs,about_id,locus_kind,address",
    LOCUS_CASES,
    ids=[c[0] for c in LOCUS_CASES],
)
def test_store_locus_evidence_links_supports_edge_when_claim_given(
    method_name, kwargs, about_id, locus_kind, address
):
    """Same `relationship: "SUPPORTS"` convention as the PageRegion seam,
    for every other locus kind."""
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
        {"relationship": "SUPPORTS"},
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
