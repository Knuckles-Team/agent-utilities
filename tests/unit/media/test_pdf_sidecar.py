"""TCK-style conformance test for the PDF sidecar modality
(CONCEPT:AU-KG.ingest.media-sidecar-delegation): artifact -> bundle -> loci ->
embeddings -> queryable, proven end-to-end against a FAKE engine client (the
SAME ``_FakeClient``/``_FakeCompute`` shape
``tests/unit/knowledge_graph/test_media_store_evidence_spine.py`` uses) with
the fleet sidecar call MOCKED at the seam — no live fleet service required,
matching the task's own acceptance bar. Fixture PDF bytes only (no real PDF
parsing happens here — the sidecar's decoded pages/boxes are supplied
directly, exactly like a real sidecar's JSON response would be).
"""

from __future__ import annotations

import hashlib

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.knowledge_graph.memory.media_store import MediaStore
from agent_utilities.media import pdf_sidecar
from agent_utilities.media.sidecar_delegate import SidecarDelegationResult
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext

# -- the SAME fake engine-client shape as test_media_store_evidence_spine.py,
# duplicated (not imported) — this codebase's own established convention for
# these small MediaStore test doubles (see that file's docstring). --


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
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.blob_refs: list[tuple[str, str]] = []
        self.embeddings: dict[str, list[float]] = {}
        self._n = 0

    def begin(self, graph=None) -> str:
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


class _FakeClaimEngine:
    def __init__(self) -> None:
        self.nodes: list[tuple[str, object, dict]] = []

    def add_node(self, node_id, node_type, properties):
        self.nodes.append((node_id, node_type, dict(properties)))


def _session() -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id="user:1",
            actor_type=ActorType.HUMAN,
            tenant_id="acme",
            authenticated=True,
        ),
        tenant="acme",
    )


FIXTURE_PDF = b"%PDF-1.4 fixture born-digital text\n" + bytes(range(64))


def _mock_pages_result(pages):
    return SidecarDelegationResult(
        available=True,
        modality="pdf",
        provider="stirlingpdf-mcp",
        tool="pdf_action",
        action="ocr_pdf",
        activity_id="activity:media_sidecar:stirlingpdf-mcp:pdf:1:aaaa1111",
        raw={"pages": pages},
    )


def test_pdf_conformance_artifact_bundle_loci_embeddings_queryable(monkeypatch):
    """The full TCK chain for PDF: artifact -> bundle -> loci -> embeddings ->
    queryable, proven against the exact node/edge shape
    ``eg_epistemic::BeliefGraph::from_graph_view`` decodes — the SAME bar
    ``test_media_store_evidence_spine.py`` holds every other locus producer
    to."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    claim_engine = _FakeClaimEngine()

    monkeypatch.setattr(pdf_sidecar, "claim_engine", lambda: claim_engine)
    monkeypatch.setattr(
        pdf_sidecar,
        "delegate_extract",
        lambda *a, **k: _mock_pages_result(
            [
                {
                    "page": 1,
                    "width": 612.0,
                    "height": 792.0,
                    "text": "Quarterly revenue rose 12%.",
                    "embedding": [0.1, 0.2, 0.3],
                    "boxes": [
                        {
                            "x": 10.0,
                            "y": 20.0,
                            "width": 100.0,
                            "height": 12.0,
                            "text": "Quarterly",
                        },
                        {
                            "x": 10.0,
                            "y": 34.0,
                            "width": 90.0,
                            "height": 12.0,
                            "text": "revenue rose",
                        },
                    ],
                },
                {
                    "page": 2,
                    "width": 612.0,
                    "height": 792.0,
                    "text": "See appendix for detail.",
                    "boxes": [],
                },
            ]
        ),
    )

    result = pdf_sidecar.ingest_pdf_via_sidecar(
        FIXTURE_PDF,
        document_id="doc-quarterly-report",
        source="test",
        session=_session(),
        media_store=store,
    )

    # -- artifact -> bundle: ONE :AssetOccurrence -> :Blob for the whole PDF --
    assert result.available is True
    assert result.occurrence_id is not None
    assert result.occurrence_id.startswith("occurrence:")
    assert result.blob_id is not None
    assert result.blob_id.startswith("blob:")
    occ_props = client.nodes.properties(result.occurrence_id)
    assert occ_props["node_type"] == "AssetOccurrence"
    assert occ_props["media_type"] == "application/pdf"
    assert occ_props["content_digest"] == _digest(FIXTURE_PDF)

    # -- bundle -> loci: PageBox per page, DocumentSpan per page, ImageRegion
    # per OCR box --
    assert result.page_count == 2
    assert len(result.page_evidence_ids) == 2
    assert len(result.span_evidence_ids) == 2
    assert len(result.ocr_box_evidence_ids) == 2  # page 1's two boxes only

    # Governed `eg_modality::EvidenceLocus` — W3.3 (au-loci) retired the
    # externally-tagged `evidence_span` shape in favour of `_governed_locus`'s
    # `{id, subject, address, policy_ref, derivation_ref}` (identity lives in
    # `subject`/`about`, so the address carries the variant's OWN fields only —
    # no `document_id`/`image_id`; the kind is the snake_case `EvidenceAddress`
    # variant tag). The sidecar producers auto-inherit this via MediaStore.
    page1_evidence = client.nodes.properties(result.page_evidence_ids[0])
    page1_locus_token = result.page_evidence_ids[0].split(":", 1)[1]
    page1_occurrence_token = page1_evidence["occurrence_id"].split(":", 1)[1]
    assert page1_evidence["evidence_locus"] == {
        "id": f"eg:locus:{page1_locus_token}",
        "subject": {
            "kind": "occurrence",
            "id": f"eg:occurrence:{page1_occurrence_token}",
        },
        "address": {
            "kind": "page_region",
            "page": 1,
            "x": 0.0,
            "y": 0.0,
            "width": 612.0,
            "height": 792.0,
        },
        "policy_ref": f"eg:policy:{page1_locus_token}",
        "derivation_ref": f"eg:derivation:{page1_locus_token}",
    }

    span1_evidence = client.nodes.properties(result.span_evidence_ids[0])
    span1_locus_token = result.span_evidence_ids[0].split(":", 1)[1]
    span1_occurrence_token = span1_evidence["occurrence_id"].split(":", 1)[1]
    assert span1_evidence["evidence_locus"] == {
        "id": f"eg:locus:{span1_locus_token}",
        "subject": {
            "kind": "occurrence",
            "id": f"eg:occurrence:{span1_occurrence_token}",
        },
        "address": {
            "kind": "character_range",
            "start": 0,
            "end": len("Quarterly revenue rose 12%."),
        },
        "policy_ref": f"eg:policy:{span1_locus_token}",
        "derivation_ref": f"eg:derivation:{span1_locus_token}",
    }

    box_evidence = client.nodes.properties(result.ocr_box_evidence_ids[0])
    box_locus_token = result.ocr_box_evidence_ids[0].split(":", 1)[1]
    box_occurrence_token = box_evidence["occurrence_id"].split(":", 1)[1]
    assert box_evidence["evidence_locus"] == {
        "id": f"eg:locus:{box_locus_token}",
        "subject": {
            "kind": "occurrence",
            "id": f"eg:occurrence:{box_occurrence_token}",
        },
        "address": {
            "kind": "image_region",
            "x": 10.0,
            "y": 20.0,
            "width": 100.0,
            "height": 12.0,
        },
        "policy_ref": f"eg:policy:{box_locus_token}",
        "derivation_ref": f"eg:derivation:{box_locus_token}",
    }

    # -- embeddings: page 1's embedding rode through to its PageBox occurrence --
    page1_occurrence_id = page1_evidence["occurrence_id"]
    assert client.txn.embeddings[page1_occurrence_id] == [0.1, 0.2, 0.3]

    # -- queryable: every locus is linked SUPPORTS -> the ONE governed claim,
    # exactly the convention eg_epistemic::evidence_citations's support-walk
    # recognizes, with no engine-side change --
    assert result.claim_id is not None
    assert len(claim_engine.nodes) == 1
    claim_node_id, claim_node_type, claim_props = claim_engine.nodes[0]
    assert claim_node_id == result.claim_id
    assert claim_node_type == "Claim"
    assert claim_props["is_verified"] is True
    assert claim_props["confidence"] == 1.0
    assert claim_props["metadata"]["was_generated_by"] == "agent:stirlingpdf-mcp"

    supports_edges = [
        (s, t)
        for s, t, props in client.edges.edges
        if props.get("relationship") == "SUPPORTS"
    ]
    assert {t for _s, t in supports_edges} == {result.claim_id}
    assert len(supports_edges) == (
        len(result.page_evidence_ids)
        + len(result.span_evidence_ids)
        + len(result.ocr_box_evidence_ids)
    )


def test_pdf_sidecar_degrades_cleanly_when_delegation_fails(monkeypatch):
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))

    monkeypatch.setattr(
        pdf_sidecar,
        "delegate_extract",
        lambda *a, **k: SidecarDelegationResult(
            available=False,
            modality="pdf",
            provider="stirlingpdf-mcp",
            tool="pdf_action",
            action="ocr_pdf",
            activity_id=None,
            raw={},
            error="ConnectionError: sidecar unreachable",
        ),
    )

    result = pdf_sidecar.ingest_pdf_via_sidecar(
        FIXTURE_PDF,
        document_id="doc-unreachable",
        session=_session(),
        media_store=store,
    )
    assert result.available is False
    assert result.error == "ConnectionError: sidecar unreachable"
    # the artifact bundle was still stored (a valid PDF was received; only
    # the delegated extraction failed) — occurrence/blob exist even though
    # no locus was written.
    assert result.occurrence_id is not None
    assert result.page_count == 0


def test_pdf_sidecar_respects_the_fail_closed_capability_manifest(monkeypatch):
    """A provider not manifest-declared for ImageRegion (paperless-ngx-mcp,
    the 'pdf_documents' alternate) never writes OCR-box loci, even when the
    (mocked) sidecar response includes boxes."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    monkeypatch.setattr(pdf_sidecar, "claim_engine", lambda: _FakeClaimEngine())
    monkeypatch.setattr(
        pdf_sidecar,
        "delegate_extract",
        lambda *a, **k: SidecarDelegationResult(
            available=True,
            modality="pdf",
            provider="paperless-ngx-mcp",
            tool="document_operations",
            action="post_document",
            activity_id="activity:1",
            raw={
                "pages": [
                    {
                        "page": 1,
                        "width": 612.0,
                        "height": 792.0,
                        "text": "hi",
                        "boxes": [
                            {"x": 0, "y": 0, "width": 1, "height": 1, "text": "hi"}
                        ],
                    }
                ]
            },
        ),
    )

    result = pdf_sidecar.ingest_pdf_via_sidecar(
        FIXTURE_PDF,
        document_id="doc-2",
        provider="pdf_documents",
        session=_session(),
        media_store=store,
    )
    assert result.available is True
    assert len(result.page_evidence_ids) == 1
    assert len(result.span_evidence_ids) == 1
    assert result.ocr_box_evidence_ids == []  # gated out — not manifest-declared


def test_pdf_sidecar_never_raises_when_media_store_write_fails(monkeypatch):
    class _FailingBlob(_FakeBlob):
        def store(self, data: bytes) -> str:  # noqa: D102
            raise RuntimeError("simulated blob store failure")

    client = _FakeClient()
    client.blob = _FailingBlob()
    store = MediaStore(_FakeCompute(client))

    result = pdf_sidecar.ingest_pdf_via_sidecar(
        FIXTURE_PDF, document_id="doc-3", session=_session(), media_store=store
    )
    assert result.available is False
    assert "failed to store the PDF artifact bundle" in result.error
