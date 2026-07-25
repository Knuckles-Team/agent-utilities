"""TCK-style conformance test for the JPEG sidecar modality
(CONCEPT:AU-KG.ingest.media-sidecar-delegation): artifact -> pHash/thumbnail ->
region loci -> embedding -> queryable, proven end-to-end against a FAKE
engine client with the fleet sidecar call MOCKED at the seam — no live fleet
service required. Fixture JPEG bytes only (an opaque byte blob is all
``MediaStore`` needs; no real JPEG decode happens in this suite — the
sidecar's decoded pHash/thumbnail/regions are supplied directly).
"""

from __future__ import annotations

import base64
import hashlib

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.knowledge_graph.memory.media_store import MediaStore
from agent_utilities.media import image_sidecar
from agent_utilities.media.sidecar_delegate import SidecarDelegationResult
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext

# -- the SAME fake engine-client shape as test_pdf_sidecar.py /
# test_media_store_evidence_spine.py (this codebase's established per-file
# MediaStore test-double convention). --


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


FIXTURE_JPEG = b"\xff\xd8\xff\xe0 fixture jpeg bytes " + bytes(range(32)) + b"\xff\xd9"
FIXTURE_THUMBNAIL_PNG = b"\x89PNG\r\n\x1a\n fixture decoded thumbnail"


def _mock_decode_result(**overrides):
    raw = {
        "phash": "9f3a1c7e0b2d4f6a",
        "thumbnail_png_b64": base64.b64encode(FIXTURE_THUMBNAIL_PNG).decode("ascii"),
        "embedding": [0.4, 0.5, 0.6],
        "regions": [
            {
                "x": 5.0,
                "y": 6.0,
                "width": 40.0,
                "height": 40.0,
                "label": "face",
                "confidence": 0.97,
            },
            {
                "x": 60.0,
                "y": 10.0,
                "width": 20.0,
                "height": 20.0,
                "label": "logo",
                "confidence": 0.81,
            },
        ],
    }
    raw.update(overrides)
    return SidecarDelegationResult(
        available=True,
        modality="jpeg",
        provider="data-science-mcp",
        tool="image_decode",
        action="analyze_image",
        activity_id="activity:media_sidecar:data-science-mcp:jpeg:1:bbbb2222",
        raw=raw,
    )


def test_jpeg_conformance_artifact_phash_regions_embedding_queryable(monkeypatch):
    """The full TCK chain for JPEG: artifact -> bundle -> pHash/thumbnail ->
    region loci -> embedding -> queryable."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    claim_engine = _FakeClaimEngine()

    monkeypatch.setattr(image_sidecar, "claim_engine", lambda: claim_engine)
    monkeypatch.setattr(
        image_sidecar, "delegate_extract", lambda *a, **k: _mock_decode_result()
    )

    result = image_sidecar.ingest_jpeg_via_sidecar(
        FIXTURE_JPEG,
        image_id="img-portrait-1",
        source="test",
        session=_session(),
        media_store=store,
    )

    # -- artifact -> bundle: the raw JPEG stored as-is (opaque blob, no decode
    # needed to store bytes) --
    assert result.available is True
    assert result.occurrence_id is not None
    assert result.occurrence_id.startswith("occurrence:")
    assert result.blob_id is not None
    occ_props = client.nodes.properties(result.occurrence_id)
    assert occ_props["type"] == "AssetOccurrence"
    assert occ_props["media_type"] == "image/jpeg"
    assert occ_props["content_digest"] == _digest(FIXTURE_JPEG)

    # -- pHash + embedding: recorded via MediaStore's existing async
    # extraction seam (record_extraction) onto the SAME occurrence, not a
    # new engine surface --
    assert result.phash == "9f3a1c7e0b2d4f6a"
    occ_props_after = client.nodes.properties(result.occurrence_id)
    assert occ_props_after["phash"] == "9f3a1c7e0b2d4f6a"
    assert occ_props_after["extraction_model"] == "data-science-mcp"
    assert client.txn.embeddings[result.occurrence_id] == [0.4, 0.5, 0.6]

    # -- decoded thumbnail -> PNG :Rendition (re-entering the engine's own
    # native codec) --
    assert result.rendition_id is not None
    assert result.rendition_id.startswith("rendition:")
    rendition_props = client.nodes.properties(result.rendition_id)
    assert rendition_props["type"] == "Rendition"
    assert rendition_props["rendition_type"] == "thumbnail"
    assert rendition_props["mime_type"] == "image/png"
    assert rendition_props["derived_from_digest"] == _digest(FIXTURE_JPEG)
    assert rendition_props["content_digest"] == _digest(FIXTURE_THUMBNAIL_PNG)

    # -- region loci: one ImageRegion per detected region --
    assert len(result.region_evidence_ids) == 2
    region1 = client.nodes.properties(result.region_evidence_ids[0])
    assert region1["evidence_span"] == {
        "ImageRegion": {
            "image_id": "img-portrait-1",
            "x": 5.0,
            "y": 6.0,
            "width": 40.0,
            "height": 40.0,
        }
    }

    # -- queryable: both region loci SUPPORT the ONE governed claim --
    assert result.claim_id is not None
    assert len(claim_engine.nodes) == 1
    claim_node_id, claim_node_type, claim_props = claim_engine.nodes[0]
    assert claim_node_id == result.claim_id
    assert claim_node_type == "Claim"
    assert claim_props["is_verified"] is True
    assert claim_props["metadata"]["was_generated_by"] == "agent:data-science-mcp"

    supports_edges = [
        (s, t)
        for s, t, props in client.edges.edges
        if props.get("relationship_type") == "SUPPORTS"
    ]
    assert {t for _s, t in supports_edges} == {result.claim_id}
    assert len(supports_edges) == len(result.region_evidence_ids)


def test_jpeg_sidecar_honest_degrade_when_no_crop_bytes_are_returned(monkeypatch):
    """A region with no crop_png_b64 (the common case — full-image crops
    would bloat the sidecar payload) stores a compact JSON descriptor
    instead of fabricating pixel bytes (the no-invented-data charter)."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    monkeypatch.setattr(image_sidecar, "claim_engine", lambda: _FakeClaimEngine())
    monkeypatch.setattr(
        image_sidecar,
        "delegate_extract",
        lambda *a, **k: _mock_decode_result(
            regions=[{"x": 1.0, "y": 1.0, "width": 2.0, "height": 2.0, "label": "dog"}],
            thumbnail_png_b64=None,
        ),
    )

    result = image_sidecar.ingest_jpeg_via_sidecar(
        FIXTURE_JPEG, image_id="img-2", session=_session(), media_store=store
    )
    assert result.available is True
    assert result.rendition_id is None  # no thumbnail bytes -> no rendition
    assert len(result.region_evidence_ids) == 1
    # the stored bytes for the region are a real (if minimal) JSON
    # descriptor, not fabricated pixels — recoverable via the blob digest.
    region_evidence = client.nodes.properties(result.region_evidence_ids[0])
    stored_bytes = client.blob.fetch(
        client.nodes.properties(region_evidence["occurrence_id"])["content_digest"]
    )
    assert b"dog" in stored_bytes


def test_jpeg_sidecar_degrades_cleanly_when_delegation_fails(monkeypatch):
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))

    monkeypatch.setattr(
        image_sidecar,
        "delegate_extract",
        lambda *a, **k: SidecarDelegationResult(
            available=False,
            modality="jpeg",
            provider="data-science-mcp",
            tool="image_decode",
            action="analyze_image",
            activity_id=None,
            raw={},
            error="ConnectionError: sidecar unreachable",
        ),
    )

    result = image_sidecar.ingest_jpeg_via_sidecar(
        FIXTURE_JPEG, image_id="img-unreachable", session=_session(), media_store=store
    )
    assert result.available is False
    assert result.error == "ConnectionError: sidecar unreachable"
    assert result.occurrence_id is not None  # the raw bytes were still stored
    assert result.region_evidence_ids == []


def test_jpeg_sidecar_never_raises_when_media_store_write_fails(monkeypatch):
    class _FailingBlob(_FakeBlob):
        def store(self, data: bytes) -> str:  # noqa: D102
            raise RuntimeError("simulated blob store failure")

    client = _FakeClient()
    client.blob = _FailingBlob()
    store = MediaStore(_FakeCompute(client))

    result = image_sidecar.ingest_jpeg_via_sidecar(
        FIXTURE_JPEG, image_id="img-3", session=_session(), media_store=store
    )
    assert result.available is False
    assert "failed to store the JPEG artifact bundle" in result.error
