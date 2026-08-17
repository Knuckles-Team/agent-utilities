"""TCK-style conformance test for the video sidecar modality
(CONCEPT:AU-KG.ingest.media-sidecar-delegation, GOC-07): artifact -> shots +
keyframes -> VideoShot/VideoFrameRange loci -> queryable, proven end-to-end
against a FAKE engine client with the fleet sidecar call MOCKED at the seam —
no live fleet service required. Fixture video bytes only (an opaque byte
blob is all ``MediaStore`` needs; no real container parsing happens in this
suite — the sidecar's shots/keyframes are supplied directly. Bounded
malformed-container rejection for the video bytes THEMSELVES is proven at
the Rust ``eg-video`` layer, not here — see
``crates/eg-video/src/runtime.rs::malformed_or_metadata_only_container_is_rejected``.)
"""

from __future__ import annotations

import base64
import hashlib

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.knowledge_graph.memory.media_store import MediaStore
from agent_utilities.media import video_sidecar
from agent_utilities.media.sidecar_delegate import SidecarDelegationResult
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext

# -- the SAME fake engine-client shape as test_image_sidecar.py /
# test_audio_sidecar.py (this codebase's established per-file MediaStore
# test-double convention). --


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


FIXTURE_MP4 = b"\x00\x00\x00\x18ftypisom" + bytes(range(16))
FIXTURE_KEYFRAME_PNG = b"\x89PNG\r\n\x1a\n fixture keyframe"


def _mock_decode_result(**overrides):
    raw = {
        "shots": [
            {"start_ms": 0, "end_ms": 2000, "label": "intro", "confidence": 0.9},
            {"start_ms": 2000, "end_ms": 5000, "label": "main", "confidence": 0.86},
        ],
        "keyframes": [
            {
                "start_frame": 1,
                "end_frame": 1,
                "png_b64": base64.b64encode(FIXTURE_KEYFRAME_PNG).decode("ascii"),
            },
            {"frame_number": 30},
        ],
    }
    raw.update(overrides)
    return SidecarDelegationResult(
        available=True,
        modality="video",
        provider="data-science-mcp",
        tool="video_keyframes",
        action="extract_keyframes",
        activity_id="activity:media_sidecar:data-science-mcp:video:1:dddd4444",
        raw=raw,
    )


def test_video_conformance_artifact_shots_keyframes_loci_queryable(monkeypatch):
    """The full TCK chain for video: artifact -> bundle -> shots -> VideoShot
    loci + keyframes -> VideoFrameRange loci -> queryable."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    claim_engine = _FakeClaimEngine()

    monkeypatch.setattr(video_sidecar, "claim_engine", lambda: claim_engine)
    monkeypatch.setattr(
        video_sidecar, "delegate_extract", lambda *a, **k: _mock_decode_result()
    )

    result = video_sidecar.ingest_video_via_sidecar(
        FIXTURE_MP4,
        video_id="video-standup-1",
        source="test",
        session=_session(),
        media_store=store,
    )

    assert result.available is True
    assert result.occurrence_id is not None
    assert result.occurrence_id.startswith("occurrence:")
    occ_props = client.nodes.properties(result.occurrence_id)
    assert occ_props["media_type"] == "video/mp4"
    assert occ_props["content_digest"] == _digest(FIXTURE_MP4)

    # -- shot loci: one VideoShot per detected shot boundary (a wall-clock
    # time range) --
    assert result.shot_count == 2
    assert len(result.shot_evidence_ids) == 2
    shot1 = client.nodes.properties(result.shot_evidence_ids[0])
    assert shot1["evidence_locus"]["address"] == {
        "kind": "video_time_range",
        "start_ms": 0,
        "end_ms": 2000,
    }

    # -- frame-range loci: one VideoFrameRange per keyframe (a distinct
    # decoded-frame index range, never conflated with the shot's wall-clock
    # range) --
    assert result.keyframe_count == 2
    assert len(result.frame_range_evidence_ids) == 2
    frame1 = client.nodes.properties(result.frame_range_evidence_ids[0])
    assert frame1["evidence_locus"]["address"] == {
        "kind": "frame_range",
        "start_frame": 1,
        "end_frame": 1,
    }
    stored_frame_bytes = client.blob.fetch(
        client.nodes.properties(frame1["occurrence_id"])["content_digest"]
    )
    assert stored_frame_bytes == FIXTURE_KEYFRAME_PNG

    # second keyframe had no png_b64 -> honest JSON descriptor, not
    # fabricated pixels
    frame2 = client.nodes.properties(result.frame_range_evidence_ids[1])
    assert frame2["evidence_locus"]["address"] == {
        "kind": "frame_range",
        "start_frame": 30,
        "end_frame": 30,
    }
    stored_frame2_bytes = client.blob.fetch(
        client.nodes.properties(frame2["occurrence_id"])["content_digest"]
    )
    assert b"30" in stored_frame2_bytes

    # -- queryable: all loci SUPPORT the ONE governed claim --
    assert result.claim_id is not None
    assert len(claim_engine.nodes) == 1
    claim_node_id, claim_node_type, claim_props = claim_engine.nodes[0]
    assert claim_node_id == result.claim_id
    assert claim_node_type == "Claim"
    assert claim_props["metadata"]["was_generated_by"] == "agent:data-science-mcp"

    supports_edges = [
        (s, t)
        for s, t, props in client.edges.edges
        if props.get("relationship") == "SUPPORTS"
    ]
    assert {t for _s, t in supports_edges} == {result.claim_id}
    assert len(supports_edges) == len(result.shot_evidence_ids) + len(
        result.frame_range_evidence_ids
    )


def test_video_sidecar_skips_malformed_shots_and_keyframes_without_fabricating_bounds(
    monkeypatch,
):
    """A malformed sidecar payload — a shot missing bounds, an inverted
    range, a non-dict entry, and a keyframe with a non-positive frame
    number — is skipped rather than guessed at or partially materialized
    into a plausible-looking (but fabricated) evidence locus."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    monkeypatch.setattr(video_sidecar, "claim_engine", lambda: _FakeClaimEngine())
    monkeypatch.setattr(
        video_sidecar,
        "delegate_extract",
        lambda *a, **k: _mock_decode_result(
            shots=[
                {"start_ms": 0, "end_ms": 1000, "label": "ok"},
                {"label": "missing bounds"},
                {"start_ms": 9000, "end_ms": 1000, "label": "inverted"},
                "not-even-a-dict",
            ],
            keyframes=[
                {"frame_number": 1},
                {"frame_number": 0},
                {"start_frame": 5, "end_frame": 3},
                42,
            ],
        ),
    )

    result = video_sidecar.ingest_video_via_sidecar(
        FIXTURE_MP4, video_id="video-hostile-1", session=_session(), media_store=store
    )
    assert result.available is True
    assert result.shot_count == 4  # the sidecar's reported count is honest
    assert result.keyframe_count == 4
    assert len(result.shot_evidence_ids) == 1  # only the well-formed shot was written
    assert len(result.frame_range_evidence_ids) == 1  # only the well-formed keyframe


def test_video_sidecar_degrades_cleanly_when_delegation_fails(monkeypatch):
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))

    monkeypatch.setattr(
        video_sidecar,
        "delegate_extract",
        lambda *a, **k: SidecarDelegationResult(
            available=False,
            modality="video",
            provider="data-science-mcp",
            tool="video_keyframes",
            action="extract_keyframes",
            activity_id=None,
            raw={},
            error="ConnectionError: sidecar unreachable",
        ),
    )

    result = video_sidecar.ingest_video_via_sidecar(
        FIXTURE_MP4, video_id="video-unreachable", session=_session(), media_store=store
    )
    assert result.available is False
    assert result.error == "ConnectionError: sidecar unreachable"
    assert result.occurrence_id is not None  # the raw bytes were still stored
    assert result.shot_evidence_ids == []
    assert result.frame_range_evidence_ids == []


def test_video_sidecar_never_raises_when_media_store_write_fails(monkeypatch):
    class _FailingBlob(_FakeBlob):
        def store(self, data: bytes) -> str:  # noqa: D102
            raise RuntimeError("simulated blob store failure")

    client = _FakeClient()
    client.blob = _FailingBlob()
    store = MediaStore(_FakeCompute(client))

    result = video_sidecar.ingest_video_via_sidecar(
        FIXTURE_MP4, video_id="video-3", session=_session(), media_store=store
    )
    assert result.available is False
    assert "failed to store the video artifact bundle" in result.error
