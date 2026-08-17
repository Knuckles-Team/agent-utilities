"""TCK-style conformance test for the audio sidecar modality
(CONCEPT:AU-KG.ingest.media-sidecar-delegation, GOC-07): artifact ->
transcript segments -> AudioSegment loci -> queryable, proven end-to-end
against a FAKE engine client with the fleet sidecar call MOCKED at the seam —
no live fleet service required. Fixture audio bytes only (an opaque byte
blob is all ``MediaStore`` needs; no real audio decode happens in this
suite — the sidecar's transcript segments are supplied directly. Bounded
malformed-container rejection for the audio bytes THEMSELVES is proven at
the Rust ``eg-audio`` layer, not here — see
``crates/eg-audio/src/runtime.rs::malformed_or_unsupported_audio_is_rejected``.)
"""

from __future__ import annotations

import hashlib

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.knowledge_graph.memory.media_store import MediaStore
from agent_utilities.media import audio_sidecar
from agent_utilities.media.sidecar_delegate import SidecarDelegationResult
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext

# -- the SAME fake engine-client shape as test_image_sidecar.py /
# test_pdf_sidecar.py (this codebase's established per-file MediaStore
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


# A minimal (but not decode-attempted by this suite) RIFF/WAVE header.
FIXTURE_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt " + bytes(range(32))


def _mock_decode_result(**overrides):
    raw = {
        "segments": [
            {"start_ms": 0, "end_ms": 1500, "text": "hello there", "confidence": 0.95},
            {"start_ms": 1500, "end_ms": 3200, "text": "general kenobi", "confidence": 0.88},
        ],
    }
    raw.update(overrides)
    return SidecarDelegationResult(
        available=True,
        modality="audio",
        provider="audio-transcriber-mcp",
        tool="transcribe_media",
        action="transcribe_segments",
        activity_id="activity:media_sidecar:audio-transcriber-mcp:audio:1:cccc3333",
        raw=raw,
    )


def test_audio_conformance_artifact_segments_loci_queryable(monkeypatch):
    """The full TCK chain for audio: artifact -> bundle -> transcript
    segments -> AudioSegment loci -> queryable."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    claim_engine = _FakeClaimEngine()

    monkeypatch.setattr(audio_sidecar, "claim_engine", lambda: claim_engine)
    monkeypatch.setattr(
        audio_sidecar, "delegate_extract", lambda *a, **k: _mock_decode_result()
    )

    result = audio_sidecar.ingest_audio_via_sidecar(
        FIXTURE_WAV,
        audio_id="audio-briefing-1",
        source="test",
        session=_session(),
        media_store=store,
    )

    # -- artifact -> bundle: the raw audio stored as-is (opaque blob, no
    # decode needed to store bytes) --
    assert result.available is True
    assert result.occurrence_id is not None
    assert result.occurrence_id.startswith("occurrence:")
    assert result.blob_id is not None
    occ_props = client.nodes.properties(result.occurrence_id)
    assert occ_props["node_type"] == "AssetOccurrence"
    assert occ_props["media_type"] == "audio/wav"
    assert occ_props["content_digest"] == _digest(FIXTURE_WAV)

    # -- segment loci: one AudioSegment per transcribed segment, never a
    # bare opaque transcript reference masquerading as resolved text --
    assert result.segment_count == 2
    assert len(result.segment_evidence_ids) == 2
    seg1 = client.nodes.properties(result.segment_evidence_ids[0])
    seg1_locus_token = result.segment_evidence_ids[0].split(":", 1)[1]
    seg1_occurrence_token = seg1["occurrence_id"].split(":", 1)[1]
    assert seg1["evidence_locus"] == {
        "id": f"eg:locus:{seg1_locus_token}",
        "subject": {
            "kind": "occurrence",
            "id": f"eg:occurrence:{seg1_occurrence_token}",
        },
        "address": {"kind": "audio_range", "start_ms": 0, "end_ms": 1500},
        "policy_ref": f"eg:policy:{seg1_locus_token}",
        "derivation_ref": f"eg:derivation:{seg1_locus_token}",
    }
    stored_text = client.blob.fetch(
        client.nodes.properties(seg1["occurrence_id"])["content_digest"]
    )
    assert stored_text == b"hello there"

    # -- queryable: both segment loci SUPPORT the ONE governed claim --
    assert result.claim_id is not None
    assert len(claim_engine.nodes) == 1
    claim_node_id, claim_node_type, claim_props = claim_engine.nodes[0]
    assert claim_node_id == result.claim_id
    assert claim_node_type == "Claim"
    assert claim_props["is_verified"] is True
    assert claim_props["metadata"]["was_generated_by"] == "agent:audio-transcriber-mcp"

    supports_edges = [
        (s, t)
        for s, t, props in client.edges.edges
        if props.get("relationship") == "SUPPORTS"
    ]
    assert {t for _s, t in supports_edges} == {result.claim_id}
    assert len(supports_edges) == len(result.segment_evidence_ids)


def test_audio_sidecar_skips_malformed_segments_without_fabricating_bounds(monkeypatch):
    """A malformed sidecar payload — a segment missing start_ms/end_ms, one
    with end_ms <= start_ms, and one that isn't even a dict — is skipped
    rather than guessed at or partially materialized into a plausible-
    looking (but fabricated) evidence locus. Only the one well-formed
    segment is written."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    monkeypatch.setattr(audio_sidecar, "claim_engine", lambda: _FakeClaimEngine())
    monkeypatch.setattr(
        audio_sidecar,
        "delegate_extract",
        lambda *a, **k: _mock_decode_result(
            segments=[
                {"start_ms": 0, "end_ms": 1000, "text": "well formed"},
                {"text": "missing bounds entirely"},
                {"start_ms": 5000, "end_ms": 4000, "text": "inverted range"},
                "not-even-a-dict",
                {"start_ms": -1, "end_ms": 100, "text": "negative start"},
            ]
        ),
    )

    result = audio_sidecar.ingest_audio_via_sidecar(
        FIXTURE_WAV, audio_id="audio-hostile-1", session=_session(), media_store=store
    )
    assert result.available is True
    assert result.segment_count == 5  # the sidecar's reported count is honest
    assert len(result.segment_evidence_ids) == 1  # only the well-formed one was written
    kept = client.nodes.properties(result.segment_evidence_ids[0])
    assert kept["evidence_locus"]["address"] == {
        "kind": "audio_range",
        "start_ms": 0,
        "end_ms": 1000,
    }


def test_audio_sidecar_degrades_cleanly_when_delegation_fails(monkeypatch):
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))

    monkeypatch.setattr(
        audio_sidecar,
        "delegate_extract",
        lambda *a, **k: SidecarDelegationResult(
            available=False,
            modality="audio",
            provider="audio-transcriber-mcp",
            tool="transcribe_media",
            action="transcribe_segments",
            activity_id=None,
            raw={},
            error="ConnectionError: sidecar unreachable",
        ),
    )

    result = audio_sidecar.ingest_audio_via_sidecar(
        FIXTURE_WAV, audio_id="audio-unreachable", session=_session(), media_store=store
    )
    assert result.available is False
    assert result.error == "ConnectionError: sidecar unreachable"
    assert result.occurrence_id is not None  # the raw bytes were still stored
    assert result.segment_evidence_ids == []


def test_audio_sidecar_never_raises_when_media_store_write_fails(monkeypatch):
    class _FailingBlob(_FakeBlob):
        def store(self, data: bytes) -> str:  # noqa: D102
            raise RuntimeError("simulated blob store failure")

    client = _FakeClient()
    client.blob = _FailingBlob()
    store = MediaStore(_FakeCompute(client))

    result = audio_sidecar.ingest_audio_via_sidecar(
        FIXTURE_WAV, audio_id="audio-3", session=_session(), media_store=store
    )
    assert result.available is False
    assert "failed to store the audio artifact bundle" in result.error
