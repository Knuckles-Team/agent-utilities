"""Unit tests for the reusable sidecar delegate loop
(CONCEPT:AU-KG.ingest.media-sidecar-delegation) — the ``graph_mine_deep``-style
component both shipped modalities (PDF, JPEG) share.

No live fleet service is required: the fleet call (``call_tool_once``) and the
PROV-O activity write (``lineage.record_media_sidecar_activity``) are MOCKED
at the seam, matching the task's own acceptance bar.
"""

from __future__ import annotations

import base64
import json

import pytest

from agent_utilities.knowledge_graph.etl import lineage
from agent_utilities.media import sidecar_delegate as sd


class _FakeClaimEngine:
    def __init__(self) -> None:
        self.nodes: list[tuple[str, object, dict]] = []

    def add_node(self, node_id, node_type, properties):
        self.nodes.append((node_id, node_type, dict(properties)))


@pytest.fixture
def fake_engine(monkeypatch) -> _FakeClaimEngine:
    engine = _FakeClaimEngine()
    monkeypatch.setattr(sd, "claim_engine", lambda: engine)
    return engine


def test_delegate_extract_unknown_modality_degrades_cleanly_without_a_fleet_call(
    monkeypatch,
):
    called = False

    async def _boom(**kwargs):
        nonlocal called
        called = True
        raise AssertionError(
            "call_tool_once must not be reached for an unknown modality"
        )

    monkeypatch.setattr(sd, "call_tool_once", _boom)

    result = sd.delegate_extract(
        b"bytes", digest="deadbeef", media_type="image/tiff", modality="tiff"
    )
    assert result.available is False
    assert "no declared sidecar capability" in result.error
    assert called is False


def test_delegate_extract_ships_the_expected_wire_shape(monkeypatch, fake_engine):
    captured: dict = {}

    async def _fake_call_tool_once(**kwargs):
        captured.update(kwargs)
        return {"available": True, "pages": []}

    monkeypatch.setattr(sd, "call_tool_once", _fake_call_tool_once)

    data = b"%PDF-1.4 fixture bytes"
    result = sd.delegate_extract(
        data, digest="abc123", media_type="application/pdf", modality="pdf"
    )

    assert result.available is True
    assert result.provider == "stirlingpdf-mcp"
    assert result.tool == "pdf_action"
    assert result.action == "ocr_pdf"
    assert captured["server"] == "stirlingpdf-mcp"
    assert captured["tool"] == "pdf_action"
    assert captured["action"] == "ocr_pdf"
    assert captured["params_style"] == "json"
    assert captured["params"]["digest"] == "abc123"
    assert captured["params"]["media_type"] == "application/pdf"
    assert base64.b64decode(captured["params"]["artifact_b64"]) == data


def test_delegate_extract_merges_extra_params(monkeypatch, fake_engine):
    captured: dict = {}

    async def _fake_call_tool_once(**kwargs):
        captured.update(kwargs)
        return {"available": True}

    monkeypatch.setattr(sd, "call_tool_once", _fake_call_tool_once)

    sd.delegate_extract(
        b"jpeg-bytes",
        digest="d1",
        media_type="image/jpeg",
        modality="jpeg",
        extra_params={"phash_size": 8},
    )
    assert captured["params"]["phash_size"] == 8
    # the universal payload keys are still present alongside the tuning knob
    assert "artifact_b64" in captured["params"]


def test_delegate_extract_records_one_prov_o_activity_node(monkeypatch, fake_engine):
    async def _fake_call_tool_once(**kwargs):
        return {"available": True, "pages": []}

    monkeypatch.setattr(sd, "call_tool_once", _fake_call_tool_once)

    result = sd.delegate_extract(
        b"data", digest="digest1", media_type="application/pdf", modality="pdf"
    )
    assert result.activity_id is not None
    assert len(fake_engine.nodes) == 1
    node_id, node_type, props = fake_engine.nodes[0]
    assert node_id == result.activity_id
    assert props["kind"] == "media_sidecar"
    assert props["sidecar"] == "stirlingpdf-mcp"
    assert props["tool"] == "pdf_action"
    assert props["modality"] == "pdf"
    assert props["action"] == "ocr_pdf"


def test_delegate_extract_degrades_when_sidecar_unreachable(monkeypatch, fake_engine):
    async def _boom(**kwargs):
        raise ConnectionError("no route to stirlingpdf-mcp")

    monkeypatch.setattr(sd, "call_tool_once", _boom)

    result = sd.delegate_extract(
        b"data", digest="d", media_type="application/pdf", modality="pdf"
    )
    assert result.available is False
    assert "sidecar unreachable" in result.error
    # a PROV-O activity was still recorded (the call was attempted, even
    # though it failed) — provenance covers attempted delegations too.
    assert result.activity_id is not None


def test_delegate_extract_activity_recording_failure_never_blocks_the_call(
    monkeypatch,
):
    def _boom_engine():
        raise RuntimeError("engine not active")

    monkeypatch.setattr(sd, "claim_engine", _boom_engine)

    async def _fake_call_tool_once(**kwargs):
        return {"available": True, "pages": []}

    monkeypatch.setattr(sd, "call_tool_once", _fake_call_tool_once)

    result = sd.delegate_extract(
        b"data", digest="d", media_type="application/pdf", modality="pdf"
    )
    assert result.available is True
    assert result.activity_id is None  # provenance best-effort, never raised


def test_delegate_extract_passes_through_sidecar_reported_unavailable(
    monkeypatch, fake_engine
):
    async def _fake_call_tool_once(**kwargs):
        return {"available": False, "error": "OCR engine not configured"}

    monkeypatch.setattr(sd, "call_tool_once", _fake_call_tool_once)

    result = sd.delegate_extract(
        b"data", digest="d", media_type="application/pdf", modality="pdf"
    )
    assert result.available is False
    assert result.error == "OCR engine not configured"


# ── BUG-7-style: an already-JSON-encoded STRING response must still parse ──
def test_delegate_extract_parses_a_json_encoded_string_response(
    monkeypatch, fake_engine
):
    async def _fake_call_tool_once(**kwargs):
        return json.dumps({"available": True, "pages": [{"page": 1, "text": "hi"}]})

    monkeypatch.setattr(sd, "call_tool_once", _fake_call_tool_once)

    result = sd.delegate_extract(
        b"data", digest="d", media_type="application/pdf", modality="pdf"
    )
    assert result.available is True
    assert result.raw["pages"][0]["text"] == "hi"


def test_delegate_extract_reports_a_genuinely_non_json_string_as_a_shape_error(
    monkeypatch, fake_engine
):
    async def _fake_call_tool_once(**kwargs):
        return "not json at all"

    monkeypatch.setattr(sd, "call_tool_once", _fake_call_tool_once)

    result = sd.delegate_extract(
        b"data", digest="d", media_type="application/pdf", modality="pdf"
    )
    assert result.available is False
    assert "unexpected sidecar response shape" in result.error


def test_delegate_extract_resolves_an_explicit_alternate_provider(
    monkeypatch, fake_engine
):
    captured: dict = {}

    async def _fake_call_tool_once(**kwargs):
        captured.update(kwargs)
        return {"available": True, "pages": []}

    monkeypatch.setattr(sd, "call_tool_once", _fake_call_tool_once)

    result = sd.delegate_extract(
        b"data",
        digest="d",
        media_type="application/pdf",
        modality="pdf",
        provider="pdf_documents",
    )
    assert result.provider == "paperless-ngx-mcp"
    assert captured["server"] == "paperless-ngx-mcp"
    assert captured["action"] == "post_document"


def test_claim_engine_resolves_the_default_connection_registry_engine(monkeypatch):
    """claim_engine() reaches the SAME default engine graph_write's
    non-fan-out path resolves — never MediaStore's own GraphComputeEngine
    currency (see the module docstring)."""
    from agent_utilities.mcp import kg_server

    sentinel = object()

    class _FakeRegistry:
        def get_engine(self, name):
            assert name == ""
            return sentinel

    monkeypatch.setattr(kg_server, "get_connection_registry", lambda: _FakeRegistry())
    assert sd.claim_engine() is sentinel


def test_record_media_sidecar_claim_links_activity_and_carries_prov_o_metadata():
    engine = _FakeClaimEngine()
    claim_id = lineage.record_media_sidecar_claim(
        engine,
        sidecar="stirlingpdf-mcp",
        modality="pdf",
        artifact_id="doc-1",
        summary="stirlingpdf-mcp extracted 3 page(s) from doc-1",
        activity_id="activity:media_sidecar:stirlingpdf-mcp:pdf:123:abcd1234",
    )
    assert claim_id is not None
    assert len(engine.nodes) == 1
    node_id, node_type, props = engine.nodes[0]
    assert node_id == claim_id
    assert node_type == "Claim"
    assert props["confidence"] == 1.0
    assert props["is_verified"] is True
    assert props["source_ids"] == [
        "activity:media_sidecar:stirlingpdf-mcp:pdf:123:abcd1234"
    ]
    assert props["metadata"]["was_generated_by"] == "agent:stirlingpdf-mcp"
    assert props["metadata"]["artifact_id"] == "doc-1"
    assert "generated_at_time" in props["metadata"]


def test_record_media_sidecar_activity_and_claim_are_best_effort():
    class _NoAddNode:
        pass

    assert (
        lineage.record_media_sidecar_activity(
            _NoAddNode(), sidecar="s", tool="t", modality="pdf"
        )
        is None
    )
    assert (
        lineage.record_media_sidecar_claim(
            _NoAddNode(),
            sidecar="s",
            modality="pdf",
            artifact_id="a",
            summary="x",
            activity_id=None,
        )
        is None
    )
    assert (
        lineage.record_media_sidecar_activity(
            None, sidecar="s", tool="t", modality="pdf"
        )
        is None
    )
