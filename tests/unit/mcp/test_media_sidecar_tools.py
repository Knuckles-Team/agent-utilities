"""Unit tests for the ``graph_media_sidecar`` MCP tool
(CONCEPT:AU-KG.ingest.media-sidecar-delegation) — the two-surface (MCP + REST)
entry point for the W4.6 sidecar delegate pattern. The adapter functions
(``ingest_pdf_via_sidecar``/``ingest_jpeg_via_sidecar``) are MOCKED here; their
own write-back behavior is proven by the TCK conformance tests in
``tests/unit/media/``. No live engine or fleet service is required.
"""

from __future__ import annotations

import base64
import json

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import media_sidecar_tools
from agent_utilities.media.image_sidecar import ImageSidecarResult
from agent_utilities.media.pdf_sidecar import PdfSidecarResult


class _CollectingMCP:
    """Minimal FastMCP stand-in that captures ``@mcp.tool``-registered functions."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            return fn

        return _deco


@pytest.fixture
def tools() -> dict[str, object]:
    mcp = _CollectingMCP()
    media_sidecar_tools.register_media_sidecar_tools(mcp)
    return mcp.tools


def test_graph_media_sidecar_registered_with_rest_twin(tools):
    assert "graph_media_sidecar" in tools
    assert kg_server.REGISTERED_TOOLS.get("graph_media_sidecar") is not None
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_media_sidecar") == "/media/sidecar"


def test_graph_media_sidecar_ingest_pdf_dispatches_to_the_pdf_adapter(
    monkeypatch, tools
):
    captured: dict = {}

    def _fake_ingest(data, *, document_id, source, provider):
        captured["data"] = data
        captured["document_id"] = document_id
        captured["source"] = source
        captured["provider"] = provider
        return PdfSidecarResult(
            available=True,
            document_id=document_id,
            occurrence_id="occurrence:1",
            blob_id="blob:1",
            claim_id="claim:1",
            page_count=1,
            page_evidence_ids=["evidence:1"],
        )

    monkeypatch.setattr(media_sidecar_tools, "ingest_pdf_via_sidecar", _fake_ingest)

    out = json.loads(
        tools["graph_media_sidecar"](
            action="ingest_pdf",
            artifact_id="doc-1",
            data_b64=base64.b64encode(b"%PDF-1.4 bytes").decode("ascii"),
            source="unit-test",
        )
    )
    assert out["available"] is True
    assert out["document_id"] == "doc-1"
    assert out["page_count"] == 1
    assert captured["data"] == b"%PDF-1.4 bytes"
    assert captured["document_id"] == "doc-1"
    assert captured["source"] == "unit-test"


def test_graph_media_sidecar_ingest_jpeg_dispatches_to_the_image_adapter(
    monkeypatch, tools
):
    def _fake_ingest(data, *, image_id, source, provider):
        return ImageSidecarResult(
            available=True,
            image_id=image_id,
            occurrence_id="occurrence:2",
            blob_id="blob:2",
            rendition_id="rendition:1",
            phash="deadbeef",
            claim_id="claim:2",
            region_evidence_ids=["evidence:2", "evidence:3"],
        )

    monkeypatch.setattr(media_sidecar_tools, "ingest_jpeg_via_sidecar", _fake_ingest)

    out = json.loads(
        tools["graph_media_sidecar"](
            action="ingest_jpeg",
            artifact_id="img-1",
            data_b64=base64.b64encode(b"\xff\xd8\xff\xe0jpeg").decode("ascii"),
        )
    )
    assert out["available"] is True
    assert out["image_id"] == "img-1"
    assert out["phash"] == "deadbeef"
    assert len(out["region_evidence_ids"]) == 2


@pytest.mark.parametrize(
    "action,modality", [("ingest_audio", "audio"), ("ingest_video", "video")]
)
def test_graph_media_sidecar_stub_actions_degrade_cleanly(tools, action, modality):
    out = json.loads(
        tools["graph_media_sidecar"](action=action, artifact_id="x", data_b64="Zm9v")
    )
    assert out["available"] is False
    assert out["stub"] is True
    assert modality in out["error"]
    assert "DESIGN STUB" in out["error"]


def test_graph_media_sidecar_unknown_action_is_reported(tools):
    out = json.loads(tools["graph_media_sidecar"](action="ingest_gif"))
    assert "unknown action" in out["error"]


def test_graph_media_sidecar_missing_artifact_id_is_reported(tools):
    out = json.loads(tools["graph_media_sidecar"](action="ingest_pdf", data_b64="Zm9v"))
    assert out["error"] == "artifact_id is required"


def test_graph_media_sidecar_missing_data_is_reported(tools):
    out = json.loads(
        tools["graph_media_sidecar"](action="ingest_pdf", artifact_id="doc-1")
    )
    assert out["error"] == "data_b64 is required"


def test_graph_media_sidecar_invalid_base64_is_reported(tools):
    out = json.loads(
        tools["graph_media_sidecar"](
            action="ingest_pdf", artifact_id="doc-1", data_b64="not-valid-base64!!"
        )
    )
    assert "error" in out


def test_graph_media_sidecar_adapter_exception_degrades_cleanly(monkeypatch, tools):
    def _boom(data, *, document_id, source, provider):
        raise RuntimeError("boom")

    monkeypatch.setattr(media_sidecar_tools, "ingest_pdf_via_sidecar", _boom)

    out = json.loads(
        tools["graph_media_sidecar"](
            action="ingest_pdf",
            artifact_id="doc-1",
            data_b64=base64.b64encode(b"data").decode("ascii"),
        )
    )
    assert "error" in out
    assert out["error"]["code"] == "operation_failed"
