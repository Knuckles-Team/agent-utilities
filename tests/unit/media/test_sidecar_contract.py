"""Unit tests for the fail-closed media-sidecar capability manifest
(CONCEPT:AU-KG.ingest.media-sidecar-delegation).
"""

from __future__ import annotations

import pytest

from agent_utilities.media import sidecar_contract as sc


def test_every_declared_capability_produces_only_known_locus_kinds():
    for key, capability in sc.SIDECAR_CAPABILITIES.items():
        assert capability.produces <= sc.LOCUS_KINDS, key


def test_capability_for_resolves_the_default_provider():
    capability = sc.capability_for("pdf")
    assert capability.server == "stirlingpdf-mcp"
    assert capability.modality == "pdf"


def test_capability_for_resolves_an_explicit_alternate_provider():
    capability = sc.capability_for("pdf", provider="pdf_documents")
    assert capability.server == "paperless-ngx-mcp"


def test_capability_for_rejects_unknown_modality():
    with pytest.raises(sc.SidecarContractError):
        sc.capability_for("tiff")


def test_capability_for_rejects_a_provider_for_the_wrong_modality():
    with pytest.raises(sc.SidecarContractError):
        sc.capability_for("jpeg", provider="pdf_documents")


def test_assert_capable_passes_for_a_declared_locus_kind():
    capability = sc.capability_for("pdf")
    sc.assert_capable(capability, "PageBox")  # must not raise


def test_assert_capable_fails_closed_for_an_undeclared_locus_kind():
    capability = sc.capability_for("pdf_documents")  # no ImageRegion declared
    with pytest.raises(sc.SidecarContractError):
        sc.assert_capable(capability, "ImageRegion")


def test_is_capable_is_the_non_raising_twin():
    capability = sc.capability_for("pdf_documents")
    assert sc.is_capable(capability, "PageBox") is True
    assert sc.is_capable(capability, "ImageRegion") is False


def test_implemented_and_stub_modalities_are_disjoint_and_declared():
    assert sc.IMPLEMENTED_MODALITIES.isdisjoint(sc.STUB_MODALITIES)
    for modality in sc.IMPLEMENTED_MODALITIES | sc.STUB_MODALITIES:
        assert sc.capability_for(modality) is not None


def test_audio_and_video_are_implemented_capabilities():
    """GOC-07 (audio/video modality stack) wired the adapters the W4.6 ADR
    declared as design stubs — both modalities are now in
    IMPLEMENTED_MODALITIES with a real adapter module
    (``audio_sidecar.py``/``video_sidecar.py``), and STUB_MODALITIES is
    empty."""
    assert sc.STUB_MODALITIES == frozenset()
    assert {"audio", "video"} <= sc.IMPLEMENTED_MODALITIES
    audio = sc.capability_for("audio")
    assert audio.produces == frozenset({"AudioSegment"})
    video = sc.capability_for("video")
    assert video.produces == frozenset({"VideoShot", "VideoFrameRange"})


def test_a_capability_declaring_an_unknown_locus_kind_fails_closed_at_construction():
    with pytest.raises(sc.SidecarContractError):
        sc.SidecarCapability(
            modality="bogus",
            server="bogus-mcp",
            tool="bogus_tool",
            action="bogus_action",
            produces=frozenset({"NotARealLocusKind"}),
        )
