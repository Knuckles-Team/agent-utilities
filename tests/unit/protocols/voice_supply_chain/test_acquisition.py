"""Tests for governed Piper voice-model acquisition (GOC-36-W03).

Uses ``httpx.MockTransport`` — the same test seam ``http_safety``'s own suite uses
(``tests/unit/protocols/test_source_connectors.py``) — so no real network call is ever
made, and the fetcher's DNS-pinning hop is bypassed exactly the way a ``transport=``
override already does for every other connector's tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from agent_utilities.protocols.voice_supply_chain import acquisition as acq
from agent_utilities.protocols.voice_supply_chain.acquisition import (
    PinnedVoiceSource,
    UnsupportedVoiceAssetFormat,
    VoiceAssetDigestMismatch,
    VoiceSourcePinConflict,
    acquire_voice_config,
    acquire_voice_model,
)
from agent_utilities.protocols.voice_supply_chain.manifest import VoiceManifestStatus

_REVISION = "0123456789abcdef0123456789abcdef01234567"
_MODEL_BYTES = b"\x00fake-onnx-graph-bytes\x01" * 4
_MODEL_SHA = hashlib.sha256(_MODEL_BYTES).hexdigest()
_CONFIG_JSON = json.dumps(
    {"audio": {"sample_rate": 22050}, "phoneme_id_map": {"a": [1]}, "espeak": {"voice": "en"}}
).encode()
_CONFIG_SHA = hashlib.sha256(_CONFIG_JSON).hexdigest()


@pytest.fixture(autouse=True)
def _isolated_quarantine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the quarantine/manifest store to a per-test tmp dir — never touch the
    real XDG data dir, and never let one test's manifests leak into another's."""
    monkeypatch.setattr(acq, "data_dir", lambda: tmp_path)


def _model_source(**overrides: object) -> PinnedVoiceSource:
    fields: dict[str, object] = {
        "repo_id": "rhasspy/piper-voices",
        "revision": _REVISION,
        "path": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "expected_sha256": _MODEL_SHA,
    }
    fields.update(overrides)
    return PinnedVoiceSource(**fields)


def _config_source(**overrides: object) -> PinnedVoiceSource:
    fields: dict[str, object] = {
        "repo_id": "rhasspy/piper-voices",
        "revision": _REVISION,
        "path": "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        "expected_sha256": _CONFIG_SHA,
    }
    fields.update(overrides)
    return PinnedVoiceSource(**fields)


def _transport_returning(content: bytes) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(200, content=content, request=request))


# ── Pin validation (DEF-017 + immutability) ────────────────────────────


def test_mutable_revision_is_rejected() -> None:
    with pytest.raises(ValueError, match="commit SHA"):
        _model_source(revision="main").validate()


def test_non_piper_asset_is_unsupported_format() -> None:
    with pytest.raises(UnsupportedVoiceAssetFormat, match="unsupported_format"):
        _model_source(path="pytorch_model.bin", expected_sha256=_MODEL_SHA).validate()


def test_immutable_url_uses_pinned_revision_not_a_branch() -> None:
    source = _model_source()
    assert f"/resolve/{_REVISION}/" in source.immutable_url
    assert "/resolve/main/" not in source.immutable_url


# ── Acquisition: happy path, fail-closed digest mismatch, idempotency ──


@pytest.mark.asyncio
async def test_acquire_voice_model_verifies_digest_and_quarantines() -> None:
    source = _model_source()
    manifest = await acquire_voice_model(source, transport=_transport_returning(_MODEL_BYTES))

    assert manifest.sha256 == _MODEL_SHA
    assert manifest.status == VoiceManifestStatus.QUARANTINED
    assert manifest.source_revision == _REVISION
    assert (acq.quarantine_dir() / f"{_MODEL_SHA}.onnx").read_bytes() == _MODEL_BYTES


@pytest.mark.asyncio
async def test_digest_mismatch_fails_closed_and_writes_nothing() -> None:
    """A mismatched digest must raise, and NEVER leave a quarantined file or manifest
    behind — an unverified copy must never become available to a later reader."""
    source = _model_source(expected_sha256="f" * 64)  # wrong on purpose
    with pytest.raises(VoiceAssetDigestMismatch):
        await acquire_voice_model(source, transport=_transport_returning(_MODEL_BYTES))

    assert list(acq.quarantine_dir().iterdir()) == []
    assert list(acq.manifest_index_dir().iterdir()) == []


@pytest.mark.asyncio
async def test_byte_length_mismatch_fails_closed() -> None:
    source = _model_source(expected_byte_length=len(_MODEL_BYTES) + 1)
    with pytest.raises(VoiceAssetDigestMismatch, match="bytes"):
        await acquire_voice_model(source, transport=_transport_returning(_MODEL_BYTES))
    assert list(acq.quarantine_dir().iterdir()) == []


@pytest.mark.asyncio
async def test_reacquiring_same_digest_is_idempotent_no_second_fetch() -> None:
    source = _model_source()
    first = await acquire_voice_model(source, transport=_transport_returning(_MODEL_BYTES))

    def _boom(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not re-fetch an already-quarantined identical digest")

    second = await acquire_voice_model(source, transport=httpx.MockTransport(_boom))
    assert second == first


@pytest.mark.asyncio
async def test_reacquiring_same_source_different_digest_conflicts() -> None:
    """A source coordinate, once quarantined, is an immutable pin — a later acquisition
    claiming a DIFFERENT expected digest for the same repo/revision/path is a hard
    error, never a silent overwrite."""
    source = _model_source()
    await acquire_voice_model(source, transport=_transport_returning(_MODEL_BYTES))

    other_bytes = _MODEL_BYTES + b"tampered"
    other_source = _model_source(expected_sha256=hashlib.sha256(other_bytes).hexdigest())
    with pytest.raises(VoiceSourcePinConflict):
        await acquire_voice_model(other_source, transport=_transport_returning(other_bytes))


# ── Config pair validation ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_voice_config_pairs_with_model_and_validates_piper_schema() -> None:
    model = await acquire_voice_model(
        _model_source(), transport=_transport_returning(_MODEL_BYTES)
    )
    config = await acquire_voice_config(
        _config_source(),
        model_manifest=model,
        transport=_transport_returning(_CONFIG_JSON),
    )
    assert config.model_manifest_id == model.manifest_id
    assert config.sample_rate == 22050
    assert config.phoneme_id_map_present is True


@pytest.mark.asyncio
async def test_config_missing_phoneme_map_is_rejected() -> None:
    model = await acquire_voice_model(
        _model_source(), transport=_transport_returning(_MODEL_BYTES)
    )
    bad_json = json.dumps({"audio": {"sample_rate": 22050}}).encode()
    source = _config_source(expected_sha256=hashlib.sha256(bad_json).hexdigest())
    with pytest.raises(UnsupportedVoiceAssetFormat, match="phoneme_id_map"):
        await acquire_voice_config(
            source, model_manifest=model, transport=_transport_returning(bad_json)
        )


@pytest.mark.asyncio
async def test_config_malformed_json_is_rejected() -> None:
    model = await acquire_voice_model(
        _model_source(), transport=_transport_returning(_MODEL_BYTES)
    )
    bad_bytes = b"{not json"
    source = _config_source(expected_sha256=hashlib.sha256(bad_bytes).hexdigest())
    with pytest.raises(UnsupportedVoiceAssetFormat, match="not valid JSON"):
        await acquire_voice_config(
            source, model_manifest=model, transport=_transport_returning(bad_bytes)
        )
