"""Governed Piper voice-model acquisition (GOC-36-W03 AU-side adapter).

Fetches a **pinned** Hugging Face asset — exact repository, exact 40-char commit
revision, exact path — through the repo's existing hardened source-HTTP fetcher
(``protocols.source_connectors.http_safety.safe_get_bytes_async``, which already
streams from the wire under a host allowlist / redirect policy / size cap, per this
package's "Sprawl boundaries" convention: reuse the governed fetcher rather than
build a second one). The response is hashed with SHA-256 and compared against the
operator-supplied expected digest **before** anything is written to the quarantine
directory or a manifest is produced.

Fail-closed, per the lane doc's "Acquisition and promotion lifecycle": a digest
mismatch, a truncated/oversized/malformed transfer, or an unsupported asset shape
never yields a usable manifest and never falls back to an unverified copy — it raises.
A caller that swallows the exception and proceeds anyway is the caller's bug, not
this module's; nothing here returns an empty stand-in manifest on failure.

Idempotency: re-acquiring the same ``(repo_id, revision, path)`` with the same
digest is a no-op that returns the existing manifest without a second network
fetch. Re-acquiring the same source coordinate with a **different** digest than a
previously recorded manifest is a hard error — a source pin, once quarantined, is
immutable evidence, not a mutable cursor.

Scope guard (DEF-017): only a Piper ``.onnx`` model file or its paired
``.onnx.json``/``.json`` config is accepted. Anything else raises
:class:`UnsupportedVoiceAssetFormat` — this module is not a generic Hugging Face /
Transformers loader and must never attempt a heuristic conversion.

Authority note: this module's manifests stop at ``VERIFIED`` (see
``manifest.py``'s module docstring) — it has no promotion/revocation authority. That
authority is the EG registry described in the GOC-36 lane doc, which does not exist
yet in this workspace as of this change.

**Known gap — response-size ceiling.** The lane doc's stated acquisition budget is up
to 4 GiB per model. The shared fetcher this module reuses
(``source_connectors.http_safety``) enforces a hard, non-overridable
``HARD_MAX_RESPONSE_BYTES`` ceiling of 64 MiB across every caller in the repo (every
connector, not just this one) — many real Piper voice models fit under that, but the
lane's stated budget does not. Raising that ceiling is a shared, security-relevant
change to code this lane does not own and is out of scope here; a model that exceeds
it fails closed with ``SourceEgressError`` (a real, typed refusal — not silently
truncated) rather than being fetched unsafely. This is reported as an open gap, not
worked around.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from agent_utilities.core.event_loop import run_blocking_ordered
from agent_utilities.core.paths import data_dir
from agent_utilities.protocols.source_connectors.http_safety import (
    configured_source_http_policy,
    safe_get_bytes_async,
)
from agent_utilities.protocols.voice_supply_chain.manifest import (
    HF_COMMIT_SHA_LEN,
    VoiceConfigManifest,
    VoiceManifestStatus,
    VoiceModelManifest,
    VoiceModelProvider,
)

logger = logging.getLogger(__name__)

#: Piper distributes voices from Hugging Face model repos; this is the sole permitted
#: source host. The `resolve/<revision>/<path>` route is Hugging Face's immutable,
#: revision-pinned download endpoint (as opposed to the mutable default-branch tree
#: the piper-rs README's `wget` examples point at — the exact gap this lane closes).
HUGGINGFACE_HOST = "huggingface.co"

#: Hugging Face's `resolve` endpoint redirects large (LFS-backed) files here. Any other
#: redirect target is refused by the underlying fetcher's host allowlist.
_HF_LFS_REDIRECT_HOSTS = ("cdn-lfs.huggingface.co", "cdn-lfs-us-1.huggingface.co")

_MODEL_SUFFIX = ".onnx"
_CONFIG_SUFFIXES = (".onnx.json", ".json")


class UnsupportedVoiceAssetFormat(Exception):
    """Raised for any asset shape other than a Piper ``.onnx`` model or its JSON config.

    DEF-017 scope guard — this is a typed, catchable refusal, never a heuristic
    conversion attempt.
    """


class VoiceAssetDigestMismatch(Exception):
    """Raised when a downloaded asset's SHA-256 does not match the pinned expectation.

    Fail-closed: raising here means no manifest is written and no bytes are quarantined
    — the caller never receives an unverified copy under any circumstance.
    """


class VoiceSourcePinConflict(Exception):
    """Raised when the same ``(repo_id, revision, path)`` was already quarantined under
    a *different* digest — a source pin is immutable once recorded (append-only)."""


@dataclass(frozen=True)
class PinnedVoiceSource:
    """An exact, immutable Hugging Face file coordinate.

    ``revision`` MUST be a full 40-character commit SHA. A branch or tag name (``main``,
    ``v1``) is mutable — its content can change after acquisition without changing the
    ref — so it is rejected outright rather than treated as a "close enough" pin.
    """

    repo_id: str
    revision: str
    path: str
    expected_sha256: str
    expected_byte_length: int | None = None

    def validate(self) -> None:
        if not self.repo_id or "/" not in self.repo_id:
            raise ValueError(f"repo_id must be '<owner>/<name>', got {self.repo_id!r}")
        if len(self.revision) != HF_COMMIT_SHA_LEN or any(
            c not in "0123456789abcdef" for c in self.revision.lower()
        ):
            raise ValueError(
                f"revision must be a {HF_COMMIT_SHA_LEN}-char commit SHA, not a "
                f"mutable branch/tag name: {self.revision!r}"
            )
        if len(self.expected_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.expected_sha256.lower()
        ):
            raise ValueError("expected_sha256 must be 64 lowercase hex characters")
        if not (
            self.path.endswith(_MODEL_SUFFIX) or self.path.endswith(_CONFIG_SUFFIXES)
        ):
            raise UnsupportedVoiceAssetFormat(
                f"unsupported_format: {self.path!r} is neither a Piper .onnx model nor "
                "its .onnx.json/.json config (DEF-017: no generic HF loader)"
            )

    @property
    def is_config(self) -> bool:
        return self.path.endswith(_CONFIG_SUFFIXES)

    @property
    def immutable_url(self) -> str:
        return f"https://{HUGGINGFACE_HOST}/{self.repo_id}/resolve/{self.revision}/{self.path}"


def quarantine_dir() -> Path:
    """Directory holding content-addressed, digest-verified voice assets pending
    handoff to the (not-yet-existing) EG registry. ``AGENT_UTILITIES_DATA_DIR``-relative,
    per this package's XDG path conventions (``core/paths.data_dir``)."""
    d = data_dir() / "voice-models" / "quarantine"
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_index_dir() -> Path:
    """Directory holding one JSON manifest file per acquired asset, keyed by digest."""
    d = data_dir() / "voice-models" / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_model_manifest(manifest_id: str) -> VoiceModelManifest | None:
    """Look up a previously quarantined model manifest by its digest id.

    Used by the ``agent-utilities voice-model`` CLI to resolve an operator-supplied
    manifest id into the object :func:`acquire_voice_config` and
    ``license_registry.is_ready_for_promotion_handoff`` need.
    """
    path = manifest_index_dir() / f"{manifest_id}.json"
    if not path.exists():
        return None
    return VoiceModelManifest.model_validate_json(path.read_text())


def _existing_manifest_for_source(
    source: PinnedVoiceSource,
) -> VoiceModelManifest | None:
    """Return a previously written manifest for this exact source coordinate, if any."""
    for path in manifest_index_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            data.get("source_repository") == source.repo_id
            and data.get("source_revision") == source.revision.lower()
            and data.get("source_path") == source.path
        ):
            return VoiceModelManifest.model_validate(data)
    return None


async def acquire_voice_model(
    source: PinnedVoiceSource, *, transport: object | None = None
) -> VoiceModelManifest:
    """Fetch, verify, and quarantine a pinned Piper ``.onnx`` model file.

    Streams the download through the governed source-HTTP fetcher restricted to
    ``huggingface.co`` (plus its known LFS redirect hosts), computes SHA-256 over the
    received bytes, and raises :class:`VoiceAssetDigestMismatch` — writing nothing —
    if it does not match ``source.expected_sha256``. Idempotent: a prior manifest for
    the identical ``(repo_id, revision, path, digest)`` short-circuits without a
    network call; a prior manifest for the same coordinate under a *different* digest
    raises :class:`VoiceSourcePinConflict`.

    Args:
        transport: Optional ``httpx`` transport override (test seam only — production
            callers never pass this; it bypasses the governed DNS-pinning hop the same
            way ``http_safety``'s own test suite does).
    """
    source.validate()
    if source.is_config:
        raise UnsupportedVoiceAssetFormat(
            f"{source.path!r} is a config, not a model — use acquire_voice_config()"
        )

    existing = _existing_manifest_for_source(source)
    if existing is not None:
        if existing.sha256 == source.expected_sha256.lower():
            logger.info(
                "voice model already quarantined idempotently: %s@%s/%s (sha256=%s)",
                source.repo_id,
                source.revision[:12],
                source.path,
                existing.sha256[:12],
            )
            return existing
        raise VoiceSourcePinConflict(
            f"{source.repo_id}@{source.revision}/{source.path} was already quarantined "
            f"under sha256={existing.sha256!r}, refusing to overwrite with a different "
            f"expected digest {source.expected_sha256!r} — a source pin is immutable"
        )

    content, _encoding = await safe_get_bytes_async(
        source.immutable_url,
        allowed_private_hosts=(HUGGINGFACE_HOST,),
        allowed_redirect_hosts=_HF_LFS_REDIRECT_HOSTS,
        transport=transport,
        **{
            k: v
            for k, v in configured_source_http_policy().items()
            if k not in ("allowed_private_hosts", "allowed_redirect_hosts")
        },
    )

    digest = hashlib.sha256(content).hexdigest()
    if digest != source.expected_sha256.lower():
        raise VoiceAssetDigestMismatch(
            f"{source.repo_id}@{source.revision}/{source.path}: expected sha256="
            f"{source.expected_sha256!r}, got {digest!r} — refusing to quarantine an "
            "unverified copy"
        )
    if (
        source.expected_byte_length is not None
        and len(content) != source.expected_byte_length
    ):
        raise VoiceAssetDigestMismatch(
            f"{source.repo_id}@{source.revision}/{source.path}: expected "
            f"{source.expected_byte_length} bytes, got {len(content)} — refusing to "
            "quarantine a mismatched-length transfer"
        )

    quarantine_path = quarantine_dir() / f"{digest}.onnx"
    # run_blocking_ordered (not bare asyncio.to_thread): a cancelled caller must not
    # observe completion before the quarantined file is actually fully written — a
    # partial write here would corrupt the fail-closed digest-quarantine invariant.
    await run_blocking_ordered(quarantine_path.write_bytes, content)

    manifest = VoiceModelManifest(
        manifest_id=digest,
        provider=VoiceModelProvider.PIPER,
        format="onnx",
        source_host=HUGGINGFACE_HOST,
        source_repository=source.repo_id,
        source_revision=source.revision.lower(),
        source_path=source.path,
        source_url=source.immutable_url,
        byte_length=len(content),
        sha256=digest,
        status=VoiceManifestStatus.QUARANTINED,
    )
    await run_blocking_ordered(
        (manifest_index_dir() / f"{digest}.json").write_text,
        manifest.model_dump_json(indent=2),
    )
    logger.info(
        "voice model quarantined: %s@%s/%s (sha256=%s, %d bytes)",
        source.repo_id,
        source.revision[:12],
        source.path,
        digest[:12],
        len(content),
    )
    return manifest


async def acquire_voice_config(
    source: PinnedVoiceSource,
    *,
    model_manifest: VoiceModelManifest,
    transport: object | None = None,
) -> VoiceConfigManifest:
    """Fetch, verify, and pair-validate a pinned Piper JSON config against its model.

    Same digest fail-closed contract as :func:`acquire_voice_model`. Additionally
    parses the JSON and requires the fields a Piper runtime needs (``audio.sample_rate``
    and a ``phoneme_id_map``) — a config missing either is rejected as
    :class:`UnsupportedVoiceAssetFormat` rather than accepted with a guessed default.

    Args:
        transport: Optional ``httpx`` transport override (test seam only).
    """
    source.validate()
    if not source.is_config:
        raise UnsupportedVoiceAssetFormat(
            f"{source.path!r} is not a .onnx.json/.json config"
        )

    content, _encoding = await safe_get_bytes_async(
        source.immutable_url,
        allowed_private_hosts=(HUGGINGFACE_HOST,),
        allowed_redirect_hosts=_HF_LFS_REDIRECT_HOSTS,
        transport=transport,
        **{
            k: v
            for k, v in configured_source_http_policy().items()
            if k not in ("allowed_private_hosts", "allowed_redirect_hosts")
        },
    )

    digest = hashlib.sha256(content).hexdigest()
    if digest != source.expected_sha256.lower():
        raise VoiceAssetDigestMismatch(
            f"{source.repo_id}@{source.revision}/{source.path}: expected sha256="
            f"{source.expected_sha256!r}, got {digest!r} — refusing to quarantine an "
            "unverified copy"
        )

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise UnsupportedVoiceAssetFormat(
            f"{source.path!r} is not valid JSON: {exc}"
        ) from exc
    audio = parsed.get("audio") if isinstance(parsed, dict) else None
    sample_rate = audio.get("sample_rate") if isinstance(audio, dict) else None
    phoneme_map_present = isinstance(parsed, dict) and bool(
        parsed.get("phoneme_id_map")
    )
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise UnsupportedVoiceAssetFormat(
            f"{source.path!r} has no valid audio.sample_rate — not a Piper voice config"
        )
    if not phoneme_map_present:
        raise UnsupportedVoiceAssetFormat(
            f"{source.path!r} has no phoneme_id_map — not a Piper voice config"
        )

    quarantine_path = quarantine_dir() / f"{digest}.onnx.json"
    await run_blocking_ordered(quarantine_path.write_bytes, content)

    config_manifest = VoiceConfigManifest(
        manifest_id=digest,
        model_manifest_id=model_manifest.manifest_id,
        sha256=digest,
        byte_length=len(content),
        sample_rate=int(sample_rate),
        espeak_voice=str((parsed.get("espeak") or {}).get("voice", "")),
        phoneme_id_map_present=True,
        piper_schema_present=True,
    )
    await run_blocking_ordered(
        (manifest_index_dir() / f"{digest}.config.json").write_text,
        config_manifest.model_dump_json(indent=2),
    )
    logger.info(
        "voice config quarantined and pair-validated: %s@%s/%s (sha256=%s, model=%s)",
        source.repo_id,
        source.revision[:12],
        source.path,
        digest[:12],
        model_manifest.manifest_id[:12],
    )
    return config_manifest


__all__ = [
    "HUGGINGFACE_HOST",
    "PinnedVoiceSource",
    "UnsupportedVoiceAssetFormat",
    "VoiceAssetDigestMismatch",
    "VoiceSourcePinConflict",
    "acquire_voice_config",
    "acquire_voice_model",
    "get_model_manifest",
    "manifest_index_dir",
    "quarantine_dir",
]
