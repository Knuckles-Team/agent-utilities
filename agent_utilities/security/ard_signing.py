"""CONCEPT:OS-5.60 — ARD datapoint signing (Ed25519 publisher verification).

Agentic Resource Discovery (ARD) treats publisher verification as a first-class
citizen: each catalog datapoint is signed with the publisher's Ed25519 key and an
agent verifies the signature against the publisher's exposed public key (anchored to
the publishing domain) *before* it trusts a discovered capability. This module is the
sign/verify primitive for both directions — we sign the entries we publish in
``/.well-known/ai-catalog.json`` (ECO-4.95) and verify the entries we ingest from
external registries (ECO-4.96).

Custody mirrors the run-token secret (:mod:`security.run_token`): the private key is
**only** ever read from ``setting("ARD_SIGNING_PRIVATE_KEY")`` — injected from OpenBao
(``apps/agent-utilities/ard``) into the served stack env at deploy, never committed and
never logged. With no key configured (dev/tests) a stable per-process ephemeral key is
synthesized so sign/verify round-trips, and the manifest advertises ``signed: false``.

Dependency discipline: ``cryptography`` is import-guarded and lazy. With it absent the
lean serving image still boots — :func:`sign_datapoint` returns ``None`` (the manifest
is served unsigned) and :func:`verify_datapoint` returns ``False`` (an explicit "cannot
verify"), so a missing dependency never crashes a serving path.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

#: Per-process private-key seed fallback (dev/test); env-provided in production.
_EPHEMERAL_SEED: bytes | None = None


def _ed25519():
    """Return the cryptography Ed25519 module, or ``None`` if unavailable (lazy)."""
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519

        return ed25519
    except Exception:  # noqa: BLE001 — absent dep ⇒ serve unsigned, never crash
        return None


def signing_available() -> bool:
    """Whether Ed25519 signing/verification can run in this process."""
    return _ed25519() is not None


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    s = s.strip()
    # Accept both urlsafe and standard base64 (publishers vary); pad to a multiple of 4.
    s = s.replace("-", "+").replace("_", "/")
    return base64.b64decode(s + "=" * (-len(s) % 4))


def canonical(datapoint: Any) -> str:
    """Canonical JSON for signing: sorted keys, no whitespace (stable across hosts)."""
    if isinstance(datapoint, str):
        return datapoint
    return json.dumps(datapoint, sort_keys=True, separators=(",", ":"))


def _seed() -> bytes:
    """The 32-byte Ed25519 private seed (OpenBao-injected; ephemeral dev fallback).

    Accepts the configured value as base64 of a raw 32-byte seed (the canonical form
    we document) — anything else falls back to a deterministic per-process key so
    sign/verify still round-trips in dev/tests.
    """
    val = (setting("ARD_SIGNING_PRIVATE_KEY", default="") or "").strip()
    if val:
        try:
            raw = _b64d(val)
            if len(raw) >= 32:
                return raw[:32]
            # A short/opaque value is still usable as deterministic key material.
            return hashlib.sha256(val.encode()).digest()
        except Exception:  # noqa: BLE001 — malformed key ⇒ derive deterministically
            return hashlib.sha256(val.encode()).digest()
    global _EPHEMERAL_SEED
    if _EPHEMERAL_SEED is None:
        # Stable within a process so a publish/verify in the same run agree; never persisted.
        _EPHEMERAL_SEED = hashlib.sha256(
            f"au-ard-signing:{os.getpid()}".encode()
        ).digest()
    return _EPHEMERAL_SEED


def _private_key():
    ed = _ed25519()
    if ed is None:
        return None
    return ed.Ed25519PrivateKey.from_private_bytes(_seed())


def public_key_b64() -> str | None:
    """This publisher's Ed25519 public key (base64url) for the manifest, or ``None``.

    Returns ``None`` only when ``cryptography`` is unavailable — i.e. when we serve the
    manifest unsigned. Otherwise the key derives from the configured (or ephemeral) seed.
    """
    key = _private_key()
    if key is None:
        return None
    from cryptography.hazmat.primitives import serialization

    raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return _b64e(raw)


def is_configured() -> bool:
    """Whether a real (non-ephemeral) signing key is configured for this deployment."""
    return bool((setting("ARD_SIGNING_PRIVATE_KEY", default="") or "").strip())


def sign_datapoint(datapoint: Any) -> str | None:
    """Sign a datapoint (dict canonicalized, or a pre-canonical string).

    Returns the base64url Ed25519 signature, or ``None`` when signing is unavailable
    (``cryptography`` not installed) — in which case the caller serves the entry unsigned.
    """
    key = _private_key()
    if key is None:
        return None
    sig = key.sign(canonical(datapoint).encode())
    return _b64e(sig)


def verify_datapoint(datapoint: Any, signature_b64: str, public_key_b64_: str) -> bool:
    """Verify a datapoint's Ed25519 signature against a publisher's public key.

    Returns ``False`` on any failure (bad signature, malformed key/signature, or
    ``cryptography`` unavailable) — a fail-closed "cannot verify". ``datapoint`` may be a
    dict (canonicalized here) or the exact canonical string that was signed.
    """
    ed = _ed25519()
    if ed is None or not signature_b64 or not public_key_b64_:
        return False
    try:
        pub = ed.Ed25519PublicKey.from_public_bytes(_b64d(public_key_b64_))
        pub.verify(_b64d(signature_b64), canonical(datapoint).encode())
        return True
    except Exception:  # noqa: BLE001 — any verify failure is a rejection, not an error
        return False
