"""Ontology supply-chain integrity — canonical hash, signature, lockfile (X6).

CONCEPT:AU-KG.ontology.supply-chain-integrity — fleet-wide guarantee that a generated/
reconciled ``ontology_<source>.ttl`` (or federated ``<pkg>/ontology/*.ttl``) is exactly
the graph a trusted party produced: a **canonical, serialization-order-invariant hash**
(URDNA2015-equivalent, via rdflib's own RDF-canonicalization) plus an Ed25519 release
signature whose public key is independently pinned and can be verified without access
to the private runtime signing secret.

Release generation requires an explicitly injected private key or secret reference and
never uses a synthesized fallback.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "DEFAULT_SIGNER_ID",
    "DEFAULT_TRUSTED_SIGNERS",
    "active_release_public_key",
    "assert_signing_key_matches_locks",
    "canonical_hash",
    "canonical_manifest_hash",
    "canonical_signed_document_hash",
    "default_dependency_lock_path",
    "dependency_lock_digest",
    "DURABLE_SECRET_SCHEMES",
    "ReleaseSigner",
    "ReleaseSigningError",
    "release_signer_for_publication",
    "release_trusted_public_keys",
    "verify_release_signature",
    "load_lock",
    "lock_pinned_public_keys",
    "rotation_record",
    "save_lock",
    "update_lock_entry",
]

# The one signer the deterministic generator produces manifests as. A genuinely new
# trusted signer (a human reviewer, a CI service account) is a real second value —
# add it to the allowlist passed to :func:`verify_release_signature`, don't grow
# this constant.
DEFAULT_SIGNER_ID = "ontology-manifest-generator"
DEFAULT_TRUSTED_SIGNERS: tuple[str, ...] = (DEFAULT_SIGNER_ID,)
RELEASE_SIGNATURE_ALGORITHM = "ed25519"


class ReleaseSigningError(RuntimeError):
    """A release artifact cannot be signed with an explicit, durable key."""


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64url_decode(value: str, *, expected_bytes: int) -> bytes:
    rendered = str(value or "").strip()
    if not rendered or len(rendered) > 4_096 or any(ch.isspace() for ch in rendered):
        raise ValueError("release signing material is invalid")
    try:
        decoded = base64.b64decode(
            rendered + "=" * (-len(rendered) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("release signing material is invalid") from exc
    if len(decoded) != expected_bytes:
        raise ValueError("release signing material has an invalid size")
    # Base64's unused trailing bits can otherwise make multiple textual values
    # decode to the same bytes. Release artifacts use one canonical, unpadded
    # base64url representation so textual tampering is rejected fail-closed.
    if _b64url_encode(decoded) != rendered:
        raise ValueError("release signing material is not canonical base64url")
    return decoded


def _runtime_release_private_key() -> str:
    """Resolve one release-signing key from the configured secret provider."""
    from agent_utilities.core.config import setting

    reference = str(
        setting("ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF", "") or ""
    ).strip()
    if not reference:
        raise ReleaseSigningError(
            "ontology release signing requires a private-key reference"
        )
    try:
        from agent_utilities.security.cli_secrets import (
            resolve_runtime_secret_reference,
        )

        resolved = resolve_runtime_secret_reference(reference)
    except Exception as exc:
        raise ReleaseSigningError(
            "ontology release signing key reference is unavailable"
        ) from exc
    private_key = str(resolved or "").strip()
    if not private_key:
        raise ReleaseSigningError(
            "ontology release signing key reference did not resolve"
        )
    return private_key


@dataclass(frozen=True, slots=True)
class ReleaseSigner:
    """Explicit Ed25519 signer for release/certification artifacts.

    This class never synthesizes a key. Its public key is persisted with the
    artifact so verification does not need access to the private release secret.
    """

    signer_id: str
    public_key: str
    algorithm: str = RELEASE_SIGNATURE_ALGORITHM
    _private_key: Any = field(repr=False, compare=False, default=None)

    @classmethod
    def from_runtime(cls, *, signer_id: str = DEFAULT_SIGNER_ID) -> ReleaseSigner:
        if signer_id not in DEFAULT_TRUSTED_SIGNERS:
            raise ReleaseSigningError("ontology release signer is not trusted")
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )

            seed = _b64url_decode(_runtime_release_private_key(), expected_bytes=32)
            private_key = Ed25519PrivateKey.from_private_bytes(seed)
            public = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        except ReleaseSigningError:
            raise
        except Exception as exc:
            raise ReleaseSigningError(
                "ontology release signing key is invalid"
            ) from exc
        return cls(
            signer_id=signer_id,
            public_key=_b64url_encode(public),
            _private_key=private_key,
        )

    def sign(self, digest_hex: str) -> str:
        if self._private_key is None or not _valid_digest(digest_hex):
            raise ReleaseSigningError("ontology release digest is invalid")
        message = f"{self.signer_id}:{digest_hex}".encode("ascii")
        return _b64url_encode(self._private_key.sign(message))


def _valid_digest(value: str) -> bool:
    return len(str(value)) == 64 and all(ch in "0123456789abcdef" for ch in str(value))


def _rotation_record_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "deploy"
        / "release"
        / "signing-key-rotation.yml"
    )


def rotation_record(path: str | Path | None = None) -> dict[str, Any]:
    """The committed, auditable trust-anchor rotation ledger.

    CONCEPT:AU-KG.ontology.release-key-rotation — the release trust anchor lives in
    a reviewed repository artifact, not only in an operator environment variable.
    On 2026-07-28 the ``Qkigd…`` seed was destroyed while every lock entry and all
    68 provider attestations still pinned it, and nothing in the repository recorded
    that this had happened. A rotation is a governed, ordered ledger entry naming
    the key rotated FROM, the key rotated TO, the authorisation, and the exact
    artifact set that moved — never a silent substitution.
    """
    resolved = Path(path) if path else _rotation_record_path()
    if not resolved.is_file():
        return {}
    try:
        document = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ReleaseSigningError("release key rotation record is unreadable") from exc
    if not isinstance(document, dict):
        raise ReleaseSigningError("release key rotation record is malformed")
    return document


def active_release_public_key(path: str | Path | None = None) -> str | None:
    """The key the newest recorded rotation moved the trust anchor TO."""

    rotations = rotation_record(path).get("rotations") or []
    if not isinstance(rotations, list) or not rotations:
        return None
    newest = rotations[-1]
    if not isinstance(newest, dict):
        raise ReleaseSigningError("release key rotation record is malformed")
    key = str(newest.get("to_public_key") or "")
    try:
        _b64url_decode(key, expected_bytes=32)
    except ValueError as exc:
        raise ReleaseSigningError("rotated release public key is invalid") from exc
    return key


def release_trusted_public_keys() -> tuple[str, ...]:
    """Ed25519 keys accepted outside a signed release lock.

    The union of the operator-pinned setting and the key the committed rotation
    ledger currently designates. The ledger participates so a governed rotation is
    a reviewable repository change rather than an invisible environment edit.
    """
    from agent_utilities.core.config import setting

    raw = str(setting("ONTOLOGY_RELEASE_TRUSTED_PUBLIC_KEYS", "") or "")
    keys: list[str] = []
    for candidate in raw.split(","):
        value = candidate.strip()
        if not value:
            continue
        try:
            _b64url_decode(value, expected_bytes=32)
        except ValueError as exc:
            raise ReleaseSigningError(
                "ontology trusted release public key is invalid"
            ) from exc
        keys.append(value)
    active = active_release_public_key()
    if active:
        keys.append(active)
    return tuple(dict.fromkeys(keys))


def lock_pinned_public_keys(path: str | Path) -> frozenset[str]:
    """Every distinct release public key an ``ontology.lock`` currently pins."""

    pinned: set[str] = set()
    for entry in load_lock(path).values():
        if not isinstance(entry, dict):
            continue
        for field_name in ("signing_public_key", "certification_signing_public_key"):
            value = entry.get(field_name)
            if isinstance(value, str) and value:
                pinned.add(value)
    return frozenset(pinned)


def assert_signing_key_matches_locks(
    signer: ReleaseSigner,
    *,
    lock_path: str | Path,
    rotating: bool = False,
) -> None:
    """Refuse to sign with a key the locks do not already pin.

    CONCEPT:AU-KG.ontology.release-key-rotation. D-OB-5 was only discovered at
    *admission* time — long after artifacts had been produced with a seed nobody
    had noticed was different from the one every lock entry pinned. Detecting the
    same condition here, before a single byte is written, converts an unrecoverable
    fleet-wide invalidation into a loud refusal at the one moment it is cheap to
    fix. A genuine rotation passes ``rotating=True``, which is what the recorded
    rotation tool does — and only after writing the ledger entry that explains it.
    """
    pinned = lock_pinned_public_keys(lock_path)
    if not pinned or signer.public_key in pinned:
        return
    if rotating:
        return
    raise ReleaseSigningError(
        "the release signing key does not match the key pinned by ontology.lock; "
        "signing would invalidate every locked artifact — record a rotation instead"
    )


#: Reference schemes whose custody is *versioned*, so an overwrite is recoverable.
DURABLE_SECRET_SCHEMES = ("vault://", "secret://")


def default_dependency_lock_path() -> Path:
    """The ecosystem-workspace ``uv.lock`` this repo's own dependency pins live in."""

    return Path(__file__).resolve().parents[3] / "uv.lock"


def dependency_lock_digest(lock_path: str | Path | None = None) -> str:
    """Deterministic SHA-256 over the exact ``(name, version)`` pins in a ``uv.lock``.

    CONCEPT:AU-KG.ontology.connector-manifest-gate — GOC-84 names "dependency-lock
    drift" as one of the adversarial cases a signed connector manifest must catch:
    a connector's declared behavior can depend on a third-party library version even
    when neither the manifest YAML nor the connector's own source changed a single
    byte. Binding this digest into ``ProvenanceSpec`` (and therefore into the signed
    ``canonical_manifest_hash``) makes that class of drift provable, not assumed.

    Hashes the *parsed* ``(name, version)`` pairs from every ``[[package]]`` table,
    sorted, rather than the raw file bytes — invariant to comment/whitespace/
    reordering churn in the lock file, while any real dependency add/remove/version
    change still changes the digest. Raises :class:`ReleaseSigningError` if the lock
    cannot be read or parsed: a manifest must never be signed (or verified) against
    dependency state nobody could actually confirm.
    """
    import tomllib

    path = Path(lock_path) if lock_path is not None else default_dependency_lock_path()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReleaseSigningError(
            f"dependency lock is unreadable: {path}"
        ) from exc
    try:
        document = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        raise ReleaseSigningError(f"dependency lock is malformed: {path}") from exc
    packages = document.get("package")
    if not isinstance(packages, list):
        raise ReleaseSigningError("dependency lock package inventory is invalid")
    pins: list[tuple[str, str]] = []
    for item in packages:
        if not isinstance(item, dict):
            raise ReleaseSigningError("dependency lock package inventory is invalid")
        name = item.get("name")
        version = item.get("version")
        if not isinstance(name, str) or not isinstance(version, str) or not name or not version:
            raise ReleaseSigningError(
                "dependency lock contains an invalid package identity"
            )
        pins.append((name, version))
    payload = json.dumps(sorted(pins), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def release_signer_for_publication(
    *, lock_path: str | Path, rotating: bool = False
) -> ReleaseSigner:
    """The ONLY signer a publication path may use.

    CONCEPT:AU-KG.ontology.release-key-rotation. Two custody invariants, both of
    which D-OB-5 violated:

    * **Versioned custody.** The seed must resolve through a versioned secret store
      (``vault://`` / ``secret://``), never an ``env://`` reference backed by a
      local ``runtime-secrets.json``. That file was overwritten in place on
      2026-07-28 with a different seed, with no history and no audit trail; the
      same seed written to OpenBao KV v2 minutes earlier survived untouched,
      because KV v2 keeps every version. A store that can be silently overwritten
      beyond recovery is not custody.
    * **Lock agreement.** The seed that would sign must be the seed the locks pin.
      Checked here, before any artifact is produced, rather than at admission time
      when the only remedy left is a fleet-wide rotation.
    """
    from agent_utilities.core.config import setting

    reference = str(
        setting("ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF", "") or ""
    ).strip()
    if not reference.startswith(DURABLE_SECRET_SCHEMES):
        raise ReleaseSigningError(
            "release signing requires versioned secret custody "
            "(vault:// or secret://); an env:// reference has no version history "
            "and an overwrite is unrecoverable"
        )
    signer = ReleaseSigner.from_runtime()
    assert_signing_key_matches_locks(signer, lock_path=lock_path, rotating=rotating)
    return signer


def verify_release_signature(
    digest_hex: str,
    signature: str | None,
    *,
    signer_id: str | None,
    algorithm: str | None,
    public_key: str | None,
    trusted_public_keys: tuple[str, ...] | None = None,
) -> bool:
    """Verify a release signature with an explicitly trusted Ed25519 key."""
    if (
        not _valid_digest(digest_hex)
        or signer_id not in DEFAULT_TRUSTED_SIGNERS
        or algorithm != RELEASE_SIGNATURE_ALGORITHM
        or not signature
        or not public_key
    ):
        return False
    trusted = (
        release_trusted_public_keys()
        if trusted_public_keys is None
        else trusted_public_keys
    )
    if public_key not in trusted:
        return False
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        key = Ed25519PublicKey.from_public_bytes(
            _b64url_decode(public_key, expected_bytes=32)
        )
        key.verify(
            _b64url_decode(signature, expected_bytes=64),
            f"{signer_id}:{digest_hex}".encode("ascii"),
        )
        return True
    except Exception:  # noqa: BLE001 - malformed/tampered signatures are rejections
        return False


def canonical_hash(graph: Any) -> tuple[str, int]:
    """A URDNA2015-equivalent canonical hash of an RDF graph.

    Uses :func:`rdflib.compare.to_canonical_graph` (deterministic SHA-256 bnode
    labeling correlated with graph contents — the same guarantee URDNA2015 gives),
    then hashes the *sorted* N3 serialization of every canonicalized triple. Sorting
    makes the result invariant to the graph's internal/serialization triple order —
    parsing the same ontology from Turtle vs. N-Triples vs. JSON-LD yields the same hash.

    Returns:
        ``(hex_digest, triple_count)``.
    """
    from rdflib.compare import to_canonical_graph

    canon = to_canonical_graph(graph)
    lines = sorted(f"{s.n3()} {p.n3()} {o.n3()} ." for s, p, o in canon)
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(lines)


def canonical_manifest_hash(manifest: Any) -> str:
    """Canonical SHA-256 of a complete connector manifest, excluding its signature.

    The ontology graph hash intentionally covers only compiled RDF.  Connector
    authorization, source presets, field mappings, and governance declarations
    are security-relevant too, so the runtime signature/pin is bound to this
    complete canonical document hash instead.  ``manifest`` may be a Pydantic
    model or an already-decoded mapping.
    """
    if hasattr(manifest, "model_dump"):
        data = manifest.model_dump(mode="json")
    elif isinstance(manifest, dict):
        data = dict(manifest)
    else:
        raise TypeError("connector manifest must be a model or mapping")

    provenance = dict(data.get("provenance") or {})
    provenance["signature"] = None
    data["provenance"] = provenance
    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_signed_document_hash(
    document: dict[str, Any], *, signature_field: str = "signature"
) -> str:
    """Canonical SHA-256 for a signed JSON document, excluding its signature."""

    data = dict(document)
    data[signature_field] = None
    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ── ontology.lock — pins the canonical hash per artifact, fleet-wide ──────────────


def load_lock(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read ``ontology.lock`` (``{artifact_path: {hash, algorithm, ...}}``); empty if absent."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} does not contain a mapping at the top level")
    return data


def save_lock(path: str | Path, entries: dict[str, dict[str, Any]]) -> None:
    """Write ``ontology.lock`` with stable key ordering (byte-stable across runs)."""
    p = Path(path)
    ordered = {k: entries[k] for k in sorted(entries)}
    p.write_text(
        yaml.safe_dump(ordered, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )


def update_lock_entry(
    path: str | Path,
    artifact: str,
    digest_hex: str,
    *,
    algorithm: str = "urdna2015-sha256",
    triple_count: int = 0,
    manifest_hash: str | None = None,
    signer: str | None = None,
    signature: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Pin/update one artifact's canonical hash in ``ontology.lock``; returns the full lock."""
    entries = load_lock(path)
    entry: dict[str, Any] = {
        "hash": digest_hex,
        "algorithm": algorithm,
        "triple_count": triple_count,
    }
    if manifest_hash is not None:
        entry["manifest_hash"] = manifest_hash
    if signer is not None:
        entry["signer"] = signer
    if signature is not None:
        entry["signature"] = signature
    for key, value in sorted((metadata or {}).items()):
        if value is not None:
            entry[str(key)] = value
    entries[artifact] = entry
    save_lock(path, entries)
    return entries
