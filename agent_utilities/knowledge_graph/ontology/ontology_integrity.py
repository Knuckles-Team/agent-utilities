"""Ontology supply-chain integrity — canonical hash, signature, lockfile (X6).

CONCEPT:AU-KG.ontology.supply-chain-integrity — fleet-wide guarantee that a generated/
reconciled ``ontology_<source>.ttl`` (or federated ``<pkg>/ontology/*.ttl``) is exactly
the graph a trusted party produced: a **canonical, serialization-order-invariant hash**
(URDNA2015-equivalent, via rdflib's own RDF-canonicalization — no new crypto/deps) plus
an **HMAC signature** over that hash, checked against a pinned trusted-signer allowlist.

Reuses the fleet's existing HMAC/secret primitive rather than inventing one:
:func:`agent_utilities.security.run_token._secret` already resolves a signing secret
from ``AGENT_UTILITIES_TOKEN_SECRET`` (env/config.json) with a stable per-process
fallback for dev/test — the same pattern :mod:`agent_utilities.security.permissions_kernel`
uses for agent-identity signatures. This module signs/verifies *ontology* hashes with
that same secret, scoped per signer id, instead of a parallel crypto stack.
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Any

import yaml

from ...security.run_token import (
    _secret,  # reused HMAC secret (AGENT_UTILITIES_TOKEN_SECRET)
)

__all__ = [
    "DEFAULT_SIGNER_ID",
    "DEFAULT_TRUSTED_SIGNERS",
    "canonical_hash",
    "sign",
    "verify",
    "load_lock",
    "save_lock",
    "update_lock_entry",
]

# The one signer the deterministic generator produces manifests as. A genuinely new
# trusted signer (a human reviewer, a CI service account) is a real second value —
# add it to the allowlist passed to :func:`verify`, don't grow this constant.
DEFAULT_SIGNER_ID = "ontology-manifest-generator"
DEFAULT_TRUSTED_SIGNERS: tuple[str, ...] = (DEFAULT_SIGNER_ID,)


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


def sign(digest_hex: str, *, signer_id: str = DEFAULT_SIGNER_ID) -> str:
    """HMAC-SHA256 sign a canonical hash, scoped to ``signer_id``."""
    message = f"{signer_id}:{digest_hex}".encode()
    return hmac.new(_secret(), message, hashlib.sha256).hexdigest()


def verify(
    digest_hex: str,
    signature: str | None,
    *,
    signer_id: str | None,
    allowlist: tuple[str, ...] = DEFAULT_TRUSTED_SIGNERS,
) -> bool:
    """Fail-closed verification: unsigned, unknown-signer, or tampered all return False.

    Never raises on a bad/missing signature — callers (``apply_manifest``,
    ``check_connector_manifests.py``) decide whether to hard-fail or soft-warn.
    """
    if not signature or not signer_id:
        return False
    if signer_id not in allowlist:
        return False
    expected = sign(digest_hex, signer_id=signer_id)
    return hmac.compare_digest(signature, expected)


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
) -> dict[str, dict[str, Any]]:
    """Pin/update one artifact's canonical hash in ``ontology.lock``; returns the full lock."""
    entries = load_lock(path)
    entries[artifact] = {
        "hash": digest_hex,
        "algorithm": algorithm,
        "triple_count": triple_count,
    }
    save_lock(path, entries)
    return entries
