#!/usr/bin/python
from __future__ import annotations

"""Entropy-gated entity resolution (CONCEPT:AHE-3.69).

A cheap, deterministic, LLM-free entity-dedup fast-path absorbed from Zep's
Graphiti (its ``dedup_helpers`` ladder, arXiv:2501.13956) and folded in front of
our embedding/engine similarity tier (:mod:`.dedup`). The common case — the same
entity named near-identically across sources — is resolved here without ever
computing an embedding or calling an LLM; only genuinely-ambiguous names escalate
to the expensive vector/LLM tier.

The ladder, cheapest first:

1. **Exact normalized-name match.** Names are normalized (lower-cased, punctuation
   folded, common corporate suffixes dropped, whitespace removed) to a canonical
   key; equal keys are a certain merge. This alone collapses
   ``"OpenAI" / "Open AI, Inc." / "OPENAI"``.
2. **Shannon-entropy gate.** A name is only trusted for *name-only* merge if its
   canonical key carries enough information (length + per-character entropy).
   Generic low-entropy names (``"data"``, ``"system"``, ``"ai"``) FALL THROUGH to
   the embedding tier instead of being merged on the name alone — the guard
   against over-merging distinct entities that happen to share a vague label.
3. **MinHash + LSH banding + Jaccard ≥ threshold** over character-shingles of the
   canonical key catches fuzzy near-duplicates (typos, suffix drift) among the
   high-entropy names without an O(n²) all-pairs comparison.
4. **Residual.** Whatever is not merged here (low-entropy names, and high-entropy
   names with no exact/fuzzy match) is returned as the *ambiguous* set for the
   caller to escalate to :func:`.dedup.dedup_features`' embedding/engine pass.

Pure-Python (``hashlib`` only) — no heavy/native dependency, so it is safe on the
lean serving plane (see *Dependency discipline*). Deterministic: the MinHash
permutation coefficients are seeded from a fixed constant, so repeated runs
converge (idempotent, like the MERGE-on-write dedup it feeds).
"""

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

# --- tunables (module constants per Configuration discipline — not env flags) ---

#: Canonical-key tokens dropped as non-identifying corporate/legal suffixes.
_SUFFIXES: frozenset[str] = frozenset(
    {
        "inc",
        "incorporated",
        "llc",
        "ltd",
        "limited",
        "corp",
        "corporation",
        "co",
        "company",
        "gmbh",
        "plc",
        "sa",
        "ag",
        "nv",
        "bv",
        "group",
        "holdings",
        "labs",
        "lab",
    }
)

#: A canonical key must be at least this long to be eligible for name-only merge.
_MIN_KEY_LEN: int = 4
#: ...and carry at least this many bits of per-character Shannon entropy.
_MIN_ENTROPY: float = 1.5

#: Generic single-word labels that are unsafe to merge on the name alone even
#: though their character entropy is high — two unrelated entities are often both
#: called "system"/"service" (e.g. truncated "Payment System" vs "Auth System").
#: A canonical key equal to one of these is rejected by the entropy gate and
#: escalated to the embedding tier, where the surrounding context disambiguates.
_GENERIC_KEYS: frozenset[str] = frozenset(
    {
        "data",
        "system",
        "service",
        "server",
        "platform",
        "api",
        "app",
        "application",
        "tool",
        "model",
        "agent",
        "user",
        "client",
        "file",
        "node",
        "object",
        "item",
        "thing",
        "test",
        "demo",
        "example",
        "default",
        "main",
        "core",
        "base",
        "module",
        "component",
        "process",
        "task",
        "job",
        "engine",
    }
)

#: MinHash / LSH parameters. bands * rows == num_perm. b=4, r=16 over 64 perms
#: puts the LSH S-curve threshold near ~0.92, matching the Jaccard cutoff below.
_NUM_PERM: int = 64
_LSH_BANDS: int = 4
_LSH_ROWS: int = 16
#: Character-shingle size for MinHash.
_SHINGLE_K: int = 3
#: Exact-Jaccard cutoff for a fuzzy (LSH-candidate) merge.
_JACCARD_THRESHOLD: float = 0.9

_MERSENNE_PRIME: int = (1 << 61) - 1


@dataclass
class ResolutionResult:
    """Outcome of an entropy-gated resolution pass.

    Attributes:
        merge_pairs: ``(survivor_id, duplicate_id, score, tier)`` for every
            name-only merge decided here. ``tier`` is ``"exact"`` or ``"lsh"``.
        resolved_ids: ids that participated in a merge (the fast path handled
            them — the caller need not embed them to find *these* duplicates).
        residual_ids: ambiguous ids (low-entropy names, or high-entropy names
            with no name match) the caller should escalate to the embedding tier.
        exact_merges: count of pairs decided by tier 1.
        lsh_merges: count of pairs decided by tier 3.
        low_entropy: count of ids rejected by the entropy gate (tier 2).
    """

    merge_pairs: list[tuple[str, str, float, str]] = field(default_factory=list)
    resolved_ids: set[str] = field(default_factory=set)
    residual_ids: set[str] = field(default_factory=set)
    exact_merges: int = 0
    lsh_merges: int = 0
    low_entropy: int = 0


def normalize_name(name: str) -> str:
    """Fold a display name to a canonical comparison key.

    Lower-cases, replaces every non-alphanumeric run with a token boundary, drops
    standalone corporate/legal suffix tokens, and concatenates the survivors in
    order (no separators). Order is preserved so ``"Sun Microsystems"`` and
    ``"Microsystems Sun"`` do NOT collide.
    """
    if not name:
        return ""
    folded = []
    cur = []
    for ch in name.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            folded.append("".join(cur))
            cur = []
    if cur:
        folded.append("".join(cur))
    tokens = [t for t in folded if t and t not in _SUFFIXES]
    return "".join(tokens)


def shannon_entropy(text: str) -> float:
    """Per-character Shannon entropy (bits) of ``text``; ``0.0`` when empty."""
    if not text:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for ch in text:
        counts[ch] += 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def has_high_entropy(key: str) -> bool:
    """True if a canonical ``key`` is specific enough to trust for name-only merge.

    Requires a minimum length, a minimum per-character entropy, and that the key
    is not a known generic label, so short and/or generic names (``"ai"``,
    ``"data"``, ``"system"``) are rejected and fall through to the embedding tier.
    """
    if key in _GENERIC_KEYS:
        return False
    return len(key) >= _MIN_KEY_LEN and shannon_entropy(key) >= _MIN_ENTROPY


def _shingles(key: str, k: int = _SHINGLE_K) -> frozenset[str]:
    """Character k-gram shingle set of a canonical key (the key itself if short)."""
    if len(key) <= k:
        return frozenset({key})
    return frozenset(key[i : i + k] for i in range(len(key) - k + 1))


def _base_hash(shingle: str) -> int:
    """A stable 64-bit hash of a shingle (blake2b — deterministic across runs)."""
    return int.from_bytes(
        hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big"
    )


def _perm_coeffs(num_perm: int = _NUM_PERM) -> list[tuple[int, int]]:
    """Deterministic ``(a, b)`` coefficients for ``num_perm`` hash permutations."""
    rng = random.Random(0xA5F00D)  # nosec B311 — fixed-seed MinHash signature coefficients, not cryptographic
    return [
        (rng.randrange(1, _MERSENNE_PRIME), rng.randrange(0, _MERSENNE_PRIME))
        for _ in range(num_perm)
    ]


_COEFFS = _perm_coeffs()


def _minhash(shingles: frozenset[str]) -> tuple[int, ...]:
    """MinHash signature of a shingle set using the shared permutation family."""
    sig = [_MERSENNE_PRIME] * len(_COEFFS)
    for sh in shingles:
        h = _base_hash(sh)
        for i, (a, b) in enumerate(_COEFFS):
            v = ((a * h + b) % _MERSENNE_PRIME) & 0xFFFFFFFFFFFFFFFF
            v %= _MERSENNE_PRIME
            if v < sig[i]:
                sig[i] = v
    return tuple(sig)


def _lsh_bands(sig: tuple[int, ...]) -> list[tuple[int, int]]:
    """Banded LSH keys ``(band_index, band_hash)`` for an LSH candidate index."""
    out: list[tuple[int, int]] = []
    for band in range(_LSH_BANDS):
        start = band * _LSH_ROWS
        chunk = sig[start : start + _LSH_ROWS]
        out.append((band, hash(chunk)))
    return out


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def resolve_entities(items: list[tuple[str, str]]) -> ResolutionResult:
    """Resolve a batch of ``(id, display_name)`` entities, LLM-free.

    Runs the entropy-gated exact + MinHash/LSH ladder and returns a
    :class:`ResolutionResult`. Computes **no** embeddings and makes **no** LLM
    calls — the whole point is to short-circuit those for the common case.

    Args:
        items: ``(id, name)`` pairs (duplicate ids are tolerated; first wins).

    Returns:
        A :class:`ResolutionResult`. ``merge_pairs`` are high-confidence name-only
        duplicates; ``residual_ids`` is what should be escalated to the embedding
        tier.
    """
    result = ResolutionResult()
    seen: set[str] = set()
    eligible: list[tuple[str, str]] = []  # (id, canonical_key)
    for nid, name in items:
        if nid in seen:
            continue
        seen.add(nid)
        key = normalize_name(name)
        if has_high_entropy(key):
            eligible.append((nid, key))
        else:
            result.low_entropy += 1
            result.residual_ids.add(nid)

    # --- tier 1: exact canonical-key match ---
    by_key: dict[str, list[str]] = defaultdict(list)
    for nid, key in eligible:
        by_key[key].append(nid)

    for ids in by_key.values():
        survivor = ids[0]
        for dup in ids[1:]:
            result.merge_pairs.append((survivor, dup, 1.0, "exact"))
            result.resolved_ids.update((survivor, dup))
            result.exact_merges += 1

    # --- tier 3: MinHash + LSH over the distinct canonical keys ---
    distinct_keys = list(by_key.keys())
    if len(distinct_keys) > 1:
        shingles_by_key = {k: _shingles(k) for k in distinct_keys}
        sig_by_key = {k: _minhash(shingles_by_key[k]) for k in distinct_keys}
        buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
        for k in distinct_keys:
            for band_key in _lsh_bands(sig_by_key[k]):
                buckets[band_key].append(k)

        # candidate key-pairs that share at least one band
        candidates: set[tuple[str, str]] = set()
        for members in buckets.values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = sorted((members[i], members[j]))
                    candidates.add((a, b))

        # union-find over keys that pass the exact-Jaccard cutoff
        parent = {k: k for k in distinct_keys}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        fuzzy_scores: dict[tuple[str, str], float] = {}
        for a, b in candidates:
            jac = _jaccard(shingles_by_key[a], shingles_by_key[b])
            if jac >= _JACCARD_THRESHOLD:
                fuzzy_scores[(a, b)] = jac
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

        # emit a merge pair linking the two key-groups' survivors
        key_groups: dict[str, list[str]] = defaultdict(list)
        for k in distinct_keys:
            key_groups[find(k)].append(k)
        for group_keys in key_groups.values():
            if len(group_keys) < 2:
                continue
            survivor = by_key[group_keys[0]][0]
            for other_key in group_keys[1:]:
                dup = by_key[other_key][0]
                # best fuzzy score we saw involving this key pair
                pair = tuple(sorted((group_keys[0], other_key)))
                score = fuzzy_scores.get(pair, _JACCARD_THRESHOLD)  # type: ignore[arg-type]
                result.merge_pairs.append((survivor, dup, float(score), "lsh"))
                result.resolved_ids.update((survivor, dup))
                result.lsh_merges += 1

    # high-entropy keys that matched nothing are still ambiguous → embedding tier
    for nid, _key in eligible:
        if nid not in result.resolved_ids:
            result.residual_ids.add(nid)

    return result


__all__ = [
    "ResolutionResult",
    "resolve_entities",
    "normalize_name",
    "shannon_entropy",
    "has_high_entropy",
]
