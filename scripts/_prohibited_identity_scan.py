"""Bounded scanner for an operator-owned prohibited-identity catalog."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

READ_SIZE = 64 * 1024
MAX_CATALOG_BYTES = 64 * 1024
MAX_GROUP_BYTES = 64 * 1024
_CAMEL_SEGMENT_RE = re.compile(
    rb"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)
_SEPARATORS = frozenset(b"_-")


class IdentityPolicyError(ValueError):
    """The external policy or scanned input violates a bounded contract."""


@dataclass(frozen=True, slots=True)
class IdentityFinding:
    path: str
    line: int
    evidence: str


def load_identity_catalog(path: Path) -> tuple[bytes, ...]:
    """Load a bounded, versioned external identity catalog."""
    with path.open("rb") as stream:
        payload = stream.read(MAX_CATALOG_BYTES + 1)
    if len(payload) > MAX_CATALOG_BYTES:
        raise IdentityPolicyError("identity catalog exceeds the size limit")
    raw = json.loads(payload)
    return _validated_catalog_terms(raw)


def _validated_catalog_terms(raw: object) -> tuple[bytes, ...]:
    identities = _catalog_identities(raw)
    return tuple(sorted({_validated_identity(identity) for identity in identities}))


def _catalog_identities(raw: object) -> list[object]:
    if not isinstance(raw, dict) or set(raw) != {"version", "identities"}:
        raise IdentityPolicyError(
            "identity catalog must contain version and identities"
        )
    if not isinstance(raw["version"], str) or not raw["version"].strip():
        raise IdentityPolicyError("identity catalog version must be non-empty")
    identities = raw["identities"]
    if not isinstance(identities, list) or not identities:
        raise IdentityPolicyError(
            "identity catalog identities must be a non-empty list"
        )
    return identities


def _validated_identity(identity: object) -> bytes:
    if (
        not isinstance(identity, str)
        or not identity.isascii()
        or not identity.isalpha()
    ):
        raise IdentityPolicyError("identity catalog entries must be ASCII letters")
    encoded = identity.casefold().encode("ascii")
    if not 2 <= len(encoded) <= 64:
        raise IdentityPolicyError("identity catalog entry length is outside 2..64")
    return encoded


def scan_prohibited_identities(
    root: Path, paths: Iterable[Path], identities: tuple[bytes, ...]
) -> list[IdentityFinding]:
    """Scan tracked paths and file bodies without loading a whole file."""
    findings: list[IdentityFinding] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if _group_matches(relative.encode(), identities):
            findings.append(IdentityFinding(relative, 0, relative))
        with path.open("rb") as stream:
            findings.extend(
                IdentityFinding(relative, line, group.decode("utf-8", "replace"))
                for line, group in _matching_stream_groups(stream, identities)
            )
    return findings


def _matching_stream_groups(
    stream: BinaryIO, identities: tuple[bytes, ...]
) -> Iterable[tuple[int, bytes]]:
    group = bytearray()
    line = 1
    group_line = 1
    while chunk := stream.read(READ_SIZE):
        matches, line, group_line = _consume_chunk(
            chunk,
            group,
            line=line,
            group_line=group_line,
            identities=identities,
        )
        yield from matches
    if group and _group_matches(bytes(group), identities):
        yield group_line, bytes(group)


def _consume_chunk(
    chunk: bytes,
    group: bytearray,
    *,
    line: int,
    group_line: int,
    identities: tuple[bytes, ...],
) -> tuple[list[tuple[int, bytes]], int, int]:
    matches: list[tuple[int, bytes]] = []
    for value in chunk:
        if _is_group_byte(value):
            group_line = line if not group else group_line
            if len(group) >= MAX_GROUP_BYTES:
                raise IdentityPolicyError(
                    "tracked lexical group exceeds the size limit"
                )
            group.append(value)
            continue
        if group:
            rendered = bytes(group)
            if _group_matches(rendered, identities):
                matches.append((group_line, rendered))
            group.clear()
        line += value == 10
    return matches, line, group_line


def _is_group_byte(value: int) -> bool:
    return (
        48 <= value <= 57
        or 65 <= value <= 90
        or 97 <= value <= 122
        or value in _SEPARATORS
    )


def _group_matches(group: bytes, identities: tuple[bytes, ...]) -> bool:
    positions = [index for index, value in enumerate(group) if value not in _SEPARATORS]
    normalized = bytes(group[index] for index in positions).lower()
    for identity in identities:
        start = normalized.find(identity)
        while start >= 0:
            raw_start = positions[start]
            raw_end = positions[start + len(identity) - 1] + 1
            if not _cross_segment_coincidence(group, raw_start, raw_end):
                return True
            start = normalized.find(identity, start + 1)
    return False


def _cross_segment_coincidence(group: bytes, start: int, end: int) -> bool:
    """Allow a match formed only by the interiors of adjacent lexical segments."""
    overlapping = [
        match.span()
        for match in _CAMEL_SEGMENT_RE.finditer(group)
        if match.start() < end and match.end() > start
    ]
    if len(overlapping) < 2:
        return False
    return start > overlapping[0][0] and end < overlapping[-1][1]
