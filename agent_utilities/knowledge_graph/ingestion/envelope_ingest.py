"""Engine-native atomic ingestion for :class:`ChangeEnvelope`.

(CONCEPT:AU-KG.ingest.envelope-atomic-transaction, AU-P1-5).

**The gap this closes.** AU-P1-6 introduced :class:`ChangeEnvelope` as the one
typed unit-of-change every connector shape (connector push, MCP pull,
fleet-package pull, CDC/webhook, bulk snapshot) *could* emit, but deliberately
left ``source_sync``'s ~20 per-connector handlers on the old ad hoc
``{"id", "type", **props}`` + a single unguarded
``engine.ingest_external_batch(...)`` call — no shared validation, no
per-record lineage, no crash-safe watermark advance. This module is the single
place that turns one ``ChangeEnvelope`` into a durable KG change, as ONE unit.

The production path renders the connector DTO into Epistemic Graph's governed
``ApplyChangeEnvelope`` wire contract.  Graph rows, blob/feature/evidence
material, policy, lineage, typed content version, typed source cursor, and the
CDC/projection outbox are committed by one authoritative redb transaction (and
one Raft entry when clustered) before acknowledgement.  The current
``GraphSession`` supplies verified identity/policy/trace authority; the routed
client binds the catalog epoch and fencing token and retries one stale route
with the same idempotency key.

There is no fallback to a Python multi-step write sequence. An engine without
the authoritative redb capability fails closed.

**Ambient epistemics (W3.4, CONCEPT:AU-KG.temporal.ambient-connector-valid-time).**
A row's bitemporal ``valid_from`` is mapped from the envelope's own
``valid_time``/``event_time`` (the source's reported modification/creation
timestamp) with no per-connector code change, and a delete/reconcile-tombstone
closes ``valid_to`` at the same instant — so an ``as_of`` read over
connector-ingested data answers ``VALID AS OF`` out of the box. Flag-gated
(``KG_AMBIENT_EPISTEMIC``, default ON; per-source opt-out via
``KG_AMBIENT_EPISTEMIC_DISABLED_SOURCES``) and never-fabricating: a source with
no usable timestamp leaves the row exactly as it was before this feature
existed. See :func:`_stamp_ambient_valid_time` / :func:`_stamp_ambient_valid_until`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from importlib.resources import files
from typing import Any
from urllib.parse import quote

from .change_envelope import ChangeEnvelope

logger = logging.getLogger(__name__)

#: Opaque digest lengths this gate exempts from the full privacy-pattern scan
#: (D-GM-3): 24-hex/96-bit (``engine.py``'s ``_ingest_connector`` object_key —
#: ``sha256(portable_uri).hexdigest()[:24]`` — and the matching
#: ``GitMarkdownConnector._document_node_id``/``_object_key`` truncation),
#: 32-hex/128-bit, 40-hex/160-bit (``evidence_spine.py``'s
#: ``artifact_id_for``/``fragment_id_for`` — ``sha256(...).hexdigest()[:40]``),
#: and 64-hex/256-bit full digests. A truncated digest previously fell through
#: to the full scan and could trip the case-insensitive IBAN pattern by chance
#: (~1 in 20 sha256 digests), rejecting a genuine connector-owned document/
#: record id with no PII involved.
#:
#: Also exempts an optional trailing ``#<16-hex>`` content-pin suffix
#: (``Fragment.version_id`` — ``f"{fragment_id}#{content_hash[7:23]}"``, a
#: content-pinned citation address; see the module docstring on
#: ``Fragment.version_id`` in ``evidence_spine.py`` for why the separator is
#: ``#`` and not ``@``: the engine's ``ApplyChangeEnvelope`` privacy guard
#: rejects any ``@`` outright, D-GM-4/D-GS856-6/D-MW-1/D-MW-2). Same class of
#: defect as D-GM-3, on the same shared gate: without this, a fragment's
#: ``fragment:<40-hex>#<16-hex>`` version_id falls through to the full scan and
#: can reproducibly trip the IBAN pattern, rejecting entire connector documents
#: with no PII involved.
_OPAQUE_DIGEST = r"(?:[0-9a-f]{24}|[0-9a-f]{32}|[0-9a-f]{40}|[0-9a-f]{64})"
_OPAQUE_VERSION_PIN = r"(?:#[0-9a-f]{16})?"
_OPAQUE_INTERNAL_ID_RE = re.compile(f"^{_OPAQUE_DIGEST}{_OPAQUE_VERSION_PIN}$")
_OPAQUE_NAMESPACED_ID_RE = re.compile(
    r"^(?P<namespace>[a-z][a-z0-9._-]{0,63}"
    r"(?::[a-z][a-z0-9._-]{0,63}){0,7}):"
    f"{_OPAQUE_DIGEST}{_OPAQUE_VERSION_PIN}$"
)

__all__ = [
    "NativeChangeEnvelopeEngineProxy",
    "NativeChangeEnvelopeUnavailable",
    "ingest_envelope",
    "ingest_graph_slice",
    "read_change_cursor",
    "validate_envelope",
    "validate_rows_against_shacl",
]


class NativeChangeEnvelopeUnavailable(RuntimeError):
    """The authoritative engine cannot commit a native ChangeEnvelope."""


class _NativeOccRetryBudgetExhausted(RuntimeError):
    """Bounded guaranteed-precommit conflicts exhausted their retry budget."""

    def __init__(self, conflicts: list[str]) -> None:
        super().__init__("native ChangeEnvelope OCC retry budget exhausted")
        self.conflicts = tuple(conflicts)


#: Greppable marker embedded in the message of
#: :class:`_PartialMaterializationRetriesExhausted` so a caller reached only
#: through the synthesized ``RuntimeError`` that :func:`ingest_graph_slice`
#: raises on a non-success status (e.g. ``source_sync._write_fleet_slice``)
#: can still tell "the engine was still materializing and gave up after
#: bounded retries" apart from a genuine content rejection, without
#: re-parsing engine wire payloads. Exported (not underscore-only in intent,
#: just following this module's existing private-cross-import convention —
#: see ``_retryable_partial_materialization`` reused from ``engine_tasks.py``)
#: for ``source_sync.py`` to import and match against ``str(exc)``.
PARTIAL_MATERIALIZATION_RETRIES_EXHAUSTED_MARKER = (
    "PARTIAL_MATERIALIZATION_RETRIES_EXHAUSTED"
)


class _PartialMaterializationRetriesExhausted(RuntimeError):
    """A retryable PARTIAL_MATERIALIZATION signal never cleared in time.

    Raised by :func:`ingest_envelope` only after
    :func:`~agent_utilities.knowledge_graph.core.engine_tasks._retryable_partial_materialization`
    matched the engine's exact wire payload on every attempt, and one of the
    three bounded-resume stop conditions fired (mirrors
    ``pipeline/runner.py``'s ``_MATERIALIZATION_MAX_ATTEMPTS`` resume loop):
    the attempt budget ran out, the ``completeness_cursor`` stopped advancing,
    or ``source_snapshot_version`` changed mid-resume. This is DELIBERATELY
    NOT the same outcome as a genuine engine rejection (bad content, policy
    denial): the row was never judged unacceptable, the engine just never
    finished materializing within budget. Callers MUST NOT treat this the
    same as a content rejection — e.g. never cache the row as permanently
    known-bad from this signal alone.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            f"{PARTIAL_MATERIALIZATION_RETRIES_EXHAUSTED_MARKER}: {message}"
        )


class NativeChangeEnvelopeEngineProxy:
    """Make batch-producing connectors use the native envelope boundary.

    Hydration connectors were intentionally written against the small
    ``engine.ingest_external_batch(domain, entities, relationships)`` protocol.
    Rewriting every connector would duplicate governance logic and make future
    additions easy to get wrong.  This transparent proxy preserves their read
    and orchestration surface while replacing that one durable method with a
    native :func:`ingest_graph_slice` commit. It is safe to wrap more than once.
    """

    __slots__ = ("_authority",)

    def __init__(self, authority: Any) -> None:
        self._authority = (
            authority.authority
            if isinstance(authority, NativeChangeEnvelopeEngineProxy)
            else authority
        )

    @property
    def authority(self) -> Any:
        """The real engine authority used for the atomic commit."""

        return self._authority

    def __getattr__(self, name: str) -> Any:
        return getattr(self._authority, name)

    def ingest_external_batch(
        self,
        domain: str,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Translate connector DTO fields into canonical graph properties."""

        canonical_entities = []
        for item in entities:
            row = dict(item)
            row["node_type"] = row.pop("type")
            canonical_entities.append(row)
        canonical_relationships = []
        for item in relationships or []:
            row = dict(item)
            row["relationship"] = row.pop("type")
            canonical_relationships.append(row)

        return ingest_graph_slice(
            self._authority,
            domain,
            canonical_entities,
            canonical_relationships,
            source_instance=domain,
        )


@dataclass(frozen=True)
class _NativeAuthority:
    compute: Any
    backend: Any


_NATIVE_LOCKS_GUARD = threading.Lock()
_NATIVE_LOCKS: dict[tuple[str, str], threading.RLock] = {}
_NATIVE_GRAPH_VERSIONS: dict[tuple[str, str], int] = {}

# Sentinel telling `_native_material` to read the live source cursor itself (the
# single-envelope path). The batch path reads the page's cursor ONCE and passes the
# CHAINED position instead, because envelopes in a page apply sequentially inside ONE
# transaction — a live per-record read would see the page-start cursor and STALE
# every envelope after the first.
_CURSOR_READ_LIVE = object()
_NATIVE_OCC_MAX_ATTEMPTS = 8
_NATIVE_OCC_BACKOFF_BASE_SECONDS = 0.001
_NATIVE_OCC_BACKOFF_CAP_SECONDS = 0.01

# Bounded resume for the engine's retryable PARTIAL_MATERIALIZATION signal
# (a catalog-known graph deliberately rejects every operation while its
# bounded lazy-open rebuild is incomplete — availability state, not an
# ingestion failure). Same fixed-constant discipline, and the SAME values, as
# ``pipeline/runner.py``'s ``_MATERIALIZATION_MAX_ATTEMPTS`` /
# ``_MATERIALIZATION_RETRY_DELAY_S`` (a sibling lane's fix for the identical
# wire payload at the pipeline-phase granularity) — reused by value rather
# than imported so this module keeps its existing no-cross-layer-import shape
# (``pipeline`` depends on ``ingestion``, not the reverse). A hardcoded module
# constant, never an env knob (Configuration discipline).
#
# This retry is paid ONCE per ``ingest_envelope`` call, whatever granularity
# that call happens to be invoked at. ``source_sync._write_fleet_slice``
# calls into this through ``ingest_graph_slice`` with the ENTIRE fleet-catalog
# slice packed into ONE envelope at the top of its bisection tree, so in the
# steady state (the engine finishes materializing within the budget below)
# the whole ~1,372-row slice lands after this ONE bounded wait — never a
# per-row retry loop. ``_write_fleet_slice`` deliberately does not bisect
# further on this exact signal (seeing
# ``PARTIAL_MATERIALIZATION_RETRIES_EXHAUSTED_MARKER`` in the propagated
# error) precisely so that, even in the worst case, this budget is paid a
# small constant number of times — not once per row.
_MATERIALIZATION_MAX_ATTEMPTS = 8
_MATERIALIZATION_RETRY_DELAY_S = 1.0
_MATERIALIZATION_UNSET = object()
_NATIVE_OCC_CONFLICT_RE = re.compile(
    r"\b(?:STALE_GRAPH_VERSION|STALE_VERSION|STALE_CONTENT_VERSION|STALE_CURSOR)\b",
    re.IGNORECASE,
)
_STALE_GRAPH_VERSION_RE = re.compile(
    r"(?:current\s+|authoritative version is\s+)(\d+)", re.IGNORECASE
)


# ── mandatory engine-native SHACL admission ──


_KG_NS = "http://knuckles.team/kg#"
_SHACL_SKIP_PROPERTIES = {"embedding", "ewc_fisher_diag", "node_type"}


def _shacl_class_name(node_type: Any) -> str:
    cleaned = str(node_type or "").strip()
    if not cleaned:
        return "Thing"
    if any(separator in cleaned for separator in (" ", "_", "-")):
        parts = cleaned.replace("-", " ").replace("_", " ").split()
        return "".join(part[:1].upper() + part[1:] for part in parts)
    return cleaned[:1].upper() + cleaned[1:]


def _shacl_iri(value: Any) -> str:
    return f"<{_KG_NS}{quote(str(value), safe='')}>"


def _shacl_data_graph(rows: list[tuple[str, dict[str, Any]]]) -> str:
    """Render deterministic, injection-safe Turtle for connector rows."""
    triples = [
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
    ]
    for node_id, row in rows:
        subject = _shacl_iri(f"node/{node_id}")
        class_iri = _shacl_iri(_shacl_class_name(row.get("node_type")))
        triples.append(f"{subject} rdf:type {class_iri} .")
        for key, value in sorted(row.items(), key=lambda item: str(item[0])):
            if key in _SHACL_SKIP_PROPERTIES or isinstance(value, bool):
                continue
            predicate = _shacl_iri(key)
            if isinstance(value, str) and value:
                triples.append(
                    f"{subject} {predicate} {json.dumps(value, ensure_ascii=False)} ."
                )
            elif isinstance(value, int):
                triples.append(f'{subject} {predicate} "{value}"^^xsd:integer .')
            elif isinstance(value, float) and math.isfinite(value):
                triples.append(f'{subject} {predicate} "{value}"^^xsd:double .')
    return "\n".join(triples) + "\n"


def _shacl_validate_rows(
    client: Any,
    rows: list[tuple[str, dict[str, Any]]],
) -> None:
    """Admit connector rows only after native canonical SHACL validation.

    Connector material is an external trust boundary. Validation therefore is
    unconditional: an unavailable engine validator, missing packaged shapes,
    malformed report, or non-conforming data fails closed before the native
    ChangeEnvelope is constructed. Invalid rows are never materialized.
    """
    try:
        shapes = (
            files("agent_utilities.knowledge_graph")
            .joinpath("shapes", "governance.shapes.ttl")
            .read_text(encoding="utf-8")
        )
        rdf = getattr(client, "rdf", None)
        validate = getattr(rdf, "validate_shacl", None)
        if not callable(validate):
            raise NativeChangeEnvelopeUnavailable(
                "connector SHACL validation support is unavailable"
            )
        report = validate(shapes, _shacl_data_graph(rows))
    except (NativeChangeEnvelopeUnavailable, ValueError):
        raise
    except Exception as exc:
        raise NativeChangeEnvelopeUnavailable(
            "connector SHACL validation could not complete"
        ) from exc
    if not isinstance(report, dict) or not isinstance(report.get("conforms"), bool):
        raise NativeChangeEnvelopeUnavailable(
            "connector SHACL validator returned an invalid report"
        )
    if not report["conforms"]:
        # Name the failing shapes. "violates the governed ontology" without the
        # violations is unactionable: it cannot distinguish a missing required
        # property from a bad datatype on a single row out of hundreds.
        raise ValueError(
            "connector material violates the governed ontology: "
            f"{_shacl_violation_summary(report)}"
        )


def validate_rows_against_shacl(
    client: Any, rows: list[tuple[str, dict[str, Any]]]
) -> None:
    """Public reuse point for :func:`_shacl_validate_rows` (CONCEPT:AU-KG.ingest.governed-claim-promotion).

    The connector ingestion boundary's fail-closed SHACL gate — unavailable
    validator, missing shapes, malformed report, or non-conforming data all
    ``raise`` — exposed so another governed-write caller (e.g.
    ``knowledge_graph.ingestion.promotion``'s candidate-claim promotion gate)
    can reuse the SAME unconditional validate-or-raise contract instead of a
    second SHACL implementation.
    """
    _shacl_validate_rows(client, rows)


def _shacl_violation_detail(item: Any) -> str:
    """Render one SHACL result row, or ``""`` when it carries no usable detail."""
    if not isinstance(item, dict):
        return ""
    return " ".join(
        str(item[key])
        for key in ("focusNode", "resultPath", "sourceShape", "resultMessage")
        if item.get(key)
    )


def _shacl_violation_summary(report: dict[str, Any], *, limit: int = 5) -> str:
    """Summarize a non-conforming SHACL report as a short, bounded string."""
    results = report.get("results")
    if not isinstance(results, list) or not results:
        return str(report.get("message") or "no violation detail reported")
    seen: list[str] = []
    for item in results:
        detail = _shacl_violation_detail(item)
        if detail and detail not in seen:
            seen.append(detail)
        if len(seen) >= limit:
            break
    extra = len(results) - len(seen)
    return "; ".join(seen) + (f" (+{extra} more)" if extra > 0 else "")


# ── validate ─────────────────────────────────────────────────────────────


def _is_sha256_digest(digest: str) -> bool:
    """``sha256:`` + exactly 64 lowercase hex characters."""
    return (
        digest.startswith("sha256:")
        and len(digest) == len("sha256:") + 64
        and all(char in "0123456789abcdef" for char in digest.removeprefix("sha256:"))
    )


def _is_non_negative_int(value: Any) -> bool:
    """A real ``int`` (never a ``bool``) that is zero or greater."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_blob_fields(envelope: ChangeEnvelope) -> list[str]:
    """Per-field blob checks, emitted in the original digest/length/type order."""
    checks = (
        (
            _is_sha256_digest(envelope.blob_digest or ""),
            "blob_digest must be a sha256 digest",
        ),
        (
            _is_non_negative_int(envelope.blob_length),
            "blob_length must be non-negative",
        ),
        (
            bool(
                isinstance(envelope.blob_media_type, str) and envelope.blob_media_type
            ),
            "blob_media_type must be non-empty",
        ),
    )
    return [message for ok, message in checks if not ok]


def _validate_blob_metadata(envelope: ChangeEnvelope) -> list[str]:
    """Blob-metadata completeness/shape checks, in their original order."""
    present = [
        envelope.blob_digest is not None,
        envelope.blob_length is not None,
        envelope.blob_media_type is not None,
    ]
    if not any(present):
        return []
    if envelope.blob_ref is None or not all(present):
        return ["blob_digest, blob_length and blob_media_type must accompany blob_ref"]
    return _validate_blob_fields(envelope)


def _validate_structured_evidence(envelope: ChangeEnvelope) -> list[str]:
    """Structured evidence must be a canonical-JSON object within the size bound."""
    if envelope.structured_evidence is None:
        return []
    if not isinstance(envelope.structured_evidence, dict):
        return ["structured_evidence must be an object"]
    try:
        evidence_bytes = json.dumps(
            envelope.structured_evidence,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return ["structured_evidence must be canonical JSON"]
    if len(evidence_bytes) > 4 * 1024 * 1024:
        return ["structured_evidence exceeds the bounded size"]
    return []


def _validate_envelope_policy(envelope: ChangeEnvelope) -> list[str]:
    """Fail-closed connector-permission policy checks (CONCEPT:AU-P0-4).

    An "unknown" ACL must never silently become "public": a ``PUBLIC``
    classification requires an explicit ``source_acl.is_public=True`` proof and
    the converse must hold too, so durable policy and the runtime ACL
    projection cannot diverge. Order matches the original inline chain.
    """
    from ...models.company_brain import DataClassification

    violations: list[str] = []
    if envelope.operation == "upsert" and envelope.source_acl is None:
        violations.append(
            "upsert envelope has no source ACL proof and was not quarantined"
        )
    if not 0.0 <= envelope.confidence <= 1.0:
        violations.append(f"confidence {envelope.confidence!r} out of range [0.0, 1.0]")
    if envelope.classification == DataClassification.PUBLIC and not (
        envelope.source_acl is not None and envelope.source_acl.is_public
    ):
        violations.append(
            "classification=PUBLIC requires an explicit source_acl.is_public=True "
            "(CONCEPT:AU-P0-4 fail-closed connector permissions) — refusing to "
            "publish an object with no proof of public access"
        )
    if (
        envelope.source_acl is not None
        and envelope.source_acl.is_public
        and envelope.classification != DataClassification.PUBLIC
    ):
        violations.append(
            "source_acl.is_public=True requires classification=PUBLIC so durable "
            "policy and runtime ACL projection cannot diverge"
        )
    return violations


def _validate_envelope(envelope: ChangeEnvelope) -> list[str]:
    """Schema + fail-closed policy checks. Empty list = OK.

    Re-affirms the invariants ``ChangeEnvelope.__post_init__`` already enforces
    (defense-in-depth against a future mutation) and adds the policy check
    ``__post_init__`` can't: CONCEPT:AU-P0-4 fail-closed connector permissions
    — a ``PUBLIC``-classified object must carry an explicit
    ``source_acl.is_public=True`` proof; "unknown" must never silently become
    "public" just because a connector forgot to set an ACL.
    """
    violations: list[str] = []
    if envelope.operation not in ("upsert", "delete", "snapshot_complete"):
        violations.append(f"invalid operation {envelope.operation!r}")
    if envelope.typed_payload is not None and envelope.blob_ref is not None:
        violations.append(
            "typed_payload and blob_ref are mutually exclusive on this envelope"
        )
    violations.extend(_validate_blob_metadata(envelope))
    violations.extend(_validate_structured_evidence(envelope))
    if (
        envelope.operation == "upsert"
        and envelope.typed_payload is None
        and envelope.blob_ref is None
    ):
        violations.append(
            "upsert envelope carries neither typed_payload nor blob_ref — nothing to write"
        )
    violations.extend(_validate_envelope_policy(envelope))
    return violations


def validate_envelope(envelope: ChangeEnvelope) -> list[str]:
    """Public reuse point for :func:`_validate_envelope` (CONCEPT:AU-KG.ingest.governed-claim-promotion).

    Lets another governed-write caller (e.g. a candidate-claim promotion gate
    deciding whether a claim's proposed materialization is even well-formed
    before a steward reviews it) run the SAME fail-closed schema + policy
    checks ``ingest_envelope`` itself enforces at the write boundary — never a
    second, potentially-drifting policy implementation. Empty list = OK.
    """
    return _validate_envelope(envelope)


#: Private "this entry was not masked" marker for :class:`_OpaqueIdentityVault`.
#: A unique object so no payload value can ever collide with it.
_UNMASKED = object()


def _assert_safe_identity(guard: Any, value: Any) -> None:
    """Reject sensitive identities without scanning opaque digest material.

    Deterministic HMAC/SHA identifiers are random-looking tokens, so a
    runtime deny term can occur in their hexadecimal material by chance.
    Treat only exact lowercase 128/256-bit digests as opaque.  For a
    namespaced identifier, the bounded namespace is still scanned while
    the digest suffix is not; arbitrary namespaced content receives the
    normal full privacy scan.
    """
    rendered = str(value or "")
    if _OPAQUE_INTERNAL_ID_RE.fullmatch(rendered):
        return
    namespaced = _OPAQUE_NAMESPACED_ID_RE.fullmatch(rendered)
    candidate = namespaced.group("namespace") if namespaced else rendered
    _, report = guard.sanitize_text(candidate)
    if report.changed:
        raise ValueError("unsafe envelope identity")


def _is_identity_field(key: Any, *, in_links: bool) -> bool:
    """Is ``key`` a routing/identity key that must not be rewritten?

    ``id``/``*_id``/``*Id``/``*ID`` are unconditionally node-identity
    keys everywhere in the payload. ``source``/``target`` are ONLY
    node-id references when they appear inside an edge record (an
    entry of a ``_links`` list) — a :class:`DocumentChunk`/Document
    record legitimately reuses the bare key ``source`` for a
    human-readable provenance label (e.g. a filesystem path or URL),
    which is exactly the kind of content the privacy gate commits
    to "deeply sanitizing" rather than rejecting outright. Treating
    that provenance field as an opaque identity previously made
    ``ingest_envelope`` reject ordinary document/skill ingestion
    whenever the source path matched a PII pattern (e.g. a POSIX
    local path) instead of just redacting it.
    """
    rendered = str(key)
    normalized = re.sub(r"[^a-z0-9]+", "_", rendered.casefold()).strip("_")
    if (
        normalized == "id"
        or normalized.endswith("_id")
        or rendered.endswith(("Id", "ID"))
    ):
        return True
    return in_links and normalized in {"source", "target"}


class _OpaqueIdentityVault:
    """Hide opaque identity strings from one payload sanitize pass.

    Identity and routing keys cannot be rewritten safely, but a deterministic
    digest identifier is random-looking enough to trip a privacy pattern by
    chance. Each opaque identity is swapped for a private negative-int
    sentinel before the scan and restored verbatim afterwards, so the scan
    never sees (and so can never rewrite) the identity.
    """

    def __init__(self, guard: Any) -> None:
        self._guard = guard
        self._values: dict[int, Any] = {}
        self._next_sentinel = -(1 << 255)

    def _reserve(self, item: Any) -> int:
        sentinel = self._next_sentinel
        self._next_sentinel -= 1
        self._values[sentinel] = item
        return sentinel

    def _mask_entry(self, key: Any, item: Any, *, in_links: bool) -> Any:
        """Sentinel for one opaque identity entry, else :data:`_UNMASKED`.

        :data:`_UNMASKED` means "not an opaque identity" and the caller must
        recurse into the value — the SAME fall-through the previous inline
        implementation took for a non-opaque identity field.
        """
        if not _is_identity_field(key, in_links=in_links) or item in (None, ""):
            return _UNMASKED
        _assert_safe_identity(self._guard, item)
        rendered = str(item)
        if _OPAQUE_INTERNAL_ID_RE.fullmatch(
            rendered
        ) or _OPAQUE_NAMESPACED_ID_RE.fullmatch(rendered):
            return self._reserve(item)
        return _UNMASKED

    def mask(self, value: Any, *, in_links: bool = False) -> Any:
        if isinstance(value, dict):
            masked: dict[Any, Any] = {}
            for key, item in value.items():
                replacement = self._mask_entry(key, item, in_links=in_links)
                masked[key] = (
                    replacement
                    if replacement is not _UNMASKED
                    else self.mask(item, in_links=in_links or key == "_links")
                )
            return masked
        if isinstance(value, list | tuple | set | frozenset):
            return [self.mask(item, in_links=in_links) for item in value]
        return value

    def _is_sentinel(self, key: Any, item: Any, *, in_links: bool) -> bool:
        return (
            _is_identity_field(key, in_links=in_links)
            and isinstance(item, int)
            and not isinstance(item, bool)
            and item in self._values
        )

    def restore(self, value: Any, *, in_links: bool = False) -> Any:
        if isinstance(value, dict):
            return {
                key: (
                    self._values[item]
                    if self._is_sentinel(key, item, in_links=in_links)
                    else self.restore(item, in_links=in_links or key == "_links")
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.restore(item, in_links=in_links) for item in value]
        return value


def _sanitize_typed_payload(
    guard: Any, payload: Any
) -> tuple[dict[str, Any] | None, Any]:
    """Deeply sanitize a typed payload without rewriting its identity keys."""
    if payload is None:
        _, report = guard.sanitize(None)
        return None, report
    vault = _OpaqueIdentityVault(guard)
    try:
        masked_payload = vault.mask(payload)
    except ValueError:
        raise ValueError("unsafe payload identity") from None
    clean_payload, report = guard.sanitize(masked_payload)
    if not isinstance(clean_payload, dict):
        raise ValueError("invalid sanitized payload")
    return vault.restore(clean_payload), report


def _sanitize_envelope_acl(guard: Any, access: Any) -> tuple[Any, int]:
    """Return ``(access, redaction_count)`` — fail CLOSED, never widened.

    An ACL carrying an unsafe principal is replaced wholesale by the deny-all
    quarantine (the principals are not merely dropped, which would silently
    widen the object to whatever remained). Stripping user emails from a
    non-public ACL that then has no remaining principal quarantines for the
    same reason.
    """
    from ...protocols.source_connectors.base import ExternalAccess

    if access is None:
        return None, 0
    unsafe_acl = False
    for principal in (*access.group_ids, *access.read_roles, *access.markings):
        _, report = guard.sanitize_text(str(principal))
        unsafe_acl = unsafe_acl or report.changed
    acl_redactions = len(access.user_emails)
    if unsafe_acl:
        acl_redactions += (
            len(access.group_ids) + len(access.read_roles) + len(access.markings)
        )
        return ExternalAccess.quarantined(), acl_redactions
    if access.user_emails:
        access = access.model_copy(update={"user_emails": []})
        if not access.is_public and not (
            access.group_ids or access.read_roles or access.markings
        ):
            return ExternalAccess.quarantined(), acl_redactions
    return access, acl_redactions


def _privacy_redaction_summary(
    reports: tuple[Any, ...], acl_redactions: int
) -> dict[str, Any] | None:
    """Non-sensitive redaction summary, or ``None`` when nothing was redacted."""
    redactions = sum(report.redactions for report in reports) + acl_redactions
    if not redactions:
        return None
    categories: set[str] = set()
    for report in reports:
        categories.update(report.detected_types)
    if acl_redactions:
        categories.add("acl_principal")
    return {"redactions": redactions, "detected_types": sorted(categories)}


def _privacy_gate(envelope: ChangeEnvelope) -> ChangeEnvelope:
    """Remove PII and machine locations before any durable envelope step.

    Identity and ACL routing keys cannot be rewritten safely. An unsafe object
    identity is rejected; unsafe ACL principals are replaced by the deny-all
    quarantine. Content/provenance fields are deeply sanitized and only a
    non-sensitive redaction summary is retained.
    """
    from ...security.persistence_privacy import PersistencePrivacyGuard

    guard = PersistencePrivacyGuard()

    for value in (envelope.envelope_id, envelope.idempotency_key):
        _assert_safe_identity(guard, value)
    for value in (
        envelope.connector,
        envelope.tenant,
        envelope.source_instance,
        envelope.source_object_id,
        envelope.source_version,
        *envelope.live_ids,
    ):
        _assert_safe_identity(guard, value)

    clean_payload, payload_report = _sanitize_typed_payload(
        guard, envelope.typed_payload
    )

    clean_provenance, provenance_report = guard.sanitize(envelope.provenance)
    if not isinstance(clean_provenance, dict):
        clean_provenance = {}
    clean_evidence, evidence_report = guard.sanitize(envelope.structured_evidence)
    if clean_evidence is not None and not isinstance(clean_evidence, dict):
        raise ValueError("invalid sanitized structured evidence")
    clean_operational, operational_report = guard.sanitize(
        {
            "blob_ref": envelope.blob_ref,
            "checkpoint": envelope.checkpoint,
            "trace_context": envelope.trace_context,
        }
    )

    access, acl_redactions = _sanitize_envelope_acl(guard, envelope.source_acl)

    summary = _privacy_redaction_summary(
        (payload_report, provenance_report, evidence_report, operational_report),
        acl_redactions,
    )
    if summary is not None:
        clean_provenance = dict(clean_provenance)
        clean_provenance["persistence_privacy"] = summary

    return replace(
        envelope,
        typed_payload=clean_payload,
        blob_ref=clean_operational.get("blob_ref"),
        checkpoint=clean_operational.get("checkpoint"),
        trace_context=clean_operational.get("trace_context"),
        provenance=clean_provenance,
        structured_evidence=clean_evidence,
        source_acl=access,
    )


# ── identity resolution ─────────────────────────────────────────────────


def _resolve_identity(
    envelope: ChangeEnvelope,
) -> tuple[str | None, dict[str, Any] | None]:
    """Return ``(node_id, row)`` — ``row`` is the rendered entity dict for an
    upsert with an inline ``typed_payload`` (``None`` for a blob-backed upsert,
    a delete, or a snapshot_complete marker, each handled without a row)."""
    if envelope.operation == "snapshot_complete":
        return None, None
    if envelope.operation == "upsert" and envelope.typed_payload is not None:
        row = envelope.to_entity_dict()
        node_id = str(
            row.get("id") or envelope.source_object_id or envelope.idempotency_key
        )
        return node_id, row
    node_id = envelope.source_object_id or envelope.idempotency_key
    return node_id, None


def _native_lock(scope: tuple[str, str]) -> threading.RLock:
    with _NATIVE_LOCKS_GUARD:
        return _NATIVE_LOCKS.setdefault(scope, threading.RLock())


def _native_occ_backoff(attempt: int) -> None:
    """Yield briefly between guaranteed-precommit OCC conflicts.

    The delay is deliberately tiny: the version learned from a conflict ages
    while this process waits. Full jitter still prevents independent connector
    processes from retrying in lockstep, and the cap keeps synchronous ingest
    latency bounded below the surrounding tool timeout.
    """

    ceiling = min(
        _NATIVE_OCC_BACKOFF_CAP_SECONDS,
        _NATIVE_OCC_BACKOFF_BASE_SECONDS * (2 ** max(0, attempt)),
    )
    time.sleep(random.SystemRandom().uniform(0.0, ceiling))


def _has_change_client(value: Any) -> bool:
    client = getattr(value, "client", None)
    return client is not None and getattr(client, "changes", None) is not None


def _resolve_native_authority(engine: Any) -> _NativeAuthority:
    """Resolve the Epistemic Graph authority without selecting a mirror."""
    if _has_change_client(engine):
        return _NativeAuthority(engine, None)

    backend = getattr(engine, "backend", None)
    authority_backend = getattr(backend, "_authority", backend)
    compute = getattr(authority_backend, "graph", None)
    if _has_change_client(compute):
        # Keep the OUTER backend as the publication seam. In a fan-out topology
        # ``authority_backend`` is intentionally unwrapped only to reach the
        # native ChangeEnvelope client; later embedding publication must traverse
        # the outer fan-out so its mirror outbox receives the committed vector.
        return _NativeAuthority(compute, backend)

    # A compute scratch graph is not an authority when another backend exists.
    if backend is None:
        compute = getattr(engine, "graph", None)
        if _has_change_client(compute):
            return _NativeAuthority(compute, None)

    raise NativeChangeEnvelopeUnavailable(
        "the configured write authority is not Epistemic Graph"
    )


def _json_value(value: Any) -> Any:
    """Return deterministic MessagePack-JSON material (no Python-only values)."""
    return json.loads(
        json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    )


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _native_envelope_id(envelope: ChangeEnvelope) -> str:
    """Stable engine identity for every delivery of one logical change."""
    return f"envelope:{envelope.idempotency_key}"


def _pack(value: Any) -> bytes:
    import msgpack

    return msgpack.packb(_json_value(value), use_bin_type=True)


def _observed_at_ms(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.timestamp() * 1000)
    except (TypeError, ValueError, OverflowError):
        # Constructor-owned and therefore stable for every retry of this DTO.
        return int(_digest(str(value))[:12], 16)


def _typed_position(value: str | None, *, content: bool) -> dict[str, Any]:
    raw = str(value or "")
    if raw.isdecimal():
        return {"kind": "sequence", "value": int(raw)}
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return {"kind": "timestamp_millis", "value": int(parsed.timestamp() * 1000)}
    except (TypeError, ValueError, OverflowError):
        discriminator = "version_type" if content else "cursor_type"
        return {
            "kind": "opaque",
            "value": {discriminator: "connector_opaque_v1", "value": raw},
        }


def _numeric_position_advances(left: Any, right: Any) -> bool:
    """Strictly-greater comparison for a sequence/timestamp position."""
    if left is None or right is None:
        return False
    try:
        return int(left) > int(right)
    except (TypeError, ValueError):
        return False


def _opaque_position_advances(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """A connector-opaque position advances only within the same cursor type."""
    left_type = left.get("cursor_type", left.get("version_type"))
    right_type = right.get("cursor_type", right.get("version_type"))
    return left_type == right_type and bool(left.get("value")) and left != right


def _position_advances(next_value: dict[str, Any], prior: dict[str, Any]) -> bool:
    if next_value.get("kind") != prior.get("kind"):
        return False
    kind = next_value.get("kind")
    left = next_value.get("value")
    right = prior.get("value")
    if kind in {"sequence", "timestamp_millis"}:
        return _numeric_position_advances(left, right)
    if kind == "opaque" and isinstance(left, dict) and isinstance(right, dict):
        return _opaque_position_advances(left, right)
    return False


def _cursor_partition(source_instance: str) -> str:
    return (
        hashlib.sha256(source_instance.encode("utf-8")).hexdigest()[:32]
        if source_instance
        else ""
    )


def _checkpoint_from_sequence(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return None


def _checkpoint_from_timestamp_millis(value: Any) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromtimestamp(int(value) / 1000, tz=UTC)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _checkpoint_from_opaque(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("value")
    return str(raw) if raw not in (None, "") else None


#: Typed-position ``kind`` -> reader. A kind with no reader (or a reader that
#: cannot decode its value) yields ``None``, exactly as the previous if-chain's
#: terminal ``return None`` did: an undecodable position is never a checkpoint.
_CHECKPOINT_READERS: dict[str, Callable[[Any], str | None]] = {
    "sequence": _checkpoint_from_sequence,
    "timestamp_millis": _checkpoint_from_timestamp_millis,
    "opaque": _checkpoint_from_opaque,
}


def _checkpoint_from_position(position: Any) -> str | None:
    if not isinstance(position, dict):
        return None
    reader = _CHECKPOINT_READERS.get(str(position.get("kind") or ""))
    return reader(position.get("value")) if reader is not None else None


def _advanced_content_position(
    prior: dict[str, Any], material_digest: str
) -> dict[str, Any] | None:
    """Advance a prior typed content position, or ``None`` if it cannot be read.

    ``None`` means "no advancing position derivable from the prior value" and
    the caller falls back to the digest-derived position — the SAME outcome the
    previous inline chain reached by falling through its ``except``/``if`` arms.
    """
    kind = prior.get("kind")
    value = prior.get("value")
    if kind in {"sequence", "timestamp_millis"} and value is not None:
        try:
            return {"kind": kind, "value": int(value) + 1}
        except (TypeError, ValueError):
            return None
    if kind == "opaque" and isinstance(value, dict):
        version_type = str(value.get("version_type") or "connector_opaque_v1")
        return {
            "kind": "opaque",
            "value": {"version_type": version_type, "value": material_digest},
        }
    return None


def _content_position(
    explicit: str | None,
    current: dict[str, Any] | None,
    material_digest: str,
) -> dict[str, Any]:
    """Choose an advancing content version without inventing wall-clock state."""
    if explicit:
        return _typed_position(explicit, content=True)
    prior = (
        current.get("source_version")
        if isinstance(current, dict) and isinstance(current.get("source_version"), dict)
        else None
    )
    if isinstance(prior, dict):
        advanced = _advanced_content_position(prior, material_digest)
        if advanced is not None:
            return advanced
    return _typed_position(material_digest, content=True)


def _node_properties(client: Any, node_id: str) -> dict[str, Any]:
    value = client.nodes.properties(node_id)
    if isinstance(value, dict):
        return _json_value(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def _node_properties_verified(client: Any, node_id: str) -> dict[str, Any]:
    """Point-hydrate one node, rejecting ambiguous malformed responses."""
    value = client.nodes.properties(node_id)
    if value is None:
        return {}
    if isinstance(value, dict):
        return _json_value(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "node property hydration returned malformed JSON"
            ) from exc
        if isinstance(decoded, dict):
            return decoded
    raise RuntimeError("node property hydration returned an invalid payload")


def _node_properties_point_reads(
    client: Any, node_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Fail-closed compatibility fallback: verified point read per requested id."""
    return {node_id: _node_properties_verified(client, node_id) for node_id in node_ids}


def _decode_batch_properties(properties: Any) -> dict[str, Any] | None:
    """Decode one batch entry.

    ``{}`` means "the engine reported this node has no properties"; ``None``
    means "this entry is malformed/ambiguous" and the caller MUST re-read the
    node individually rather than treat it as absent — an ambiguous entry that
    silently became ``{}`` would make a partial upsert look like a new entity
    and clear a healthy vector.
    """
    if isinstance(properties, dict):
        return _json_value(properties)
    if isinstance(properties, str):
        try:
            decoded = json.loads(properties)
        except (TypeError, ValueError):
            return None
        return decoded if isinstance(decoded, dict) else None
    if properties is None:
        return {}
    return None


def _node_properties_batch(
    client: Any, node_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Read a bounded node set in one RPC, with a compatibility fallback."""
    if not node_ids:
        return {}
    properties_batch = getattr(client.nodes, "properties_batch", None)
    if not callable(properties_batch):
        return _node_properties_point_reads(client, node_ids)
    raw = properties_batch(node_ids)
    if not isinstance(raw, dict):
        # A malformed/degraded batch response is not evidence that every node is
        # absent.  Falling through as ``{}`` makes a partial non-text upsert look
        # like a new entity and clears a healthy vector.  Bound the compatibility
        # fallback to exactly the requested IDs and fail closed via point reads.
        return _node_properties_point_reads(client, node_ids)
    result: dict[str, dict[str, Any]] = {}
    for node_id, properties in raw.items():
        decoded = _decode_batch_properties(properties)
        if decoded is not None:
            result[str(node_id)] = decoded
    # A partial batch response is equally ambiguous: hydrate omitted requested
    # IDs individually instead of treating them as non-existent nodes.
    for node_id in node_ids:
        if node_id not in result:
            result[node_id] = _node_properties_verified(client, node_id)
    return result


def _snapshot_rows(
    client: Any, connector: str, source_instance: str = ""
) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for raw_id, raw_properties in client.nodes.list():
        properties: Any = raw_properties
        if isinstance(properties, str):
            try:
                properties = json.loads(properties)
            except (TypeError, ValueError):
                properties = {}
        if (
            isinstance(properties, dict)
            and str(properties.get("domain") or "") == connector
            and (
                not source_instance
                or str(properties.get("source_instance") or "") == source_instance
            )
        ):
            rows.append((str(raw_id), _json_value(properties)))
    return rows


def _native_session(
    authority: _NativeAuthority, envelope: ChangeEnvelope
) -> tuple[_NativeAuthority, Any]:
    from ..core.session import current_session, resolve_session

    ambient = current_session()
    if ambient is None:
        raise NativeChangeEnvelopeUnavailable(
            "ingestion requires a middleware-minted GraphSession"
        )
    session = resolve_session(ambient, required_scope="kg:write")
    if not session.graph or not session.tenant:
        raise NativeChangeEnvelopeUnavailable(
            "verified GraphSession lacks an explicit graph or tenant"
        )
    if envelope.tenant and session.tenant != envelope.tenant:
        raise PermissionError("ChangeEnvelope tenant does not match GraphSession")
    graph = session.graph
    compute = authority.compute
    view_factory = getattr(compute, "for_graph", None)
    if callable(view_factory):
        compute = view_factory(graph)
    return _NativeAuthority(compute, authority.backend), session


# ── ambient epistemics (CONCEPT:AU-KG.temporal.ambient-connector-valid-time) ──
#
# W3.4: connector-ingested rows carry epistemic value BY DEFAULT — the source's
# own reported timestamp becomes the row's bitemporal ``valid_from`` (and a
# delete/tombstone's ``valid_to``) with no per-connector code change required.
# Flag-gated so a deployment can restore byte-identical legacy behavior (no
# ``valid_from``/``valid_to`` at all) globally or for one noisy/untrusted
# source, per the repo's configuration-discipline convention of a CSV
# per-source allowlist (mirrors ``source_sync._reconcile_allowed_empty_sources``).


def _ambient_epistemic_enabled(connector: str) -> bool:
    """Whether ambient bitemporal/provenance stamping applies to ``connector``.

    Default ON (``KG_AMBIENT_EPISTEMIC``); a source named in the CSV
    ``KG_AMBIENT_EPISTEMIC_DISABLED_SOURCES`` opts out even when the global
    default stays ON.
    """
    from ...core.config import setting

    # ``cast`` omitted: auto-inferred as ``to_boolean`` from the ``True``
    # default (see ``core._env.setting``'s docstring) — also keeps this call
    # compatible with a caller-test's simpler ``setting`` monkeypatch that
    # only accepts ``(key, default)``.
    if not bool(setting("KG_AMBIENT_EPISTEMIC", True)):
        return False
    disabled = setting("KG_AMBIENT_EPISTEMIC_DISABLED_SOURCES", "") or ""
    disabled_sources = {
        s.strip().lower() for s in str(disabled).split(",") if s.strip()
    }
    return (connector or "").strip().lower() not in disabled_sources


def _stamp_ambient_valid_time(row: dict[str, Any], envelope: ChangeEnvelope) -> None:
    """Map the source's own reported timestamp onto ``valid_from`` (in place).

    Prefers the envelope's explicit ``valid_time`` (the fact's own asserted
    domain validity, e.g. a backdated ticket) over ``event_time`` (the
    connector's per-record modification/version timestamp); never invents a
    value — a source with neither leaves ``row`` untouched, exactly the legacy
    shape (:func:`~..core.bitemporal.stamp_valid_from_source`'s contract).
    """
    if not _ambient_epistemic_enabled(envelope.connector):
        return
    from ..core.bitemporal import stamp_valid_from_source

    stamp_valid_from_source(row, valid_from=envelope.valid_time or envelope.event_time)


def _stamp_ambient_valid_until(row: dict[str, Any], envelope: ChangeEnvelope) -> None:
    """Close the bitemporal validity interval on a tombstone/archive write (in place).

    Uses the envelope's own ``event_time`` when the source reported a real
    supersession instant, else ``observed_time`` (always populated — this
    system's own observation instant is never a fabrication: it really did
    learn of the removal at that moment).
    """
    if not _ambient_epistemic_enabled(envelope.connector):
        return
    from ..core.bitemporal import stamp_valid_until_from_source

    stamp_valid_until_from_source(
        row, valid_until=envelope.event_time or envelope.observed_time
    )


def _verified_live_ids(
    envelope: ChangeEnvelope, current_rows: list[tuple[str, dict[str, Any]]]
) -> set[str]:
    """Live-id set for a snapshot reconcile — FAIL CLOSED on a degraded read.

    An EMPTY live-id set is honoured (i.e. allowed to tombstone everything the
    connector omitted) only when the connector both reported a successful fetch
    AND is approved to declare an authoritative empty source. Otherwise every
    currently-stored id is treated as still live, so the verified snapshot
    decision is committed WITHOUT tombstoning: a failed live-id fetch must
    never be read as "the source is genuinely empty".
    """
    live_ids = set(envelope.live_ids)
    if live_ids:
        return live_ids
    from ..core.source_sync import _reconcile_allowed_empty_sources

    fetch_ok = bool(envelope.provenance.get("fetch_ok", True))
    allowed = (
        envelope.provenance.get("authoritative_empty_approved") is True
        or envelope.connector.lower() in _reconcile_allowed_empty_sources()
    )
    if fetch_ok and allowed:
        return live_ids
    # Commit the verified snapshot decision without tombstoning.
    return {
        str(properties.get("externalToolId"))
        for _, properties in current_rows
        if properties.get("externalToolId")
    }


def _archived_snapshot_row(
    properties: dict[str, Any], envelope: ChangeEnvelope
) -> dict[str, Any]:
    """Archive one row a verified snapshot omitted."""
    updated = dict(properties)
    updated["archived"] = True
    updated["current"] = False
    updated["deprecated"] = True
    updated["lifecycle_state"] = "archived"
    # Retrieval's legacy default gate keys on ``status``.  Keep
    # the explicit lifecycle fields above while making a verified
    # snapshot omission impossible to rank as current.
    updated["status"] = "archived"
    updated["archivedReason"] = f"absent-from-{envelope.connector}"
    _stamp_ambient_valid_until(updated, envelope)
    return updated


def _snapshot_complete_rows(
    client: Any, envelope: ChangeEnvelope
) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """The snapshot marker row plus every row this snapshot archives."""
    fetch_ok = bool(envelope.provenance.get("fetch_ok", True))
    current_rows = _snapshot_rows(client, envelope.connector, envelope.source_instance)
    live_ids = _verified_live_ids(envelope, current_rows)
    stale: list[tuple[str, dict[str, Any]]] = []
    for existing_id, properties in current_rows:
        external_id = str(properties.get("externalToolId") or "")
        if external_id and external_id not in live_ids:
            stale.append((existing_id, _archived_snapshot_row(properties, envelope)))
    marker_digest = _digest(
        {
            "connector": envelope.connector,
            "source_instance": envelope.source_instance,
        }
    )
    node_id = f"snapshot:{marker_digest}"
    marker = {
        "id": node_id,
        "node_type": "SourceSnapshot",
        "domain": envelope.connector,
        "source_system": envelope.connector,
        "source_instance": envelope.source_instance,
        "live_count": len(live_ids),
        "fetch_verified": fetch_ok,
        "content_digest": marker_digest,
    }
    return node_id, [(node_id, marker), *stale]


def _tombstone_row(
    client: Any, envelope: ChangeEnvelope, node_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Merge the delete tombstone onto the current row; return it and its evidence."""
    current = _node_properties(client, node_id)
    provenance_evidence = envelope.provenance.get("evidence")
    evidence = (
        [item for item in provenance_evidence if isinstance(item, dict)]
        if isinstance(provenance_evidence, list)
        else []
    )
    current.update(
        {
            "id": node_id,
            "archived": True,
            "current": False,
            "deprecated": True,
            "lifecycle_state": "tombstoned",
            # Retrieval's default query excludes archived rows; callers
            # must opt into temporal/history state to see this tombstone.
            "status": "archived",
            "archivedReason": f"tombstoned-by-{envelope.connector}",
        }
    )
    _stamp_ambient_valid_until(current, envelope)
    return current, evidence


def _blob_backed_row(envelope: ChangeEnvelope, node_id: str) -> dict[str, Any]:
    """Default row for an upsert whose material is a blob, not a typed payload."""
    row: dict[str, Any] = {
        "id": node_id,
        "node_type": envelope.payload_type or "Artifact",
        "blob_ref": envelope.blob_ref,
        "classification": envelope.classification.value,
        "tenant": envelope.tenant,
        "source_instance": envelope.source_instance,
        "retention": envelope.retention,
        "legal_hold": envelope.legal_hold,
    }
    if envelope.blob_digest is not None:
        row.update(
            {
                "blob_digest": envelope.blob_digest,
                "blob_length": envelope.blob_length,
                "blob_media_type": envelope.blob_media_type,
            }
        )
    return row


def _pop_sidecar_list(row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Pop a ``_links``/``_features``/``_evidence`` sidecar, keeping dict entries."""
    value = row.pop(key, None)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _auxiliary_node_rows(
    client: Any, envelope: ChangeEnvelope, auxiliary_value: Any, seen_ids: set[str]
) -> list[tuple[str, dict[str, Any]]]:
    """Hydrate and stamp the envelope's ``_nodes`` auxiliary rows."""
    from ..enrichment.provenance import stamp_source

    rows: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(auxiliary_value, list):
        return rows
    for raw_auxiliary in auxiliary_value:
        if not isinstance(raw_auxiliary, dict):
            continue
        auxiliary = dict(raw_auxiliary)
        auxiliary_id = str(auxiliary.pop("id", "") or "")
        if not auxiliary_id or auxiliary_id in seen_ids:
            raise ValueError(
                "ChangeEnvelope auxiliary node ids must be unique and non-empty"
            )
        seen_ids.add(auxiliary_id)
        stamp_source(auxiliary, envelope.connector)
        # Deliberately NOT ``_stamp_ambient_valid_time`` here: an auxiliary
        # node is a distinct entity (e.g. "this image's repo") whose own
        # validity start is unknown to this envelope — the primary row's
        # event_time is not evidence about when the auxiliary became true,
        # so stamping it would be a fabrication, not an inference.
        existing = _node_properties(client, auxiliary_id)
        existing.update(auxiliary)
        existing["id"] = auxiliary_id
        rows.append((auxiliary_id, existing))
    return rows


def _upsert_node_rows(
    client: Any, envelope: ChangeEnvelope, node_id: str, row: dict[str, Any] | None
) -> tuple[
    list[tuple[str, dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Merged primary/auxiliary rows plus the envelope's sidecar material."""
    from ..enrichment.provenance import stamp_source

    row = _blob_backed_row(envelope, node_id) if row is None else dict(row)
    links = _pop_sidecar_list(row, "_links")
    features = _pop_sidecar_list(row, "_features")
    evidence = _pop_sidecar_list(row, "_evidence")
    auxiliary_value = row.pop("_nodes", None)

    stamp_source(row, envelope.connector)
    _stamp_ambient_valid_time(row, envelope)
    current = _node_properties(client, node_id)
    current.update(row)
    current["id"] = node_id
    node_rows = [(node_id, current)]
    node_rows.extend(_auxiliary_node_rows(client, envelope, auxiliary_value, {node_id}))
    _project_relations_into(client, node_rows, links)
    return node_rows, links, features, evidence


def _prepare_node_rows(
    client: Any, envelope: ChangeEnvelope
) -> tuple[
    str,
    list[tuple[str, dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Resolve merge/tombstone/snapshot rows from one OCC-fenced read."""
    node_id, row = _resolve_identity(envelope)
    node_id = str(node_id or "")

    if envelope.operation == "snapshot_complete":
        marker_id, node_rows = _snapshot_complete_rows(client, envelope)
        return marker_id, node_rows, [], [], []

    if envelope.operation == "delete":
        tombstone, evidence = _tombstone_row(client, envelope, node_id)
        return node_id, [(node_id, tombstone)], [], [], evidence

    node_rows, links, features, evidence = _upsert_node_rows(
        client, envelope, node_id, row
    )
    return node_id, node_rows, links, features, evidence


def _collect_projected_relations(
    node_rows: list[tuple[str, dict[str, Any]]],
) -> tuple[
    list[tuple[str, dict[str, Any]]],
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
    set[str],
]:
    """Run the shared relation projection over ``node_rows``.

    Returns ``(derived_nodes, edges, candidate_edges, known_ids)``. A candidate
    edge is one whose target this projection did not itself write, so it still
    needs an existence probe before it may be committed.
    """
    from ..enrichment.relation_projection import project_relations, projects_anything

    derived: list[tuple[str, dict[str, Any]]] = []
    edges: list[tuple[str, str, str]] = []
    candidates: list[tuple[str, str, str]] = []
    known = {identifier for identifier, _ in node_rows}
    for node_id, properties in list(node_rows):
        if not projects_anything(properties.get("node_type")):
            continue
        projected = project_relations(node_id, properties)
        for identifier, row in projected.nodes:
            if identifier in known:
                continue
            known.add(identifier)
            derived.append((identifier, row))
        edges.extend(projected.edges)
        candidates.extend(projected.candidate_edges)
    return derived, edges, candidates, known


def _resolvable_candidate_edges(
    client: Any, candidates: list[tuple[str, str, str]], known: set[str]
) -> list[tuple[str, str, str]]:
    """Keep only candidate edges whose target is known-present.

    Fail closed: when the batched existence probe itself fails, ``present``
    stays EMPTY and every unverified candidate is DROPPED, never committed —
    the engine refuses an envelope whose edge endpoints do not exist, so a
    failed probe must not be read as "the target is there".
    """
    targets = sorted({target for _, target, _ in candidates} - known)
    present: dict[str, bool] = {}
    try:
        present = dict(client.nodes.has_batch(targets)) if targets else {}
    except Exception as exc:  # noqa: BLE001 — an unverifiable reference is dropped, never a failed envelope
        logger.debug("relation projection: endpoint check failed: %s", exc)
    return [edge for edge in candidates if edge[1] in known or present.get(edge[1])]


def _project_relations_into(
    client: Any,
    node_rows: list[tuple[str, dict[str, Any]]],
    links: list[dict[str, Any]],
) -> None:
    """Materialise the edges these rows' own properties encode, in place.

    CONCEPT:AU-KG.enrichment.relation-projection — materialise the edges a node's
    own properties already encode. This is the connector leg of the SAME
    projection ``EpistemicGraphBackend.add_node`` applies, sharing one
    :func:`~...enrichment.relation_projection.project_relations` implementation
    rather than a second copy of the convention. Injected here (not in
    :func:`_graph_operations`) so derived rows are SHACL-validated and folded
    into the envelope's content digest exactly like the rows that caused them.

    Free on the wire: the derived nodes/edges travel inside the SAME native
    ``ChangeEnvelope`` mutation, and a node type with no projection rule costs
    one dict lookup. A reference property whose targets this projection did not
    write needs one batched existence probe, because the engine refuses an
    envelope whose edge endpoints do not exist.
    """
    derived, edges, candidates, known = _collect_projected_relations(node_rows)
    if candidates:
        edges.extend(_resolvable_candidate_edges(client, candidates, known))

    node_rows.extend(derived)
    links.extend(
        {
            "source": source,
            "target": target,
            "relationship": relationship,
            "projected_from": "node_property",
        }
        for source, target, relationship in edges
    )


def _graph_operations(
    node_rows: list[tuple[str, dict[str, Any]]],
    links: list[dict[str, Any]],
    default_source: str,
) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for node_id, properties in node_rows:
        operations.append(
            {
                "ordinal": len(operations),
                "surface": "graph",
                "domain": "graph_rows",
                "method": {
                    "method": "AddNode",
                    "params": {
                        "node_id": node_id,
                        "properties_msgpack": _pack(properties),
                    },
                },
            }
        )
    for link in links:
        source = str(link.get("source") or default_source)
        target = str(link.get("target") or "")
        if not target:
            raise ValueError("ChangeEnvelope relationship has no target")
        properties = {
            key: value for key, value in link.items() if key not in {"source", "target"}
        }
        operations.append(
            {
                "ordinal": len(operations),
                "surface": "graph",
                "domain": "graph_rows",
                "method": {
                    "method": "AddEdge",
                    "params": {
                        "source_id": source,
                        "target_id": target,
                        "properties_msgpack": _pack(properties),
                    },
                },
            }
        )
    return operations


def _native_feature_rows(
    raw_features: list[dict[str, Any]], node_id: str, operation: str
) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    governed: set[str] = set()
    for index, item in enumerate(raw_features):
        object_id = str(item.get("object_id") or node_id)
        governed.add(object_id)
        value = item.get("value", item.get("value_msgpack", {}))
        rows.append(
            {
                "feature_id": str(
                    item.get("feature_id")
                    or f"feature:{_digest([node_id, index, item])}"
                ),
                "operation": operation,
                "object_id": object_id,
                "kind": str(item.get("kind") or "derived"),
                "value_msgpack": _pack(value),
                "model_version": str(item.get("model_version") or "unversioned"),
            }
        )
    return rows, governed


def _native_evidence_rows(
    raw_evidence: list[dict[str, Any]],
    node_id: str,
    operation: str,
    content_digest: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    governed: set[str] = set()
    for index, item in enumerate(raw_evidence):
        object_id = str(item.get("object_id") or node_id)
        governed.add(object_id)
        locus = item.get("locus", item.get("locus_msgpack", {}))
        rows.append(
            {
                "evidence_id": str(
                    item.get("evidence_id")
                    or f"evidence:{_digest([node_id, index, item])}"
                ),
                "operation": operation,
                "object_id": object_id,
                "modality": str(item.get("modality") or "structured"),
                "locus_msgpack": _pack(locus),
                "content_digest": str(item.get("content_digest") or content_digest),
            }
        )
    return rows, governed


def _native_content_state(
    client: Any, envelope: ChangeEnvelope, node_id: str, material: dict[str, Any]
) -> tuple[str, str, dict[str, Any], str | None]:
    """``(material_digest, content_digest, source_position, previous_digest)``."""
    material_digest = _digest(material)
    current_version = client.changes.content_version(node_id)
    source_position = _content_position(
        envelope.source_version or envelope.checkpoint,
        current_version,
        material_digest,
    )
    content_digest = _digest(
        {
            "material": material_digest,
            "source_position": source_position,
            "operation": envelope.operation,
        }
    )
    previous_digest = (
        str(current_version.get("digest"))
        if isinstance(current_version, dict) and current_version.get("digest")
        else None
    )
    return material_digest, content_digest, source_position, previous_digest


def _native_cursor(
    client: Any, envelope: ChangeEnvelope, chained_cursor_position: Any
) -> tuple[dict[str, Any] | None, bool]:
    """The source-cursor advance this envelope commits, if any.

    ``(None, False)`` means "do NOT advance the watermark": an envelope with no
    checkpoint, or one whose checkpoint does not strictly advance the current
    position, must never move the cursor forward.
    """
    if not envelope.checkpoint:
        return None, False
    partition = _cursor_partition(envelope.source_instance)
    next_position = _typed_position(envelope.checkpoint, content=False)
    if chained_cursor_position is _CURSOR_READ_LIVE:
        current_cursor = client.changes.cursor(envelope.connector, partition)
        current_position = (
            current_cursor.get("position")
            if isinstance(current_cursor, dict)
            and isinstance(current_cursor.get("position"), dict)
            else None
        )
    else:
        current_position = chained_cursor_position
    if current_position is not None and not _position_advances(
        next_position, current_position
    ):
        return None, False
    cursor: dict[str, Any] = {
        "source": envelope.connector,
        "partition": partition,
        "position": next_position,
    }
    if current_position is not None:
        cursor["expected_previous"] = current_position
    return cursor, True


def _structured_evidence_row(node_id: str, structured_evidence: Any) -> dict[str, Any]:
    """The evidence row an envelope's inline ``structured_evidence`` contributes."""
    return {
        "evidence_id": f"evidence:{_digest([node_id, structured_evidence])}",
        "object_id": node_id,
        "modality": "structured",
        "locus": structured_evidence,
        "content_digest": _digest(structured_evidence),
    }


def _native_policies(
    session: Any, envelope: ChangeEnvelope, governed_ids: set[str]
) -> list[dict[str, Any]]:
    """One durable policy row per governed object, fail-closed on a missing ACL.

    An envelope with no ``source_acl`` yields a ``{"deny_all": True}`` subject
    set — never an empty/permissive one.
    """
    access = envelope.source_acl
    access_material = access.model_dump() if access is not None else {"deny_all": True}
    subject_set_digest = _digest(access_material)
    tenant = str(session.tenant)
    policy_version = str(session.policy_version or "unversioned")
    return [
        {
            "policy_id": (
                f"policy:{_digest([tenant, object_id, policy_version, subject_set_digest])}"
            ),
            "operation": "upsert",
            "object_id": object_id,
            "tenant": tenant,
            "classification": envelope.classification.value,
            "policy_version": policy_version,
            "subject_set_digest": subject_set_digest,
            "retention_policy": str(envelope.retention or "unspecified"),
            "legal_hold": bool(envelope.legal_hold),
        }
        for object_id in sorted(governed_ids)
    ]


def _native_blob_rows(
    envelope: ChangeEnvelope, node_id: str, operation_name: str
) -> list[dict[str, Any]]:
    """The blob row this envelope commits, or ``[]`` when it carries no blob."""
    if not envelope.blob_ref:
        return []
    return [
        {
            "blob_id": node_id,
            "operation": operation_name,
            "digest_algorithm": "sha256",
            "digest": (
                envelope.blob_digest.removeprefix("sha256:")
                if envelope.blob_digest
                else hashlib.sha256(envelope.blob_ref.encode("utf-8")).hexdigest()
            ),
            "media_type": str(
                envelope.blob_media_type
                or envelope.payload_type
                or "application/octet-stream"
            ),
            "length": int(envelope.blob_length or 0),
        }
    ]


def _native_placement(authority: _NativeAuthority, session: Any) -> tuple[int, Any]:
    """``(placement_epoch, placement_group)`` — verified session first, compute next."""
    placement_epoch = int(
        session.catalog_epoch
        if session.catalog_epoch is not None
        else (getattr(authority.compute, "catalog_epoch", 0) or 0)
    )
    placement_group = (
        session.placement_group
        if session.placement_group is not None
        else getattr(authority.compute, "placement_group", None)
    )
    return placement_epoch, placement_group


def _native_mutation(
    session: Any,
    envelope: ChangeEnvelope,
    operations: list[dict[str, Any]],
    content_digest: str,
    *,
    expected_graph_version: int,
    created_at_ms: int,
    placement: tuple[int, Any],
) -> dict[str, Any]:
    """The engine's durable mutation DTO for this envelope."""
    placement_epoch, placement_group = placement
    policy_version = str(session.policy_version or "unversioned")
    mutation: dict[str, Any] = {
        # Epistemic Graph's current-only durable mutation contract is v2.
        # The enclosing ChangeEnvelope remains v1; these are distinct wire
        # schemas and neither side accepts the retired mutation v1 shape.
        "schema_version": 2,
        "batch_id": f"batch:{envelope.idempotency_key}",
        "context": {
            "request_id": 0,
            "principal": "",
            "purpose": "external_change_ingestion",
            "policy_fingerprint": policy_version,
            "trace_id": str(session.trace_context or envelope.trace_context or ""),
        },
        "tenant": str(session.tenant),
        "graph": str(session.graph),
        "placement_epoch": placement_epoch,
        "idempotency_key": envelope.idempotency_key,
        "expected_graph_version": int(expected_graph_version),
        "operations": operations,
        "outbox": [
            {
                "topic": "kg.mutations",
                "key": _native_envelope_id(envelope),
                "payload": _pack(
                    {
                        "schema": "agent-utilities.change-committed.v1",
                        "content_digest": content_digest,
                        "operation": envelope.operation,
                    }
                ),
                "headers": {"schema": "agent-utilities.change-committed.v1"},
            }
        ],
        "created_at_ms": created_at_ms,
    }
    if placement_group is not None:
        mutation["fencing_token"] = int(placement_group)
    return mutation


def _native_counts(
    envelope: ChangeEnvelope,
    node_rows: list[tuple[str, dict[str, Any]]],
    links: list[dict[str, Any]],
    features: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    blobs: list[dict[str, Any]],
) -> dict[str, int]:
    """Per-envelope write counts reported back to the caller."""
    return {
        "nodes": len(node_rows),
        "edges": len(links),
        "features": len(features),
        "evidence": len(evidence),
        "blobs": len(blobs),
        "tombstoned": (
            max(0, len(node_rows) - 1)
            if envelope.operation == "snapshot_complete"
            else int(envelope.operation == "delete")
        ),
    }


def _native_material(
    authority: _NativeAuthority,
    session: Any,
    envelope: ChangeEnvelope,
    *,
    expected_graph_version: int,
    created_at_ms: int,
    chained_cursor_position: Any = _CURSOR_READ_LIVE,
) -> tuple[dict[str, Any], dict[str, int], list[str], bool]:
    """Render the complete engine-native wire DTO from one OCC-fenced read.

    ``chained_cursor_position`` is :data:`_CURSOR_READ_LIVE` for the single-envelope
    path (reads the live source cursor here). The batch path passes the already-read,
    client-chained current position (or ``None`` for no prior cursor) so it is not
    re-read per envelope and the intra-batch cursor chain stays correct.
    """
    client = authority.compute.client
    node_id, node_rows, links, raw_features, raw_evidence = _prepare_node_rows(
        client, envelope
    )
    # The shared-graph read boundary scopes on the canonical ``tenant_id``
    # property. Bind every primary and auxiliary row to the already-verified
    # GraphSession tenant, overriding any untrusted source value.
    for _current_id, properties in node_rows:
        properties["tenant_id"] = str(session.tenant)
    _shacl_validate_rows(client, node_rows)
    operations = _graph_operations(node_rows, links, node_id)
    material_digest, content_digest, source_position, previous_digest = (
        _native_content_state(
            client,
            envelope,
            node_id,
            {
                "operation": envelope.operation,
                "nodes": node_rows,
                "links": links,
                "blob_ref": envelope.blob_ref,
                "blob_digest": envelope.blob_digest,
                "blob_length": envelope.blob_length,
                "blob_media_type": envelope.blob_media_type,
                "features": raw_features,
                "evidence": raw_evidence,
                "structured_evidence": envelope.structured_evidence,
            },
        )
    )
    cursor, cursor_advanced = _native_cursor(client, envelope, chained_cursor_position)

    operation_name = "delete" if envelope.operation == "delete" else "upsert"
    features, feature_objects = _native_feature_rows(
        raw_features, node_id, operation_name
    )
    if envelope.structured_evidence is not None:
        raw_evidence = [
            *raw_evidence,
            _structured_evidence_row(node_id, envelope.structured_evidence),
        ]
    evidence, evidence_objects = _native_evidence_rows(
        raw_evidence, node_id, operation_name, content_digest
    )
    governed_ids = {current_id for current_id, _ in node_rows}
    governed_ids.update(feature_objects)
    governed_ids.update(evidence_objects)

    blobs = _native_blob_rows(envelope, node_id, operation_name)
    native: dict[str, Any] = {
        "schema_version": 1,
        "envelope_id": _native_envelope_id(envelope),
        "mutation": _native_mutation(
            session,
            envelope,
            operations,
            content_digest,
            expected_graph_version=expected_graph_version,
            created_at_ms=created_at_ms,
            placement=_native_placement(authority, session),
        ),
        "content_version": {
            "object_id": node_id,
            "digest_algorithm": "sha256",
            "digest": content_digest,
            "source_version": source_position,
        },
        "blobs": blobs,
        "features": features,
        "evidence": evidence,
        "policies": _native_policies(session, envelope, governed_ids),
        "lineage": [
            {
                "lineage_id": f"lineage:{envelope.idempotency_key}",
                "operation": operation_name,
                "object_id": node_id,
                "source_artifact_digest": material_digest,
                "transform_name": envelope.connector,
                "transform_version": str(
                    envelope.ontology_mapping_version or envelope.schema_version or "1"
                ),
                "parent_content_digests": [],
            }
        ],
        "privacy": {
            "policy_version": "persistence-privacy-v1",
            "sanitizer_version": "agent-utilities-v1",
            "sanitized_payload_digest": material_digest,
        },
    }
    if previous_digest:
        native["content_version"]["previous_digest"] = previous_digest
    if cursor is not None:
        native["cursor"] = cursor
    counts = _native_counts(envelope, node_rows, links, features, evidence, blobs)
    return native, counts, sorted(governed_ids), cursor_advanced


def _unpacked_properties(packed: Any) -> dict[str, Any] | None:
    """Unpack a wire ``properties_msgpack`` blob, or ``None`` when it is not a dict."""
    import msgpack

    properties = msgpack.unpackb(packed, raw=False)
    return properties if isinstance(properties, dict) else None


def _mirror_node_operation(params: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild one ``upsert_node`` mirror op, or ``None`` if it cannot be replayed."""
    node_id = str(params.get("node_id") or "").strip()
    packed = params.get("properties_msgpack")
    if not node_id or not packed:
        return None
    properties = _unpacked_properties(packed)
    if properties is None:
        return None
    # ``ChangeEnvelope.to_entity_dict()``/auxiliary ``_nodes`` rows carry
    # the connector's own ``type`` key verbatim (never renamed) — only
    # ``ingest_graph_slice``'s stricter external contract commits the
    # canonical ``node_type`` key. Accept either, matching the AddEdge
    # ``type``/``relationship`` normalization below: the mirror op
    # must resolve the SAME label the authority just committed.
    node_type = str(properties.get("node_type") or properties.get("type") or "").strip()
    if not node_type:
        # An untyped node cannot be replayed through the typed mirror
        # seam; reconcile() is the backstop for it.
        return None
    return {
        "op": "upsert_node",
        "id": node_id,
        "properties": {**properties, "node_type": node_type},
    }


def _mirror_edge_operation(params: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild one ``upsert_edge`` mirror op, or ``None`` if it cannot be replayed."""
    source_id = str(params.get("source_id") or "").strip()
    target_id = str(params.get("target_id") or "").strip()
    packed = params.get("properties_msgpack")
    if not source_id or not target_id or not packed:
        return None
    properties = _unpacked_properties(packed)
    if properties is None:
        return None
    # A ChangeEnvelope ``_links`` item's edge-type key is caller-shaped:
    # ``ingest_graph_slice`` requires the canonical ``relationship`` key,
    # but a raw ``_links`` entry (as most connectors emit it) commits
    # ``type`` straight onto the wire the same way ``AddNode`` commits
    # ``node_type`` — neither is renamed before packing. Accept either so
    # the mirror op always resolves the SAME edge label the authority
    # just committed, never a second independently-guessed one.
    relationship = str(
        properties.get("relationship") or properties.get("type") or ""
    ).strip()
    if not relationship:
        return None
    return {
        "op": "upsert_edge",
        "source": source_id,
        "target": target_id,
        "properties": {**properties, "relationship": relationship},
    }


#: Committed wire method -> mirror-op builder. A method with no builder is not
#: replayable through the typed mirror seam and is skipped, exactly as the
#: previous ``if/elif`` chain's absent ``else`` did.
_MIRROR_REPLAY_BUILDERS: dict[
    str, Callable[[dict[str, Any]], dict[str, Any] | None]
] = {
    "AddNode": _mirror_node_operation,
    "AddEdge": _mirror_edge_operation,
}


def _mirror_replay_operations(native: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild fan-out typed-batch mirror ops from an already-committed native DTO.

    D-BFR-12: a native ``ApplyChangeEnvelope(s)`` commit calls the unwrapped
    engine client directly (see ``_resolve_native_authority``), so the graph
    slice it commits never passes through ``FanOutBackend.execute``/
    ``apply_typed_batch`` and therefore never reaches the mirror outbox on its
    own. ``native["mutation"]["operations"]`` is the exact ``AddNode``/
    ``AddEdge`` wire list the engine just committed (built by
    ``_graph_operations``); reusing it here — rather than re-deriving node/edge
    state — guarantees the mirror replay is the SAME material the authority
    accepted, not a second independently-computed mutation.
    """
    operations = ((native.get("mutation") or {}).get("operations")) or []
    replay: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        method = operation.get("method") or {}
        builder = _MIRROR_REPLAY_BUILDERS.get(str(method.get("method") or ""))
        if builder is None:
            continue
        mirrored = builder(method.get("params") or {})
        if mirrored is not None:
            replay.append(mirrored)
    return replay


def _replay_native_graph_mutation(
    authority: _NativeAuthority, native: dict[str, Any]
) -> None:
    """Enqueue mirror replay for one just-committed native ChangeEnvelope.

    Best-effort by the SAME contract ``FanOutBackend.apply_typed_batch`` already
    uses for an authority-committed mutation: the authoritative graph is the
    durable acknowledgement source, so a mirror handoff failure is logged loudly
    (reconcile() is the backstop) rather than raised back through the ingest
    caller and made to look like the source write itself failed.
    """
    publisher = getattr(authority, "backend", None)
    replay = getattr(publisher, "replay_committed_change", None)
    if not callable(replay):
        return
    operations = _mirror_replay_operations(native)
    if not operations:
        return
    from ..backends.fanout_backend import AuthorityCommittedMirrorHandoffError

    try:
        replay(operations)
    except AuthorityCommittedMirrorHandoffError as exc:
        logger.critical(
            "native ChangeEnvelope committed but mirror replay failed; "
            "reconciliation required: %s",
            exc,
        )


def _filtered_receipt(receipt: Any, envelope: ChangeEnvelope) -> dict[str, Any]:
    source = receipt if isinstance(receipt, dict) else {}
    allowed = {
        key: source[key]
        for key in (
            "batch_id",
            "replayed",
            "projection_pending",
            "outbox_count",
            "replicated",
            "group",
            "epoch",
            "fencing_token",
        )
        if key in source
    }
    allowed.setdefault("envelope_id", envelope.envelope_id)
    return allowed


def _recover_native_receipt(
    client: Any, envelope: ChangeEnvelope
) -> dict[str, Any] | None:
    """Reconcile an ambiguous transport failure without replaying a mutation."""
    try:
        record = client.changes.get(_native_envelope_id(envelope))
    except Exception:  # noqa: BLE001 - ambiguity must remain failed closed
        return None
    if not isinstance(record, dict):
        return None
    return {
        "envelope_id": _native_envelope_id(envelope),
        "batch_id": f"batch:{envelope.idempotency_key}",
        "replayed": True,
        "projection_pending": False,
    }


def _policy_cache_object_ids(client: Any, envelope: ChangeEnvelope) -> list[str]:
    """Reconstruct the governed object set for idempotent cache projection.

    The durable ChangeEnvelope is authoritative. A process restart can lose the
    in-memory ACL projection before an identical delivery is retried, so replay
    must rebuild the same object set without reapplying the mutation.
    """

    node_id, node_rows, _links, features, evidence = _prepare_node_rows(
        client, envelope
    )
    governed = {current_id for current_id, _properties in node_rows if current_id}
    for item in (*features, *evidence):
        object_id = str(item.get("object_id") or node_id or "").strip()
        if object_id:
            governed.add(object_id)
    return sorted(governed)


#: Engine error markers meaning "this deployment cannot serve the native
#: single-envelope change contract at all" — a capability gap, never a
#: retryable conflict, so they fail closed instead of being retried.
_NATIVE_UNAVAILABLE_MARKERS = (
    "requires the authoritative redb backend",
    "requires a configured persistence backend",
    "Unknown method",
    "unknown method",
)

#: The same class of capability gap for the batch contract, plus the
#: placement-scoped refusal that only the batch method can hit.
_NATIVE_BATCH_UNAVAILABLE_MARKERS = (
    *_NATIVE_UNAVAILABLE_MARKERS,
    "CHANGE_BATCH_UNAVAILABLE_UNDER_PLACEMENT",
)


# NB: the two guards below deliberately spell their capability literally rather
# than share one parameterised helper. ``scripts/check_native_change_envelope_
# boundary.py`` greps this module for the exact ``supports("ApplyChangeEnvelope")``
# source marker, so a parameterised guard would silently disarm that fail-closed
# architectural gate while still behaving correctly at runtime.
def _require_apply_change_envelope(client: Any) -> None:
    """Fail closed unless the engine EXPLICITLY advertises ApplyChangeEnvelope.

    A client with no ``supports`` method is treated as not advertising it — an
    unknown engine is never assumed to be capable.
    """
    supports = getattr(client, "supports", None)
    if not callable(supports) or not bool(supports("ApplyChangeEnvelope")):
        raise NativeChangeEnvelopeUnavailable(
            "engine does not advertise ApplyChangeEnvelope"
        )


def _require_apply_change_envelopes(client: Any) -> None:
    """Fail closed unless the engine EXPLICITLY advertises ApplyChangeEnvelopes."""
    supports = getattr(client, "supports", None)
    if not callable(supports) or not bool(supports("ApplyChangeEnvelopes")):
        raise NativeChangeEnvelopeUnavailable(
            "engine does not advertise ApplyChangeEnvelopes"
        )


def _native_occ_next_expected(
    exc: BaseException,
    scope: tuple[str, str],
    expected: int,
    conflict_sequence: list[str],
) -> int | None:
    """Classify a failed apply as a GUARANTEED pre-commit conflict.

    Returns the ``expected_graph_version`` a retry must use, or ``None`` when
    the failure is not retryable and the caller must re-raise the original
    exception. Raises :class:`NativeChangeEnvelopeUnavailable` for a capability
    gap. Nothing else is retried: this is the authoritative commit path, so an
    ambiguous failure must never be replayed into a duplicate write.
    """
    message = str(exc)
    conflict = _NATIVE_OCC_CONFLICT_RE.search(message)
    stale = _STALE_GRAPH_VERSION_RE.search(message)
    if conflict is not None and stale is not None:
        conflict_sequence.append(conflict.group(0).upper())
        _NATIVE_GRAPH_VERSIONS[scope] = int(stale.group(1))
        return int(stale.group(1))
    if conflict is not None and conflict.group(0).upper() in {
        "STALE_CONTENT_VERSION",
        "STALE_CURSOR",
    }:
        conflict_sequence.append(conflict.group(0).upper())
        # Both are guaranteed pre-commit; re-read authoritative
        # material and let graph OCC fence the rebuilt request.
        return expected
    if any(marker in message for marker in _NATIVE_UNAVAILABLE_MARKERS):
        raise NativeChangeEnvelopeUnavailable(
            "engine lacks authoritative ChangeEnvelope persistence"
        ) from exc
    return None


class _NativeOccAttempt:
    """Bounded OCC state for ONE single-envelope native commit.

    Owns the retry budget, the tracked ``expected_graph_version`` and the
    material of whichever attempt the engine accepted. It deliberately performs
    NO durable write itself: the caller issues the single authoritative
    ``client.changes.apply(...)`` and hands the outcome back here, so the
    durable write stays at the exact boundary the native ChangeEnvelope gate
    (``scripts/check_native_change_envelope_boundary.py``) inspects.

    Construct it inside the scope lock — it reads the shared tracked graph
    version on construction.
    """

    def __init__(self, scope: tuple[str, str]) -> None:
        self.scope = scope
        self.expected = _NATIVE_GRAPH_VERSIONS.get(scope, 0)
        self.attempt = -1
        self.conflicts: list[str] = []
        self.receipt: Any = None
        self.native: dict[str, Any] | None = None
        self.counts: dict[str, int] = {}
        self.governed_ids: list[str] = []
        self.cursor_advanced = False

    def recovered(self, receipt: Any, governed_ids: list[str]) -> None:
        """Adopt a receipt reconciled from an ambiguous earlier transport failure.

        ``native`` stays ``None`` so the caller does NOT re-enqueue a mirror
        replay for material the committing attempt already published.
        """
        self.receipt = receipt
        self.governed_ids = governed_ids

    def pending(self) -> bool:
        """Is another apply attempt needed — and still within the retry budget?"""
        if self.receipt is not None:
            return False
        self.attempt += 1
        if self.attempt >= _NATIVE_OCC_MAX_ATTEMPTS:
            raise _NativeOccRetryBudgetExhausted(self.conflicts)
        return True

    def render(
        self,
        authority: _NativeAuthority,
        session: Any,
        envelope: ChangeEnvelope,
        created_at_ms: int,
    ) -> dict[str, Any]:
        """Re-read authoritative material against the currently-expected version."""
        self.native, self.counts, self.governed_ids, self.cursor_advanced = (
            _native_material(
                authority,
                session,
                envelope,
                expected_graph_version=self.expected,
                created_at_ms=created_at_ms,
            )
        )
        return self.native

    def _advance_graph_version(self) -> None:
        _NATIVE_GRAPH_VERSIONS[self.scope] = max(
            _NATIVE_GRAPH_VERSIONS.get(self.scope, 0), self.expected + 1
        )

    def applied(self, receipt: Any) -> None:
        """Record an accepted apply."""
        self.receipt = receipt
        self._advance_graph_version()

    def failed(self, client: Any, envelope: ChangeEnvelope, exc: BaseException) -> None:
        """Classify a failed apply; returns only when a retry may proceed.

        An ambiguous transport failure is reconciled against the durable record
        rather than replayed. Anything that is not a GUARANTEED pre-commit
        conflict is re-raised: this is the authoritative commit path, so an
        ambiguous failure must never be retried into a duplicate write.
        """
        recovered = _recover_native_receipt(client, envelope)
        if recovered is not None:
            self.receipt = recovered
            self._advance_graph_version()
            return
        retry_expected = _native_occ_next_expected(
            exc, self.scope, self.expected, self.conflicts
        )
        if retry_expected is None:
            raise exc
        self.expected = retry_expected
        if self.attempt + 1 < _NATIVE_OCC_MAX_ATTEMPTS:
            _native_occ_backoff(self.attempt)


def _native_commit_result(
    authority: _NativeAuthority, envelope: ChangeEnvelope, attempt: _NativeOccAttempt
) -> dict[str, Any]:
    """Mirror, refresh the policy cache, and shape the caller-facing result."""
    receipt = attempt.receipt
    replayed = bool(receipt.get("replayed")) if isinstance(receipt, dict) else False
    # D-BFR-12: mirror the node/edge delta ONLY for a genuine first commit. A
    # recovered/idempotent replay means the mirror outbox already saw this
    # exact material on the attempt that actually committed it — re-enqueueing
    # here would be a no-op delta at best, a wasted duplicate outbox write at
    # worst (the "replay-skip" contract).
    if attempt.native is not None and not replayed:
        _replay_native_graph_mutation(authority, attempt.native)
    # This is a read-policy cache refresh, never the durability authority.
    # Both first apply and idempotent replay project it; replay is the bounded
    # recovery path after process-local cache loss.
    _sync_policy_cache(attempt.governed_ids, envelope)
    return {
        "status": "skipped" if replayed else "success",
        "reason": (
            "idempotent replay — envelope already applied" if replayed else None
        ),
        "native_atomic": True,
        "envelope_id": envelope.envelope_id,
        "idempotency_key": envelope.idempotency_key,
        "connector": envelope.connector,
        "operation": envelope.operation,
        "node_id": _resolve_identity(envelope)[0],
        "write_result": {
            **attempt.counts,
            "receipt": _filtered_receipt(receipt, envelope),
        },
        "watermark_advanced": bool(attempt.cursor_advanced and not replayed),
        "checkpoint": envelope.checkpoint,
    }


def _sync_policy_cache(governed_ids: list[str], envelope: ChangeEnvelope) -> None:
    """Refresh the read-policy cache; NEVER the durability authority.

    Durable policy stays fail-closed regardless, so the first failure logs once
    and stops rather than raising back through the ingest caller.
    """
    from ...protocols.source_connectors.permission_sync import sync_access

    for object_id in governed_ids:
        try:
            sync_access(
                object_id,
                envelope.source_acl,
                classification=envelope.classification,
            )
        except Exception:  # noqa: BLE001 - durable policy remains fail-closed
            logger.warning("native policy cache refresh is pending")
            break


def _apply_native_change_envelope(
    authority: _NativeAuthority, session: Any, envelope: ChangeEnvelope
) -> dict[str, Any]:
    """Commit ONE envelope through the engine's native ``ApplyChangeEnvelope``.

    The bounded OCC bookkeeping lives in :class:`_NativeOccAttempt`; the single
    authoritative durable write stays here, so there is exactly one
    ``changes.apply`` call on this path and no sequential Python fallback.
    """
    from ..core.session import use_session

    client = authority.compute.client
    scope = (str(session.tenant), str(session.graph))
    created_at_ms = _observed_at_ms(envelope.observed_time)
    with use_session(session), _native_lock(scope):
        _require_apply_change_envelope(client)
        attempt = _NativeOccAttempt(scope)
        recovered = _recover_native_receipt(client, envelope)
        if recovered is not None:
            attempt.recovered(recovered, _policy_cache_object_ids(client, envelope))
        while attempt.pending():
            native = attempt.render(authority, session, envelope, created_at_ms)
            try:
                attempt.applied(client.changes.apply(native))
            except Exception as exc:  # noqa: BLE001 - only explicit pre-commit retries
                attempt.failed(client, envelope, exc)
    return _native_commit_result(authority, envelope, attempt)


def _batch_entry_result(
    envelope: ChangeEnvelope,
    engine_result: dict[str, Any],
    counts: dict[str, int],
    node_id: str | None,
    cursor_advanced: bool,
) -> dict[str, Any]:
    """Map one engine per-envelope batch result onto the SAME result-dict shape the
    single :func:`ingest_envelope` produces, so callers treat both paths identically."""
    status = str(engine_result.get("status"))
    if status == "applied":
        replayed = False
        out_status = "success"
        reason: str | None = None
    elif status == "idempotent_skip":
        replayed = True
        out_status = "skipped"
        reason = "idempotent replay — envelope already applied"
    else:
        return {
            "status": "failed",
            "error": engine_result.get("error") or "change envelope conflict",
            "native_atomic": True,
            "envelope_id": envelope.envelope_id,
            "idempotency_key": envelope.idempotency_key,
            "connector": envelope.connector,
            "operation": envelope.operation,
            "node_id": node_id,
            "watermark_advanced": False,
            "checkpoint": envelope.checkpoint,
        }
    return {
        "status": out_status,
        "reason": reason,
        "native_atomic": True,
        "envelope_id": envelope.envelope_id,
        "idempotency_key": envelope.idempotency_key,
        "connector": envelope.connector,
        "operation": envelope.operation,
        "node_id": node_id,
        "write_result": {
            **counts,
            "receipt": _filtered_receipt(engine_result, envelope),
        },
        "watermark_advanced": bool(cursor_advanced and not replayed),
        "checkpoint": envelope.checkpoint,
    }


@dataclass
class _BatchMaterial:
    """Per-envelope native DTOs and bookkeeping for one page, in input order."""

    natives: list[dict[str, Any]] = field(default_factory=list)
    counts: list[dict[str, int]] = field(default_factory=list)
    node_ids: list[str | None] = field(default_factory=list)
    cursor_advanced: list[bool] = field(default_factory=list)
    governed: list[list[str]] = field(default_factory=list)


def _page_cursor_position(client: Any, envelope: ChangeEnvelope) -> Any:
    """Read the page's source cursor ONCE.

    It is then chained client-side across the sequentially-applied envelopes:
    a live per-record read would STALE the 2nd+ envelope inside the shared
    transaction.
    """
    partition = _cursor_partition(envelope.source_instance)
    page_cursor = client.changes.cursor(envelope.connector, partition)
    return (
        page_cursor.get("position")
        if isinstance(page_cursor, dict)
        and isinstance(page_cursor.get("position"), dict)
        else None
    )


def _batch_native_material(
    authority: _NativeAuthority,
    session: Any,
    envelopes: list[ChangeEnvelope],
    expected: int,
) -> _BatchMaterial:
    """Render every envelope in the page, chaining the source cursor client-side."""
    chained: Any = _page_cursor_position(authority.compute.client, envelopes[0])
    material = _BatchMaterial()
    for offset, envelope in enumerate(envelopes):
        native, counts, governed_ids, cursor_advanced = _native_material(
            authority,
            session,
            envelope,
            expected_graph_version=expected + offset,
            created_at_ms=_observed_at_ms(envelope.observed_time),
            chained_cursor_position=chained,
        )
        material.natives.append(native)
        material.counts.append(counts)
        material.node_ids.append(_resolve_identity(envelope)[0])
        material.cursor_advanced.append(cursor_advanced)
        material.governed.append(governed_ids)
        if cursor_advanced and envelope.checkpoint:
            chained = _typed_position(envelope.checkpoint, content=False)
    return material


def _root_batch_conflict(engine_results: list[dict[str, Any]]) -> str | None:
    """The ROOT failure (a non-aborted conflict), or ``None``.

    Sibling entries carry the atomic-batch abort note; only the root error can
    decide whether an OCC retry could make progress.
    """
    for result in engine_results:
        if str(result.get("status")) == "conflict":
            error = str(result.get("error") or "")
            if not error.startswith("ABORTED_ATOMIC_GRAPH_BATCH"):
                return error
    return None


def _retry_batch_conflict(
    root_error: str,
    scope: tuple[str, str],
    conflict_sequence: list[str],
    attempt: int,
) -> bool:
    """``True`` when an OCC retry may make progress on this whole-batch failure.

    Raises when the retry budget is exhausted, or when the error means the
    engine cannot serve the native batch contract at all. ``False`` means the
    error is not an OCC conflict and the caller reports the engine's per-entry
    results unchanged.
    """
    conflict = _NATIVE_OCC_CONFLICT_RE.search(root_error)
    stale = _STALE_GRAPH_VERSION_RE.search(root_error)
    if conflict is not None:
        conflict_sequence.append(conflict.group(0).upper())
        if stale is not None:
            _NATIVE_GRAPH_VERSIONS[scope] = int(stale.group(1))
        if attempt + 1 < _NATIVE_OCC_MAX_ATTEMPTS:
            _native_occ_backoff(attempt)
            return True
        raise _NativeOccRetryBudgetExhausted(conflict_sequence)
    if any(marker in root_error for marker in _NATIVE_BATCH_UNAVAILABLE_MARKERS):
        raise NativeChangeEnvelopeUnavailable(
            "engine lacks authoritative ApplyChangeEnvelopes"
        )
    return False


def _advance_batch_graph_version(
    scope: tuple[str, str], engine_results: list[dict[str, Any]], expected: int
) -> None:
    """Advance the tracked graph version by the number of applied (non-replay)
    envelopes so a subsequent page fences correctly."""
    applied = sum(
        1 for result in engine_results if str(result.get("status")) == "applied"
    )
    if applied:
        _NATIVE_GRAPH_VERSIONS[scope] = max(
            _NATIVE_GRAPH_VERSIONS.get(scope, 0), expected + applied
        )


def _sync_batch_policy_cache(
    envelopes: list[ChangeEnvelope],
    engine_results: list[dict[str, Any]],
    governed: list[list[str]],
) -> None:
    """Refresh the read-policy cache for governed objects that committed, each
    under ITS OWN envelope's source ACL/classification (a page may mix them)."""
    from ...protocols.source_connectors.permission_sync import sync_access

    for offset, envelope in enumerate(envelopes):
        if str(engine_results[offset].get("status")) != "applied":
            continue
        try:
            for object_id in sorted(set(governed[offset])):
                sync_access(
                    object_id,
                    envelope.source_acl,
                    classification=envelope.classification,
                )
        except Exception:  # noqa: BLE001 - durable policy stays fail-closed
            logger.warning("native policy cache refresh is pending")
            break


def _apply_native_batch_attempt(
    authority: _NativeAuthority,
    session: Any,
    envelopes: list[ChangeEnvelope],
    scope: tuple[str, str],
    conflict_sequence: list[str],
    attempt: int,
) -> list[dict[str, Any]] | None:
    """One OCC attempt at the page. ``None`` means "retry"; a list is final."""
    expected = _NATIVE_GRAPH_VERSIONS.get(scope, 0)
    material = _batch_native_material(authority, session, envelopes, expected)
    engine_results = authority.compute.client.changes.apply_batch(material.natives)
    if not isinstance(engine_results, list) or len(engine_results) != len(
        material.natives
    ):
        raise RuntimeError("ApplyChangeEnvelopes returned a malformed result set")

    root_error = _root_batch_conflict(engine_results)
    if root_error is not None and _retry_batch_conflict(
        root_error, scope, conflict_sequence, attempt
    ):
        return None

    _advance_batch_graph_version(scope, engine_results, expected)

    # D-BFR-12: mirror each genuinely-applied envelope's node/edge delta,
    # in the SAME page order the authority just committed it, so the
    # mirror outbox preserves batch ordering. A replayed/idempotent-skip
    # or conflicted entry is never re-enqueued (the "replay-skip"
    # contract — see the single-envelope path for why).
    for offset, result in enumerate(engine_results):
        if str(result.get("status")) == "applied":
            _replay_native_graph_mutation(authority, material.natives[offset])

    _sync_batch_policy_cache(envelopes, engine_results, material.governed)

    return [
        _batch_entry_result(
            envelope,
            engine_results[offset],
            material.counts[offset],
            material.node_ids[offset],
            material.cursor_advanced[offset],
        )
        for offset, envelope in enumerate(envelopes)
    ]


def _apply_native_change_envelopes(
    authority: _NativeAuthority, session: Any, envelopes: list[ChangeEnvelope]
) -> list[dict[str, Any]]:
    """Commit a page of envelopes (all one graph) through native ``ApplyChangeEnvelopes``.

    The whole page is ONE round-trip and ONE coalesced engine transaction. Envelopes
    are given sequential ``expected_graph_version``s and a client-chained source cursor
    (read once for the page) so the shared transaction's read-your-writes chain is
    correct. Returns per-envelope result dicts in input order. A whole-batch version
    conflict re-reads the authoritative version and retries (bounded), mirroring the
    single-envelope OCC loop.
    """
    from ..core.session import use_session

    client = authority.compute.client
    scope = (str(session.tenant), str(session.graph))
    with use_session(session), _native_lock(scope):
        _require_apply_change_envelopes(client)
        conflict_sequence: list[str] = []
        for attempt in range(_NATIVE_OCC_MAX_ATTEMPTS):
            results = _apply_native_batch_attempt(
                authority, session, envelopes, scope, conflict_sequence, attempt
            )
            if results is not None:
                return results
        raise _NativeOccRetryBudgetExhausted(conflict_sequence)


# ── D-EMB / D-PERF-5: ingest-time embedding chokepoint ──────────────────────
#
# Every typed-entity connector (ServiceNow, LeanIX, GitHub, Twenty, ...) funnels
# its writes through `_ingest_entities_via_envelope` (source_sync.py) into
# `ingest_envelope`/`ingest_envelopes` here — this is the one point where EVERY
# ChangeEnvelope-based write converges (grepped: 15+ production callers beyond
# source_sync alone — domain_packs, external_graph, promotion, supersession,
# document_processing, worldmodel_pipeline, feed_sources, research/loop_controller,
# engine_surface_tools, neural/governance). Only ~6 document-shaped connectors
# computed their own embedding before calling in; every other entity landed with
# no vector (26,680 KG nodes, 136 embedded — 0.5%). This hooks embedding
# generation HERE, once, instead of at each connector, so no future connector can
# bypass it the way the ACL fix's 4 write-chokepoint precedent established.
#
# Best-effort by design: a down/unconfigured embedding endpoint must never fail
# an entity's write (embedding is a retrieval nicety, not a durability gate) —
# every failure path below degrades to "no vector this write" and is logged at
# most once per call, never raised.
def _primary_upsert_targets(
    envelopes: list[ChangeEnvelope],
) -> list[tuple[int, str, dict[str, Any]]]:
    """``(position, node_id, rendered row)`` for each primary typed upsert."""
    primary: list[tuple[int, str, dict[str, Any]]] = []
    for position, envelope in enumerate(envelopes):
        if envelope.operation != "upsert" or envelope.typed_payload is None:
            continue
        node_id, row = _resolve_identity(envelope)
        if node_id and row is not None:
            primary.append((position, str(node_id), row))
    return primary


def _stage_embedding_change(
    payload: dict[str, Any], current: dict[str, Any], row: dict[str, Any]
) -> tuple[list[float] | None, str] | None:
    """Stage one envelope's fail-closed embedding change, in place on ``payload``.

    ``None`` means "leave the durable vector exactly as it is". Otherwise
    ``payload["embedding"]`` has been nulled and the index-ready flag cleared —
    a stale vector can never outlive the text it described — and the result is
    ``(supplied_vector_or_None, effective_text)``: a non-``None`` vector was
    supplied by the caller, while ``None`` with a non-empty text means "generate
    one" and ``None`` with an empty text means "nothing to embed".
    """
    from ..enrichment.semantic import (
        EMBEDDING_BACKFILL_STATE_FIELD,
        EMBEDDING_INDEX_READY_FIELD,
        derive_entity_text,
    )

    payload[EMBEDDING_BACKFILL_STATE_FIELD] = None
    effective = dict(current)
    effective.update(row)
    old_text = derive_entity_text(current)
    new_text = derive_entity_text(effective)

    if "embedding" in payload:
        # Explicit null/empty is an invalidation request. A supplied vector
        # also passes through null first, then becomes visible atomically with
        # its ANN replacement after the source envelope commits.
        incoming_embedding = payload.get("embedding")
        payload["embedding"] = None
        payload[EMBEDDING_INDEX_READY_FIELD] = False
        if incoming_embedding:
            return list(incoming_embedding), new_text
        return None

    if current.get("embedding") and old_text == new_text:
        # D-BFR-10: a partial ACL/classification/operational field merge did
        # not alter the effective embedding text, so preserve the current
        # vector and avoid needless embedder + ANN work.
        return None

    # New/missing vectors and real text changes are fail-closed. If the
    # embedder is unavailable the source write still lands, but the obsolete
    # ANN candidate is rejected because its durable vector property is null.
    payload["embedding"] = None
    payload[EMBEDDING_INDEX_READY_FIELD] = False
    return None, new_text


def _auto_embed_enabled() -> bool:
    """Is ingest-time auto-embedding on? An unreadable config defaults to ON."""
    try:
        from agent_utilities.core.config import config

        return bool(getattr(config, "kg_ingest_auto_embed", True))
    except Exception:  # noqa: BLE001 - config unavailable defaults to enabled
        return True


def _generate_pending_vectors(
    pending: list[tuple[int, str]],
) -> list[list[float]] | None:
    """Batch-generate replacement vectors, or ``None`` when the embedder failed.

    ``None`` is deliberately distinct from an empty list: it means "no vectors
    this write" and the caller keeps whatever was already durable, rather than
    zipping an empty result against ``pending`` as if the embedder had answered.
    """
    try:
        from ..enrichment.semantic import make_embed_fn, validate_embedding_vectors

        embed_fn = make_embed_fn()
        return validate_embedding_vectors(
            embed_fn([text for _, text in pending]),
            expected_count=len(pending),
        )
    except Exception as exc:  # noqa: BLE001 - embedding is not a durability gate
        logger.debug("ingest-time auto-embed skipped (%s): %s", type(exc).__name__, exc)
        return None


def _prepare_embedding_envelopes(
    client: Any, envelopes: list[ChangeEnvelope]
) -> dict[int, tuple[list[float], str]]:
    """Stage fail-closed embedding changes and batch-generate replacements.

    Native entity writes are field merges.  For each primary upsert this compares
    the durable entity text with the effective post-merge text. An omitted vector
    therefore preserves a current embedding when only non-text fields changed,
    while a real text change commits ``embedding = null`` in the source envelope.
    Replacement vectors are returned for a later atomic field+ANN transaction;
    they are deliberately *not* made durable in the source mutation first.
    """
    primary = _primary_upsert_targets(envelopes)
    existing = _node_properties_batch(client, [node_id for _, node_id, _ in primary])

    supplied: dict[int, tuple[list[float], str]] = {}
    pending: list[tuple[int, str]] = []
    for position, node_id, row in primary:
        payload = envelopes[position].typed_payload
        assert payload is not None
        staged = _stage_embedding_change(payload, existing.get(node_id, {}), row)
        if staged is None:
            continue
        vector, new_text = staged
        if vector is not None:
            supplied[position] = (vector, new_text)
        elif new_text:
            pending.append((position, new_text))

    if not pending or not _auto_embed_enabled():
        return supplied
    vectors = _generate_pending_vectors(pending)
    if vectors is None:
        return supplied

    embedded = dict(supplied)
    for (position, text), vector in zip(pending, vectors, strict=True):
        embedded[position] = (list(vector), text)
    return embedded


def _atomic_embedding_fn(
    authority: Any,
) -> Callable[[str, dict[str, Any], dict[str, Any], list[float]], bool] | None:
    """The authority's atomic field+ANN embedding transaction, or ``None``.

    ``None`` means the capability is absent, which the caller reports and
    skips — it never falls back to a non-atomic property write.
    """
    compute = getattr(authority, "compute", None)
    publisher = getattr(authority, "backend", None)
    scoped_atomic_embedding = getattr(
        publisher, "compare_and_set_node_embedding_for_graph", None
    )
    if callable(scoped_atomic_embedding):
        graph_name = str(getattr(compute, "graph_name", "") or "")

        def _scoped_atomic_embedding(
            node_id: str,
            conditions: dict[str, Any],
            updates: dict[str, Any],
            vector: list[float],
        ) -> bool:
            return bool(
                scoped_atomic_embedding(
                    graph_name, node_id, conditions, updates, vector
                )
            )

        return _scoped_atomic_embedding
    candidate = getattr(compute, "compare_and_set_node_embedding", None)
    return candidate if callable(candidate) else None


def _commit_one_embedded_vector(
    atomic_embedding: Callable[
        [str, dict[str, Any], dict[str, Any], list[float]], bool
    ],
    node_id: str,
    current: dict[str, Any],
    vector: list[float],
    expected_text: str,
) -> None:
    """Cross-modal CAS for one node.

    A text change between generation and this transaction loses the exact-field
    CAS and applies NEITHER side; a failed commit leaves the durable vector null
    (the source write itself stays valid).
    """
    from ..enrichment.semantic import (
        EMBEDDING_BACKFILL_STATE_FIELD,
        EMBEDDING_INDEX_READY_FIELD,
        derive_entity_text_snapshot,
    )

    text, text_conditions = derive_entity_text_snapshot(current)
    if text != expected_text:
        logger.debug("ingest-time embedding text changed before atomic commit")
        return
    conditions = {
        "embedding": None,
        EMBEDDING_BACKFILL_STATE_FIELD: None,
        EMBEDDING_INDEX_READY_FIELD: False,
        **text_conditions,
    }
    updates = {
        "embedding": list(vector),
        EMBEDDING_BACKFILL_STATE_FIELD: None,
    }
    try:
        atomic_embedding(str(node_id), conditions, updates, list(vector))
    except Exception as exc:  # noqa: BLE001 - source write remains valid and vector stays null
        logger.debug(
            "ingest-time atomic embedding commit skipped for %s (%s): %s",
            node_id,
            type(exc).__name__,
            exc,
        )


def _commit_embedded_vectors(
    authority: Any,
    node_ids: dict[int, str | None],
    vectors: dict[int, tuple[list[float], str]],
) -> None:
    """Atomically publish freshly embedded vectors to properties and ANN.

    ``node_ids`` is deliberately ``str | None``-valued: callers build it from a
    result payload's ``node_id``, which is absent for a skipped/failed record.
    The source envelope has already committed the effective text with a null
    embedding. Each successful cross-modal CAS commits the vector property and
    native ANN replacement durably, with served visibility fenced until ANN
    projection completes. A text change between generation and this transaction
    loses the exact-field CAS and applies neither side.
    """
    compute = getattr(authority, "compute", None)
    atomic_embedding = _atomic_embedding_fn(authority)
    if atomic_embedding is None:
        logger.warning(
            "ingest-time embedding remains unavailable: authority lacks atomic "
            "field+ANN transactions"
        )
        return
    client = getattr(compute, "client", None)
    if client is None:
        logger.warning(
            "ingest-time embedding remains unavailable: authority lacks a native client"
        )
        return
    positioned_ids = {
        position: str(node_id)
        for position, node_id in node_ids.items()
        if node_id and position in vectors
    }
    properties = _node_properties_batch(client, list(positioned_ids.values()))
    for position, (vector, expected_text) in vectors.items():
        node_id = node_ids.get(position)
        if not node_id:
            continue
        _commit_one_embedded_vector(
            atomic_embedding,
            str(node_id),
            properties.get(str(node_id), {}),
            vector,
            expected_text,
        )


def _gate_and_validate(
    envelope: ChangeEnvelope,
) -> tuple[dict[str, Any] | None, ChangeEnvelope | None]:
    """``(rejection_result, gated_envelope)`` — exactly one is not ``None``."""
    try:
        gated = _privacy_gate(envelope)
    except ValueError:
        logger.warning("native ChangeEnvelope privacy gate rejected identity")
        return {
            "status": "rejected",
            "reason": "persistence privacy gate rejected an unsafe identity",
            "watermark_advanced": False,
        }, None
    violations = _validate_envelope(gated)
    if violations:
        return {
            "status": "rejected",
            "envelope_id": gated.envelope_id,
            "idempotency_key": gated.idempotency_key,
            "connector": gated.connector,
            "operation": gated.operation,
            "watermark_advanced": False,
            "violations": violations,
        }, None
    return None, gated


def _prepare_batch_envelopes(
    envelopes: list[ChangeEnvelope], results: list[dict[str, Any]]
) -> list[tuple[int, ChangeEnvelope]]:
    """Privacy-gate + validate the page, writing rejections into ``results``.

    Stops at the FIRST rejection and marks every LATER envelope
    ``skipped_not_attempted``: envelopes after a rejection are not attempted,
    which is what preserves the contiguous-prefix watermark contract — the
    watermark can never jump past a record that was never committed.
    """
    prepared: list[tuple[int, ChangeEnvelope]] = []
    stopped_at: int | None = None
    for index, envelope in enumerate(envelopes):
        rejection, gated = _gate_and_validate(envelope)
        if rejection is not None or gated is None:
            results[index] = rejection or {}
            stopped_at = index
            break
        prepared.append((index, gated))

    # Envelopes AFTER the first rejection are not attempted — the contiguous-prefix
    # watermark contract stops there.
    if stopped_at is not None:
        for index in range(stopped_at + 1, len(envelopes)):
            results[index] = {
                "status": "skipped_not_attempted",
                "reason": "batch stopped at an earlier envelope",
                "watermark_advanced": False,
            }
    return prepared


def _batch_failure_results(
    prepared: list[tuple[int, ChangeEnvelope]],
    results: list[dict[str, Any]],
    status: str,
    error: str,
) -> list[dict[str, Any]]:
    """Mark every prepared envelope with the same whole-batch failure.

    ``watermark_advanced`` stays ``False`` on every entry: a failed page must
    never move a source cursor.
    """
    for index, envelope in prepared:
        results[index] = {
            "status": status,
            "error": error,
            "envelope_id": envelope.envelope_id,
            "watermark_advanced": False,
        }
    return results


def _commit_native_batch(
    engine: Any, prepared: list[tuple[int, ChangeEnvelope]]
) -> tuple[Any, dict[int, tuple[list[float], str]], list[dict[str, Any]]]:
    """Resolve the authority, stage embeddings, and commit the page natively."""
    authority = _resolve_native_authority(engine)
    authority, session = _native_session(authority, prepared[0][1])
    _require_apply_change_envelopes(authority.compute.client)
    # Mutate private payload copies so a capability fallback can safely
    # re-enter the single-envelope path with the original DTOs.
    prepared_envelopes = [
        replace(
            envelope,
            typed_payload=(
                dict(envelope.typed_payload)
                if envelope.typed_payload is not None
                else None
            ),
        )
        for _, envelope in prepared
    ]
    embedded_by_position = _prepare_embedding_envelopes(
        authority.compute.client, prepared_envelopes
    )
    batch_results = _apply_native_change_envelopes(
        authority, session, prepared_envelopes
    )
    return authority, embedded_by_position, batch_results


def _batch_commit_failure(
    exc: BaseException,
    prepared: list[tuple[int, ChangeEnvelope]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Log and shape a whole-batch commit failure.

    There is deliberately NO fallback write path here: this is the
    authoritative commit, so a genuine failure surfaces as a failed/rejected
    status on every prepared envelope with ``watermark_advanced=False``, never
    a retry through a second, weaker mechanism.
    """
    if isinstance(exc, PermissionError | ValueError):
        # Log the real cause, not just the exception type: type(exc).__name__
        # alone would be the exact swallowed-error antipattern
        # scripts/check_swallowed_errors.py exists to catch. Pass exc itself
        # (not a second, redundant type(exc).__name__ arg) -- core/log_privacy.py's
        # _sanitize_value already renders a raw exception as "Type: message"
        # while still redacting paths/etc.
        logger.warning("native ChangeEnvelope batch rejected (%s)", exc)
        return _batch_failure_results(prepared, results, "rejected", type(exc).__name__)
    if isinstance(exc, _NativeOccRetryBudgetExhausted):
        logger.warning(
            "native ChangeEnvelope batch OCC retry budget exhausted (conflict_sequence=%s)",
            ",".join(exc.conflicts),
        )
        return _batch_failure_results(
            prepared, results, "failed", "NativeChangeEnvelopeConflictExhausted"
        )
    logger.warning("native ChangeEnvelope batch commit failed (%s)", type(exc).__name__)
    return _batch_failure_results(prepared, results, "failed", type(exc).__name__)


def _commit_batch_embeddings(
    authority: Any,
    prepared: list[tuple[int, ChangeEnvelope]],
    results: list[dict[str, Any]],
    embedded_by_position: dict[int, tuple[list[float], str]],
) -> None:
    """Publish the page's staged vectors for the entries that actually committed."""
    if not embedded_by_position:
        return
    node_ids_by_position = {
        position: results[index].get("node_id")
        for position, (index, _envelope) in enumerate(prepared)
        if results[index].get("status") in {"success", "skipped"}
    }
    _commit_embedded_vectors(authority, node_ids_by_position, embedded_by_position)


def ingest_envelopes(
    engine: Any, envelopes: list[ChangeEnvelope]
) -> list[dict[str, Any]]:
    """Commit a batch of external changes through native ``ApplyChangeEnvelopes``.

    Batches the per-record :func:`ingest_envelope` flow: the whole page becomes ONE
    engine round-trip and ONE coalesced transaction. Returns one result dict per input
    envelope, in order, each with the SAME shape/status vocabulary the single path
    produces (``success`` / ``skipped`` / ``failed`` / ``rejected``). A client-side
    validation/privacy rejection stops the batch at that envelope (contiguous prefix),
    preserving the break-on-first-failure watermark guarantee. An engine without the
    batch method transparently falls back to the per-record path (logged once).
    """
    if isinstance(engine, NativeChangeEnvelopeEngineProxy):
        engine = engine.authority
    if not envelopes:
        return []

    results: list[dict[str, Any]] = [{} for _ in envelopes]
    prepared = _prepare_batch_envelopes(envelopes, results)
    if not prepared:
        return results

    try:
        authority, embedded_by_position, batch_results = _commit_native_batch(
            engine, prepared
        )
    except NativeChangeEnvelopeUnavailable:
        logger.info(
            "native ApplyChangeEnvelopes unavailable; falling back to per-record ingestion"
        )
        return [ingest_envelope(engine, envelope) for envelope in envelopes]
    except Exception as exc:  # noqa: BLE001 - never fall back after a native failure
        return _batch_commit_failure(exc, prepared, results)

    for (index, _envelope), result in zip(prepared, batch_results, strict=True):
        results[index] = result
    _commit_batch_embeddings(authority, prepared, results, embedded_by_position)
    return results


def read_change_cursor(
    engine: Any,
    connector: str,
    *,
    source_instance: str = "",
) -> str | None:
    """Read the engine-native checkpoint used by an envelope connector.

    The cursor is tenant/graph scoped by the same verified ``GraphSession`` as
    writes. A missing native capability is an explicit error; callers may elect
    to perform a safe full pull, but must not substitute a separately-written
    watermark as the durability authority.
    """
    if isinstance(engine, NativeChangeEnvelopeEngineProxy):
        engine = engine.authority
    probe = _privacy_gate(
        ChangeEnvelope(
            connector=connector,
            operation="snapshot_complete",
            source_instance=source_instance,
        )
    )
    authority = _resolve_native_authority(engine)
    authority, session = _native_session(authority, probe)
    client = authority.compute.client
    supports = getattr(client, "supports", None)
    if not callable(supports) or not bool(supports("GetChangeCursor")):
        raise NativeChangeEnvelopeUnavailable(
            "engine does not advertise GetChangeCursor"
        )

    from ..core.session import use_session

    with use_session(session):
        cursor = client.changes.cursor(connector, _cursor_partition(source_instance))
    if not isinstance(cursor, dict):
        return None
    return _checkpoint_from_position(cursor.get("position"))


def _validate_graph_slice(
    entities: list[dict[str, Any]], relationships: list[dict[str, Any]]
) -> None:
    """Enforce the graph slice's canonical-key contract, fail closed on aliases."""
    for entity in entities:
        if "type" in entity or not str(entity.get("node_type") or "").strip():
            raise ValueError(
                "graph-slice nodes require canonical node_type and may not use type"
            )
    for relationship in relationships:
        if (
            any(
                key in relationship
                for key in ("type", "rel_type", "relationship_type", "relation")
            )
            or not str(relationship.get("relationship") or "").strip()
        ):
            raise ValueError(
                "graph-slice edges require canonical relationship and no aliases"
            )


def _slice_digest(payload: dict[str, Any]) -> str:
    """Deterministic sha256 over a canonical JSON rendering of ``payload``."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _edge_only_marker_entity(
    connector: str, source_instance: str, relationships: list[dict[str, Any]]
) -> dict[str, Any]:
    """Governed primary object for an edge-only derived/extractor batch.

    The deterministic marker owns delivery identity without inventing a source
    cursor or leaking endpoint material into its id.
    """
    marker_digest = _slice_digest(
        {
            "connector": connector,
            "source_instance": source_instance,
            "relationships": relationships,
        }
    )
    return {
        "id": f"source-materialization:{marker_digest}",
        "node_type": "SourceMaterialization",
        "source_system": connector,
        "relationship_count": len(relationships),
    }


def _graph_slice_primary(
    entities: list[dict[str, Any]], relationships: list[dict[str, Any]]
) -> dict[str, Any]:
    """The slice's primary row, carrying the rest as governed auxiliary material."""
    primary = dict(entities[0])
    if len(entities) > 1:
        primary["_nodes"] = [dict(item) for item in entities[1:]]
    if relationships:
        primary["_links"] = [dict(item) for item in relationships]
    return primary


def ingest_graph_slice(
    engine: Any,
    connector: str,
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    source_instance: str = "",
    checkpoint: str | None = None,
    version_field: str = "updatedAt",
    ontology_mapping_version: str = "",
    classification: str = "",
    idempotency_key: str = "",
) -> dict[str, Any]:
    """Commit a connector-produced multi-node graph slice atomically.

    The first entity is the envelope's primary object; remaining entities and
    all relationships are attached as governed auxiliary material.  When the
    source has no explicit version, a deterministic content digest supplies the
    idempotent version without advancing a source cursor. The Epistemic Graph
    authority is resolved from ``engine`` and missing native support fails closed.

    ``idempotency_key`` (B-11, CONCEPT:AU-KG.ingest.envelope-atomic-transaction):
    a non-empty value is used EXACTLY as given — the caller owns replay
    detection for this delivery, honestly enforced by the engine's own
    ``(tenant, graph, idempotency_key)``-scoped ``ApplyChangeEnvelope`` dedup
    (the result's ``status`` comes back ``"skipped"`` on a genuine replay, never
    silently reported as a fresh ``"success"``). Defaults to ``""``, which
    preserves every existing caller's behavior unchanged: the whole-slice
    content digest below supplies the idempotent version instead.

    ``ontology_mapping_version`` (CONCEPT:AU-KG.ingest.domain-pack-framework)
    stamps the resulting envelope's ``ChangeEnvelope.ontology_mapping_version``
    — the domain-pack framework's caller (``domain_packs.envelope_bridge``)
    passes its pack's ``"<pack>@<version>"`` so every fact this slice produces
    traces back to the exact mapping revision that produced it. Defaults to
    ``""`` (unchanged behavior for every existing caller).

    ``classification`` (CONCEPT:AU-KG.ingest.domain-pack-framework), one of
    ``DataClassification``'s lowercase values (``"public"``/``"internal"``/
    ``"confidential"``/``"restricted"``), overrides the envelope's
    ``ChangeEnvelope.classification``. Defaults to ``""``, which leaves
    ``from_connector_record``'s own fail-closed default (``PUBLIC`` only when
    the record's ``external_access`` says so, else ``INTERNAL``) unchanged for
    every existing caller.
    """
    relationships = relationships or []
    if not entities and not relationships:
        return {"status": "skipped", "reason": "empty graph slice"}
    _validate_graph_slice(entities, relationships)
    if isinstance(engine, NativeChangeEnvelopeEngineProxy):
        engine = engine.authority

    if not entities:
        entities = [_edge_only_marker_entity(connector, source_instance, relationships)]

    primary = _graph_slice_primary(entities, relationships)
    material_version = _slice_digest(
        {"entities": entities, "relationships": relationships}
    )
    overrides: dict[str, Any] = {}
    if classification:
        from ...models.company_brain import DataClassification

        overrides["classification"] = DataClassification(classification)
    envelope = ChangeEnvelope.from_connector_record(
        primary,
        connector=connector,
        source_instance=source_instance,
        id_field="id",
        version_field=version_field,
        checkpoint=checkpoint,
        ontology_mapping_version=ontology_mapping_version,
        **overrides,
    )
    # Batch identity must cover every auxiliary node and relationship. Keeping
    # only the primary row's upstream timestamp would replay-skip a changed
    # child/edge whenever that first row stayed stable. Preserve the upstream
    # marker in the payload/event time, but use the whole-slice digest as the
    # authoritative content version and idempotency input.
    # ``dataclasses.replace`` carries the already-derived idempotency key unless
    # it is explicitly cleared. Recompute it from the whole-slice version so a
    # changed auxiliary node/edge cannot be mistaken for a replay merely because
    # the primary row's upstream timestamp stayed stable.
    envelope = replace(
        envelope,
        source_version=material_version,
        idempotency_key=idempotency_key,
    )
    result = ingest_envelope(engine, envelope)
    if result.get("status") not in {"success", "skipped"}:
        raise RuntimeError(
            "native ChangeEnvelope graph slice failed: "
            f"{result.get('error') or result.get('status')}"
            # The rejection reason is the only actionable part of this failure;
            # dropping it forces every caller to re-derive it from logs.
            + (f" ({result['reason']})" if result.get("reason") else "")
        )
    return result


def _partial_materialization_outcome(
    envelope: ChangeEnvelope,
    exc: BaseException,
    resume: dict[str, Any],
    attempt: int,
) -> BaseException | None:
    """``None`` means "resumable partial materialization — retry".

    Otherwise the exception the caller must report: the ORIGINAL one when this
    was not a partial materialization at all, or the bounded give-up error.

    Only the EXACT retryable wire payload (PARTIAL_MATERIALIZATION,
    retryable=True) resumes; every other exception — malformed, stale, or
    terminal — falls straight through to unchanged failure handling. This is
    the SAME strictness ``_retryable_partial_materialization`` already
    enforces; it is not re-implemented or broadened here.
    """
    from ..core.engine_tasks import _retryable_partial_materialization

    materialization = _retryable_partial_materialization(exc)
    if materialization is None:
        return exc
    return _materialization_resume_error(envelope, materialization, resume, attempt)


def _native_result_base(envelope: ChangeEnvelope) -> dict[str, Any]:
    """The fields every single-envelope result carries, watermark held back."""
    return {
        "envelope_id": envelope.envelope_id,
        "idempotency_key": envelope.idempotency_key,
        "connector": envelope.connector,
        "operation": envelope.operation,
        "watermark_advanced": False,
        "native_atomic": True,
    }


def _gate_envelope_for_single_ingest(
    envelope: ChangeEnvelope,
) -> tuple[ChangeEnvelope, dict[str, Any] | None]:
    """``(gated_envelope, rejection)``; ``rejection`` is ``None`` when it passed.

    On a privacy-gate rejection the ORIGINAL envelope comes back untouched —
    the caller returns the rejection and never writes it.
    """
    try:
        gated = _privacy_gate(envelope)
    except ValueError:
        logger.warning("native ChangeEnvelope privacy gate rejected identity")
        return envelope, {
            "status": "rejected",
            "reason": "persistence privacy gate rejected an unsafe identity",
            "watermark_advanced": False,
        }
    violations = _validate_envelope(gated)
    if violations:
        return gated, {
            **_native_result_base(gated),
            "status": "rejected",
            "violations": violations,
        }
    return gated, None


def _publish_envelope_embedding(
    authority: Any,
    result: dict[str, Any],
    embedded_by_position: dict[int, tuple[list[float], str]],
) -> None:
    """Publish a staged vector ONLY for a genuinely committed write.

    A failed/rejected write must never leave a vector describing text that was
    not durably stored.
    """
    if embedded_by_position and result.get("status") in {"success", "skipped"}:
        _commit_embedded_vectors(
            authority, {0: result.get("node_id")}, embedded_by_position
        )


def _materialization_resume_error(
    envelope: ChangeEnvelope,
    materialization: dict[str, Any],
    resume: dict[str, Any],
    attempt: int,
) -> _PartialMaterializationRetriesExhausted | None:
    """``None`` means "the resume is still valid, retry"; else the give-up error.

    Three bounded give-up conditions, in the original order: the engine moved to
    a different ``source_snapshot_version`` (a completeness cursor is only valid
    against the snapshot it was issued for), the cursor stopped advancing, or
    the attempt budget ran out. ``resume`` is mutated in place to carry the
    first-seen snapshot version and the last-seen cursor across attempts.
    """
    cursor = materialization.get("completeness_cursor")
    snapshot_version = materialization.get("source_snapshot_version")
    if resume["snapshot_version"] is _MATERIALIZATION_UNSET:
        resume["snapshot_version"] = snapshot_version
    if snapshot_version != resume["snapshot_version"]:
        # A completeness_cursor is only valid against the snapshot
        # it was issued for; the engine moved to a different
        # snapshot mid-resume, so the cursor no longer means what
        # it did.
        return _PartialMaterializationRetriesExhausted(
            f"envelope {envelope.envelope_id} partial-materialization "
            "resume aborted: source_snapshot_version changed from "
            f"{resume['snapshot_version']!r} to {snapshot_version!r} "
            f"while resuming from cursor={cursor!r}."
        )
    if resume["cursor"] is not _MATERIALIZATION_UNSET and cursor == resume["cursor"]:
        return _PartialMaterializationRetriesExhausted(
            f"envelope {envelope.envelope_id} partial-materialization "
            f"cursor stopped advancing at {cursor!r} "
            f"(snapshot={snapshot_version!r}) after {attempt} "
            "attempt(s); giving up instead of retrying forever."
        )
    if attempt >= _MATERIALIZATION_MAX_ATTEMPTS:
        return _PartialMaterializationRetriesExhausted(
            f"envelope {envelope.envelope_id} did not finish "
            f"materializing within {_MATERIALIZATION_MAX_ATTEMPTS} "
            f"attempts (cursor={cursor!r}, "
            f"snapshot={snapshot_version!r})."
        )
    resume["cursor"] = cursor
    logger.info(
        "native ChangeEnvelope commit for %s hit a retryable "
        "partial materialization (cursor=%s snapshot=%s); "
        "resuming (attempt %d/%d)",
        envelope.envelope_id,
        cursor,
        snapshot_version,
        attempt,
        _MATERIALIZATION_MAX_ATTEMPTS,
    )
    return None


def ingest_envelope(engine: Any, envelope: ChangeEnvelope) -> dict[str, Any]:
    """Commit one external change through native ``ApplyChangeEnvelope``.

    Validation and persistence privacy run before authority resolution. Missing
    native capability fails closed.
    """
    if isinstance(engine, NativeChangeEnvelopeEngineProxy):
        engine = engine.authority
    envelope, rejection = _gate_envelope_for_single_ingest(envelope)
    if rejection is not None:
        return rejection
    base = _native_result_base(envelope)

    # Bounded resume state for THIS envelope's own attempts — see the
    # _MATERIALIZATION_* constants above for the shared rationale/values with
    # ``pipeline/runner.py``'s identical resume loop.
    attempt = 0
    resume: dict[str, Any] = {
        "snapshot_version": _MATERIALIZATION_UNSET,
        "cursor": _MATERIALIZATION_UNSET,
    }

    while True:
        attempt += 1
        try:
            authority = _resolve_native_authority(engine)
            authority, session = _native_session(authority, envelope)
            embedded_by_position = _prepare_embedding_envelopes(
                authority.compute.client, [envelope]
            )
            result = _apply_native_change_envelope(authority, session, envelope)
            _publish_envelope_embedding(authority, result, embedded_by_position)
            return result
        except NativeChangeEnvelopeUnavailable:
            logger.warning("native ChangeEnvelope capability is unavailable")
            return {
                **base,
                "status": "failed",
                "error": "NativeChangeEnvelopeUnavailable",
                "reason": "authoritative native ChangeEnvelope commit is unavailable",
            }
        except (PermissionError, ValueError) as exc:
            # D-DSTK + D-DG (reconciliation-gate-2: two lanes fixed this same
            # defect independently). Collapsing to type(exc).__name__ alone dropped
            # the actual rejection reason (which field/tenant/identity was invalid)
            # — "rejected (ValueError)" is equally true of a bad property type, an
            # over-long id and a policy denial, so every caller across the fleet
            # (source_sync, external_graph, document_processing, ...) reported an
            # unactionable failure. `error` stays the class name (some callers match
            # on it); the message now travels too, under the `reason` key this
            # function already uses everywhere else (including the sibling
            # "unavailable" return in this very except-chain).
            # NB: no exc_info — core/log_privacy.py nulls it on every
            # agent_utilities.* record, so the interpolated message is the only
            # channel that actually carries the cause.
            logger.warning(
                "native ChangeEnvelope rejected (%s): %s", type(exc).__name__, exc
            )
            return {
                **base,
                "status": "rejected",
                "error": type(exc).__name__,
                "reason": str(exc),
            }
        except _NativeOccRetryBudgetExhausted as exc:
            logger.warning(
                "native ChangeEnvelope OCC retry budget exhausted (conflict_sequence=%s)",
                ",".join(exc.conflicts),
            )
            return {
                **base,
                "status": "failed",
                "error": "NativeChangeEnvelopeConflictExhausted",
            }
        except Exception as exc:  # noqa: BLE001 — never fall back after native failure (this is the authoritative commit path, so a genuine failure must surface as failed status, not be retried on a different path); `error`/`reason` below now carry the real cause instead of only the exception class name
            effective_exc = _partial_materialization_outcome(
                envelope, exc, resume, attempt
            )
            if effective_exc is None:
                time.sleep(_MATERIALIZATION_RETRY_DELAY_S)
                continue

            # Same reasoning as the rejection path above: the class name alone
            # cannot tell an operator WHICH commit failed or why.
            logger.warning(
                "native ChangeEnvelope commit failed (%s): %s",
                type(effective_exc).__name__,
                effective_exc,
            )
            return {
                **base,
                "status": "failed",
                "error": type(effective_exc).__name__,
                "reason": str(effective_exc),
                "retryable": isinstance(
                    effective_exc, _PartialMaterializationRetriesExhausted
                ),
            }
