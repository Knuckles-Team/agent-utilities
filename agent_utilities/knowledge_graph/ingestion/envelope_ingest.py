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
from dataclasses import dataclass, replace
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


def _shacl_violation_summary(report: dict[str, Any], *, limit: int = 5) -> str:
    """Summarize a non-conforming SHACL report as a short, bounded string."""
    results = report.get("results")
    if not isinstance(results, list) or not results:
        return str(report.get("message") or "no violation detail reported")
    seen: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        detail = " ".join(
            str(item[key])
            for key in ("focusNode", "resultPath", "sourceShape", "resultMessage")
            if item.get(key)
        )
        if detail and detail not in seen:
            seen.append(detail)
        if len(seen) >= limit:
            break
    extra = len(results) - len(seen)
    return "; ".join(seen) + (f" (+{extra} more)" if extra > 0 else "")


# ── validate ─────────────────────────────────────────────────────────────


def _validate_envelope(envelope: ChangeEnvelope) -> list[str]:
    """Schema + fail-closed policy checks. Empty list = OK.

    Re-affirms the invariants ``ChangeEnvelope.__post_init__`` already enforces
    (defense-in-depth against a future mutation) and adds the policy check
    ``__post_init__`` can't: CONCEPT:AU-P0-4 fail-closed connector permissions
    — a ``PUBLIC``-classified object must carry an explicit
    ``source_acl.is_public=True`` proof; "unknown" must never silently become
    "public" just because a connector forgot to set an ACL.
    """
    from ...models.company_brain import DataClassification

    violations: list[str] = []
    if envelope.operation not in ("upsert", "delete", "snapshot_complete"):
        violations.append(f"invalid operation {envelope.operation!r}")
    if envelope.typed_payload is not None and envelope.blob_ref is not None:
        violations.append(
            "typed_payload and blob_ref are mutually exclusive on this envelope"
        )
    if (
        envelope.operation == "upsert"
        and envelope.typed_payload is None
        and envelope.blob_ref is None
    ):
        violations.append(
            "upsert envelope carries neither typed_payload nor blob_ref — nothing to write"
        )
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


def validate_envelope(envelope: ChangeEnvelope) -> list[str]:
    """Public reuse point for :func:`_validate_envelope` (CONCEPT:AU-KG.ingest.governed-claim-promotion).

    Lets another governed-write caller (e.g. a candidate-claim promotion gate
    deciding whether a claim's proposed materialization is even well-formed
    before a steward reviews it) run the SAME fail-closed schema + policy
    checks ``ingest_envelope`` itself enforces at the write boundary — never a
    second, potentially-drifting policy implementation. Empty list = OK.
    """
    return _validate_envelope(envelope)


def _privacy_gate(envelope: ChangeEnvelope) -> ChangeEnvelope:
    """Remove PII and machine locations before any durable envelope step.

    Identity and ACL routing keys cannot be rewritten safely. An unsafe object
    identity is rejected; unsafe ACL principals are replaced by the deny-all
    quarantine. Content/provenance fields are deeply sanitized and only a
    non-sensitive redaction summary is retained.
    """
    from ...protocols.source_connectors.base import ExternalAccess
    from ...security.persistence_privacy import PersistencePrivacyGuard

    guard = PersistencePrivacyGuard()

    def assert_safe_identity(value: Any) -> None:
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

    for value in (envelope.envelope_id, envelope.idempotency_key):
        assert_safe_identity(value)
    for value in (
        envelope.connector,
        envelope.tenant,
        envelope.source_instance,
        envelope.source_object_id,
        envelope.source_version,
        *envelope.live_ids,
    ):
        assert_safe_identity(value)

    payload = envelope.typed_payload
    if payload is not None:
        opaque_values: dict[int, Any] = {}
        next_sentinel = -(1 << 255)

        def identity_field(key: Any) -> bool:
            rendered = str(key)
            normalized = re.sub(r"[^a-z0-9]+", "_", rendered.casefold()).strip("_")
            return (
                normalized in {"id", "source", "target"}
                or normalized.endswith("_id")
                or rendered.endswith(("Id", "ID"))
            )

        def mask_opaque_identities(value: Any) -> Any:
            nonlocal next_sentinel
            if isinstance(value, dict):
                masked: dict[Any, Any] = {}
                for key, item in value.items():
                    if identity_field(key) and item not in (None, ""):
                        assert_safe_identity(item)
                        rendered = str(item)
                        if _OPAQUE_INTERNAL_ID_RE.fullmatch(
                            rendered
                        ) or _OPAQUE_NAMESPACED_ID_RE.fullmatch(rendered):
                            sentinel = next_sentinel
                            next_sentinel -= 1
                            opaque_values[sentinel] = item
                            masked[key] = sentinel
                            continue
                    masked[key] = mask_opaque_identities(item)
                return masked
            if isinstance(value, list | tuple | set | frozenset):
                return [mask_opaque_identities(item) for item in value]
            return value

        def restore_opaque_identities(value: Any) -> Any:
            if isinstance(value, dict):
                restored: dict[Any, Any] = {}
                for key, item in value.items():
                    if (
                        identity_field(key)
                        and isinstance(item, int)
                        and not isinstance(item, bool)
                        and item in opaque_values
                    ):
                        restored[key] = opaque_values[item]
                    else:
                        restored[key] = restore_opaque_identities(item)
                return restored
            if isinstance(value, list):
                return [restore_opaque_identities(item) for item in value]
            return value

        try:
            masked_payload = mask_opaque_identities(payload)
        except ValueError:
            raise ValueError("unsafe payload identity") from None
        clean_payload, payload_report = guard.sanitize(masked_payload)
        if not isinstance(clean_payload, dict):
            raise ValueError("invalid sanitized payload")
        clean_payload = restore_opaque_identities(clean_payload)
    else:
        clean_payload = None
        _, payload_report = guard.sanitize(None)

    clean_provenance, provenance_report = guard.sanitize(envelope.provenance)
    if not isinstance(clean_provenance, dict):
        clean_provenance = {}
    clean_operational, operational_report = guard.sanitize(
        {
            "blob_ref": envelope.blob_ref,
            "checkpoint": envelope.checkpoint,
            "trace_context": envelope.trace_context,
        }
    )

    access = envelope.source_acl
    acl_redactions = 0
    if access is not None:
        unsafe_acl = False
        for principal in (*access.group_ids, *access.read_roles, *access.markings):
            _, report = guard.sanitize_text(str(principal))
            unsafe_acl = unsafe_acl or report.changed
        acl_redactions = len(access.user_emails)
        if unsafe_acl:
            acl_redactions += (
                len(access.group_ids) + len(access.read_roles) + len(access.markings)
            )
            access = ExternalAccess.quarantined()
        elif access.user_emails:
            access = access.model_copy(update={"user_emails": []})
            if not access.is_public and not (
                access.group_ids or access.read_roles or access.markings
            ):
                access = ExternalAccess.quarantined()

    redactions = (
        payload_report.redactions
        + provenance_report.redactions
        + operational_report.redactions
        + acl_redactions
    )
    if redactions:
        categories = {
            *payload_report.detected_types,
            *provenance_report.detected_types,
            *operational_report.detected_types,
        }
        if acl_redactions:
            categories.add("acl_principal")
        clean_provenance = dict(clean_provenance)
        clean_provenance["persistence_privacy"] = {
            "redactions": redactions,
            "detected_types": sorted(categories),
        }

    return replace(
        envelope,
        typed_payload=clean_payload,
        blob_ref=clean_operational.get("blob_ref"),
        checkpoint=clean_operational.get("checkpoint"),
        trace_context=clean_operational.get("trace_context"),
        provenance=clean_provenance,
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
        return _NativeAuthority(compute, authority_backend)

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


def _position_advances(next_value: dict[str, Any], prior: dict[str, Any]) -> bool:
    if next_value.get("kind") != prior.get("kind"):
        return False
    kind = next_value.get("kind")
    left = next_value.get("value")
    right = prior.get("value")
    if kind in {"sequence", "timestamp_millis"}:
        if left is None or right is None:
            return False
        try:
            return int(left) > int(right)
        except (TypeError, ValueError):
            return False
    if kind == "opaque" and isinstance(left, dict) and isinstance(right, dict):
        left_type = left.get("cursor_type", left.get("version_type"))
        right_type = right.get("cursor_type", right.get("version_type"))
        return left_type == right_type and bool(left.get("value")) and left != right
    return False


def _cursor_partition(source_instance: str) -> str:
    return (
        hashlib.sha256(source_instance.encode("utf-8")).hexdigest()[:32]
        if source_instance
        else ""
    )


def _checkpoint_from_position(position: Any) -> str | None:
    if not isinstance(position, dict):
        return None
    kind = position.get("kind")
    value = position.get("value")
    if kind == "sequence":
        if value is None:
            return None
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return None
    if kind == "timestamp_millis":
        if value is None:
            return None
        try:
            parsed = datetime.fromtimestamp(int(value) / 1000, tz=UTC)
        except (TypeError, ValueError, OverflowError):
            return None
        return parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if kind == "opaque" and isinstance(value, dict):
        raw = value.get("value")
        return str(raw) if raw not in (None, "") else None
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
        kind = prior.get("kind")
        value = prior.get("value")
        if kind in {"sequence", "timestamp_millis"} and value is not None:
            try:
                return {"kind": kind, "value": int(value) + 1}
            except (TypeError, ValueError):
                pass
        if kind == "opaque" and isinstance(value, dict):
            version_type = str(value.get("version_type") or "connector_opaque_v1")
            return {
                "kind": "opaque",
                "value": {"version_type": version_type, "value": material_digest},
            }
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
    links: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []

    if envelope.operation == "snapshot_complete":
        fetch_ok = bool(envelope.provenance.get("fetch_ok", True))
        live_ids = set(envelope.live_ids)
        current_rows = _snapshot_rows(
            client, envelope.connector, envelope.source_instance
        )
        if not live_ids:
            from ..core.source_sync import _reconcile_allowed_empty_sources

            allowed = (
                envelope.provenance.get("authoritative_empty_approved") is True
                or envelope.connector.lower() in _reconcile_allowed_empty_sources()
            )
            if not fetch_ok or not allowed:
                # Commit the verified snapshot decision without tombstoning.
                live_ids = {
                    str(properties.get("externalToolId"))
                    for _, properties in current_rows
                    if properties.get("externalToolId")
                }
        stale: list[tuple[str, dict[str, Any]]] = []
        for existing_id, properties in current_rows:
            external_id = str(properties.get("externalToolId") or "")
            if external_id and external_id not in live_ids:
                updated = dict(properties)
                updated["archived"] = True
                updated["archivedReason"] = f"absent-from-{envelope.connector}"
                _stamp_ambient_valid_until(updated, envelope)
                stale.append((existing_id, updated))
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
        return node_id, [(node_id, marker), *stale], links, features, evidence

    if envelope.operation == "delete":
        current = _node_properties(client, node_id)
        current.update(
            {
                "id": node_id,
                "archived": True,
                "archivedReason": f"tombstoned-by-{envelope.connector}",
            }
        )
        _stamp_ambient_valid_until(current, envelope)
        return node_id, [(node_id, current)], links, features, evidence

    if row is None:
        row = {
            "id": node_id,
            "node_type": envelope.payload_type or "Artifact",
            "blob_ref": envelope.blob_ref,
            "classification": envelope.classification.value,
            "tenant": envelope.tenant,
            "source_instance": envelope.source_instance,
            "retention": envelope.retention,
            "legal_hold": envelope.legal_hold,
        }
    else:
        row = dict(row)
    links_value = row.pop("_links", None)
    if isinstance(links_value, list):
        links = [item for item in links_value if isinstance(item, dict)]
    features_value = row.pop("_features", None)
    if isinstance(features_value, list):
        features = [item for item in features_value if isinstance(item, dict)]
    evidence_value = row.pop("_evidence", None)
    if isinstance(evidence_value, list):
        evidence = [item for item in evidence_value if isinstance(item, dict)]
    auxiliary_value = row.pop("_nodes", None)

    from ..enrichment.provenance import stamp_source

    stamp_source(row, envelope.connector)
    _stamp_ambient_valid_time(row, envelope)
    current = _node_properties(client, node_id)
    current.update(row)
    current["id"] = node_id
    node_rows = [(node_id, current)]
    seen_ids = {node_id}
    if isinstance(auxiliary_value, list):
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
            node_rows.append((auxiliary_id, existing))
    return node_id, node_rows, links, features, evidence


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
    material_digest = _digest(
        {
            "operation": envelope.operation,
            "nodes": node_rows,
            "links": links,
            "blob_ref": envelope.blob_ref,
            "features": raw_features,
            "evidence": raw_evidence,
        }
    )
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

    cursor: dict[str, Any] | None = None
    cursor_advanced = False
    if envelope.checkpoint:
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
        if current_position is None or _position_advances(
            next_position, current_position
        ):
            cursor = {
                "source": envelope.connector,
                "partition": partition,
                "position": next_position,
            }
            if current_position is not None:
                cursor["expected_previous"] = current_position
            cursor_advanced = True

    operation_name = "delete" if envelope.operation == "delete" else "upsert"
    features, feature_objects = _native_feature_rows(
        raw_features, node_id, operation_name
    )
    evidence, evidence_objects = _native_evidence_rows(
        raw_evidence, node_id, operation_name, content_digest
    )
    governed_ids = {current_id for current_id, _ in node_rows}
    governed_ids.update(feature_objects)
    governed_ids.update(evidence_objects)

    access = envelope.source_acl
    access_material = access.model_dump() if access is not None else {"deny_all": True}
    subject_set_digest = _digest(access_material)
    tenant = str(session.tenant)
    graph = str(session.graph)
    policy_version = str(session.policy_version or "unversioned")
    policies = [
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

    blobs: list[dict[str, Any]] = []
    if envelope.blob_ref:
        blobs.append(
            {
                "blob_id": node_id,
                "operation": operation_name,
                "digest_algorithm": "sha256",
                "digest": hashlib.sha256(envelope.blob_ref.encode("utf-8")).hexdigest(),
                "media_type": str(envelope.payload_type or "application/octet-stream"),
                "length": 0,
            }
        )

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
        "tenant": tenant,
        "graph": graph,
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

    native: dict[str, Any] = {
        "schema_version": 1,
        "envelope_id": _native_envelope_id(envelope),
        "mutation": mutation,
        "content_version": {
            "object_id": node_id,
            "digest_algorithm": "sha256",
            "digest": content_digest,
            "source_version": source_position,
        },
        "blobs": blobs,
        "features": features,
        "evidence": evidence,
        "policies": policies,
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
    counts = {
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
    return native, counts, sorted(governed_ids), cursor_advanced


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


def _apply_native_change_envelope(
    authority: _NativeAuthority, session: Any, envelope: ChangeEnvelope
) -> dict[str, Any]:
    from ..core.session import use_session

    client = authority.compute.client
    scope = (str(session.tenant), str(session.graph))
    created_at_ms = _observed_at_ms(envelope.observed_time)
    with use_session(session), _native_lock(scope):
        supports = getattr(client, "supports", None)
        if not callable(supports) or not bool(supports("ApplyChangeEnvelope")):
            raise NativeChangeEnvelopeUnavailable(
                "engine does not advertise ApplyChangeEnvelope"
            )
        expected = _NATIVE_GRAPH_VERSIONS.get(scope, 0)
        counts: dict[str, int] = {}
        governed_ids: list[str] = []
        cursor_advanced = False
        conflict_sequence: list[str] = []
        receipt: Any = _recover_native_receipt(client, envelope)
        if receipt is not None:
            governed_ids = _policy_cache_object_ids(client, envelope)
        if receipt is None:
            for attempt in range(_NATIVE_OCC_MAX_ATTEMPTS):
                native, counts, governed_ids, cursor_advanced = _native_material(
                    authority,
                    session,
                    envelope,
                    expected_graph_version=expected,
                    created_at_ms=created_at_ms,
                )
                try:
                    receipt = client.changes.apply(native)
                except Exception as exc:  # noqa: BLE001 - only explicit pre-commit retries
                    recovered = _recover_native_receipt(client, envelope)
                    if recovered is not None:
                        receipt = recovered
                        _NATIVE_GRAPH_VERSIONS[scope] = max(
                            _NATIVE_GRAPH_VERSIONS.get(scope, 0), expected + 1
                        )
                        break
                    message = str(exc)
                    conflict = _NATIVE_OCC_CONFLICT_RE.search(message)
                    stale = _STALE_GRAPH_VERSION_RE.search(message)
                    if conflict is not None and stale is not None:
                        conflict_sequence.append(conflict.group(0).upper())
                        expected = int(stale.group(1))
                        _NATIVE_GRAPH_VERSIONS[scope] = expected
                        if attempt + 1 < _NATIVE_OCC_MAX_ATTEMPTS:
                            _native_occ_backoff(attempt)
                        continue
                    if conflict is not None and conflict.group(0).upper() in {
                        "STALE_CONTENT_VERSION",
                        "STALE_CURSOR",
                    }:
                        conflict_sequence.append(conflict.group(0).upper())
                        # Both are guaranteed pre-commit; re-read authoritative
                        # material and let graph OCC fence the rebuilt request.
                        if attempt + 1 < _NATIVE_OCC_MAX_ATTEMPTS:
                            _native_occ_backoff(attempt)
                        continue
                    if any(
                        marker in message
                        for marker in (
                            "requires the authoritative redb backend",
                            "requires a configured persistence backend",
                            "Unknown method",
                            "unknown method",
                        )
                    ):
                        raise NativeChangeEnvelopeUnavailable(
                            "engine lacks authoritative ChangeEnvelope persistence"
                        ) from exc
                    raise
                else:
                    _NATIVE_GRAPH_VERSIONS[scope] = max(
                        _NATIVE_GRAPH_VERSIONS.get(scope, 0), expected + 1
                    )
                    break
            else:
                raise _NativeOccRetryBudgetExhausted(conflict_sequence)

    replayed = bool(receipt.get("replayed")) if isinstance(receipt, dict) else False
    # This is a read-policy cache refresh, never the durability authority.
    # Both first apply and idempotent replay project it; replay is the bounded
    # recovery path after process-local cache loss.
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
            **counts,
            "receipt": _filtered_receipt(receipt, envelope),
        },
        "watermark_advanced": bool(cursor_advanced and not replayed),
        "checkpoint": envelope.checkpoint,
    }


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
        supports = getattr(client, "supports", None)
        if not callable(supports) or not bool(supports("ApplyChangeEnvelopes")):
            raise NativeChangeEnvelopeUnavailable(
                "engine does not advertise ApplyChangeEnvelopes"
            )
        conflict_sequence: list[str] = []
        for attempt in range(_NATIVE_OCC_MAX_ATTEMPTS):
            expected = _NATIVE_GRAPH_VERSIONS.get(scope, 0)
            # Read the page's source cursor ONCE, then chain it client-side across the
            # sequentially-applied envelopes (a live per-record read would STALE the
            # 2nd+ envelope inside the shared transaction).
            partition = _cursor_partition(envelopes[0].source_instance)
            page_cursor = client.changes.cursor(envelopes[0].connector, partition)
            chained: Any = (
                page_cursor.get("position")
                if isinstance(page_cursor, dict)
                and isinstance(page_cursor.get("position"), dict)
                else None
            )
            natives: list[dict[str, Any]] = []
            per_counts: list[dict[str, int]] = []
            per_node: list[str | None] = []
            per_cursor_advanced: list[bool] = []
            per_governed: list[list[str]] = []
            for offset, envelope in enumerate(envelopes):
                created_at_ms = _observed_at_ms(envelope.observed_time)
                native, counts, governed_ids, cursor_advanced = _native_material(
                    authority,
                    session,
                    envelope,
                    expected_graph_version=expected + offset,
                    created_at_ms=created_at_ms,
                    chained_cursor_position=chained,
                )
                natives.append(native)
                per_counts.append(counts)
                per_node.append(_resolve_identity(envelope)[0])
                per_cursor_advanced.append(cursor_advanced)
                per_governed.append(governed_ids)
                if cursor_advanced and envelope.checkpoint:
                    chained = _typed_position(envelope.checkpoint, content=False)

            engine_results = client.changes.apply_batch(natives)
            if not isinstance(engine_results, list) or len(engine_results) != len(
                natives
            ):
                raise RuntimeError(
                    "ApplyChangeEnvelopes returned a malformed result set"
                )

            # Find the ROOT failure (a non-aborted conflict) to decide whether an OCC
            # retry can make progress. Sibling entries carry the atomic-batch abort note.
            root_error: str | None = None
            for result in engine_results:
                if str(result.get("status")) == "conflict":
                    error = str(result.get("error") or "")
                    if not error.startswith("ABORTED_ATOMIC_GRAPH_BATCH"):
                        root_error = error
                        break
            if root_error is not None:
                conflict = _NATIVE_OCC_CONFLICT_RE.search(root_error)
                stale = _STALE_GRAPH_VERSION_RE.search(root_error)
                if conflict is not None:
                    conflict_sequence.append(conflict.group(0).upper())
                    if stale is not None:
                        _NATIVE_GRAPH_VERSIONS[scope] = int(stale.group(1))
                    if attempt + 1 < _NATIVE_OCC_MAX_ATTEMPTS:
                        _native_occ_backoff(attempt)
                        continue
                    raise _NativeOccRetryBudgetExhausted(conflict_sequence)
                if any(
                    marker in root_error
                    for marker in (
                        "requires the authoritative redb backend",
                        "requires a configured persistence backend",
                        "Unknown method",
                        "unknown method",
                        "CHANGE_BATCH_UNAVAILABLE_UNDER_PLACEMENT",
                    )
                ):
                    raise NativeChangeEnvelopeUnavailable(
                        "engine lacks authoritative ApplyChangeEnvelopes"
                    )

            # Advance the tracked graph version by the number of applied (non-replay)
            # envelopes so a subsequent page fences correctly.
            applied = sum(
                1 for result in engine_results if str(result.get("status")) == "applied"
            )
            if applied:
                _NATIVE_GRAPH_VERSIONS[scope] = max(
                    _NATIVE_GRAPH_VERSIONS.get(scope, 0), expected + applied
                )

            # Refresh the read-policy cache for governed objects that committed, each
            # under ITS OWN envelope's source ACL/classification (a page may mix them).
            from ...protocols.source_connectors.permission_sync import sync_access

            for offset, envelope in enumerate(envelopes):
                if str(engine_results[offset].get("status")) != "applied":
                    continue
                try:
                    for object_id in sorted(set(per_governed[offset])):
                        sync_access(
                            object_id,
                            envelope.source_acl,
                            classification=envelope.classification,
                        )
                except Exception:  # noqa: BLE001 - durable policy stays fail-closed
                    logger.warning("native policy cache refresh is pending")
                    break

            return [
                _batch_entry_result(
                    envelope,
                    engine_results[offset],
                    per_counts[offset],
                    per_node[offset],
                    per_cursor_advanced[offset],
                )
                for offset, envelope in enumerate(envelopes)
            ]
        raise _NativeOccRetryBudgetExhausted(conflict_sequence)


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
    prepared: list[tuple[int, ChangeEnvelope]] = []
    stopped_at: int | None = None
    for index, envelope in enumerate(envelopes):
        try:
            gated = _privacy_gate(envelope)
        except ValueError:
            logger.warning("native ChangeEnvelope privacy gate rejected identity")
            results[index] = {
                "status": "rejected",
                "reason": "persistence privacy gate rejected an unsafe identity",
                "watermark_advanced": False,
            }
            stopped_at = index
            break
        violations = _validate_envelope(gated)
        if violations:
            results[index] = {
                "status": "rejected",
                "envelope_id": gated.envelope_id,
                "idempotency_key": gated.idempotency_key,
                "connector": gated.connector,
                "operation": gated.operation,
                "watermark_advanced": False,
                "violations": violations,
            }
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

    if not prepared:
        return results

    try:
        authority = _resolve_native_authority(engine)
        authority, session = _native_session(authority, prepared[0][1])
        batch_results = _apply_native_change_envelopes(
            authority, session, [envelope for _, envelope in prepared]
        )
    except NativeChangeEnvelopeUnavailable:
        logger.info(
            "native ApplyChangeEnvelopes unavailable; falling back to per-record ingestion"
        )
        return [ingest_envelope(engine, envelope) for envelope in envelopes]
    except (PermissionError, ValueError) as exc:
        # Log the real cause, not just the exception type: type(exc).__name__
        # alone would be the exact swallowed-error antipattern
        # scripts/security/check_swallowed_errors.py exists to catch. Pass exc itself
        # (not a second, redundant type(exc).__name__ arg) -- core/log_privacy.py's
        # _sanitize_value already renders a raw exception as "Type: message"
        # while still redacting paths/etc.
        logger.warning("native ChangeEnvelope batch rejected (%s)", exc)
        for index, envelope in prepared:
            results[index] = {
                "status": "rejected",
                "error": type(exc).__name__,
                "envelope_id": envelope.envelope_id,
                "watermark_advanced": False,
            }
        return results
    except _NativeOccRetryBudgetExhausted as exc:
        logger.warning(
            "native ChangeEnvelope batch OCC retry budget exhausted (conflict_sequence=%s)",
            ",".join(exc.conflicts),
        )
        for index, envelope in prepared:
            results[index] = {
                "status": "failed",
                "error": "NativeChangeEnvelopeConflictExhausted",
                "envelope_id": envelope.envelope_id,
                "watermark_advanced": False,
            }
        return results
    except Exception as exc:  # noqa: BLE001 - never fall back after a native failure
        logger.warning(
            "native ChangeEnvelope batch commit failed (%s)", type(exc).__name__
        )
        for index, envelope in prepared:
            results[index] = {
                "status": "failed",
                "error": type(exc).__name__,
                "envelope_id": envelope.envelope_id,
                "watermark_advanced": False,
            }
        return results

    for (index, _envelope), result in zip(prepared, batch_results, strict=True):
        results[index] = result
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
) -> dict[str, Any]:
    """Commit a connector-produced multi-node graph slice atomically.

    The first entity is the envelope's primary object; remaining entities and
    all relationships are attached as governed auxiliary material.  When the
    source has no explicit version, a deterministic content digest supplies the
    idempotent version without advancing a source cursor. The Epistemic Graph
    authority is resolved from ``engine`` and missing native support fails closed.

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
    if isinstance(engine, NativeChangeEnvelopeEngineProxy):
        engine = engine.authority

    if not entities:
        # Edge-only derived/extractor batches still need a governed primary
        # object. The deterministic marker owns delivery identity without
        # inventing a source cursor or leaking endpoint material into its id.
        marker_digest = hashlib.sha256(
            json.dumps(
                {
                    "connector": connector,
                    "source_instance": source_instance,
                    "relationships": relationships,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        entities = [
            {
                "id": f"source-materialization:{marker_digest}",
                "node_type": "SourceMaterialization",
                "source_system": connector,
                "relationship_count": len(relationships),
            }
        ]

    primary = dict(entities[0])
    if len(entities) > 1:
        primary["_nodes"] = [dict(item) for item in entities[1:]]
    if relationships:
        primary["_links"] = [dict(item) for item in relationships]
    material_version = hashlib.sha256(
        json.dumps(
            {"entities": entities, "relationships": relationships},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
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
        idempotency_key="",
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


def ingest_envelope(engine: Any, envelope: ChangeEnvelope) -> dict[str, Any]:
    """Commit one external change through native ``ApplyChangeEnvelope``.

    Validation and persistence privacy run before authority resolution. Missing
    native capability fails closed.
    """
    if isinstance(engine, NativeChangeEnvelopeEngineProxy):
        engine = engine.authority
    try:
        envelope = _privacy_gate(envelope)
    except ValueError:
        logger.warning("native ChangeEnvelope privacy gate rejected identity")
        return {
            "status": "rejected",
            "reason": "persistence privacy gate rejected an unsafe identity",
            "watermark_advanced": False,
        }
    base = {
        "envelope_id": envelope.envelope_id,
        "idempotency_key": envelope.idempotency_key,
        "connector": envelope.connector,
        "operation": envelope.operation,
        "watermark_advanced": False,
        "native_atomic": True,
    }
    violations = _validate_envelope(envelope)
    if violations:
        return {**base, "status": "rejected", "violations": violations}

    try:
        authority = _resolve_native_authority(engine)
        authority, session = _native_session(authority, envelope)
        return _apply_native_change_envelope(authority, session, envelope)
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
        # Same reasoning as the rejection path above: the class name alone
        # cannot tell an operator WHICH commit failed or why.
        logger.warning(
            "native ChangeEnvelope commit failed (%s): %s", type(exc).__name__, exc
        )
        return {
            **base,
            "status": "failed",
            "error": type(exc).__name__,
            "reason": str(exc),
        }
