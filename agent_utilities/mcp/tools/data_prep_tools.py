"""Governed data-preparation tools over the NE-108 Arrow kernel.

``graph_data_prep`` is the one MCP/REST action-routed surface for
``profile_dataset``, ``clean_dataset``, ``validate_prepared`` and
``commit_prepared``.  The module deliberately owns only the served boundary:
the Arrow transforms, model validation and evidence format remain in
``agent_utilities.data_prep`` and persistence remains in the native
``ChangeEnvelope`` authority.

The artifact, model and policy authorities are composed at the process-owned
engine startup boundary.  The graph-native provider reads durable
``AssetOccurrence``/``Artifact`` metadata and content-addressed blobs, and
accepts row models only through a typed, digest-pinned runtime configuration.
It never falls back to a caller-selected import path, an inline Arrow IPC
value, or a second store.  Missing native/model/policy registration fails
closed at the served boundary.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Literal, NoReturn, Protocol

from pydantic import Field, ValidationError, model_validator

from agent_utilities.data_prep import (
    ArrowAdapter,
    CleanPipeline,
    CleanPlan,
    Digest,
    LocalProfile,
    OpaqueReference,
    PrepEvidence,
    ProfileResult,
    RowModelRegistry,
    schema_digest,
)
from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
from agent_utilities.mcp import kg_server
from agent_utilities.models.company_brain import DataClassification
from agent_utilities.protocols.epistemic_operations import ProtocolModel
from agent_utilities.protocols.source_connectors.base import ExternalAccess

logger = logging.getLogger(__name__)

# Request and artifact limits are deliberately below the much larger kernel
# maxima.  A served caller must not turn one MCP request into a memory-pressure
# or response-amplification primitive.
_MAX_PARAMS_BYTES = 4 * 1024 * 1024
_MAX_PLAN_BYTES = 128 * 1024
_MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
_MAX_INLINE_ROWS = 10_000
_MAX_INLINE_COLUMNS = 256
_MAX_INLINE_CELL_BYTES = 64 * 1024
_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_ARTIFACT_ROWS = 1_000_000
_MAX_ARTIFACT_COLUMNS = 1_024
_MAX_ARTIFACT_DEPTH = 16
_MAX_WALL_TIME_MS = 300_000
_PREPARED_REF_MAX = 4_096
_RECEIPT_VERSION = "data-prep-receipt.v1"
_RECEIPT_ENDPOINT = "data-prep"
_RECEIPT_TTL_MS = 5 * 60 * 1000
_RUNTIME_PROVIDER_ATTR = "_data_prep_runtime_provider"


class DataPrepToolError(ValueError):
    """A request failed the governed data-prep boundary."""


class ArtifactAuthorityUnavailable(DataPrepToolError):
    """The required tenant-bound artifact authority is not installed."""


class NativeCommitUnavailable(DataPrepToolError):
    """Native ChangeEnvelope or ICV admission is unavailable."""


def _canonical_json(value: Any) -> bytes:
    """Serialize a JSON-compatible value without locale or spacing variance."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataPrepToolError("value is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _shape_digest(table: Any) -> str:
    """Digest only the bounded output shape, not row values or column names."""

    return _sha256_bytes(
        _canonical_json(
            {
                "schema_digest": schema_digest(table),
                "rows": int(table.num_rows),
                "columns": int(table.num_columns),
            }
        )
    )


def _shape_ref(digest: str) -> str:
    return f"shape:prepared:{digest.removeprefix('sha256:')}"


def _schema_ref(digest: str) -> str:
    return f"schema:prepared:{digest.removeprefix('sha256:')}"


def _canonical_arrow_bytes(table: Any) -> bytes:
    """Render an Arrow stream deterministically for digest and native storage.

    Metadata is deliberately removed and chunks are combined before writing.
    The result is held only for this request; clean/validate never hand these
    bytes to an authority or persist them.
    """

    try:
        import pyarrow as pa

        normalized = table.combine_chunks().replace_schema_metadata(None)
        sink = pa.BufferOutputStream()
        options = pa.ipc.IpcWriteOptions(compression=None)
        with pa.ipc.new_stream(sink, normalized.schema, options=options) as writer:
            writer.write_table(normalized)
        return sink.getvalue().to_pybytes()
    except Exception as exc:  # noqa: BLE001 - normalize Arrow dependency failures
        raise DataPrepToolError(
            "prepared output cannot be serialized as Arrow"
        ) from exc


def _evidence_payload(evidence: PrepEvidence) -> tuple[dict[str, Any], str]:
    payload = evidence.model_dump(mode="json")
    encoded = _canonical_json(payload)
    if len(encoded) > _MAX_EVIDENCE_BYTES:
        raise DataPrepToolError("preparation evidence exceeds the bounded size")
    return payload, _sha256_bytes(encoded)


class DataPrepAction(StrEnum):
    """The closed action vocabulary harvested into the ACP/MCP manifest."""

    PROFILE = "profile_dataset"
    CLEAN = "clean_dataset"
    VALIDATE = "validate_prepared"
    COMMIT = "commit_prepared"


class PrepBudget(ProtocolModel):
    """Per-call bounds checked before an artifact is decoded or transformed."""

    max_rows: int = Field(ge=1, le=_MAX_ARTIFACT_ROWS)
    max_columns: int = Field(ge=1, le=_MAX_ARTIFACT_COLUMNS)
    max_compressed_bytes: int = Field(ge=1, le=_MAX_ARTIFACT_BYTES)
    max_decoded_bytes: int = Field(ge=1, le=_MAX_ARTIFACT_BYTES)
    max_depth: int = Field(ge=0, le=_MAX_ARTIFACT_DEPTH)
    max_wall_time_ms: int = Field(default=120_000, ge=1, le=_MAX_WALL_TIME_MS)


class PrepRequest(ProtocolModel):
    """Typed, immutable tool request; caller identity is intentionally absent."""

    schema_version: str = Field(pattern=r"^data-prep-tool\.v1$")
    plan: dict[str, Any]
    plan_ref: OpaqueReference
    plan_digest: Digest
    model_ref: OpaqueReference
    model_digest: Digest
    schema_ref: OpaqueReference
    schema_digest: Digest
    shape_ref: OpaqueReference
    shape_digest: Digest
    artifact_ref: OpaqueReference | None = None
    # Receipts carry signed, content-bound metadata and are intentionally larger
    # than ordinary opaque catalog refs.  They are not a persisted artifact key.
    prepared_ref: str | None = Field(
        default=None,
        min_length=1,
        max_length=_PREPARED_REF_MAX,
        pattern=r"^prep:v1:[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$",
    )
    records: list[dict[str, Any]] | None = None
    expected_output_schema_ref: OpaqueReference | None = None
    expected_output_schema_digest: Digest | None = None
    expected_output_shape_ref: OpaqueReference | None = None
    expected_output_shape_digest: Digest | None = None
    budget: PrepBudget

    @model_validator(mode="after")
    def input_source_is_explicit(self) -> PrepRequest:
        if (self.artifact_ref is None) == (self.records is None):
            raise ValueError(
                "exactly one tenant-bound artifact_ref or bounded records input is required"
            )
        if self.records is not None:
            _validate_inline_records(self.records)
        return self


@dataclass(frozen=True, slots=True)
class PreparedReceipt:
    """Tenant-bound, process-signed receipt for one deterministic clean result.

    The receipt contains no Arrow bytes and is never persisted.  A commit
    recomputes the result from the authoritative input and compares every
    identity/digest below before it asks native storage to retain the bytes.
    """

    tenant_id: str
    artifact_ref: str
    input_content_digest: str
    input_schema_ref: str
    input_schema_digest: str
    input_shape_ref: str
    input_shape_digest: str
    plan_ref: str
    plan_digest: str
    model_ref: str
    model_digest: str
    output_content_digest: str
    output_schema_ref: str
    output_schema_digest: str
    output_shape_ref: str
    output_shape_digest: str
    evidence_digest: str
    policy_version: str
    native_atomic: bool
    issued_at_ms: int
    actor_id: str = ""
    endpoint: str = _RECEIPT_ENDPOINT
    expires_at_ms: int = 0
    token: str = ""

    def _body(self) -> dict[str, Any]:
        return {
            "receipt_version": _RECEIPT_VERSION,
            "tenant_id": self.tenant_id,
            "artifact_ref": self.artifact_ref,
            "input_content_digest": self.input_content_digest,
            "input_schema_ref": self.input_schema_ref,
            "input_schema_digest": self.input_schema_digest,
            "input_shape_ref": self.input_shape_ref,
            "input_shape_digest": self.input_shape_digest,
            "plan_ref": self.plan_ref,
            "plan_digest": self.plan_digest,
            "model_ref": self.model_ref,
            "model_digest": self.model_digest,
            "output_content_digest": self.output_content_digest,
            "output_schema_ref": self.output_schema_ref,
            "output_schema_digest": self.output_schema_digest,
            "output_shape_ref": self.output_shape_ref,
            "output_shape_digest": self.output_shape_digest,
            "evidence_digest": self.evidence_digest,
            "policy_version": self.policy_version,
            "native_atomic": self.native_atomic,
            "issued_at_ms": self.issued_at_ms,
            "actor_id": self.actor_id,
            "endpoint": self.endpoint,
            "expires_at_ms": self.expires_at_ms,
        }

    def encode(self) -> str:
        from agent_utilities.security.run_token import mint_token

        issued_at_ms = int(self.issued_at_ms or time.time() * 1000)
        expires_at_ms = int(self.expires_at_ms or issued_at_ms + _RECEIPT_TTL_MS)
        if (
            not self.actor_id
            or not self.tenant_id
            or self.endpoint != _RECEIPT_ENDPOINT
        ):
            raise DataPrepToolError("prepared receipt authority is incomplete")
        if (
            expires_at_ms <= issued_at_ms
            or expires_at_ms - issued_at_ms > _RECEIPT_TTL_MS
        ):
            raise DataPrepToolError("prepared receipt expiry is invalid")
        body_receipt = replace(
            self,
            issued_at_ms=issued_at_ms,
            expires_at_ms=expires_at_ms,
            token="",
        )
        body = _canonical_json(body_receipt._body())
        # The existing run-token signer owns the configured secret, expiry and
        # actor/tenant binding.  Its run id is the digest of this exact receipt
        # body, so changing any content-bound field invalidates the token.
        body_digest = _sha256_bytes(body).removeprefix("sha256:")
        token = self.token or mint_token(
            run_id=body_digest,
            project=_RECEIPT_VERSION,
            endpoints=(_RECEIPT_ENDPOINT,),
            operations=("commit_prepared",),
            ttl_seconds=(expires_at_ms - issued_at_ms) / 1000,
            actor_id=self.actor_id,
            tenant_id=self.tenant_id,
            now=issued_at_ms / 1000,
        )
        encoded = base64.urlsafe_b64encode(body).decode("ascii").rstrip("=")
        token_encoded = (
            base64.urlsafe_b64encode(token.encode("utf-8")).decode("ascii").rstrip("=")
        )
        return f"prep:v1:{encoded}.{token_encoded}"

    @classmethod
    def decode(cls, value: str) -> PreparedReceipt:
        if not isinstance(value, str) or not value.startswith("prep:v1:"):
            raise DataPrepToolError("prepared receipt is malformed")
        try:
            encoded, token_encoded = value.removeprefix("prep:v1:").split(".", 1)
            body = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            if len(body) > _MAX_PARAMS_BYTES:
                raise DataPrepToolError("prepared receipt exceeds the bounded size")
            if base64.urlsafe_b64encode(body).decode("ascii").rstrip("=") != encoded:
                raise DataPrepToolError("prepared receipt encoding is non-canonical")
            payload = json.loads(body)
            token = base64.urlsafe_b64decode(
                token_encoded + "=" * (-len(token_encoded) % 4)
            ).decode("utf-8")
            if (
                base64.urlsafe_b64encode(token.encode("utf-8"))
                .decode("ascii")
                .rstrip("=")
                != token_encoded
            ):
                raise DataPrepToolError("prepared receipt encoding is non-canonical")
            from agent_utilities.security.run_token import validate_token

            run_token = validate_token(
                token,
                endpoint=_RECEIPT_ENDPOINT,
                operation="commit_prepared",
            )
        except Exception as exc:  # noqa: BLE001 - privacy-safe receipt boundary
            if isinstance(exc, DataPrepToolError):
                raise
            raise DataPrepToolError("prepared receipt is malformed") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("receipt_version") != _RECEIPT_VERSION
        ):
            raise DataPrepToolError("prepared receipt version is unsupported")
        payload = dict(payload)
        payload.pop("receipt_version", None)
        allowed = {
            "tenant_id",
            "artifact_ref",
            "input_content_digest",
            "input_schema_ref",
            "input_schema_digest",
            "input_shape_ref",
            "input_shape_digest",
            "plan_ref",
            "plan_digest",
            "model_ref",
            "model_digest",
            "output_content_digest",
            "output_schema_ref",
            "output_schema_digest",
            "output_shape_ref",
            "output_shape_digest",
            "evidence_digest",
            "policy_version",
            "native_atomic",
            "issued_at_ms",
            "actor_id",
            "endpoint",
            "expires_at_ms",
        }
        if set(payload) != allowed:
            raise DataPrepToolError("prepared receipt fields are invalid")
        try:
            receipt = cls(**payload)
        except (TypeError, ValueError) as exc:
            raise DataPrepToolError("prepared receipt fields are invalid") from exc
        if (
            not receipt.tenant_id
            or not receipt.artifact_ref
            or not receipt.policy_version
            or not receipt.actor_id
            or receipt.endpoint != _RECEIPT_ENDPOINT
        ):
            raise DataPrepToolError("prepared receipt authority is incomplete")
        body_digest = _sha256_bytes(_canonical_json(receipt._body())).removeprefix(
            "sha256:"
        )
        if run_token.run_id != body_digest:
            raise DataPrepToolError("prepared receipt token binding is invalid")
        if (
            run_token.actor_id != receipt.actor_id
            or run_token.tenant_id != receipt.tenant_id
            or run_token.project != _RECEIPT_VERSION
        ):
            raise DataPrepToolError("prepared receipt identity binding is invalid")
        if abs(int(run_token.expires_at * 1000) - receipt.expires_at_ms) > 1:
            raise DataPrepToolError("prepared receipt expiry binding is invalid")
        if (
            receipt.expires_at_ms <= receipt.issued_at_ms
            or receipt.expires_at_ms - receipt.issued_at_ms > _RECEIPT_TTL_MS
        ):
            raise DataPrepToolError("prepared receipt expiry is invalid")
        if receipt.native_atomic is not True:
            raise NativeCommitUnavailable(
                "prepared receipt does not prove native atomic admission"
            )
        return replace(receipt, token=token)


@dataclass(frozen=True, slots=True)
class ArtifactACL:
    """Resolved ACL facts; these never come from a tool payload."""

    is_public: bool
    principal_ids: tuple[str, ...] = ()
    principal_emails: tuple[str, ...] = ()
    group_ids: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    markings: tuple[str, ...] = ()

    @property
    def read_roles(self) -> tuple[str, ...]:
        """Downstream ``ExternalAccess.read_roles`` spelling."""

        return self.roles

    @classmethod
    def from_value(cls, value: Any) -> ArtifactACL:
        if not isinstance(value, Mapping):
            raise DataPrepToolError("artifact ACL proof is unavailable")
        is_public = value.get("is_public")
        principal_ids = value.get(
            "principal_ids", value.get("user_ids", value.get("principals", ()))
        )
        principal_emails = value.get("principal_emails", value.get("user_emails", ()))
        group_ids = value.get("group_ids", ())
        roles = value.get("roles", value.get("read_roles", ()))
        markings = value.get("markings", ())
        if not isinstance(is_public, bool):
            raise DataPrepToolError("artifact ACL proof is unavailable")
        fields = (principal_ids, principal_emails, group_ids, roles, markings)
        if not all(
            isinstance(items, (list, tuple))
            and all(isinstance(item, str) and item.strip() for item in items)
            for items in fields
        ):
            raise DataPrepToolError("artifact ACL proof is unavailable")
        if any("@" in item for item in principal_ids):
            raise DataPrepToolError("principal IDs must not be supplied as emails")
        if any("@" not in item for item in principal_emails):
            raise DataPrepToolError("artifact user email ACL proof is unavailable")
        return cls(
            is_public=is_public,
            principal_ids=tuple(sorted(set(principal_ids))),
            principal_emails=tuple(sorted(set(principal_emails))),
            group_ids=tuple(sorted(set(group_ids))),
            roles=tuple(sorted(set(roles))),
            markings=tuple(sorted(set(markings))),
        )


@dataclass(frozen=True, slots=True)
class ResolvedArtifact:
    """One fully governed Arrow artifact returned by the existing authority."""

    artifact_ref: str
    content_ref: str
    media_type: str
    schema_ref: str
    schema_digest: str
    shape_ref: str
    shape_digest: str
    tenant_id: str
    owner_id: str
    acl: ArtifactACL
    content_digest: str
    classification: DataClassification
    retention: str | None
    legal_hold: bool
    policy_version: str
    expires_at_ms: int
    compressed_bytes: int
    decoded_bytes: int
    rows: int
    columns: int
    nesting_depth: int
    table: Any


class DataPrepAuthority(Protocol):
    """Process-owned artifact, model and native persistence authority."""

    def artifact(
        self,
        artifact_ref: str,
        *,
        session: GraphSession,
        budget: PrepBudget,
    ) -> ResolvedArtifact: ...

    def records_artifact(
        self,
        records: list[dict[str, Any]],
        *,
        session: GraphSession,
        budget: PrepBudget,
    ) -> ResolvedArtifact: ...

    def approved_models(self, *, session: GraphSession) -> RowModelRegistry: ...

    def inline_records_policy_available(self, *, session: GraphSession) -> bool: ...

    def preview_ref(
        self,
        source: ResolvedArtifact,
        *,
        output_table: Any,
        evidence: PrepEvidence,
        request: PrepRequest,
        session: GraphSession,
    ) -> PreparedReceipt: ...

    def output_governance(
        self,
        source: ResolvedArtifact,
        *,
        output_table: Any,
        output_schema_ref: str,
        output_schema_digest: str,
        output_shape_ref: str,
        output_shape_digest: str,
        session: GraphSession,
    ) -> ResolvedArtifact: ...

    def native_engine(self, *, session: GraphSession) -> Any: ...

    def icv_policy_available(self, *, session: GraphSession) -> bool: ...

    def native_atomic_available(self, *, session: GraphSession) -> bool: ...

    def store_blob(
        self, payload: bytes, *, media_type: str, session: GraphSession
    ) -> str: ...

    def incref_blob(self, digest: str, *, session: GraphSession) -> None: ...

    def unref_blob(self, digest: str, *, session: GraphSession) -> None: ...


class _DefaultAuthority:
    """Fail closed when the process native authority is not available."""

    def _missing(self) -> NoReturn:
        raise ArtifactAuthorityUnavailable(
            "governed data-prep artifact authority is unavailable"
        )

    def artifact(
        self, artifact_ref: str, *, session: GraphSession, budget: PrepBudget
    ) -> ResolvedArtifact:
        del artifact_ref, session, budget
        self._missing()

    def records_artifact(
        self,
        records: list[dict[str, Any]],
        *,
        session: GraphSession,
        budget: PrepBudget,
    ) -> ResolvedArtifact:
        del records, session, budget
        self._missing()

    def approved_models(self, *, session: GraphSession) -> RowModelRegistry:
        del session
        self._missing()

    def inline_records_policy_available(self, *, session: GraphSession) -> bool:
        del session
        return False

    def preview_ref(
        self,
        source: ResolvedArtifact,
        *,
        output_table: Any,
        evidence: PrepEvidence,
        request: PrepRequest,
        session: GraphSession,
    ) -> PreparedReceipt:
        del source, output_table, evidence, request, session
        self._missing()

    def output_governance(
        self,
        source: ResolvedArtifact,
        *,
        output_table: Any,
        output_schema_ref: str,
        output_schema_digest: str,
        output_shape_ref: str,
        output_shape_digest: str,
        session: GraphSession,
    ) -> ResolvedArtifact:
        del (
            source,
            output_table,
            output_schema_ref,
            output_schema_digest,
            output_shape_ref,
            output_shape_digest,
            session,
        )
        self._missing()

    def native_engine(self, *, session: GraphSession) -> Any:
        del session
        self._missing()

    def icv_policy_available(self, *, session: GraphSession) -> bool:
        del session
        return False

    def native_atomic_available(self, *, session: GraphSession) -> bool:
        del session
        return False

    def store_blob(
        self, payload: bytes, *, media_type: str, session: GraphSession
    ) -> str:
        del payload, media_type, session
        self._missing()

    def incref_blob(self, digest: str, *, session: GraphSession) -> None:
        del digest, session
        self._missing()

    def unref_blob(self, digest: str, *, session: GraphSession) -> None:
        del digest, session
        self._missing()


@dataclass(frozen=True, slots=True)
class DataPrepModelAuthority:
    """Typed process configuration for one immutable row-model registry."""

    registry: RowModelRegistry
    config_digest: str
    connector_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.registry, RowModelRegistry):
            raise TypeError("data-prep model authority requires RowModelRegistry")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.config_digest):
            raise ValueError("data-prep model configuration digest is invalid")
        if not isinstance(self.connector_version, str) or not self.connector_version:
            raise ValueError("data-prep model connector version is required")


@dataclass(frozen=True, slots=True)
class DataPrepRuntimeConfig:
    """Process-owned data-prep configuration consumed at engine startup.

    The configuration contains already-trusted objects, not import paths or
    request metadata.  Deployments may expose the same object from their
    authoritative backend/configuration seam as ``data_prep_runtime_config``;
    the graph-native provider below never creates a caller-selected registry or
    payload store.
    """

    model_authority: DataPrepModelAuthority | None = None
    inline_records_policy: Mapping[str, Any] | None = None
    policy_authority: Any | None = None
    icv_policy_available: bool = False


class _GraphNativeDataPrepPolicy:
    """Minimal process-owned output policy over graph-native source metadata."""

    def allow_inline_records(self, *, session: GraphSession, policy: Any) -> bool:
        return bool(
            isinstance(policy, Mapping)
            and policy.get("enabled") is True
            and str(policy.get("tenant_id") or "") == session.tenant
        )

    def govern_output(
        self,
        source: ResolvedArtifact,
        *,
        output_table: Any,
        output_schema_ref: str,
        output_schema_digest: str,
        output_shape_ref: str,
        output_shape_digest: str,
        session: GraphSession,
    ) -> ResolvedArtifact:
        """Bind the derived bytes to the source governance snapshot.

        The input and output are distinct immutable objects: the output digest,
        schema and shape are recomputed from the actual Arrow result, while the
        source tenant/ACL/classification/retention/hold/policy are carried
        forward.  The service's independent downgrade gate then checks this
        result against the source before any native write.
        """

        if source.tenant_id != session.tenant:
            raise PermissionError("output governance tenant does not match session")
        if source.policy_version != str(session.policy_version or ""):
            raise DataPrepToolError("source policy version is stale")
        output_bytes = _canonical_arrow_bytes(output_table)
        output_digest = _sha256_bytes(output_bytes)
        output_ref = f"prepared:{output_digest.removeprefix('sha256:')}"
        return replace(
            source,
            artifact_ref=output_ref,
            content_ref=output_ref,
            schema_ref=output_schema_ref,
            schema_digest=output_schema_digest,
            shape_ref=output_shape_ref,
            shape_digest=output_shape_digest,
            content_digest=output_digest,
            compressed_bytes=len(output_bytes),
            decoded_bytes=int(output_table.nbytes),
            rows=int(output_table.num_rows),
            columns=int(output_table.num_columns),
            nesting_depth=_table_depth(output_table),
            table=output_table,
        )


def _runtime_sources(engine: Any) -> tuple[Any, ...]:
    """Return only process-owned runtime/configuration sources, de-duplicated."""

    backend = getattr(engine, "backend", None)
    sources: list[Any] = [
        getattr(engine, "data_prep_runtime_config", None),
        getattr(engine, "data_prep_runtime", None),
        engine,
        getattr(backend, "data_prep_runtime_config", None),
        getattr(backend, "data_prep_runtime", None),
        backend,
        getattr(backend, "_authority", None),
    ]
    # AgentConfig is a process-owned configuration authority.  It is inspected
    # only for explicitly exposed data-prep fields; no request can select an
    # object or import path through this fallback.
    try:
        from agent_utilities.core.config import config as process_config

        sources.append(process_config)
    except (
        Exception
    ):  # pragma: no cover - configuration may be unavailable at import time
        pass
    result: list[Any] = []
    seen: set[int] = set()
    for source in sources:
        if source is not None and id(source) not in seen:
            seen.add(id(source))
            result.append(source)
    return tuple(result)


def _runtime_model_authority(engine: Any) -> DataPrepModelAuthority | None:
    """Resolve only the typed process configuration seam for row models."""

    for source in _runtime_sources(engine):
        if isinstance(source, DataPrepModelAuthority):
            return source
        if isinstance(source, DataPrepRuntimeConfig):
            return source.model_authority
        candidate = getattr(source, "data_prep_model_authority", None)
        if isinstance(candidate, DataPrepModelAuthority):
            return candidate
    return None


def _runtime_policy(engine: Any) -> Any:
    for source in _runtime_sources(engine):
        if isinstance(source, DataPrepRuntimeConfig):
            return source.policy_authority
        candidate = getattr(source, "data_prep_policy_authority", None)
        if candidate is not None:
            return candidate
    return None


def _runtime_inline_policy(engine: Any) -> Any:
    for source in _runtime_sources(engine):
        if isinstance(source, DataPrepRuntimeConfig):
            return source.inline_records_policy
        candidate = getattr(source, "data_prep_inline_records_policy", None)
        if candidate is not None:
            return candidate
    return None


def _native_digest(value: Any) -> str:
    if not isinstance(value, str):
        raise ArtifactAuthorityUnavailable("native artifact digest is unavailable")
    digest = value if value.startswith("sha256:") else f"sha256:{value}"
    if len(digest) != len("sha256:") + 64 or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", digest
    ):
        raise ArtifactAuthorityUnavailable("native artifact digest is invalid")
    return digest


def _native_ref(value: Any, *, fallback: str) -> str:
    if value is None:
        return fallback
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}", value
    ):
        raise ArtifactAuthorityUnavailable("native artifact reference is invalid")
    return value


def _native_acl(props: Mapping[str, Any], *, owner_id: str) -> ArtifactACL:
    raw = props.get("external_access", props.get("acl"))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ArtifactAuthorityUnavailable(
                "native artifact ACL is invalid"
            ) from exc
    if raw is not None:
        return ArtifactACL.from_value(raw)
    # First-party MediaStore writes durable _owner_id/_shared_scope markers,
    # while connector artifacts carry ExternalAccess.  Translate those native
    # markers into the same explicit ACL facts without treating an owner id as
    # an email address or inferring public access from a missing ACL.
    generated = {
        "is_public": props.get("_shared_scope") == "commons"
        and props.get("classification") == DataClassification.PUBLIC.value,
        "principal_ids": [owner_id] if owner_id else [],
        "principal_emails": [],
        "group_ids": props.get("group_ids", ()),
        "roles": props.get("read_roles", props.get("roles", ())),
        "markings": props.get("markings", ()),
    }
    return ArtifactACL.from_value(generated)


class _GraphNativeDataPrepProvider:
    """Concrete provider over the authoritative graph node/blob substrate.

    No artifact bytes or rows are cached here.  Source metadata is read from a
    durable ``AssetOccurrence``/``Artifact`` node and bytes are fetched through
    the native content-addressed blob client.  Inline rows are admitted only by
    an explicit process-owned policy object; they are never stored by this
    provider.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._compute = (
            getattr(engine, "graph_compute", None)
            or getattr(engine, "graph", None)
            or engine
        )
        self._client = (
            getattr(self._compute, "client", None)
            or getattr(self._compute, "_client", None)
            or getattr(engine, "client", None)
        )
        self._media_store = None
        if getattr(self._compute, "_client", None) is not None:
            try:
                from agent_utilities.knowledge_graph.memory.media_store import (
                    MediaStore,
                )

                self._media_store = MediaStore(self._compute)
            except Exception:  # pragma: no cover - native media seam is diagnosed below
                self._media_store = None
        model_authority = _runtime_model_authority(engine)
        self._models = model_authority.registry if model_authority is not None else None
        self._inline_policy = _runtime_inline_policy(engine)
        self._policy = _runtime_policy(engine)
        self.diagnostics = self._diagnostics()

    @classmethod
    def from_engine(cls, engine: Any) -> _GraphNativeDataPrepProvider | None:
        provider = cls(engine)
        client = provider._client
        nodes = getattr(client, "nodes", None)
        blob = getattr(client, "blob", None)
        if not callable(getattr(nodes, "properties", None)):
            return None
        if not callable(getattr(blob, "fetch", None)):
            return None
        return provider

    def _diagnostics(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self._models is None:
            missing.append("approved row-model registry")
        if self._policy is None:
            missing.append("data-prep policy authority")
        if self._client is None:
            missing.append("native graph client")
        return tuple(missing)

    def artifact(
        self, artifact_ref: str, *, session: GraphSession, budget: PrepBudget
    ) -> ResolvedArtifact:
        if not isinstance(artifact_ref, str) or not re.fullmatch(
            r"(?:occurrence|artifact):[A-Za-z0-9._:/-]+", artifact_ref
        ):
            raise DataPrepToolError(
                "artifact reference is not an approved opaque graph ref"
            )
        # Native point reads and blob calls must inherit the same verified
        # GraphSession.  In particular, a caller-supplied session may never
        # cause a root-graph client to read another tenant's metadata before
        # the provider's own immutable governance checks run.
        with self._verified_session_scope(session):
            scoped_engine = self._scoped_engine(session)
            props = self._node_properties(artifact_ref, engine=scoped_engine)
            if not isinstance(props, Mapping):
                raise ArtifactAuthorityUnavailable(
                    "native artifact metadata is invalid"
                )
            node_type, tenant_id, policy_version, owner_id, classification, acl = (
                self._authorize_metadata(props, session=session, budget=budget)
            )
            digest = _native_digest(
                props.get("content_digest")
                or props.get("content_hash")
                or props.get("digest")
                or props.get("blob_digest")
            )
            payload = self._fetch_blob(
                digest.removeprefix("sha256:"), engine=scoped_engine
            )
            if not isinstance(payload, bytes):
                raise ArtifactAuthorityUnavailable("native artifact bytes are invalid")
        if len(payload) > budget.max_compressed_bytes:
            raise DataPrepToolError(
                "artifact compressed size exceeds the request budget"
            )
        media_type = str(props.get("media_type") or props.get("mime_type") or "")
        if media_type not in {
            "application/vnd.apache.arrow.stream",
            "application/vnd.apache.arrow.file",
        }:
            raise DataPrepToolError("artifact media type is not an approved Arrow type")
        table = self._decode_arrow(payload, media_type=media_type, budget=budget)
        actual_digest = _sha256_bytes(payload)
        if actual_digest != digest:
            raise ArtifactAuthorityUnavailable(
                "native artifact content fingerprint is invalid"
            )
        actual_schema = schema_digest(table)
        stored_schema = props.get("schema_digest")
        schema_value = _native_digest(stored_schema) if stored_schema else actual_schema
        if schema_value != actual_schema:
            raise DataPrepToolError(
                "artifact schema fingerprint does not match its content"
            )
        actual_shape = _shape_digest(table)
        stored_shape = props.get("shape_digest")
        shape_value = _native_digest(stored_shape) if stored_shape else actual_shape
        if shape_value != actual_shape:
            raise DataPrepToolError(
                "artifact shape fingerprint does not match its content"
            )
        compressed_bytes = self._metadata_int(
            props, "compressed_bytes", len(payload), "compressed size"
        )
        compressed_bytes = self._metadata_int(
            props, "file_size_bytes", compressed_bytes, "compressed size"
        )
        decoded_bytes = self._metadata_int(
            props, "decoded_bytes", int(table.nbytes), "decoded size"
        )
        rows = self._metadata_int(props, "rows", int(table.num_rows), "row count")
        columns = self._metadata_int(
            props, "columns", int(table.num_columns), "column count"
        )
        depth = self._metadata_int(
            props, "nesting_depth", _table_depth(table), "nesting depth"
        )
        expires_raw = props.get("expires_at_ms", 0)
        legal_hold = props.get("legal_hold", False)
        return ResolvedArtifact(
            artifact_ref=artifact_ref,
            content_ref=digest,
            media_type=media_type,
            schema_ref=_native_ref(
                props.get("schema_ref"), fallback=_schema_ref(schema_value)
            ),
            schema_digest=schema_value,
            shape_ref=_native_ref(
                props.get("shape_ref"), fallback=_shape_ref(shape_value)
            ),
            shape_digest=shape_value,
            tenant_id=tenant_id,
            owner_id=owner_id,
            acl=acl,
            content_digest=digest,
            classification=classification,
            retention=(
                str(props["retention"]) if props.get("retention") is not None else None
            ),
            legal_hold=legal_hold,
            policy_version=policy_version,
            expires_at_ms=expires_raw,
            compressed_bytes=compressed_bytes,
            decoded_bytes=decoded_bytes,
            rows=rows,
            columns=columns,
            nesting_depth=depth,
            table=table,
        )

    @contextmanager
    def _verified_session_scope(self, session: GraphSession):
        """Bind native reads to the already-verified request session.

        Served calls already have an ambient session.  Direct process-owned
        authority calls (such as a startup health check) get the same scoped
        context for the duration of the native point/blob read; no session is
        minted or widened here.  A conflicting ambient authority is denied
        before any native read can occur.
        """

        from agent_utilities.knowledge_graph.core.session import (
            current_session,
            resolve_session,
            use_session,
        )

        ambient = current_session()
        if ambient is None:
            with use_session(session):
                yield
            return
        try:
            resolve_session(session, required_scope="kg:read")
        except Exception as exc:  # noqa: BLE001 - hide authority mismatch
            raise PermissionError("artifact access is denied") from exc
        yield

    def _scoped_engine(self, session: GraphSession) -> Any:
        """Resolve the existing process engine view for the session graph."""

        target = str(getattr(session, "graph", "") or "").strip()
        compute = getattr(self._engine, "graph_compute", None) or getattr(
            self._engine, "graph", None
        )
        current = str(
            getattr(compute, "graph_name", "")
            or getattr(self._engine, "graph_name", "")
            or ""
        ).strip()
        if not target or (current and target == current):
            return self._engine
        view_factory = getattr(self._engine, "for_graph", None)
        if not callable(view_factory):
            view_factory = getattr(compute, "for_graph", None)
        if not callable(view_factory):
            raise ArtifactAuthorityUnavailable(
                "native graph view is unavailable for the verified session"
            )
        try:
            view = view_factory(target)
        except Exception as exc:  # noqa: BLE001 - graph routing details stay private
            raise PermissionError("artifact access is denied") from exc
        if view is None:
            raise PermissionError("artifact access is denied")
        return view

    @staticmethod
    def _node_properties(artifact_ref: str, *, engine: Any) -> Mapping[str, Any]:
        """Read metadata through the session/RLS-aware native point-read seam.

        ``EpistemicGraphBackend.get_node_properties(id)`` is the typed native
        point-read contract.  It performs the existence check and property
        read through the graph-scoped, session-routed compute view.  The
        Cypher path is retained only for lightweight test/deployment adapters
        that expose no typed point reader; the raw client fallback is likewise
        scoped to that exact engine view and never used in preference to RLS.
        """

        backend = getattr(engine, "backend", None)
        point_reader = getattr(backend, "get_node_properties", None)
        if callable(point_reader):
            try:
                props = point_reader(artifact_ref)
            except Exception as exc:  # noqa: BLE001 - absent and denied are one public outcome
                raise PermissionError("artifact access is denied") from exc
            if not isinstance(props, Mapping):
                raise PermissionError("artifact access is denied")
            return props

        execute_read = getattr(backend, "execute_read", None)
        if callable(execute_read):
            try:
                rows = execute_read(
                    "MATCH (n) WHERE n.id = $artifact_ref RETURN n LIMIT 1",
                    {"artifact_ref": artifact_ref},
                )
            except Exception as exc:  # noqa: BLE001 - absent and denied are one public outcome
                raise PermissionError("artifact access is denied") from exc
            if not isinstance(rows, list) or not rows:
                raise PermissionError("artifact access is denied")
            row = rows[0]
            props = row.get("n") if isinstance(row, Mapping) else None
            if not isinstance(props, Mapping):
                props = row.get("node") if isinstance(row, Mapping) else None
            if not isinstance(props, Mapping):
                raise PermissionError("artifact access is denied")
            return props
        compute = getattr(engine, "graph_compute", None) or getattr(
            engine, "graph", None
        )
        client = (
            getattr(compute, "client", None)
            or getattr(compute, "_client", None)
            or getattr(engine, "client", None)
        )
        nodes = getattr(client, "nodes", None)
        properties = getattr(nodes, "properties", None)
        if not callable(properties):
            raise ArtifactAuthorityUnavailable(
                "native graph point-read authority is unavailable"
            )
        try:
            props = properties(artifact_ref)
        except Exception as exc:  # noqa: BLE001 - hide graph existence details
            raise PermissionError("artifact access is denied") from exc
        if not isinstance(props, Mapping):
            raise PermissionError("artifact access is denied")
        return props

    def _fetch_blob(self, digest: str, *, engine: Any) -> bytes | None:
        """Fetch bytes through the scoped native content-addressed authority."""

        compute = getattr(engine, "graph_compute", None) or getattr(
            engine, "graph", None
        )
        media_store = None
        if engine is self._engine and self._media_store is not None:
            media_store = self._media_store
        elif getattr(compute, "_client", None) is not None:
            try:
                from agent_utilities.knowledge_graph.memory.media_store import (
                    MediaStore,
                )

                media_store = MediaStore(compute)
            except Exception:  # pragma: no cover - diagnosed as unavailable below
                media_store = None
        try:
            if media_store is not None:
                return media_store.fetch_bytes(digest)
            client = (
                getattr(compute, "client", None)
                or getattr(compute, "_client", None)
                or getattr(engine, "client", None)
            )
            blob = getattr(client, "blob", None)
            fetch = getattr(blob, "fetch", None)
            if not callable(fetch):
                raise ArtifactAuthorityUnavailable(
                    "native artifact blob authority is unavailable"
                )
            return fetch(digest)
        except ArtifactAuthorityUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - native dependency details stay private
            raise ArtifactAuthorityUnavailable(
                "native artifact bytes are unavailable"
            ) from exc

    def _authorize_metadata(
        self,
        props: Mapping[str, Any],
        *,
        session: GraphSession,
        budget: PrepBudget,
    ) -> tuple[str, str, str, str, DataClassification, ArtifactACL]:
        """Validate every access fact before touching native blob bytes."""

        node_type_raw = props.get("node_type")
        if node_type_raw is None:
            node_type_raw = props.get("type")
        if not isinstance(node_type_raw, str):
            raise DataPrepToolError("native artifact type authority is unavailable")
        node_type = node_type_raw
        if node_type not in {"AssetOccurrence", "Artifact"}:
            raise DataPrepToolError(
                "artifact reference is not a governed tabular artifact"
            )
        tenant_raw = props.get("tenant_id")
        if tenant_raw is None:
            tenant_raw = props.get("tenant")
        policy_raw = props.get("policy_version")
        if not isinstance(tenant_raw, str) or not isinstance(policy_raw, str):
            raise ArtifactAuthorityUnavailable(
                "native artifact tenant or policy authority is unavailable"
            )
        tenant_id = tenant_raw
        policy_version = policy_raw
        if not tenant_id or not policy_version:
            raise ArtifactAuthorityUnavailable(
                "native artifact tenant or policy authority is unavailable"
            )
        if tenant_id != session.tenant:
            raise PermissionError("artifact access is denied")
        if policy_version != str(session.policy_version or ""):
            raise PermissionError("artifact access is denied")
        expires_raw = props.get("expires_at_ms", 0)
        if isinstance(expires_raw, bool) or not isinstance(expires_raw, int):
            raise DataPrepToolError("native artifact expiry is invalid")
        if expires_raw and int(time.time() * 1000) >= expires_raw:
            raise PermissionError("artifact access is denied")
        legal_hold = props.get("legal_hold", False)
        if not isinstance(legal_hold, bool):
            raise DataPrepToolError("native artifact legal-hold policy is invalid")
        owner_raw = props.get("_owner_id")
        if owner_raw is None:
            owner_raw = props.get("owner")
        if owner_raw is not None and not isinstance(owner_raw, str):
            raise DataPrepToolError("native artifact owner authority is invalid")
        owner_id = owner_raw or ""
        classification_raw = props.get("classification")
        try:
            classification = (
                classification_raw
                if isinstance(classification_raw, DataClassification)
                else DataClassification(classification_raw)
            )
        except (TypeError, ValueError) as exc:
            raise DataPrepToolError(
                "native artifact classification authority is unavailable"
            ) from exc
        if props.get("retention") is not None and not isinstance(
            props["retention"], str
        ):
            raise DataPrepToolError("native artifact retention policy is invalid")
        acl = _native_acl(props, owner_id=owner_id)
        if classification is DataClassification.PUBLIC and not acl.is_public:
            raise DataPrepToolError("public classification lacks a public ACL proof")
        if acl.is_public and classification is not DataClassification.PUBLIC:
            raise DataPrepToolError("public ACL lacks a matching public classification")
        actor_id = str(getattr(session.actor, "actor_id", "") or "")
        roles = {str(role) for role in getattr(session.actor, "roles", ()) or ()}
        groups = {str(group) for group in getattr(session.actor, "groups", ()) or ()}
        if not (
            acl.is_public
            or actor_id == owner_id
            or actor_id in acl.principal_ids
            or groups.intersection(acl.group_ids)
            or roles.intersection(acl.roles)
        ):
            raise PermissionError("artifact access is denied")
        _native_digest(
            props.get("content_digest")
            or props.get("content_hash")
            or props.get("digest")
            or props.get("blob_digest")
        )
        media_type_raw = props.get("media_type")
        if media_type_raw is None:
            media_type_raw = props.get("mime_type")
        if not isinstance(media_type_raw, str):
            raise DataPrepToolError("native artifact media type is invalid")
        media_type = media_type_raw
        if media_type not in {
            "application/vnd.apache.arrow.stream",
            "application/vnd.apache.arrow.file",
        }:
            raise DataPrepToolError("artifact media type is not an approved Arrow type")
        for key, limit, label in (
            ("compressed_bytes", budget.max_compressed_bytes, "compressed size"),
            ("file_size_bytes", budget.max_compressed_bytes, "compressed size"),
            ("decoded_bytes", budget.max_decoded_bytes, "decoded size"),
            ("rows", budget.max_rows, "row count"),
            ("columns", budget.max_columns, "column count"),
            ("nesting_depth", budget.max_depth, "nesting depth"),
        ):
            value = props.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > limit
            ):
                raise DataPrepToolError(f"artifact {label} exceeds the request budget")
        if props.get("schema_digest") is not None:
            _native_digest(props["schema_digest"])
        if props.get("shape_digest") is not None:
            _native_digest(props["shape_digest"])
        if props.get("schema_ref") is not None:
            _native_ref(props["schema_ref"], fallback="schema:unused")
        if props.get("shape_ref") is not None:
            _native_ref(props["shape_ref"], fallback="shape:unused")
        return node_type, tenant_id, policy_version, owner_id, classification, acl

    @staticmethod
    def _metadata_int(
        props: Mapping[str, Any], key: str, actual: int, label: str
    ) -> int:
        value = props.get(key, actual)
        if isinstance(value, bool) or not isinstance(value, int) or value != actual:
            raise DataPrepToolError(f"artifact {label} metadata does not match content")
        return value

    @staticmethod
    def _decode_arrow(payload: bytes, *, media_type: str, budget: PrepBudget) -> Any:
        try:
            import pyarrow as pa

            reader = (
                pa.ipc.open_file(pa.py_buffer(payload))
                if media_type.endswith("file")
                else pa.ipc.open_stream(pa.py_buffer(payload))
            )
            table = reader.read_all()
            ArrowAdapter.as_table(
                table,
                profile=LocalProfile(
                    max_rows=budget.max_rows,
                    max_columns=budget.max_columns,
                    max_steps=64,
                    max_bytes=budget.max_decoded_bytes,
                ),
            )
            return table
        except DataPrepToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize Arrow/native errors
            raise DataPrepToolError(
                "native artifact is not valid bounded Arrow"
            ) from exc

    def records_artifact(
        self,
        records: list[dict[str, Any]],
        *,
        session: GraphSession,
        budget: PrepBudget,
    ) -> ResolvedArtifact:
        policy = self._inline_policy
        if not self.inline_records_policy_available(session=session) or not isinstance(
            policy, Mapping
        ):
            raise ArtifactAuthorityUnavailable(
                "server-owned inline records governance policy is unavailable"
            )
        try:
            import pyarrow as pa

            table = pa.Table.from_pylist(records)
        except Exception as exc:  # noqa: BLE001 - normalize Arrow dependency errors
            raise DataPrepToolError(
                "inline records cannot be converted to Arrow"
            ) from exc
        if table.num_rows > budget.max_rows or table.num_columns > budget.max_columns:
            raise DataPrepToolError("inline records exceed the request budget")
        output_bytes = _canonical_arrow_bytes(table)
        if len(output_bytes) > budget.max_compressed_bytes:
            raise DataPrepToolError("inline records exceed the compressed byte budget")
        if table.nbytes > budget.max_decoded_bytes:
            raise DataPrepToolError("inline records exceed the decoded byte budget")
        tenant_id = str(policy.get("tenant_id") or "")
        policy_version = str(policy.get("policy_version") or "")
        if tenant_id != session.tenant or policy_version != str(
            session.policy_version or ""
        ):
            raise PermissionError("inline records policy is not bound to the session")
        classification_raw = policy.get("classification")
        try:
            classification = DataClassification(str(classification_raw))
            acl = ArtifactACL.from_value(policy.get("acl"))
        except (DataPrepToolError, ValueError) as exc:
            raise ArtifactAuthorityUnavailable(
                "server-owned inline records governance policy is invalid"
            ) from exc
        output_digest = _sha256_bytes(output_bytes)
        schema_value = schema_digest(table)
        shape_value = _shape_digest(table)
        owner_id = str(
            policy.get("owner_id") or getattr(session.actor, "actor_id", "") or ""
        )
        return ResolvedArtifact(
            artifact_ref=f"inline:{output_digest.removeprefix('sha256:')}",
            content_ref=f"inline:{output_digest.removeprefix('sha256:')}",
            media_type="application/vnd.apache.arrow.stream",
            schema_ref=_native_ref(
                policy.get("schema_ref"), fallback=_schema_ref(schema_value)
            ),
            schema_digest=schema_value,
            shape_ref=_native_ref(
                policy.get("shape_ref"), fallback=_shape_ref(shape_value)
            ),
            shape_digest=shape_value,
            tenant_id=tenant_id,
            owner_id=owner_id,
            acl=acl,
            content_digest=output_digest,
            classification=classification,
            retention=(
                str(policy["retention"])
                if policy.get("retention") is not None
                else None
            ),
            legal_hold=bool(policy.get("legal_hold", False)),
            policy_version=policy_version,
            expires_at_ms=int(policy.get("expires_at_ms", 0)),
            compressed_bytes=len(output_bytes),
            decoded_bytes=int(table.nbytes),
            rows=int(table.num_rows),
            columns=int(table.num_columns),
            nesting_depth=_table_depth(table),
            table=table,
        )

    def approved_models(self, *, session: GraphSession) -> RowModelRegistry:
        del session
        if self._models is None:
            raise ArtifactAuthorityUnavailable(
                "approved row-model registry is unavailable from process configuration"
            )
        return self._models

    def inline_records_policy_available(self, *, session: GraphSession) -> bool:
        policy = self._inline_policy
        if isinstance(policy, Mapping):
            return bool(
                policy.get("enabled") is True
                and str(policy.get("tenant_id") or "") == session.tenant
                and str(policy.get("policy_version") or "")
                == str(session.policy_version or "")
            )
        method = getattr(self._policy, "allow_inline_records", None)
        if callable(method):
            try:
                return bool(method(session=session, policy=policy))
            except Exception:  # pragma: no cover - broken policy fails closed
                return False
        return False

    def icv_policy_available(self, *, session: GraphSession) -> bool:
        del session
        value = None
        for source in _runtime_sources(self._engine):
            if isinstance(source, DataPrepRuntimeConfig):
                value = source.icv_policy_available
                break
            candidate = getattr(source, "data_prep_icv_policy_available", None)
            if candidate is not None:
                value = candidate
                break
        return value is True

    def output_governance(
        self,
        source: ResolvedArtifact,
        *,
        output_table: Any,
        output_schema_ref: str,
        output_schema_digest: str,
        output_shape_ref: str,
        output_shape_digest: str,
        session: GraphSession,
    ) -> ResolvedArtifact:
        method = getattr(self._policy, "govern_output", None)
        if not callable(method):
            raise ArtifactAuthorityUnavailable(
                "output governance policy is unavailable"
            )
        try:
            result = method(
                source,
                output_table=output_table,
                output_schema_ref=output_schema_ref,
                output_schema_digest=output_schema_digest,
                output_shape_ref=output_shape_ref,
                output_shape_digest=output_shape_digest,
                session=session,
            )
        except (DataPrepToolError, PermissionError):
            raise
        except Exception as exc:  # noqa: BLE001 - policy dependency details stay private
            raise ArtifactAuthorityUnavailable(
                "output governance policy failed"
            ) from exc
        if not isinstance(result, ResolvedArtifact):
            raise ArtifactAuthorityUnavailable(
                "output governance policy returned invalid metadata"
            )
        return result


def _process_authority_factory(session: GraphSession) -> DataPrepAuthority:
    """Resolve the one process-owned native engine at served-call time."""

    try:
        from agent_utilities.knowledge_graph.core.engine import (
            IntelligenceGraphEngine,
        )

        engine = IntelligenceGraphEngine.get_active()
        if engine is None:
            # Importing this tool must not start a second graph authority.  The
            # process-owned MCP startup path is the only place allowed to open it.
            return _DefaultAuthority()
        register_process_data_prep_runtime(engine)
        return _NativeDataPrepAuthority(engine, session=session)
    except (ImportError, RuntimeError):
        return _DefaultAuthority()


_AUTHORITY_FACTORY: Callable[[GraphSession], DataPrepAuthority] = (
    _process_authority_factory
)


def register_process_data_prep_runtime(engine: Any) -> bool:
    """Compose and install the provider on the one process-owned engine.

    Startup creates the concrete graph-native provider when no deployment
    adapter was injected.  The private identity pin prevents a later request or
    runtime metadata mutation from replacing it.  A provider may be installed
    while reporting missing model/policy configuration; its individual methods
    then fail closed with a dependency diagnostic instead of silently falling
    back to an in-memory or caller-selected store.
    """

    existing = getattr(engine, _RUNTIME_PROVIDER_ATTR, None)
    if existing is not None:
        return existing is getattr(engine, "data_prep_artifact_authority", existing)

    provider = getattr(engine, "data_prep_artifact_authority", None)
    if provider is None:
        provider = getattr(
            getattr(engine, "backend", None),
            "data_prep_artifact_authority",
            None,
        )
    if provider is None:
        provider = _GraphNativeDataPrepProvider.from_engine(engine)
    if provider is None:
        logger.warning(
            "data-prep runtime provider unavailable: native graph nodes/blob "
            "authority is not ready"
        )
        return False
    setattr(engine, _RUNTIME_PROVIDER_ATTR, provider)
    # This is a startup-owned installation, never a request-selected provider.
    # Keep the public attribute for introspection/backward-compatible deployment
    # health checks, while all served calls resolve the pinned private object.
    engine.data_prep_artifact_authority = provider
    diagnostics = tuple(getattr(provider, "diagnostics", ()))
    if diagnostics:
        logger.warning(
            "data-prep runtime provider installed with unavailable dependencies: %s",
            ", ".join(diagnostics),
        )
    return not diagnostics


def register_data_prep_authority(
    factory: Callable[[GraphSession], DataPrepAuthority] | None,
) -> None:
    """Install the process-owned native artifact adapter (deployment seam)."""

    global _AUTHORITY_FACTORY
    _AUTHORITY_FACTORY = factory or _process_authority_factory


def _validate_inline_records(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) > _MAX_INLINE_ROWS:
        raise ValueError("inline records exceed the bounded row limit")
    names: set[str] = set()
    for row_index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise ValueError(f"inline record {row_index} is not an object")
        if len(row) > _MAX_INLINE_COLUMNS:
            raise ValueError("inline records exceed the bounded column limit")
        names.update(str(key) for key in row)
        for key, value in row.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise ValueError("inline record field names are invalid")
            if value is not None and not isinstance(value, (bool, int, float, str)):
                raise ValueError("inline records accept scalar values only")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("inline records require finite numeric values")
            if (
                isinstance(value, str)
                and len(value.encode("utf-8")) > _MAX_INLINE_CELL_BYTES
            ):
                raise ValueError("inline record cell exceeds the bounded size")
    if len(names) > _MAX_INLINE_COLUMNS:
        raise ValueError("inline records exceed the bounded column limit")
    try:
        if len(_canonical_json(list(records))) > _MAX_PARAMS_BYTES:
            raise ValueError("inline records exceed the bounded byte limit")
    except DataPrepToolError as exc:
        raise ValueError("inline records contain unsupported values") from exc


def _nested_depth(value: Any) -> int:
    """Return Arrow field nesting depth without inspecting field values."""

    if not hasattr(value, "type"):
        return 0
    data_type = value.type
    if not hasattr(data_type, "num_fields") or data_type.num_fields == 0:
        return 0
    try:
        child_depth = max(
            (
                _nested_depth(data_type.field(index))
                for index in range(data_type.num_fields)
            ),
            default=0,
        )
    except (AttributeError, TypeError, ValueError):
        return _MAX_ARTIFACT_DEPTH + 1
    return 1 + child_depth


def _table_depth(table: Any) -> int:
    return max((_nested_depth(field) for field in table.schema), default=0)


def _check_cancel(deadline: float) -> None:
    from agent_utilities.core.task_cancellation import raise_if_task_cancelled

    raise_if_task_cancelled()
    if time.monotonic() >= deadline:
        raise DataPrepToolError("data-prep resource budget expired")


def _digest_acl(acl: ArtifactACL) -> str:
    body = _canonical_json(
        {
            "is_public": acl.is_public,
            "principal_ids": acl.principal_ids,
            "principal_emails": acl.principal_emails,
            "group_ids": acl.group_ids,
            "roles": acl.roles,
            "markings": acl.markings,
        }
    )
    return _sha256_bytes(body)


_CLASSIFICATION_RANK = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.RESTRICTED: 3,
}


def _require_governance_not_weaker(
    source: ResolvedArtifact, output: ResolvedArtifact
) -> None:
    """Ensure a prepared artifact cannot broaden or shorten source authority."""

    if (
        _CLASSIFICATION_RANK[output.classification]
        < _CLASSIFICATION_RANK[source.classification]
    ):
        raise DataPrepToolError("prepared classification would downgrade source policy")
    if not source.acl.is_public and output.acl.is_public:
        raise DataPrepToolError("prepared ACL would broaden source visibility")
    if not source.acl.is_public:
        if not set(output.acl.principal_ids).issubset(source.acl.principal_ids):
            raise DataPrepToolError("prepared principal ACL is broader than source")
        if not set(output.acl.principal_emails).issubset(source.acl.principal_emails):
            raise DataPrepToolError("prepared email ACL is broader than source")
        if not set(output.acl.group_ids).issubset(source.acl.group_ids):
            raise DataPrepToolError("prepared group ACL is broader than source")
        if not set(output.acl.roles).issubset(source.acl.roles):
            raise DataPrepToolError("prepared role ACL is broader than source")
        if not set(output.acl.markings).issubset(source.acl.markings):
            raise DataPrepToolError("prepared markings ACL is broader than source")
    if source.expires_at_ms and (
        not output.expires_at_ms or output.expires_at_ms > source.expires_at_ms
    ):
        raise DataPrepToolError("prepared expiry would broaden source retention")
    if source.retention is not None and output.retention != source.retention:
        raise DataPrepToolError("prepared retention policy is not preserved")
    if source.legal_hold and not output.legal_hold:
        raise DataPrepToolError("prepared legal hold cannot be cleared")
    if not output.policy_version:
        raise DataPrepToolError("prepared policy version is missing")
    if output.policy_version != source.policy_version:
        raise DataPrepToolError("prepared policy version is not preserved")


def _require_artifact_access(
    artifact: ResolvedArtifact,
    *,
    request: PrepRequest,
    session: GraphSession,
    now_ms: int,
    match_input_shape: bool = True,
) -> None:
    actor = session.actor
    if not isinstance(artifact.classification, DataClassification):
        raise DataPrepToolError("artifact classification proof is unavailable")
    if not artifact.policy_version:
        raise DataPrepToolError("artifact policy version proof is unavailable")
    if (
        artifact.classification is DataClassification.PUBLIC
        and not artifact.acl.is_public
    ):
        raise DataPrepToolError("public classification lacks a public ACL proof")
    if (
        artifact.acl.is_public
        and artifact.classification is not DataClassification.PUBLIC
    ):
        raise DataPrepToolError("public ACL lacks a matching public classification")
    if (
        not isinstance(artifact.content_digest, str)
        or not artifact.content_digest.startswith("sha256:")
        or len(artifact.content_digest) != len("sha256:") + 64
    ):
        raise DataPrepToolError("artifact content fingerprint proof is unavailable")
    if any("@" in item for item in artifact.acl.principal_ids):
        raise DataPrepToolError("principal IDs must not be treated as email ACLs")
    if any("@" not in item for item in artifact.acl.principal_emails):
        raise DataPrepToolError("artifact user email ACL proof is unavailable")
    if artifact.tenant_id != session.tenant:
        raise PermissionError("artifact tenant authority does not match the session")
    if artifact.expires_at_ms < 0 or (
        artifact.expires_at_ms and now_ms >= artifact.expires_at_ms
    ):
        raise PermissionError("artifact access has expired")
    actor_id = str(getattr(actor, "actor_id", "") or "")
    roles = {str(role) for role in getattr(actor, "roles", ()) or ()}
    groups = {str(group) for group in getattr(actor, "groups", ()) or ()}
    if not (
        artifact.acl.is_public
        or actor_id == artifact.owner_id
        or actor_id in artifact.acl.principal_ids
        or groups.intersection(artifact.acl.group_ids)
        or roles.intersection(artifact.acl.roles)
    ):
        raise PermissionError(
            "artifact ACL does not grant the current principal access"
        )
    if match_input_shape and (
        artifact.schema_ref != request.schema_ref
        or artifact.schema_digest != request.schema_digest
    ):
        raise DataPrepToolError("artifact schema is not the approved immutable schema")
    if match_input_shape and (
        artifact.shape_ref != request.shape_ref
        or artifact.shape_digest != request.shape_digest
    ):
        raise DataPrepToolError("artifact shape is not the approved immutable shape")
    if artifact.compressed_bytes < 0 or artifact.compressed_bytes > _MAX_ARTIFACT_BYTES:
        raise DataPrepToolError(
            "artifact compressed size is outside the governed bound"
        )
    if artifact.decoded_bytes < 0 or artifact.decoded_bytes > _MAX_ARTIFACT_BYTES:
        raise DataPrepToolError("artifact decoded size is outside the governed bound")
    budget = request.budget
    if artifact.compressed_bytes > budget.max_compressed_bytes:
        raise DataPrepToolError("artifact compressed size exceeds the request budget")
    if artifact.decoded_bytes > budget.max_decoded_bytes:
        raise DataPrepToolError("artifact decoded size exceeds the request budget")
    if artifact.rows < 0 or artifact.rows > budget.max_rows:
        raise DataPrepToolError("artifact row count exceeds the request budget")
    if artifact.columns < 0 or artifact.columns > budget.max_columns:
        raise DataPrepToolError("artifact column count exceeds the request budget")
    if artifact.nesting_depth < 0 or artifact.nesting_depth > budget.max_depth:
        raise DataPrepToolError("artifact nesting depth exceeds the request budget")
    if artifact.media_type not in {
        "application/vnd.apache.arrow.stream",
        "application/vnd.apache.arrow.file",
    }:
        raise DataPrepToolError("artifact media type is not an approved Arrow type")
    ArrowAdapter.as_table(
        artifact.table,
        profile=LocalProfile(
            max_rows=budget.max_rows,
            max_columns=budget.max_columns,
            max_steps=64,
            max_bytes=budget.max_decoded_bytes,
        ),
    )
    actual_schema = schema_digest(artifact.table)
    if actual_schema != artifact.schema_digest:
        raise DataPrepToolError(
            "artifact schema fingerprint does not match its content"
        )
    if (
        artifact.rows != artifact.table.num_rows
        or artifact.columns != artifact.table.num_columns
    ):
        raise DataPrepToolError("artifact shape metadata does not match its content")
    if artifact.shape_digest != _shape_digest(artifact.table):
        raise DataPrepToolError("artifact shape fingerprint does not match its content")
    actual_depth = _table_depth(artifact.table)
    if artifact.nesting_depth != actual_depth:
        raise DataPrepToolError("artifact nesting metadata does not match its content")
    if actual_depth > budget.max_depth:
        raise DataPrepToolError(
            "artifact content nesting depth exceeds the request budget"
        )


def _require_table_bounds(table: Any, *, budget: PrepBudget) -> None:
    """Re-apply request bounds to transformed output before it gets named."""

    ArrowAdapter.as_table(
        table,
        profile=LocalProfile(
            max_rows=budget.max_rows,
            max_columns=budget.max_columns,
            max_steps=64,
            max_bytes=budget.max_decoded_bytes,
        ),
    )
    depth = _table_depth(table)
    if depth > budget.max_depth:
        raise DataPrepToolError(
            "prepared output nesting depth exceeds the request budget"
        )


def _expected_receipt_fields(
    source: ResolvedArtifact,
    result_table: Any,
    evidence: PrepEvidence,
    request: PrepRequest,
    *,
    session: GraphSession,
) -> dict[str, Any]:
    output_bytes = _canonical_arrow_bytes(result_table)
    output_schema_digest = schema_digest(result_table)
    output_shape_digest = _shape_digest(result_table)
    _, evidence_digest = _evidence_payload(evidence)
    return {
        "tenant_id": session.tenant,
        "artifact_ref": source.artifact_ref,
        "input_content_digest": source.content_digest,
        "input_schema_ref": request.schema_ref,
        "input_schema_digest": request.schema_digest,
        "input_shape_ref": request.shape_ref,
        "input_shape_digest": request.shape_digest,
        "plan_ref": request.plan_ref,
        "plan_digest": evidence.plan_digest,
        "model_ref": request.model_ref,
        "model_digest": evidence.model_digest,
        "output_content_digest": _sha256_bytes(output_bytes),
        "output_schema_ref": _schema_ref(output_schema_digest),
        "output_schema_digest": output_schema_digest,
        "output_shape_ref": _shape_ref(output_shape_digest),
        "output_shape_digest": output_shape_digest,
        "evidence_digest": evidence_digest,
        "policy_version": str(session.policy_version or ""),
    }


def _verify_receipt_binding(
    receipt: PreparedReceipt,
    source: ResolvedArtifact,
    result_table: Any,
    evidence: PrepEvidence,
    request: PrepRequest,
    *,
    session: GraphSession,
) -> None:
    actor_id = str(getattr(session.actor, "actor_id", "") or "")
    if receipt.tenant_id != session.tenant or receipt.actor_id != actor_id:
        raise PermissionError(
            "prepared receipt identity authority does not match the session"
        )
    if receipt.policy_version != str(session.policy_version or ""):
        raise PermissionError("prepared receipt policy version is stale")
    if receipt.expires_at_ms <= int(time.time() * 1000):
        raise PermissionError("prepared receipt has expired")
    expected = _expected_receipt_fields(
        source, result_table, evidence, request, session=session
    )
    actual = {key: getattr(receipt, key) for key in expected}
    if actual != expected:
        raise DataPrepToolError(
            "prepared receipt does not bind the deterministic output"
        )
    if receipt.native_atomic is not True:
        raise NativeCommitUnavailable(
            "prepared receipt does not prove native atomic admission"
        )


def _native_apply_change_supported(engine: Any) -> bool:
    """Probe the same compute-client shapes accepted by envelope ingestion."""

    candidates = [
        engine,
        getattr(engine, "graph_compute", None),
        getattr(engine, "graph", None),
    ]
    backend = getattr(engine, "backend", None)
    authority_backend = getattr(backend, "_authority", backend)
    candidates.append(getattr(authority_backend, "graph", None))
    seen: set[int] = set()
    supported = False
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        client = getattr(candidate, "client", None)
        supports = getattr(client, "supports", None)
        if callable(supports):
            try:
                supported = bool(supports("ApplyChangeEnvelope")) or supported
            except Exception:  # noqa: BLE001 - inspect every authority candidate
                continue
    return supported


class _NativeDataPrepAuthority:
    """Concrete adapter over the process-owned engine and native blob client.

    Artifact resolution/model registration/policy are supplied by the engine's
    process-owned ``data_prep_artifact_authority``.  This adapter never accepts
    a caller-selected provider or persistence location.  The native graph blob
    and ChangeEnvelope clients are the only commit authority.
    """

    def __init__(self, engine: Any, *, session: GraphSession) -> None:
        self._engine = engine
        self._session = session

    @property
    def _compute(self) -> Any:
        return getattr(self._engine, "graph_compute", None) or getattr(
            self._engine, "graph", None
        )

    @property
    def _client(self) -> Any:
        compute = self._compute
        client = getattr(compute, "client", None)
        if client is None:
            raise ArtifactAuthorityUnavailable("native graph client is unavailable")
        return client

    def _provider(self) -> Any:
        provider = getattr(self._engine, _RUNTIME_PROVIDER_ATTR, None)
        if provider is not None and provider is not self:
            return provider
        raise ArtifactAuthorityUnavailable(
            "process-owned data-prep artifact authority is not registered"
        )

    def _provider_call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        provider = self._provider()
        method = getattr(provider, name, None)
        if not callable(method):
            raise ArtifactAuthorityUnavailable(
                f"process-owned data-prep authority lacks {name}"
            )
        return method(*args, **kwargs)

    def artifact(
        self, artifact_ref: str, *, session: GraphSession, budget: PrepBudget
    ) -> ResolvedArtifact:
        return self._provider_call(
            "artifact", artifact_ref, session=session, budget=budget
        )

    def records_artifact(
        self,
        records: list[dict[str, Any]],
        *,
        session: GraphSession,
        budget: PrepBudget,
    ) -> ResolvedArtifact:
        return self._provider_call(
            "records_artifact", records, session=session, budget=budget
        )

    def approved_models(self, *, session: GraphSession) -> RowModelRegistry:
        registry = self._provider_call("approved_models", session=session)
        if isinstance(registry, RowModelRegistry):
            return registry
        raise ArtifactAuthorityUnavailable("approved model registry is invalid")

    def inline_records_policy_available(self, *, session: GraphSession) -> bool:
        try:
            value = self._provider_call(
                "inline_records_policy_available", session=session
            )
        except ArtifactAuthorityUnavailable:
            return False
        return bool(value)

    def preview_ref(
        self,
        source: ResolvedArtifact,
        *,
        output_table: Any,
        evidence: PrepEvidence,
        request: PrepRequest,
        session: GraphSession,
    ) -> PreparedReceipt:
        if not self.native_atomic_available(session=session):
            raise NativeCommitUnavailable(
                "native blob and ChangeEnvelope atomic capability is unavailable"
            )
        output_bytes = _canonical_arrow_bytes(output_table)
        output_schema_digest = schema_digest(output_table)
        output_shape_digest = _shape_digest(output_table)
        _, evidence_digest = _evidence_payload(evidence)
        issued_at_ms = int(time.time() * 1000)
        receipt = PreparedReceipt(
            tenant_id=session.tenant,
            artifact_ref=source.artifact_ref,
            input_content_digest=source.content_digest,
            input_schema_ref=request.schema_ref,
            input_schema_digest=request.schema_digest,
            input_shape_ref=request.shape_ref,
            input_shape_digest=request.shape_digest,
            plan_ref=request.plan_ref,
            plan_digest=evidence.plan_digest,
            model_ref=request.model_ref,
            model_digest=evidence.model_digest,
            output_content_digest=_sha256_bytes(output_bytes),
            output_schema_ref=_schema_ref(output_schema_digest),
            output_schema_digest=output_schema_digest,
            output_shape_ref=_shape_ref(output_shape_digest),
            output_shape_digest=output_shape_digest,
            evidence_digest=evidence_digest,
            policy_version=str(session.policy_version or ""),
            native_atomic=True,
            issued_at_ms=issued_at_ms,
            actor_id=str(getattr(session.actor, "actor_id", "") or ""),
            endpoint=_RECEIPT_ENDPOINT,
            expires_at_ms=issued_at_ms + _RECEIPT_TTL_MS,
        )
        return receipt

    def output_governance(
        self,
        source: ResolvedArtifact,
        *,
        output_table: Any,
        output_schema_ref: str,
        output_schema_digest: str,
        output_shape_ref: str,
        output_shape_digest: str,
        session: GraphSession,
    ) -> ResolvedArtifact:
        result = self._provider_call(
            "output_governance",
            source,
            output_table=output_table,
            output_schema_ref=output_schema_ref,
            output_schema_digest=output_schema_digest,
            output_shape_ref=output_shape_ref,
            output_shape_digest=output_shape_digest,
            session=session,
        )
        if not isinstance(result, ResolvedArtifact):
            raise ArtifactAuthorityUnavailable(
                "output governance authority returned an invalid result"
            )
        return result

    def native_engine(self, *, session: GraphSession) -> Any:
        del session
        return self._engine

    def icv_policy_available(self, *, session: GraphSession) -> bool:
        try:
            return bool(self._provider_call("icv_policy_available", session=session))
        except ArtifactAuthorityUnavailable:
            return False

    def native_atomic_available(self, *, session: GraphSession) -> bool:
        return bool(
            _native_apply_change_supported(self._engine)
            and self.icv_policy_available(session=session)
        )

    def store_blob(
        self, payload: bytes, *, media_type: str, session: GraphSession
    ) -> str:
        del media_type, session
        store = getattr(getattr(self._client, "blob", None), "store", None)
        if not callable(store):
            raise NativeCommitUnavailable(
                "native content-addressed blob store is unavailable"
            )
        try:
            digest = store(payload)
        except Exception as exc:  # noqa: BLE001 - map native dependency failures
            raise NativeCommitUnavailable(
                "native blob store rejected the payload"
            ) from exc
        if not isinstance(digest, str) or not digest:
            raise NativeCommitUnavailable(
                "native blob store returned no content digest"
            )
        return digest if digest.startswith("sha256:") else f"sha256:{digest}"

    def incref_blob(self, digest: str, *, session: GraphSession) -> None:
        del session
        incref = getattr(getattr(self._client, "blob", None), "incref", None)
        if not callable(incref):
            raise NativeCommitUnavailable(
                "native blob reference operation is unavailable"
            )
        try:
            incref(digest.removeprefix("sha256:"))
        except Exception as exc:  # noqa: BLE001
            raise NativeCommitUnavailable(
                "native blob reference was not admitted"
            ) from exc

    def unref_blob(self, digest: str, *, session: GraphSession) -> None:
        del session
        unref = getattr(getattr(self._client, "blob", None), "unref", None)
        if not callable(unref):
            raise NativeCommitUnavailable("native blob compensation is unavailable")
        try:
            unref(digest.removeprefix("sha256:"))
        except Exception as exc:  # noqa: BLE001
            raise NativeCommitUnavailable("native blob compensation failed") from exc


class DataPrepService:
    """Thin governed adapter that delegates all data work to NE-108."""

    def __init__(
        self,
        authority: DataPrepAuthority,
        *,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._authority = authority
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))

    def _request(self, payload: Mapping[str, Any]) -> PrepRequest:
        try:
            request = PrepRequest.model_validate(payload)
            if len(_canonical_json(request.plan)) > _MAX_PLAN_BYTES:
                raise DataPrepToolError("plan exceeds the bounded request size")
            plan = CleanPlan.model_validate(request.plan)
        except (ValidationError, ValueError, TypeError) as exc:
            raise DataPrepToolError(
                "data-prep request failed its typed plan gate"
            ) from exc
        if plan.plan_ref != request.plan_ref or plan.model_ref != request.model_ref:
            raise DataPrepToolError("plan references do not match the approved request")
        if plan.artifact_ref is not None and plan.artifact_ref != request.artifact_ref:
            raise DataPrepToolError("plan artifact reference is not approved")
        if plan.source_ref is not None and plan.source_ref != request.artifact_ref:
            raise DataPrepToolError("plan source reference is not approved")
        from agent_utilities.data_prep import plan_digest

        if plan_digest(plan) != request.plan_digest:
            raise DataPrepToolError("plan digest does not match the immutable plan")
        if plan.model_digest != request.model_digest:
            raise DataPrepToolError("model digest must be pinned in the immutable plan")
        return request

    def _input(
        self,
        request: PrepRequest,
        *,
        session: GraphSession,
        deadline: float,
    ) -> tuple[ResolvedArtifact, CleanPlan, RowModelRegistry]:
        _check_cancel(deadline)
        plan = CleanPlan.model_validate(request.plan)
        if request.artifact_ref is not None:
            artifact = self._authority.artifact(
                request.artifact_ref,
                session=session,
                budget=request.budget,
            )
        else:
            if request.records is None:  # pragma: no cover - Pydantic gate
                raise DataPrepToolError("data-prep input artifact is missing")
            if not self._authority.inline_records_policy_available(session=session):
                raise ArtifactAuthorityUnavailable(
                    "server-owned inline records governance policy is unavailable"
                )
            artifact = self._authority.records_artifact(
                request.records,
                session=session,
                budget=request.budget,
            )
        if not isinstance(artifact, ResolvedArtifact):
            raise ArtifactAuthorityUnavailable(
                "artifact authority returned an invalid result"
            )
        _require_artifact_access(
            artifact,
            request=request,
            session=session,
            now_ms=self._clock_ms(),
        )
        if (
            request.artifact_ref is not None
            and artifact.artifact_ref != request.artifact_ref
        ):
            raise DataPrepToolError(
                "artifact authority returned a different artifact reference"
            )
        _check_cancel(deadline)
        registry = self._authority.approved_models(session=session)
        if not isinstance(registry, RowModelRegistry):
            raise ArtifactAuthorityUnavailable(
                "approved model authority returned an invalid registry"
            )
        return artifact, plan, registry

    @staticmethod
    def _public_artifact(artifact: ResolvedArtifact) -> dict[str, Any]:
        return {
            "artifact_ref": artifact.artifact_ref,
            "content_digest": artifact.content_digest,
            "media_type": artifact.media_type,
            "schema_ref": artifact.schema_ref,
            "schema_digest": artifact.schema_digest,
            "shape_ref": artifact.shape_ref,
            "shape_digest": artifact.shape_digest,
            "tenant_id": artifact.tenant_id,
            "classification": artifact.classification.value,
            "retention": artifact.retention,
            "legal_hold": artifact.legal_hold,
            "policy_version": artifact.policy_version,
            "acl_digest": _digest_acl(artifact.acl),
            "expires_at_ms": artifact.expires_at_ms,
            "compressed_bytes": artifact.compressed_bytes,
            "decoded_bytes": artifact.decoded_bytes,
            "rows": artifact.rows,
            "columns": artifact.columns,
            "nesting_depth": artifact.nesting_depth,
        }

    @staticmethod
    def _public_evidence(evidence: PrepEvidence) -> dict[str, Any]:
        # NE-108 evidence is already content-free.  Keep the bounded ordinal
        # outcomes because they are needed for a caller to distinguish accepted,
        # dropped and quarantined rows without receiving rejected values.
        payload, digest = _evidence_payload(evidence)
        return {**payload, "evidence_digest": digest}

    def execute(
        self,
        action: str,
        payload: Mapping[str, Any],
        *,
        session: GraphSession,
    ) -> dict[str, Any]:
        try:
            operation = DataPrepAction(action.strip().lower())
        except (AttributeError, ValueError) as exc:
            raise DataPrepToolError("unknown data-prep action") from exc
        session.require_scope(
            "kg:write" if operation is DataPrepAction.COMMIT else "kg:read"
        )
        request = self._request(payload)
        deadline = time.monotonic() + request.budget.max_wall_time_ms / 1000
        artifact, plan, registry = self._input(
            request,
            session=session,
            deadline=deadline,
        )
        pipeline = CleanPipeline(plan, model_registry=registry)

        if operation is DataPrepAction.PROFILE:
            profile: ProfileResult = pipeline.profile(artifact.table)
            _check_cancel(deadline)
            return {
                "surface": "data_prep",
                "action": operation.value,
                "artifact": self._public_artifact(artifact),
                "profile": profile.model_dump(mode="json"),
                "side_effects": [],
            }

        result = pipeline.run(artifact.table)
        _require_table_bounds(result.table, budget=request.budget)
        output_bytes = _canonical_arrow_bytes(result.table)
        if len(output_bytes) > request.budget.max_compressed_bytes:
            raise DataPrepToolError("prepared output size exceeds the request budget")
        if result.table.nbytes > request.budget.max_decoded_bytes:
            raise DataPrepToolError(
                "prepared output decoded size exceeds the request budget"
            )
        _check_cancel(deadline)
        output_schema_digest = schema_digest(result.table)
        output_shape_digest = _shape_digest(result.table)
        output_schema_ref = _schema_ref(output_schema_digest)
        output_shape_ref = _shape_ref(output_shape_digest)
        output_content_digest = _sha256_bytes(output_bytes)
        output_governance = self._authority.output_governance(
            artifact,
            output_table=result.table,
            output_schema_ref=output_schema_ref,
            output_schema_digest=output_schema_digest,
            output_shape_ref=output_shape_ref,
            output_shape_digest=output_shape_digest,
            session=session,
        )
        if not isinstance(output_governance, ResolvedArtifact):
            raise ArtifactAuthorityUnavailable(
                "output governance authority returned an invalid result"
            )
        _require_artifact_access(
            output_governance,
            request=request,
            session=session,
            now_ms=self._clock_ms(),
            match_input_shape=False,
        )
        if (
            output_governance.schema_ref != output_schema_ref
            or output_governance.schema_digest != output_schema_digest
            or output_governance.shape_ref != output_shape_ref
            or output_governance.shape_digest != output_shape_digest
            or output_governance.content_digest != output_content_digest
        ):
            raise DataPrepToolError(
                "output governance does not bind the deterministic content"
            )
        _require_governance_not_weaker(artifact, output_governance)
        if operation is DataPrepAction.CLEAN:
            receipt = self._authority.preview_ref(
                artifact,
                output_table=result.table,
                evidence=result.evidence,
                request=request,
                session=session,
            )
            if not isinstance(receipt, PreparedReceipt):
                raise ArtifactAuthorityUnavailable(
                    "prepared receipt authority returned an invalid receipt"
                )
            _verify_receipt_binding(
                receipt,
                artifact,
                result.table,
                result.evidence,
                request,
                session=session,
            )
            prepared_ref = receipt.encode()
        else:
            if request.prepared_ref is None:
                raise DataPrepToolError("prepared receipt is required")
            receipt = PreparedReceipt.decode(request.prepared_ref)
            _verify_receipt_binding(
                receipt,
                artifact,
                result.table,
                result.evidence,
                request,
                session=session,
            )
            prepared_ref = request.prepared_ref
        if len(prepared_ref) > _PREPARED_REF_MAX:
            raise DataPrepToolError(
                "prepared receipt exceeds the bounded reference size"
            )
        if operation is DataPrepAction.CLEAN:
            output_shape_ref = receipt.output_shape_ref
            output_shape_digest = receipt.output_shape_digest
            return {
                "surface": "data_prep",
                "action": operation.value,
                "input_artifact": self._public_artifact(artifact),
                "prepared_artifact_ref": prepared_ref,
                "output_schema_ref": receipt.output_schema_ref,
                "output_schema_digest": receipt.output_schema_digest,
                "output_shape_ref": output_shape_ref,
                "output_shape_digest": output_shape_digest,
                "output_content_digest": receipt.output_content_digest,
                "evidence": self._public_evidence(result.evidence),
                "side_effects": [],
            }

        approvals = (
            request.expected_output_schema_ref,
            request.expected_output_schema_digest,
            request.expected_output_shape_ref,
            request.expected_output_shape_digest,
        )
        if any(value is not None for value in approvals) and not all(
            value is not None for value in approvals
        ):
            raise DataPrepToolError(
                "output schema and shape approvals must be complete"
            )
        if all(value is not None for value in approvals):
            if (
                request.expected_output_schema_ref != receipt.output_schema_ref
                or request.expected_output_schema_digest != receipt.output_schema_digest
                or request.expected_output_shape_ref != receipt.output_shape_ref
                or request.expected_output_shape_digest != receipt.output_shape_digest
            ):
                raise DataPrepToolError(
                    "prepared output schema or shape is not approved"
                )

        if operation is DataPrepAction.VALIDATE:
            return {
                "surface": "data_prep",
                "action": operation.value,
                "prepared_artifact_ref": prepared_ref,
                "output_schema_ref": receipt.output_schema_ref,
                "output_schema_digest": receipt.output_schema_digest,
                "output_shape_ref": receipt.output_shape_ref,
                "output_shape_digest": receipt.output_shape_digest,
                "output_content_digest": receipt.output_content_digest,
                "valid": result.evidence.checkpoint_eligible,
                "evidence": self._public_evidence(result.evidence),
                "side_effects": [],
            }

        if not result.evidence.checkpoint_eligible:
            raise DataPrepToolError("quarantined preparation cannot be committed")
        if not all(value is not None for value in approvals):
            raise DataPrepToolError(
                "approved output schema and shape refs/digests are required for commit"
            )
        _check_cancel(deadline)
        if not self._authority.native_atomic_available(session=session):
            raise NativeCommitUnavailable(
                "native atomic commit capability is unavailable"
            )
        if not self._authority.icv_policy_available(session=session):
            raise NativeCommitUnavailable("required ICV policy is unavailable")
        output_bytes = _canonical_arrow_bytes(result.table)
        stored_digest = self._authority.store_blob(
            output_bytes,
            media_type="application/vnd.apache.arrow.stream",
            session=session,
        )
        if not isinstance(stored_digest, str) or not stored_digest:
            raise NativeCommitUnavailable(
                "native blob store returned no opaque content reference"
            )
        if stored_digest != output_content_digest:
            raise NativeCommitUnavailable(
                "native blob store returned a digest different from the output"
            )
        ref_acquired = False
        try:
            self._authority.incref_blob(stored_digest, session=session)
            ref_acquired = True
        except Exception as exc:  # noqa: BLE001 - compensate any partial ref
            try:
                self._authority.unref_blob(stored_digest, session=session)
            except Exception:
                pass
            if isinstance(exc, NativeCommitUnavailable):
                raise
            raise NativeCommitUnavailable(
                "native blob reference was not admitted"
            ) from exc
        evidence_payload, evidence_digest = _evidence_payload(result.evidence)
        object_id = f"prepared:{receipt.output_content_digest.removeprefix('sha256:')}"
        envelope = ChangeEnvelope(
            connector="data-prep",
            operation="upsert",
            tenant=session.tenant,
            source_instance="data-prep",
            source_object_id=object_id,
            source_version=receipt.output_content_digest,
            payload_type="AssetOccurrence",
            blob_ref=stored_digest,
            blob_digest=stored_digest,
            blob_length=len(output_bytes),
            blob_media_type="application/vnd.apache.arrow.stream",
            source_acl=ExternalAccess(
                is_public=output_governance.acl.is_public,
                user_emails=list(output_governance.acl.principal_emails),
                group_ids=list(output_governance.acl.group_ids),
                read_roles=list(output_governance.acl.roles),
                markings=list(output_governance.acl.markings),
            ),
            classification=output_governance.classification,
            retention=output_governance.retention,
            legal_hold=output_governance.legal_hold,
            provenance={
                "plan_ref": plan.plan_ref,
                "plan_digest": result.evidence.plan_digest,
                "model_ref": plan.model_ref,
                "model_digest": result.evidence.model_digest,
                "input_schema_ref": request.schema_ref,
                "input_schema_digest": request.schema_digest,
                "input_shape_ref": request.shape_ref,
                "input_shape_digest": request.shape_digest,
                "output_schema_ref": receipt.output_schema_ref,
                "output_schema_digest": receipt.output_schema_digest,
                "output_shape_ref": receipt.output_shape_ref,
                "output_shape_digest": receipt.output_shape_digest,
                "output_content_digest": receipt.output_content_digest,
                "output_media_type": "application/vnd.apache.arrow.stream",
                "input_content_digest": receipt.input_content_digest,
                "policy_version": output_governance.policy_version,
                "acl_principal_ids": list(output_governance.acl.principal_ids),
                "acl_principal_emails": list(output_governance.acl.principal_emails),
                "acl_group_ids": list(output_governance.acl.group_ids),
                "acl_read_roles": list(output_governance.acl.roles),
                "acl_markings": list(output_governance.acl.markings),
                "prepared_receipt_digest": _sha256_bytes(
                    request.prepared_ref.encode("utf-8")
                ),
                "prep_evidence": evidence_payload,
                "prep_evidence_digest": evidence_digest,
            },
            structured_evidence=evidence_payload,
            trace_context=session.trace_context,
        )
        try:
            _check_cancel(deadline)
            from agent_utilities.knowledge_graph.core.session import use_session
            from agent_utilities.knowledge_graph.ingestion.envelope_ingest import (
                ingest_envelope,
            )

            engine = self._authority.native_engine(session=session)
            with use_session(session):
                commit = ingest_envelope(engine, envelope)
            if (
                not isinstance(commit, dict)
                or commit.get("status")
                not in {
                    "applied",
                    "idempotent_skip",
                    "success",
                    "skipped",
                }
                or commit.get("native_atomic") is not True
            ):
                raise NativeCommitUnavailable(
                    "native ChangeEnvelope commit was not atomically accepted"
                )
        except Exception as exc:  # noqa: BLE001 - compensate before surfacing
            if ref_acquired:
                try:
                    self._authority.unref_blob(stored_digest, session=session)
                except Exception:
                    pass
            if isinstance(exc, NativeCommitUnavailable):
                raise
            raise NativeCommitUnavailable(
                "native ChangeEnvelope commit failed"
            ) from exc
        return {
            "surface": "data_prep",
            "action": operation.value,
            "prepared_artifact_ref": prepared_ref,
            "commit": {
                "status": commit.get("status"),
                "envelope_id": commit.get("envelope_id"),
                "idempotency_key": commit.get("idempotency_key"),
                "native_atomic": commit.get("native_atomic"),
            },
            "evidence": self._public_evidence(result.evidence),
            "side_effects": ["native_change_envelope"],
        }


def _json_payload(raw: Any) -> Mapping[str, Any]:
    if not isinstance(raw, str):
        raise DataPrepToolError("params_json must be a JSON object")
    if len(raw.encode("utf-8")) > _MAX_PARAMS_BYTES:
        raise DataPrepToolError("params_json exceeds the bounded request size")
    try:
        value = json.loads(raw) if raw else {}
    except (TypeError, ValueError) as exc:
        raise DataPrepToolError("params_json must be valid JSON") from exc
    if not isinstance(value, dict):
        raise DataPrepToolError("params_json must decode to a JSON object")
    forbidden = {
        "arrow_ipc",
        "bytes",
        "data_b64",
        "file_path",
        "path",
        "url",
        "import_path",
        "python_code",
        "code",
        "checkpoint",
    }
    if forbidden.intersection(value):
        raise DataPrepToolError(
            "inline bytes, paths, executable code and checkpoints are forbidden"
        )
    plan = value.get("plan")
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except (TypeError, ValueError) as exc:
            raise DataPrepToolError("plan must be a JSON object") from exc
        value["plan"] = plan
    try:
        if len(_canonical_json(plan)) > _MAX_PLAN_BYTES:
            raise DataPrepToolError("plan exceeds the bounded request size")
    except DataPrepToolError:
        raise
    except (TypeError, ValueError) as exc:
        raise DataPrepToolError("plan must be a JSON object") from exc
    return value


def _as_str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def register_data_prep_tools(mcp: Any) -> None:
    """Register the condensed ``graph_data_prep`` MCP + REST twin."""

    @mcp.tool(
        name="graph_data_prep",
        description=(
            "Governed Arrow data preparation over the NE-108 kernel. Actions: "
            "'profile_dataset' (bounded privacy-safe profile), 'clean_dataset' "
            "(pure allow-listed preparation and opaque prepared artifact ref), "
            "'validate_prepared' (pure evidence/approval check), and "
            "'commit_prepared' (the only mutating action; native ChangeEnvelope "
            "with required ICV policy). params_json is a bounded typed object; "
            "use a tenant-bound artifact_ref or a small scalar records array. "
            "Inline Arrow IPC/bytes, paths, URLs, import paths, executable code "
            "and checkpoint fields are rejected."
        ),
        tags=["graph-os", "data-prep", "governance", "arrow"],
    )
    def graph_data_prep(
        action: Literal[
            "profile_dataset",
            "clean_dataset",
            "validate_prepared",
            "commit_prepared",
        ] = Field(
            default="profile_dataset",
            description="profile_dataset | clean_dataset | validate_prepared | commit_prepared",
        ),
        params_json: str = Field(
            default="{}",
            description=(
                "Bounded JSON object carrying the typed plan, immutable refs, "
                "artifact_ref or bounded scalar records, and resource budget."
            ),
        ),
    ) -> str:
        action = _as_str(action, "profile_dataset").strip().lower()
        try:
            payload = _json_payload(params_json)
            from agent_utilities.knowledge_graph.core.session import current_session

            session = current_session()
            if session is None:
                raise PermissionError("verified GraphSession is required")
            authority = _AUTHORITY_FACTORY(session)
            service = DataPrepService(authority)
            result = service.execute(action, payload, session=session)
            return json.dumps(result, sort_keys=True, separators=(",", ":"))
        except PermissionError as exc:
            from agent_utilities.security.error_surface import public_error_json

            return public_error_json(
                exc, code="permission_denied", context={"action": action}
            )
        except ArtifactAuthorityUnavailable as exc:
            from agent_utilities.security.error_surface import public_error_json

            return public_error_json(
                exc, code="dependency_unavailable", context={"action": action}
            )
        except NativeCommitUnavailable as exc:
            from agent_utilities.security.error_surface import public_error_json

            return public_error_json(
                exc, code="dependency_unavailable", context={"action": action}
            )
        except (DataPrepToolError, ValidationError, ValueError, TypeError) as exc:
            from agent_utilities.security.error_surface import public_error_json

            return public_error_json(
                exc, code="invalid_request", context={"action": action}
            )
        except Exception as exc:  # noqa: BLE001 - public boundary is privacy-safe
            from agent_utilities.security.error_surface import public_error_json

            return public_error_json(exc, context={"action": action})

    kg_server.REGISTERED_TOOLS["graph_data_prep"] = graph_data_prep
    kg_server.ACTION_TOOL_ROUTES["graph_data_prep"] = "/data/prep"


__all__ = [
    "ArtifactACL",
    "ArtifactAuthorityUnavailable",
    "DataPrepAction",
    "DataPrepAuthority",
    "DataPrepModelAuthority",
    "DataPrepRuntimeConfig",
    "DataPrepService",
    "NativeCommitUnavailable",
    "PrepBudget",
    "PrepRequest",
    "PreparedReceipt",
    "ResolvedArtifact",
    "register_data_prep_authority",
    "register_process_data_prep_runtime",
    "register_data_prep_tools",
]
