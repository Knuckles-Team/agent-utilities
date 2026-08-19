"""Typed connector preparation boundary.

(CONCEPT:AU-KG.ingest.change-envelope, AU-ECO.connector.incremental-poll-watermark)

This module binds the local Arrow preparation kernel to a connector's
validated mapping without adding another write path.  A connector supplies a
versioned, digest-pinned contract and an Arrow-table mapper; the preparer
returns ordinary :class:`~agent_utilities.knowledge_graph.ingestion.ChangeEnvelope`
objects for the existing native ingestion authority.

The boundary is deliberately side-effect free.  It does not call an engine,
advance a source cursor, or run SHACL/ICV itself.  Those authorities consume
the returned envelopes and remain responsible for durable admission.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, TypeAlias

from pydantic import Field, model_validator

from agent_utilities.protocols.source_connectors.checkpoint import ConnectorCheckpoint
from agent_utilities.protocols.epistemic_operations import ProtocolModel

from .kernel import (
    CleanPipeline,
    DataPrepError,
    InvalidRowsError,
    PlanExecutionError,
    PrepResult,
    RowModelRegistry,
    plan_digest,
    schema_digest,
)
from .models import CleanPlan, Digest, OpaqueReference

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope

__all__ = [
    "ConnectorArtifact",
    "ConnectorArtifacts",
    "ConnectorMapper",
    "ConnectorPreparation",
    "ConnectorPreparationError",
    "ConnectorPrepContract",
    "ConnectorPageLimits",
    "ConnectorPreparationDiagnostic",
    "PageCertification",
    "PageNotCertifiedError",
    "PreparedConnectorPage",
    "build_native_change_envelope",
]


ContractVersion: TypeAlias = Literal["connector-prep.v1"]
ValidationMode: TypeAlias = Literal["strict", "quarantine"]
PageOutcome: TypeAlias = Literal["complete", "quarantined", "partial", "failed"]
ArtifactKind: TypeAlias = Literal[
    "raw_model",
    "prep_plan",
    "arrow_schema",
    "mapping",
    "shacl",
    "icv",
]
DiagnosticCode: TypeAlias = Literal[
    "page_limit_exceeded",
    "cardinality_limit_exceeded",
    "diagnostic_limit_exceeded",
    "validation_failed",
    "rows_quarantined",
    "prep_failed",
    "mapping_failed",
    "mapping_contract_violation",
    "mapping_artifact_unpinned",
    "duplicate_idempotency_key",
    "idempotency_key_mismatch",
    "arrow_schema_mismatch",
    "checkpoint_blocked",
    "snapshot_not_verified",
    "snapshot_operation_forbidden",
    "deletion_blocked_on_partial",
]
AnnotatedReference: TypeAlias = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    ),
]

_DIAGNOSTIC_SUMMARIES: dict[DiagnosticCode, str] = {
    "page_limit_exceeded": "The source page exceeded its declared bounded limit.",
    "cardinality_limit_exceeded": "The mapped page exceeded its declared cardinality limit.",
    "diagnostic_limit_exceeded": "The diagnostic limit was reached; details were truncated.",
    "validation_failed": "The page failed the declared row-model or preparation gate.",
    "rows_quarantined": "One or more rows were explicitly quarantined.",
    "prep_failed": "The Arrow preparation operation failed closed.",
    "mapping_failed": "The connector mapping operation failed closed.",
    "mapping_contract_violation": "The mapper returned an envelope outside its typed contract.",
    "mapping_artifact_unpinned": "The mapper is not bound to the declared mapping artifact.",
    "duplicate_idempotency_key": "The mapped page contained a duplicate idempotency key.",
    "idempotency_key_mismatch": "The envelope idempotency key is not derived from source identity.",
    "arrow_schema_mismatch": "The prepared Arrow schema did not match its pinned artifact.",
    "checkpoint_blocked": "The page is not certified to advance its source checkpoint.",
    "snapshot_not_verified": "The page is not an explicitly verified authoritative snapshot.",
    "snapshot_operation_forbidden": "A snapshot marker may only be created by the page authority.",
    "deletion_blocked_on_partial": "A failed or partial page cannot emit a deletion operation.",
}


class ConnectorPreparationDiagnostic(ProtocolModel):
    """A bounded, stable and redacted preparation diagnostic.

    The summary is selected from an allow-list by code.  Callers cannot put
    exception text, row values, tokens, or provider responses into the
    diagnostic.  ``field_pointer`` is an optional JSON-pointer-like location;
    it is intentionally not a copy of the rejected value.
    """

    code: DiagnosticCode
    field_pointer: str | None = Field(
        default=None,
        max_length=128,
        pattern=r"^/[A-Za-z0-9_.~/-]*$",
    )
    row_index: int | None = Field(default=None, ge=0, le=1_000_000)
    summary: str = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def summary_is_redacted_and_stable(
        self,
    ) -> ConnectorPreparationDiagnostic:
        expected = _DIAGNOSTIC_SUMMARIES[self.code]
        if self.summary != expected:
            raise ValueError("diagnostic summaries are code-selected and redacted")
        if self.field_pointer is not None and (
            not self.field_pointer.startswith("/")
            or "\n" in self.field_pointer
            or "\r" in self.field_pointer
        ):
            raise ValueError("field pointers must be bounded JSON-pointer locations")
        return self

    @classmethod
    def for_code(
        cls,
        code: DiagnosticCode,
        *,
        field_pointer: str | None = None,
        row_index: int | None = None,
    ) -> ConnectorPreparationDiagnostic:
        """Create a diagnostic without accepting caller-supplied free text."""

        return cls(
            code=code,
            field_pointer=field_pointer,
            row_index=row_index,
            summary=_DIAGNOSTIC_SUMMARIES[code],
        )


class ConnectorArtifact(ProtocolModel):
    """One immutable opaque artifact reference and its content digest."""

    kind: ArtifactKind
    ref: OpaqueReference
    digest: Digest


class ConnectorArtifacts(ProtocolModel):
    """All artifacts that must travel with a prepared connector page."""

    raw_model: ConnectorArtifact
    prep_plan: ConnectorArtifact
    arrow_schema: ConnectorArtifact
    mapping: ConnectorArtifact
    shacl: ConnectorArtifact
    icv: ConnectorArtifact

    @model_validator(mode="after")
    def artifact_kinds_match_fields(self) -> ConnectorArtifacts:
        for field_name in (
            "raw_model",
            "prep_plan",
            "arrow_schema",
            "mapping",
            "shacl",
            "icv",
        ):
            artifact = getattr(self, field_name)
            if artifact.kind != field_name:
                raise ValueError(
                    f"connector artifact {field_name!r} has the wrong kind"
                )
        return self


class ConnectorPageLimits(ProtocolModel):
    """Finite limits for one connector page and its certification record."""

    max_rows: int = Field(ge=1, le=1_000_000)
    max_columns: int = Field(ge=1, le=1_024)
    max_cardinality: int = Field(ge=1, le=1_000_000)
    max_diagnostics: int = Field(ge=1, le=1_024)


class ConnectorPrepContract(ProtocolModel):
    """Versioned connector boundary binding prep, mapping and engine gates."""

    contract_version: ContractVersion
    connector: AnnotatedReference
    tenant: str = Field(
        default="",
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$|^$",
    )
    source_instance: str = Field(
        default="",
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$|^$",
    )
    schema_version: AnnotatedReference = Field(min_length=1, max_length=128)
    ontology_mapping_version: AnnotatedReference = Field(
        min_length=1,
        max_length=128,
    )
    plan: CleanPlan
    artifacts: ConnectorArtifacts
    limits: ConnectorPageLimits
    validation_mode: ValidationMode

    @model_validator(mode="after")
    def bindings_are_authoritative(self) -> ConnectorPrepContract:
        if self.artifacts.raw_model.ref != self.plan.model_ref:
            raise ValueError("raw-model artifact must bind the CleanPlan model_ref")
        if self.artifacts.prep_plan.ref != self.plan.plan_ref:
            raise ValueError("prep-plan artifact must bind the CleanPlan plan_ref")
        if self.artifacts.prep_plan.digest != plan_digest(self.plan):
            raise ValueError("prep-plan artifact digest does not match the CleanPlan")
        if self.plan.model_digest is not None:
            if self.artifacts.raw_model.digest != self.plan.model_digest:
                raise ValueError("raw-model artifact digest does not match the CleanPlan")
        expected_disposition = (
            "fail" if self.validation_mode == "strict" else "quarantine"
        )
        if self.plan.invalid_row_disposition != expected_disposition:
            raise ValueError(
                "validation_mode and CleanPlan.invalid_row_disposition must agree"
            )
        return self


class ConnectorPreparationError(DataPrepError):
    """Fail-closed preparation error with only stable redacted diagnostics."""

    def __init__(
        self,
        diagnostic: ConnectorPreparationDiagnostic,
        *,
        diagnostics: Sequence[ConnectorPreparationDiagnostic] = (),
    ) -> None:
        self.diagnostic = diagnostic
        self.diagnostics = tuple(diagnostics) or (diagnostic,)
        super().__init__(f"connector preparation failed: {diagnostic.code}")


class PageNotCertifiedError(ConnectorPreparationError):
    """Raised when a caller requests a checkpoint or snapshot without proof."""


class ConnectorRowMapper(Protocol):
    """Typed Arrow-table mapping callback supplied by a connector domain."""

    def __call__(self, table: Any) -> Sequence[ChangeEnvelope]:
        """Map one prepared Arrow table to native envelopes without committing."""


@dataclass(frozen=True, slots=True)
class ConnectorMapper:
    """Immutable mapping callback pinned to the declared mapping artifact."""

    artifact: ConnectorArtifact
    map_table: ConnectorRowMapper

    def __post_init__(self) -> None:
        if self.artifact.kind != "mapping":
            raise ValueError("ConnectorMapper must use a mapping artifact")
        if not callable(self.map_table):
            raise TypeError("ConnectorMapper.map_table must be callable")

    def __call__(self, table: Any) -> Sequence[ChangeEnvelope]:
        return self.map_table(table)


class PageCertification(ProtocolModel):
    """Bounded proof describing whether a page may advance a source cursor."""

    outcome: PageOutcome
    fetch_complete: bool
    checkpoint_eligible: bool
    rows_in: int = Field(ge=0, le=1_000_000)
    rows_out: int = Field(ge=0, le=1_000_000)
    mapped_rows: int = Field(ge=0, le=1_000_000)
    quarantined_rows: int = Field(ge=0, le=1_000_000)
    diagnostics: tuple[ConnectorPreparationDiagnostic, ...] = Field(
        default=(),
        max_length=1_024,
    )
    diagnostics_truncated: bool = False
    replay_digest: Digest | None = None

    @model_validator(mode="after")
    def checkpoint_requires_complete_page(self) -> PageCertification:
        expected = (
            self.outcome == "complete"
            and self.fetch_complete
            and not self.diagnostics
            and self.replay_digest is not None
        )
        if self.checkpoint_eligible != expected:
            raise ValueError(
                "only a complete, diagnostic-free page with a replay digest may "
                "be checkpoint-eligible"
            )
        if self.mapped_rows > self.rows_out:
            raise ValueError("mapped row count cannot exceed prepared row count")
        return self


@dataclass(frozen=True, slots=True)
class PreparedConnectorPage:
    """Side-effect-free result handed to the existing native ingest path."""

    table: Any
    evidence: Any
    envelopes: tuple[ChangeEnvelope, ...]
    certification: PageCertification
    checkpoint: ConnectorCheckpoint | None
    contract: ConnectorPrepContract

    @property
    def checkpoint_eligible(self) -> bool:
        """Whether the page may expose its caller-supplied checkpoint."""

        return self.certification.checkpoint_eligible

    def checkpoint_candidate(self) -> ConnectorCheckpoint | None:
        """Return a defensive checkpoint copy, or ``None`` when uncertified."""

        if not self.checkpoint_eligible or self.checkpoint is None:
            return None
        return copy.deepcopy(self.checkpoint)

    def snapshot_complete(
        self,
        *,
        live_ids: Iterable[str],
        authoritative_empty: bool = False,
    ) -> ChangeEnvelope:
        """Create the only page-level snapshot marker allowed by this seam.

        A failed, partial, or quarantined page cannot produce a deletion marker.
        An empty live-id set additionally requires an explicit
        ``authoritative_empty=True`` proof, so an unavailable/partial fetch can
        never become a verified-empty source.
        """

        if not self.checkpoint_eligible or self.checkpoint is None:
            raise PageNotCertifiedError(
                ConnectorPreparationDiagnostic.for_code("checkpoint_blocked")
            )
        id_set: set[str] = set()
        for value in live_ids:
            normalized = str(value)
            if normalized in id_set:
                continue
            if len(id_set) >= self.contract.limits.max_cardinality:
                raise PageNotCertifiedError(
                    ConnectorPreparationDiagnostic.for_code(
                        "cardinality_limit_exceeded"
                    )
                )
            id_set.add(normalized)
        ids = tuple(sorted(id_set))
        if not ids and not authoritative_empty:
            raise PageNotCertifiedError(
                ConnectorPreparationDiagnostic.for_code("snapshot_not_verified")
            )
        from agent_utilities.knowledge_graph.ingestion.change_envelope import (
            ChangeEnvelope,
        )

        checkpoint_token = self.checkpoint.to_json()
        marker = ChangeEnvelope.snapshot_complete(
            connector=self.contract.connector,
            tenant=self.contract.tenant,
            source_instance=self.contract.source_instance,
            checkpoint=checkpoint_token,
            live_ids=ids,
            fetch_ok=True,
            schema_version=self.contract.schema_version,
            ontology_mapping_version=self.contract.ontology_mapping_version,
            provenance={"authoritative_empty": bool(not ids)},
        )
        return _annotate_envelope(
            marker,
            self.contract,
            self.certification.replay_digest,
            evidence_summary={
                "outcome": self.certification.outcome,
                "checkpoint_eligible": self.certification.checkpoint_eligible,
            },
        )


class ConnectorPreparation:
    """Prepare one bounded Arrow page and map it to native envelopes."""

    __slots__ = ("_contract", "_mapper", "_model_registry")

    def __init__(
        self,
        contract: ConnectorPrepContract,
        *,
        model_registry: RowModelRegistry,
        mapper: ConnectorMapper,
    ) -> None:
        if mapper.artifact != contract.artifacts.mapping:
            raise ConnectorPreparationError(
                ConnectorPreparationDiagnostic.for_code("mapping_artifact_unpinned")
            )
        try:
            model_registry.resolve(
                contract.plan.model_ref,
                contract.artifacts.raw_model.digest,
            )
        except PlanExecutionError as exc:
            # The registry's error text is intentionally not propagated; model
            # names and implementation details are not safe diagnostics.
            raise ConnectorPreparationError(
                ConnectorPreparationDiagnostic.for_code("validation_failed")
            ) from exc
        self._contract = contract
        self._mapper = mapper
        self._model_registry = model_registry

    @property
    def contract(self) -> ConnectorPrepContract:
        """The immutable contract used by this preparer."""

        return self._contract

    def prepare(
        self,
        table: Any,
        *,
        checkpoint: ConnectorCheckpoint | None = None,
        fetch_complete: bool = False,
    ) -> PreparedConnectorPage:
        """Run prep, mapping and certification without publishing anything."""

        try:
            from .kernel import ArrowAdapter

            table = ArrowAdapter.as_table(table, profile=self._contract.plan.profile)
        except Exception as exc:  # noqa: BLE001 - convert to stable diagnostics
            return self._failure_page(
                checkpoint=checkpoint,
                fetch_complete=fetch_complete,
                diagnostic=(
                    ConnectorPreparationDiagnostic.for_code("prep_failed")
                ),
                cause=exc,
            )

        limits = self._contract.limits
        if table.num_rows > limits.max_rows or table.num_columns > limits.max_columns:
            return self._failure_page(
                checkpoint=checkpoint,
                fetch_complete=fetch_complete,
                diagnostic=ConnectorPreparationDiagnostic.for_code(
                    "page_limit_exceeded"
                ),
            )

        try:
            result = CleanPipeline(
                self._contract.plan,
                model_registry=self._model_registry,
            ).run(table)
        except InvalidRowsError as exc:
            return self._strict_or_failure(
                table=table,
                checkpoint=checkpoint,
                fetch_complete=fetch_complete,
                diagnostic=ConnectorPreparationDiagnostic.for_code(
                    "validation_failed",
                ),
                cause=exc,
            )
        except (DataPrepError, TypeError, ValueError) as exc:
            return self._strict_or_failure(
                table=table,
                checkpoint=checkpoint,
                fetch_complete=fetch_complete,
                diagnostic=ConnectorPreparationDiagnostic.for_code("prep_failed"),
                cause=exc,
            )

        actual_schema_digest = schema_digest(result.table)
        if actual_schema_digest != self._contract.artifacts.arrow_schema.digest:
            return self._strict_or_failure(
                table=result.table,
                evidence=result.evidence,
                checkpoint=checkpoint,
                fetch_complete=fetch_complete,
                diagnostic=ConnectorPreparationDiagnostic.for_code(
                    "arrow_schema_mismatch"
                ),
            )

        diagnostics: list[ConnectorPreparationDiagnostic] = []
        if result.evidence.outcome == "quarantined":
            diagnostics.append(
                ConnectorPreparationDiagnostic.for_code("rows_quarantined")
            )

        try:
            mapped_value = self._mapper(result.table)
        except Exception as exc:  # noqa: BLE001 - mapper text is untrusted
            return self._strict_or_failure(
                table=result.table,
                evidence=result.evidence,
                checkpoint=checkpoint,
                fetch_complete=fetch_complete,
                diagnostic=ConnectorPreparationDiagnostic.for_code("mapping_failed"),
                cause=exc,
            )
        if not isinstance(mapped_value, Sequence):
            return self._strict_or_failure(
                table=result.table,
                evidence=result.evidence,
                checkpoint=checkpoint,
                fetch_complete=fetch_complete,
                diagnostic=ConnectorPreparationDiagnostic.for_code(
                    "mapping_contract_violation"
                ),
            )
        if len(mapped_value) > limits.max_rows:
            return self._strict_or_failure(
                table=result.table,
                evidence=result.evidence,
                checkpoint=checkpoint,
                fetch_complete=fetch_complete,
                diagnostic=ConnectorPreparationDiagnostic.for_code(
                    "page_limit_exceeded"
                ),
            )
        mapped = tuple(mapped_value)

        mapping_diagnostic = self._validate_mapped(
            result,
            mapped,
            fetch_complete=fetch_complete,
        )
        if mapping_diagnostic is not None:
            return self._strict_or_failure(
                table=result.table,
                evidence=result.evidence,
                checkpoint=checkpoint,
                fetch_complete=fetch_complete,
                diagnostic=mapping_diagnostic,
            )

        mapped = tuple(
            sorted(
                mapped,
                key=lambda envelope: (
                    envelope.source_object_id,
                    envelope.source_version,
                    envelope.operation,
                    envelope.idempotency_key,
                ),
            )
        )
        replay = _replay_digest(
            self._contract,
            actual_schema_digest,
            mapped,
        )
        outcome: PageOutcome = (
            "quarantined" if result.evidence.outcome == "quarantined" else "complete"
        )
        bounded, truncated = _bound_diagnostics(diagnostics, limits.max_diagnostics)
        certification = PageCertification(
            outcome=outcome,
            fetch_complete=fetch_complete,
            checkpoint_eligible=(
                outcome == "complete" and fetch_complete and not bounded
            ),
            rows_in=result.evidence.rows_in,
            rows_out=result.evidence.rows_out,
            mapped_rows=len(mapped),
            quarantined_rows=result.evidence.quarantined_rows,
            diagnostics=tuple(bounded),
            diagnostics_truncated=truncated,
            replay_digest=replay,
        )
        annotated = tuple(
            _annotate_envelope(
                item,
                self._contract,
                replay,
                evidence_summary=_evidence_summary(result.evidence),
            )
            for item in mapped
        )
        return PreparedConnectorPage(
            table=result.table,
            evidence=result.evidence,
            envelopes=annotated,
            certification=certification,
            checkpoint=copy.deepcopy(checkpoint),
            contract=self._contract,
        )

    def _validate_mapped(
        self,
        result: PrepResult,
        mapped: Sequence[ChangeEnvelope],
        *,
        fetch_complete: bool,
    ) -> ConnectorPreparationDiagnostic | None:
        limits = self._contract.limits
        if len(mapped) != result.table.num_rows:
            return ConnectorPreparationDiagnostic.for_code(
                "mapping_contract_violation",
                field_pointer="/rows",
            )
        if len(mapped) > limits.max_rows:
            return ConnectorPreparationDiagnostic.for_code("page_limit_exceeded")
        if len({item.source_object_id for item in mapped}) > limits.max_cardinality:
            return ConnectorPreparationDiagnostic.for_code(
                "cardinality_limit_exceeded"
            )
        from agent_utilities.knowledge_graph.ingestion.change_envelope import (
            ChangeEnvelope,
        )

        seen: set[str] = set()
        for item in mapped:
            if not isinstance(item, ChangeEnvelope):
                return ConnectorPreparationDiagnostic.for_code(
                    "mapping_contract_violation",
                    field_pointer="/envelopes",
                )
            if item.connector != self._contract.connector:
                return ConnectorPreparationDiagnostic.for_code(
                    "mapping_contract_violation",
                    field_pointer="/connector",
                )
            if item.tenant != self._contract.tenant:
                return ConnectorPreparationDiagnostic.for_code(
                    "mapping_contract_violation",
                    field_pointer="/tenant",
                )
            if item.source_instance != self._contract.source_instance:
                return ConnectorPreparationDiagnostic.for_code(
                    "mapping_contract_violation",
                    field_pointer="/source_instance",
                )
            if item.schema_version != self._contract.schema_version:
                return ConnectorPreparationDiagnostic.for_code(
                    "mapping_contract_violation",
                    field_pointer="/schema_version",
                )
            if item.ontology_mapping_version != self._contract.ontology_mapping_version:
                return ConnectorPreparationDiagnostic.for_code(
                    "mapping_contract_violation",
                    field_pointer="/ontology_mapping_version",
                )
            if item.operation == "snapshot_complete":
                return ConnectorPreparationDiagnostic.for_code(
                    "snapshot_operation_forbidden",
                    field_pointer="/operation",
                )
            if (
                (result.evidence.outcome != "complete" or not fetch_complete)
                and item.operation == "delete"
            ):
                return ConnectorPreparationDiagnostic.for_code(
                    "deletion_blocked_on_partial",
                    field_pointer="/operation",
                )
            if item.checkpoint is not None:
                return ConnectorPreparationDiagnostic.for_code(
                    "mapping_contract_violation",
                    field_pointer="/checkpoint",
                )
            if not item.source_object_id or not item.source_version:
                return ConnectorPreparationDiagnostic.for_code(
                    "mapping_contract_violation",
                    field_pointer="/source_object_id",
                )
            expected_key = replace(item, idempotency_key="").idempotency_key
            if item.idempotency_key != expected_key:
                return ConnectorPreparationDiagnostic.for_code(
                    "idempotency_key_mismatch",
                )
            if item.idempotency_key in seen:
                return ConnectorPreparationDiagnostic.for_code(
                    "duplicate_idempotency_key",
                )
            seen.add(item.idempotency_key)
        return None

    def _strict_or_failure(
        self,
        *,
        table: Any,
        checkpoint: ConnectorCheckpoint | None,
        fetch_complete: bool,
        diagnostic: ConnectorPreparationDiagnostic,
        evidence: Any = None,
        cause: BaseException | None = None,
    ) -> PreparedConnectorPage:
        if self._contract.validation_mode == "strict":
            error = ConnectorPreparationError(diagnostic)
            if cause is not None:
                raise error from cause
            raise error
        return self._failure_page(
            checkpoint=checkpoint,
            fetch_complete=fetch_complete,
            diagnostic=diagnostic,
            evidence=evidence,
            table=table,
            outcome="partial",
        )

    def _failure_page(
        self,
        *,
        checkpoint: ConnectorCheckpoint | None,
        fetch_complete: bool,
        diagnostic: ConnectorPreparationDiagnostic,
        evidence: Any = None,
        table: Any = None,
        cause: BaseException | None = None,
        outcome: PageOutcome = "failed",
    ) -> PreparedConnectorPage:
        if self._contract.validation_mode == "strict" and outcome == "failed":
            error = ConnectorPreparationError(diagnostic)
            if cause is not None:
                raise error from cause
            raise error

        # ``cause`` is intentionally accepted only to preserve exception
        # chaining for logs owned by the caller; its text never reaches the
        # typed diagnostic or the certification record.
        del cause
        bounded, truncated = _bound_diagnostics(
            [diagnostic],
            self._contract.limits.max_diagnostics,
        )
        rows_in = int(getattr(table, "num_rows", 0) or 0)
        rows_out = int(getattr(evidence, "rows_out", 0) or 0)
        quarantined_rows = int(getattr(evidence, "quarantined_rows", 0) or 0)
        certification = PageCertification(
            outcome=outcome,
            fetch_complete=fetch_complete,
            checkpoint_eligible=False,
            rows_in=rows_in,
            rows_out=rows_out,
            mapped_rows=0,
            quarantined_rows=quarantined_rows,
            diagnostics=tuple(bounded),
            diagnostics_truncated=truncated,
            replay_digest=None,
        )
        return PreparedConnectorPage(
            table=table,
            evidence=evidence,
            envelopes=(),
            certification=certification,
            checkpoint=copy.deepcopy(checkpoint),
            contract=self._contract,
        )


def build_native_change_envelope(
    record: Mapping[str, Any],
    *,
    contract: ConnectorPrepContract,
    operation: Literal["upsert", "delete"] = "upsert",
    id_field: str = "id",
    version_field: str = "updatedAt",
    **overrides: Any,
) -> ChangeEnvelope:
    """Build one native envelope while binding it to the prep contract.

    This is a convenience seam for a domain mapper; it delegates construction
    to :meth:`ChangeEnvelope.from_connector_record` and never ingests.  Page
    checkpoint state is deliberately not accepted here because only a
    certified page may advance it.
    """

    from agent_utilities.knowledge_graph.ingestion.change_envelope import (
        ChangeEnvelope,
    )

    forbidden = {
        "connector",
        "tenant",
        "source_instance",
        "schema_version",
        "ontology_mapping_version",
        "checkpoint",
    }
    if forbidden.intersection(overrides):
        raise ValueError("native envelope authority fields are contract-owned")
    return ChangeEnvelope.from_connector_record(
        dict(record),
        connector=contract.connector,
        tenant=contract.tenant,
        source_instance=contract.source_instance,
        operation=operation,
        id_field=id_field,
        version_field=version_field,
        schema_version=contract.schema_version,
        ontology_mapping_version=contract.ontology_mapping_version,
        # An empty tenant is still explicit single-tenant scope.  Do not let
        # ambient process context silently replace the contract's tenant.
        session=object(),
        **overrides,
    )


def _bound_diagnostics(
    diagnostics: Sequence[ConnectorPreparationDiagnostic],
    limit: int,
) -> tuple[list[ConnectorPreparationDiagnostic], bool]:
    if len(diagnostics) <= limit:
        return list(diagnostics), False
    return list(diagnostics[:limit]), True


def _replay_digest(
    contract: ConnectorPrepContract,
    arrow_schema_digest: str,
    envelopes: Sequence[ChangeEnvelope],
) -> str:
    payload = {
        "contract_version": contract.contract_version,
        "artifacts": contract.artifacts.model_dump(mode="json"),
        "arrow_schema_digest": arrow_schema_digest,
        "idempotency_keys": [item.idempotency_key for item in envelopes],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _annotate_envelope(
    envelope: ChangeEnvelope,
    contract: ConnectorPrepContract,
    replay_digest: str | None,
    *,
    evidence_summary: Mapping[str, Any] | None = None,
) -> ChangeEnvelope:
    lineage = dict(envelope.provenance)
    lineage["connector_preparation"] = {
        "contract_version": contract.contract_version,
        "artifacts": contract.artifacts.model_dump(mode="json"),
        "validation_mode": contract.validation_mode,
        "replay_digest": replay_digest,
        "evidence": dict(evidence_summary or {}),
    }
    return replace(envelope, provenance=lineage)


def _evidence_summary(evidence: Any) -> dict[str, Any]:
    """Project PrepEvidence into a bounded, content-free envelope summary."""

    return {
        "evidence_version": getattr(evidence, "evidence_version", None),
        "algorithm": getattr(evidence, "algorithm", None),
        "algorithm_version": getattr(evidence, "algorithm_version", None),
        "outcome": getattr(evidence, "outcome", None),
        "checkpoint_eligible": bool(
            getattr(evidence, "checkpoint_eligible", False)
        ),
        "plan_digest": getattr(evidence, "plan_digest", None),
        "input_schema_digest": getattr(evidence, "input_schema_digest", None),
        "output_schema_digest": getattr(evidence, "output_schema_digest", None),
        "rows_in": int(getattr(evidence, "rows_in", 0) or 0),
        "rows_out": int(getattr(evidence, "rows_out", 0) or 0),
        "dropped_rows": int(getattr(evidence, "dropped_rows", 0) or 0),
        "quarantined_rows": int(getattr(evidence, "quarantined_rows", 0) or 0),
    }
