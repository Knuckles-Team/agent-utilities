"""Pure Arrow preparation, bounded profiling and privacy-safe evidence.

This module is the only execution authority for :mod:`agent_utilities.data_prep`.
It accepts an Arrow table, validates a frozen :class:`CleanPlan`, and applies
the five allow-listed verbs with explicit dispatch.  It never imports pandas,
uses dynamic method lookup, or mutates the caller's plan.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from .models import (
    CanonicalNames,
    CleanPlan,
    ColumnProfile,
    Dedupe,
    FillNulls,
    LocalProfile,
    NullPolicy,
    PrepEvidence,
    ProfileLimits,
    ProfileResult,
    ProfileSelector,
    ProfileTarget,
    ProfileTargetKind,
    ProfileWarningCode,
    QuantilePoint,
    QuarantineOutcome,
    RowOutcome,
    SafeCast,
    StepEvidence,
    TopKEntry,
)


class DataPrepError(ValueError):
    """Base class for fail-closed preparation errors."""


class DataPrepDependencyError(DataPrepError):
    """Raised when an Arrow operation is requested without ``pyarrow``."""


class ProfileLimitError(DataPrepError):
    """Raised before work starts when a table exceeds the local profile."""


class ProfileSelectorError(DataPrepError):
    """Raised when a profile selector is stale or outside the schema."""


class ProfileDeadlineError(ProfileLimitError):
    """Raised when a profile cannot finish inside its declared deadline."""


class PlanExecutionError(DataPrepError):
    """Raised when a typed plan cannot be applied to the current schema."""


class InvalidRowsError(DataPrepError):
    """Raised for rejected rows when the plan selected fail-closed ``fail``."""

    def __init__(self, row_count: int, *, reason: str = "declared validation") -> None:
        self.row_count = row_count
        self.reason = reason
        super().__init__(f"{row_count} row(s) failed {reason}")


@dataclass(frozen=True, slots=True)
class PrepResult:
    """Prepared Arrow table plus its bounded operation evidence."""

    table: Any
    evidence: PrepEvidence


_CANONICAL_NAME = re.compile(r"[^a-z0-9]+")
_MAX_BATCH_ROWS = 65_536
_ARROW_TYPE_BUILDERS = {
    "bool": lambda pa: pa.bool_(),
    "int8": lambda pa: pa.int8(),
    "int16": lambda pa: pa.int16(),
    "int32": lambda pa: pa.int32(),
    "int64": lambda pa: pa.int64(),
    "uint8": lambda pa: pa.uint8(),
    "uint16": lambda pa: pa.uint16(),
    "uint32": lambda pa: pa.uint32(),
    "uint64": lambda pa: pa.uint64(),
    "float32": lambda pa: pa.float32(),
    "float64": lambda pa: pa.float64(),
    "string": lambda pa: pa.string(),
}
_SIGNED_RANK = {"int8": 0, "int16": 1, "int32": 2, "int64": 3}
_UNSIGNED_RANK = {"uint8": 0, "uint16": 1, "uint32": 2, "uint64": 3}


def _require_pyarrow() -> Any:
    """Import optional Arrow only at the operation boundary."""

    try:
        import pyarrow as pa
        import pyarrow.compute as pc
    except ImportError as exc:  # pragma: no cover - exercised in lean installs
        raise DataPrepDependencyError(
            "pyarrow is required for the Arrow data-prep kernel; install the "
            "optional 'pyarrow' extra"
        ) from exc
    return pa, pc


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _scalar_shape(value: Any) -> str:
    """Return a non-secret type marker for a strict fill scalar."""

    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    raise TypeError("clean-plan scalar has an unsupported type")


def _structural_plan(plan: CleanPlan) -> dict[str, Any]:
    """Copy a plan while replacing fill literals with type-only markers."""

    structural = plan.model_dump(mode="json")
    for step in structural["steps"]:
        if step["verb"] == "fill_nulls":
            step["fills"] = {
                column: {"type": _scalar_shape(value)}
                for column, value in sorted(step["fills"].items())
            }
    return structural


def plan_digest(plan: CleanPlan) -> str:
    """Return a stable structural digest that never hashes fill literals."""

    return _digest(_canonical_json(_structural_plan(plan)))


@dataclass(frozen=True, slots=True)
class RegisteredRowModel:
    """Immutable approved model entry selected by reference, never import path."""

    ref: str
    model: type[BaseModel]
    digest: str


def row_model_digest(row_model: type[BaseModel]) -> str:
    """Digest a model's JSON schema for immutable registry pinning."""

    try:
        schema = row_model.model_json_schema(mode="validation")
    except (TypeError, ValueError) as exc:
        raise PlanExecutionError("row_model schema cannot be inspected") from exc
    return _digest(_canonical_json(schema))


class RowModelRegistry:
    """Small immutable registry for approved strict Pydantic row models.

    The plan supplies only ``model_ref`` (and may pin ``model_digest``).  Model
    classes are registered by trusted application code, so the plan can never
    name an import path or execute caller-selected code.
    """

    __slots__ = ("_entries",)

    def __init__(self, models: Mapping[str, type[BaseModel]]) -> None:
        entries: dict[str, RegisteredRowModel] = {}
        for ref, model in models.items():
            if not isinstance(ref, str) or not ref:
                raise PlanExecutionError(
                    "row-model registry references must be non-empty"
                )
            _validate_row_model_class(model)
            entries[ref] = RegisteredRowModel(
                ref=ref,
                model=model,
                digest=row_model_digest(model),
            )
        self._entries = MappingProxyType(entries)

    def resolve(
        self, ref: str, expected_digest: str | None = None
    ) -> RegisteredRowModel:
        """Resolve an approved model and optionally enforce its pinned digest."""

        entry = self._entries.get(ref)
        if entry is None:
            raise PlanExecutionError("row-model reference is not approved")
        _validate_row_model_class(entry.model)
        if row_model_digest(entry.model) != entry.digest:
            raise PlanExecutionError(
                "approved row-model entry changed after registration"
            )
        if expected_digest is not None and expected_digest != entry.digest:
            raise PlanExecutionError(
                "row-model digest does not match the approved registry entry"
            )
        return entry


def schema_digest(table: Any) -> str:
    """Digest field names/types/nullability, excluding Arrow metadata."""

    pa, _ = _require_pyarrow()
    if not isinstance(table, pa.Table):
        raise TypeError("schema_digest expects a pyarrow.Table")
    schema = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in table.schema
    ]
    return _digest(_canonical_json(schema))


def _check_profile(table: Any, profile: LocalProfile) -> None:
    pa, _ = _require_pyarrow()
    if not isinstance(table, pa.Table):
        raise TypeError("data-prep operations require a pyarrow.Table")
    if table.num_rows > profile.max_rows:
        raise ProfileLimitError("table exceeds the local row limit")
    if table.num_columns > profile.max_columns:
        raise ProfileLimitError("table exceeds the local column limit")
    if table.nbytes > profile.max_bytes:
        raise ProfileLimitError("table exceeds the local byte limit")


class ArrowAdapter:
    """Bounded adapters for Arrow tables and record batches.

    There is intentionally no pandas/JSON/row-list adapter.  A caller that has
    records must materialize them into Arrow at its governed boundary first.
    """

    @staticmethod
    def as_table(value: Any, *, profile: LocalProfile | None = None) -> Any:
        """Validate an existing table and optionally apply profile limits."""

        pa, _ = _require_pyarrow()
        if not isinstance(value, pa.Table):
            raise TypeError(
                "expected pyarrow.Table; row-oriented adapters are not supported"
            )
        if profile is not None:
            _check_profile(value, profile)
        return value

    @staticmethod
    def from_batches(
        batches: Sequence[Any],
        *,
        profile: LocalProfile,
    ) -> Any:
        """Create a table from a finite, profile-bounded batch sequence."""

        pa, _ = _require_pyarrow()
        materialized = []
        row_count = 0
        for batch in batches:
            if not isinstance(batch, pa.RecordBatch):
                raise TypeError("all batches must be pyarrow.RecordBatch instances")
            row_count += batch.num_rows
            if row_count > profile.max_rows:
                raise ProfileLimitError("record batches exceed the local row limit")
            materialized.append(batch)
        if not materialized:
            raise ValueError("at least one Arrow record batch is required")
        table = pa.Table.from_batches(materialized)
        _check_profile(table, profile)
        return table


def _profile_deadline(started: float, limits: ProfileLimits) -> float:
    return started + limits.deadline_ms / 1000


def _check_profile_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise ProfileDeadlineError("profile deadline exceeded")


def _profile_number(value: Any) -> int | float | None:
    """Return finite JSON-safe numeric output, never NaN or infinity."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if isinstance(value, int):
        return value
    return number


def _profile_dtype(dtype: Any, pa: Any) -> str:
    """Return a bounded dtype label without nested field names."""

    if pa.types.is_struct(dtype):
        return "struct"
    rendered = str(dtype)
    if "<" in rendered:
        rendered = rendered.split("<", 1)[0]
    return rendered[:128]


def _profile_warning(value: str) -> ProfileWarningCode:
    # The model is the allow-list authority; this cast keeps the kernel's
    # branch-local warning construction readable without accepting caller text.
    return value  # type: ignore[return-value]


def _column_profile(
    table: Any,
    ordinal: int,
    pa: Any,
    pc: Any,
    *,
    limits: ProfileLimits,
    deadline: float,
) -> tuple[ColumnProfile, int, bool]:
    """Profile one bounded Arrow column without retaining field names."""

    _check_profile_deadline(deadline)
    values = table.column(ordinal).combine_chunks()
    rows = table.num_rows
    null_count = int(pc.sum(pc.cast(pc.is_null(values), pa.int64())).as_py() or 0)
    null_rate = null_count / rows if rows else 0.0
    distinct_count: int | None
    warnings: list[ProfileWarningCode] = []
    try:
        distinct_count = int(pc.count_distinct(values).as_py() or 0)
    except (pa.ArrowException, NotImplementedError, TypeError, ValueError):
        distinct_count = None
        warnings.append(_profile_warning("cardinality_unavailable"))
    if distinct_count is not None and distinct_count > limits.max_cardinality:
        raise ProfileLimitError("profile cardinality exceeds the declared limit")
    if rows == 0 or null_count == rows:
        warnings.append(_profile_warning("all_null"))
    if rows and null_rate >= 0.5:
        warnings.append(_profile_warning("high_null_rate"))
    if rows and distinct_count == 1:
        warnings.append(_profile_warning("single_value"))
    if rows > 1 and distinct_count == rows:
        warnings.append(_profile_warning("high_cardinality"))

    min_value: int | float | None = None
    max_value: int | float | None = None
    mean: float | None = None
    quantiles: list[QuantilePoint] = []
    top_k: list[TopKEntry] = []
    truncated = False
    non_null_count = rows - null_count
    numeric = pa.types.is_integer(values.type) or pa.types.is_floating(values.type)
    if non_null_count < limits.disclosure_threshold:
        warnings.append(_profile_warning("small_group_suppressed"))
        if numeric:
            warnings.append(_profile_warning("stats_suppressed"))
        if limits.max_top_k:
            warnings.append(_profile_warning("topk_suppressed"))
        return (
            ColumnProfile(
                ordinal=ordinal,
                dtype=_profile_dtype(values.type, pa),
                null_count=null_count,
                null_rate=null_rate,
                distinct_count=distinct_count,
                warning_codes=warnings,
            ),
            null_count,
            truncated,
        )

    non_null = pc.drop_null(values)
    if numeric:
        try:
            min_value = _profile_number(pc.min(non_null).as_py())
            max_value = _profile_number(pc.max(non_null).as_py())
            mean_value = _profile_number(pc.mean(non_null).as_py())
            if min_value is None or max_value is None or mean_value is None:
                warnings.append(_profile_warning("stats_suppressed"))
            else:
                mean = float(mean_value)
        except (pa.ArrowException, NotImplementedError, TypeError, ValueError):
            warnings.append(_profile_warning("stats_suppressed"))
        _check_profile_deadline(deadline)
        probabilities = [0.25, 0.5, 0.75]
        if len(probabilities) > limits.max_quantiles:
            probabilities = probabilities[: limits.max_quantiles]
            truncated = True
            warnings.append(_profile_warning("quantiles_truncated"))
        if limits.max_quantiles:
            try:
                quantile_values = pc.quantile(
                    non_null,
                    q=probabilities,
                    interpolation="linear",
                )
                for probability, value in zip(probabilities, quantile_values):
                    number = _profile_number(value.as_py())
                    if number is not None:
                        quantiles.append(
                            QuantilePoint(
                                probability=float(probability),
                                value=float(number),
                            )
                        )
            except (pa.ArrowException, NotImplementedError, TypeError, ValueError):
                warnings.append(_profile_warning("quantiles_unavailable"))
        _check_profile_deadline(deadline)

    if limits.max_top_k:
        try:
            counts = pc.value_counts(non_null).to_pylist()
            normalized_counts = []
            for item in counts:
                if not isinstance(item, dict):
                    continue
                count = item.get("counts", item.get("count"))
                value = item.get("values")
                if (
                    not isinstance(count, int)
                    or count < limits.disclosure_threshold
                    or not isinstance(value, (bool, int, float, str))
                ):
                    continue
                normalized_counts.append({"values": value, "count": count})
            counts = normalized_counts
            counts.sort(
                key=lambda item: (
                    -int(item["count"]),
                    _canonical_json(item["values"]),
                )
            )
            if len(counts) > limits.max_top_k:
                truncated = True
                warnings.append(_profile_warning("topk_truncated"))
            for item in counts[: limits.max_top_k]:
                value = item["values"]
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                top_k.append(
                    TopKEntry(value=value, count=int(item["count"]))
                )
        except (pa.ArrowException, NotImplementedError, TypeError, ValueError):
            warnings.append(_profile_warning("topk_suppressed"))
    _check_profile_deadline(deadline)
    return (
        ColumnProfile(
            ordinal=ordinal,
            dtype=_profile_dtype(values.type, pa),
            null_count=null_count,
            null_rate=null_rate,
            distinct_count=distinct_count,
            min_value=min_value,
            max_value=max_value,
            mean=mean,
            quantiles=quantiles,
            top_k=top_k,
            warning_codes=warnings,
        ),
        null_count,
        truncated,
    )


def profile_table(
    table: Any,
    profile: LocalProfile,
    *,
    target_ref: str | None = None,
    target_kind: ProfileTargetKind = "local_arrow",
    as_of_lsn: int | None = None,
    selector: ProfileSelector | None = None,
    limits: ProfileLimits | None = None,
) -> ProfileResult:
    """Build one bounded, privacy-aware result for local or future engine use."""

    _check_profile(table, profile)
    pa, pc = _require_pyarrow()
    effective_limits = limits or ProfileLimits(
        max_columns=profile.max_columns,
        max_bytes=profile.max_bytes,
    )
    if table.nbytes > effective_limits.max_bytes:
        raise ProfileLimitError("profile bytes exceed the declared limit")
    digest = schema_digest(table)
    if selector is not None and selector.schema_digest != digest:
        raise ProfileSelectorError("profile selector schema digest is stale")
    ordinals = list(range(table.num_columns))
    if selector is not None:
        ordinals = list(selector.ordinals)
        if any(ordinal >= table.num_columns for ordinal in ordinals):
            raise ProfileSelectorError("profile selector ordinal is outside the schema")
    if len(ordinals) > effective_limits.max_columns:
        raise ProfileLimitError("profile columns exceed the declared limit")
    started = time.monotonic()
    deadline = _profile_deadline(started, effective_limits)
    null_cells = 0
    column_profiles: list[ColumnProfile] = []
    warnings: list[ProfileWarningCode] = []
    truncated = False
    if table.num_rows == 0:
        warnings.append(_profile_warning("empty_dataset"))
    for ordinal in ordinals:
        column_profile, null_count, column_truncated = _column_profile(
            table,
            ordinal,
            pa,
            pc,
            limits=effective_limits,
            deadline=deadline,
        )
        null_cells += null_count
        column_profiles.append(column_profile)
        warnings.extend(column_profile.warning_codes)
        truncated = truncated or column_truncated
    warning_values = list(dict.fromkeys(warnings))
    if len(warning_values) > effective_limits.max_warnings:
        truncated = True
        warning_values = warning_values[: effective_limits.max_warnings - 1]
        warning_values.append(_profile_warning("profile_truncated"))
    _check_profile_deadline(deadline)
    target = ProfileTarget(
        target_kind=target_kind,
        target_ref=target_ref or "local:anonymous",
        schema_digest=digest,
        as_of_lsn=as_of_lsn,
    )
    return ProfileResult(
        evidence_version="data-prep-profile.v1",
        algorithm="arrow-local-profile",
        algorithm_version="1",
        target=target,
        schema_digest=digest,
        as_of_lsn=as_of_lsn,
        rows=table.num_rows,
        columns=len(ordinals),
        null_cells=null_cells,
        bytes_in_memory=table.nbytes,
        disclosure_threshold=effective_limits.disclosure_threshold,
        truncated=truncated,
        warnings=warning_values,
        column_profiles=column_profiles,
    )


def _canonical_name(name: str) -> str:
    normalized = _CANONICAL_NAME.sub("_", name.strip().lower()).strip("_")
    if not normalized:
        raise PlanExecutionError("a field name cannot be canonicalized")
    if normalized[0].isdigit():
        normalized = f"field_{normalized}"
    if len(normalized) > 128:
        raise PlanExecutionError("canonical field name exceeds 128 characters")
    return normalized


def _require_columns(table: Any, columns: Sequence[str]) -> None:
    available = set(table.column_names)
    missing = [column for column in columns if column not in available]
    if missing:
        raise PlanExecutionError(
            "plan refers to a field absent from the current schema"
        )


def _count_true(mask: Any, pc: Any) -> int:
    value = pc.sum(pc.cast(mask, "int64")).as_py()
    return int(value or 0)


def _apply_canonical_names(table: Any, step: CanonicalNames) -> Any:
    del step
    names = tuple(_canonical_name(name) for name in table.column_names)
    if len(set(names)) != len(names):
        raise PlanExecutionError("canonical field names collide")
    return table.rename_columns(names)


def _apply_null_policy(
    table: Any,
    step: NullPolicy,
    *,
    disposition: str,
    profile: LocalProfile,
) -> tuple[Any, tuple[int, ...]]:
    _require_columns(table, step.columns)
    if step.action == "allow":
        return table, ()
    pa, pc = _require_pyarrow()
    invalid = None
    for column in step.columns:
        mask = pc.is_null(table[column])
        invalid = mask if invalid is None else pc.or_(invalid, mask)
    if invalid is None:  # pragma: no cover - columns are validated as non-empty
        return table, ()
    invalid = pc.fill_null(invalid, False)
    count = _count_true(invalid, pc)
    if count == 0:
        return table, ()
    if disposition == "fail":
        raise InvalidRowsError(count, reason="declared null policy")
    if count > min(profile.max_quarantine_rows, profile.max_rows):
        raise ProfileLimitError("quarantine rows exceed the local profile limit")
    indexes = tuple(int(index.as_py()) for index in pc.indices_nonzero(invalid))
    return table.filter(pc.invert(invalid)), indexes


def _arrow_type_name(data_type: Any, pa: Any) -> str | None:
    if pa.types.is_boolean(data_type):
        return "bool"
    if pa.types.is_int8(data_type):
        return "int8"
    if pa.types.is_int16(data_type):
        return "int16"
    if pa.types.is_int32(data_type):
        return "int32"
    if pa.types.is_int64(data_type):
        return "int64"
    if pa.types.is_uint8(data_type):
        return "uint8"
    if pa.types.is_uint16(data_type):
        return "uint16"
    if pa.types.is_uint32(data_type):
        return "uint32"
    if pa.types.is_uint64(data_type):
        return "uint64"
    if pa.types.is_float32(data_type):
        return "float32"
    if pa.types.is_float64(data_type):
        return "float64"
    if pa.types.is_string(data_type):
        return "string"
    return None


def _lossless_cast_allowed(source: str, target: str) -> bool:
    if source == target or (source == "string" and target == "string"):
        return True
    if source in _SIGNED_RANK and target in _SIGNED_RANK:
        return _SIGNED_RANK[source] <= _SIGNED_RANK[target]
    if source in _UNSIGNED_RANK and target in _UNSIGNED_RANK:
        return _UNSIGNED_RANK[source] <= _UNSIGNED_RANK[target]
    return source == "float32" and target == "float64"


def _apply_safe_cast(table: Any, step: SafeCast) -> Any:
    _require_columns(table, (step.column,))
    pa, pc = _require_pyarrow()
    actual = _arrow_type_name(table[step.column].type, pa)
    if actual != step.source_type:
        raise PlanExecutionError(
            "safe-cast source type does not match the current schema"
        )
    if not _lossless_cast_allowed(step.source_type, step.target_type):
        raise PlanExecutionError(
            "requested cast is not an allow-listed lossless widening"
        )
    if step.source_type == step.target_type:
        return table
    target = _ARROW_TYPE_BUILDERS[step.target_type](pa)
    try:
        casted = pc.cast(table[step.column], target, safe=True)
    except (pa.ArrowException, ValueError) as exc:
        raise PlanExecutionError("safe cast failed") from exc
    index = table.column_names.index(step.column)
    return table.set_column(index, step.column, casted)


def _fill_scalar(value: Any, target: Any, pa: Any) -> Any:
    """Create a target-typed scalar without a lossy conversion."""

    if pa.types.is_boolean(target):
        if not isinstance(value, bool):
            raise PlanExecutionError("boolean null fills require a strict boolean")
    elif pa.types.is_integer(target):
        if isinstance(value, bool) or not isinstance(value, int):
            raise PlanExecutionError("integer null fills require a strict integer")
    elif pa.types.is_floating(target):
        if (
            isinstance(value, bool)
            or not isinstance(value, float)
            or not math.isfinite(value)
        ):
            raise PlanExecutionError(
                "floating null fills require a finite strict float"
            )
    elif pa.types.is_string(target):
        if not isinstance(value, str):
            raise PlanExecutionError("string null fills require a strict string")
    else:
        raise PlanExecutionError("null fills are unsupported for this Arrow type")
    try:
        scalar = pa.scalar(value, type=target)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise PlanExecutionError(
            "null fill does not fit the target Arrow type"
        ) from exc
    if pa.types.is_float32(target) and float(scalar.as_py()) != value:
        raise PlanExecutionError("float32 null fill would be lossy")
    return scalar


def _apply_fill_nulls(table: Any, step: FillNulls) -> Any:
    _require_columns(table, tuple(step.fills))
    pa, pc = _require_pyarrow()
    result = table
    for column, value in step.fills.items():
        target = result[column].type
        scalar = _fill_scalar(value, target, pa)
        try:
            filled = pc.fill_null(result[column], scalar)
        except (ValueError, pa.ArrowException) as exc:
            raise PlanExecutionError("null fill failed") from exc
        index = result.column_names.index(column)
        result = result.set_column(index, column, filled)
    return result


def _apply_dedupe(table: Any, step: Dedupe) -> tuple[Any, int, tuple[int, ...]]:
    _require_columns(table, step.keys)
    if table.num_rows < 2:
        return table, 0, tuple(range(table.num_rows))
    pa, pc = _require_pyarrow()
    internal_index = "__au_data_prep_row_index__"
    if internal_index in table.column_names:
        raise PlanExecutionError("input schema uses a reserved data-prep field name")
    key_table = table.select(list(step.keys)).append_column(
        internal_index,
        pa.array(range(table.num_rows), type=pa.uint64()),
    )
    # The row index is a secondary key, so equal user keys have a deterministic
    # first occurrence even if Arrow changes its equal-key sort implementation.
    sort_keys = [(key, "ascending") for key in step.keys]
    sort_keys.append((internal_index, "ascending"))
    order = pc.sort_indices(key_table, sort_keys=sort_keys)
    ordered = table.take(order)
    row_count = ordered.num_rows
    equal_adjacent = pa.array([True] * (row_count - 1))
    for key in step.keys:
        values = ordered[key].combine_chunks()
        previous = values.slice(0, row_count - 1)
        current = values.slice(1)
        equal = pc.equal(previous, current)
        both_null = pc.and_(pc.is_null(previous), pc.is_null(current))
        key_equal = pc.fill_null(pc.or_kleene(equal, both_null), False)
        equal_adjacent = pc.and_(equal_adjacent, key_equal)
    keep = pa.concat_arrays([pa.array([True]), pc.invert(equal_adjacent)])
    selected = order.filter(keep)
    # Keep the first occurrence in original input order, not sort order.
    restore_order = pc.sort_indices(
        pa.table({"row_index": selected}),
        sort_keys=[("row_index", "ascending")],
    )
    selected = selected.take(restore_order)
    selected_indexes = tuple(int(index.as_py()) for index in selected)
    return table.take(selected), table.num_rows - len(selected), selected_indexes


def _validate_row_model_class(row_model: type[BaseModel] | Any) -> None:
    """Require an approved strict model class before it enters the registry."""

    if not isinstance(row_model, type) or not issubclass(row_model, BaseModel):
        raise PlanExecutionError("row_model must be a Pydantic BaseModel class")
    config = row_model.model_config
    if config.get("strict") is not True or config.get("extra") != "forbid":
        raise PlanExecutionError("row_model must use strict=True and extra='forbid'")
    fields = row_model.model_fields
    if not fields:
        raise PlanExecutionError("row_model must declare at least one field")
    if any(field.annotation is Any for field in fields.values()):
        raise PlanExecutionError("row_model fields cannot use unconstrained Any")


def _validate_row_model_schema(table: Any, row_model: type[BaseModel]) -> None:
    """Require an approved strict model with an exact Arrow schema."""

    _validate_row_model_class(row_model)
    fields = row_model.model_fields
    if set(fields) != set(table.column_names):
        raise PlanExecutionError("row_model fields must exactly match the Arrow schema")


def _validate_rows(
    table: Any,
    row_model: type[BaseModel],
    *,
    row_ids: Sequence[int],
    disposition: str,
    profile: LocalProfile,
) -> tuple[Any, list[RowOutcome], int]:
    """Validate bounded Arrow rows and return only redacted dispositions."""

    pa, _ = _require_pyarrow()
    _validate_row_model_schema(table, row_model)
    if len(row_ids) != table.num_rows:
        raise PlanExecutionError("row identity accounting does not match the table")
    if table.num_rows > profile.max_outcome_rows:
        raise ProfileLimitError("row outcomes exceed the local profile limit")
    valid_flags: list[bool] = []
    outcomes: list[RowOutcome] = []
    rejected = 0
    row_index = 0
    for batch in table.to_batches(max_chunksize=_MAX_BATCH_ROWS):
        for row in batch.to_pylist():
            try:
                row_model.model_validate(row)
            except ValidationError:
                rejected += 1
                valid_flags.append(False)
                outcomes.append(
                    RowOutcome(
                        row_index=row_ids[row_index],
                        status="quarantined",
                        reason_code="model_rejected",
                    )
                )
            else:
                valid_flags.append(True)
                outcomes.append(
                    RowOutcome(
                        row_index=row_ids[row_index],
                        status="accepted",
                        reason_code="accepted",
                    )
                )
            row_index += 1
    if rejected and disposition == "fail":
        raise InvalidRowsError(rejected, reason="strict row-model validation")
    if rejected > min(
        profile.max_quarantine_rows,
        profile.max_outcome_rows,
        profile.max_rows,
    ):
        raise ProfileLimitError("model quarantine rows exceed the local profile limit")
    return table.filter(pa.array(valid_flags, type=pa.bool_())), outcomes, rejected


def _apply_step(
    table: Any,
    step: Any,
    *,
    disposition: str,
    profile: LocalProfile,
) -> tuple[Any, tuple[int, ...], str | None, tuple[int, ...] | None]:
    """Apply one allow-listed verb with no dynamic dispatch."""

    if isinstance(step, CanonicalNames):
        return _apply_canonical_names(table, step), (), None, None
    if isinstance(step, NullPolicy):
        result, quarantined_indexes = _apply_null_policy(
            table,
            step,
            disposition=disposition,
            profile=profile,
        )
        return (
            result,
            quarantined_indexes,
            "null_rejected" if quarantined_indexes else None,
            None,
        )
    if isinstance(step, SafeCast):
        return _apply_safe_cast(table, step), (), None, None
    if isinstance(step, Dedupe):
        result, dropped, retained_indexes = _apply_dedupe(table, step)
        return result, (), None if dropped == 0 else "dedupe", retained_indexes
    if isinstance(step, FillNulls):
        return _apply_fill_nulls(table, step), (), None, None
    raise PlanExecutionError("unsupported clean-plan verb")


class CleanPipeline:
    """Apply a typed, deterministic clean plan to an Arrow table."""

    __slots__ = ("_plan", "_registry")

    def __init__(
        self,
        plan: CleanPlan | Mapping[str, Any],
        *,
        model_registry: RowModelRegistry | None = None,
    ) -> None:
        validated = CleanPlan.model_validate(plan)
        # Keep an independent deep copy of nested lists/maps.  The executor
        # never writes to a plan, and a caller mutating its own model after
        # construction cannot alter an in-flight pipeline's digest or policy.
        self._plan = validated.model_copy(deep=True)
        self._registry = model_registry

    @property
    def plan(self) -> CleanPlan:
        """The validated immutable plan used by this pipeline."""

        return self._plan

    def profile(self, table: Any) -> ProfileResult:
        """Return a bounded local profile without applying transformations."""

        return profile_table(
            table,
            self._plan.profile,
            target_ref=(
                self._plan.artifact_ref
                or self._plan.source_ref
                or self._plan.plan_ref
            ),
        )

    def run(self, table: Any) -> PrepResult:
        """Apply the plan and return the table plus content-free evidence."""

        ArrowAdapter.as_table(table, profile=self._plan.profile)
        current = table
        input_rows = current.num_rows
        if input_rows > self._plan.profile.max_outcome_rows:
            raise ProfileLimitError("row outcomes exceed the local profile limit")
        if self._registry is None:
            raise PlanExecutionError("an approved row-model registry is required")
        model_entry = self._registry.resolve(
            self._plan.model_ref,
            self._plan.model_digest,
        )
        row_ids = list(range(input_rows))
        input_schema = schema_digest(current)
        dropped_rows = 0
        quarantined_rows = 0
        quarantine: list[QuarantineOutcome] = []
        row_outcomes: list[RowOutcome] = []
        step_evidence: list[StepEvidence] = []

        for step in self._plan.steps:
            rows_in = current.num_rows
            before_row_ids = tuple(row_ids)
            current, quarantine_indexes, reason, retained_indexes = _apply_step(
                current,
                step,
                disposition=self._plan.invalid_row_disposition,
                profile=self._plan.profile,
            )
            _check_profile(current, self._plan.profile)
            count = len(quarantine_indexes)
            if reason == "null_rejected" and count:
                quarantined_rows += count
                if quarantined_rows > min(
                    self._plan.profile.max_quarantine_rows,
                    self._plan.profile.max_rows,
                ):
                    raise ProfileLimitError(
                        "quarantine rows exceed the local profile limit"
                    )
                quarantine.append(
                    QuarantineOutcome(reason_code="null_rejected", row_count=count)
                )
                row_outcomes.extend(
                    RowOutcome(
                        row_index=before_row_ids[index],
                        status="quarantined",
                        reason_code="null_rejected",
                    )
                    for index in quarantine_indexes
                )
                if len(row_outcomes) > self._plan.profile.max_outcome_rows:
                    raise ProfileLimitError(
                        "row outcomes exceed the local profile limit"
                    )
                invalid_positions = set(quarantine_indexes)
                row_ids = [
                    row_id
                    for index, row_id in enumerate(before_row_ids)
                    if index not in invalid_positions
                ]
            elif retained_indexes is not None:
                if len(retained_indexes) != current.num_rows:
                    raise PlanExecutionError(
                        "row identity accounting does not match the table"
                    )
                row_ids = [before_row_ids[index] for index in retained_indexes]
                dropped_indexes = set(range(rows_in)) - set(retained_indexes)
                if dropped_indexes:
                    dropped_rows += len(dropped_indexes)
                    row_outcomes.extend(
                        RowOutcome(
                            row_index=before_row_ids[index],
                            status="dropped",
                            reason_code="deduplicated",
                        )
                        for index in sorted(dropped_indexes)
                    )
                    if len(row_outcomes) > self._plan.profile.max_outcome_rows:
                        raise ProfileLimitError(
                            "row outcomes exceed the local profile limit"
                        )
            if len(row_ids) != current.num_rows:
                raise PlanExecutionError(
                    "row identity accounting does not match the table"
                )
            dropped = max(0, rows_in - current.num_rows - count)
            step_evidence.append(
                StepEvidence(
                    verb=step.verb,
                    rows_in=rows_in,
                    rows_out=current.num_rows,
                    dropped_rows=dropped,
                    quarantined_rows=count,
                )
            )

        current, model_outcomes, model_rejected = _validate_rows(
            current,
            model_entry.model,
            row_ids=row_ids,
            disposition=self._plan.invalid_row_disposition,
            profile=self._plan.profile,
        )
        row_outcomes.extend(model_outcomes)
        if len(row_outcomes) > self._plan.profile.max_outcome_rows:
            raise ProfileLimitError("row outcomes exceed the local profile limit")
        if model_rejected:
            quarantined_rows += model_rejected
            quarantine.append(
                QuarantineOutcome(
                    reason_code="model_rejected",
                    row_count=model_rejected,
                )
            )
            if quarantined_rows > min(
                self._plan.profile.max_quarantine_rows,
                self._plan.profile.max_rows,
            ):
                raise ProfileLimitError(
                    "quarantine rows exceed the local profile limit"
                )
        _check_profile(current, self._plan.profile)
        row_outcomes.sort(key=lambda item: item.row_index)

        output_schema = schema_digest(current)
        outcome: Literal["complete", "quarantined"] = (
            "quarantined" if quarantined_rows else "complete"
        )
        evidence = PrepEvidence(
            evidence_version="data-prep-evidence.v1",
            algorithm="arrow-clean-pipeline",
            algorithm_version="1",
            outcome=outcome,
            checkpoint_eligible=not quarantined_rows,
            plan_digest=plan_digest(self._plan),
            input_schema_digest=input_schema,
            output_schema_digest=output_schema,
            rows_in=input_rows,
            rows_out=current.num_rows,
            dropped_rows=dropped_rows,
            quarantined_rows=quarantined_rows,
            invalid_row_disposition=self._plan.invalid_row_disposition,
            plan_ref=self._plan.plan_ref,
            policy_ref=self._plan.policy_ref,
            model_ref=self._plan.model_ref,
            model_digest=model_entry.digest,
            source_ref=self._plan.source_ref,
            artifact_ref=self._plan.artifact_ref,
            quarantine=quarantine,
            row_outcomes=row_outcomes,
            model_rejected_rows=model_rejected,
            steps=step_evidence,
        )
        return PrepResult(table=current, evidence=evidence)


__all__ = [
    "ArrowAdapter",
    "CleanPipeline",
    "DataPrepDependencyError",
    "DataPrepError",
    "InvalidRowsError",
    "PlanExecutionError",
    "PrepResult",
    "ProfileDeadlineError",
    "ProfileLimitError",
    "ProfileSelectorError",
    "RegisteredRowModel",
    "RowModelRegistry",
    "plan_digest",
    "profile_table",
    "row_model_digest",
    "schema_digest",
]
