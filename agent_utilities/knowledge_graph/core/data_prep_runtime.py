"""Process-owned data-preparation runtime declarations.

This module is the configuration boundary for the served data-preparation
surface.  The deployment may declare model fields, policy facts and the native
ICV/shape capability in the validated process configuration.  It may not name
an import path, executable, storage location or arbitrary authority object.

The declaration is deliberately separate from the MCP request model.  A
request selects only immutable opaque references; this module is loaded once by
the process-owned graph startup and produces the trusted runtime inputs used by
the graph-native adapter.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    create_model,
    field_validator,
    model_validator,
)

from agent_utilities.data_prep import RowModelRegistry

_MODEL_REF = r"^model:[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$"
_POLICY_REF = r"^policy:[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$"
_CONNECTOR_VERSION = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$"
_FIELD_NAME = r"^[A-Za-z_][A-Za-z0-9_]{0,127}$"
_TENANT = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_PRINCIPAL = r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,255}$"
_ROLE = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"

DataPrepInlineMetadata: TypeAlias = Mapping[str, object]

_FORBIDDEN_CONFIGURATION_KEYS = frozenset(
    {
        "artifact_authority",
        "authority",
        "blob_ref",
        "checkpoint",
        "code",
        "executable",
        "file_path",
        "import_path",
        "module",
        "path",
        "python_code",
        "storage_ref",
        "trust",
        "url",
    }
)


class DataPrepRuntimeConfigError(ValueError):
    """A process data-prep declaration is invalid or unsafe."""


class _RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DataPrepFieldDeclaration(_RuntimeModel):
    """One allow-listed strict scalar field in an approved row model."""

    name: str = Field(min_length=1, max_length=128, pattern=_FIELD_NAME)
    arrow_type: Literal[
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "string",
    ]
    nullable: bool = False


class DataPrepModelDeclaration(_RuntimeModel):
    """Declarative description of a model compiled by the owner process."""

    ref: str = Field(min_length=1, max_length=256, pattern=_MODEL_REF)
    fields: tuple[DataPrepFieldDeclaration, ...] = Field(min_length=1, max_length=1_024)

    @field_validator("fields", mode="before")
    @classmethod
    def _tuple_fields(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def fields_are_unique(self) -> DataPrepModelDeclaration:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("data-prep model fields must be unique")
        return self


class DataPrepPolicyDeclaration(_RuntimeModel):
    """Owner-controlled policy facts; no caller or storage authority is held."""

    ref: str = Field(min_length=1, max_length=256, pattern=_POLICY_REF)
    allow_inline_records: bool = False
    tenant_id: str | None = Field(default=None, pattern=_TENANT)
    owner_id: str | None = Field(default=None, pattern=_PRINCIPAL)
    read_roles: tuple[str, ...] = Field(
        default=("data-prep-reader",), min_length=1, max_length=64
    )
    classification: Literal["internal", "confidential", "restricted"] = "confidential"
    retention: str | None = Field(default=None, min_length=1, max_length=128)
    legal_hold: bool = True

    @field_validator("read_roles", mode="before")
    @classmethod
    def _tuple_roles(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_inline_policy(self) -> DataPrepPolicyDeclaration:
        if any(not re.fullmatch(_ROLE, role) for role in self.read_roles):
            raise ValueError("data-prep policy roles are invalid")
        if self.allow_inline_records and not self.read_roles:
            raise ValueError("inline records require an explicit read role")
        # Inline data is never a public classification or ACL.  This is a
        # configuration invariant, not a request-level choice.
        if self.classification == "internal" and self.allow_inline_records:
            raise ValueError("inline records require confidential or restricted policy")
        return self


class DataPrepICVShapeDeclaration(_RuntimeModel):
    """Public native capability and prepared-shape contract."""

    profile: Literal["graph-native-v1"] = "graph-native-v1"
    required_capability: Literal["ApplyChangeEnvelope"] = "ApplyChangeEnvelope"
    schema_ref_prefix: Literal["schema:prepared:"] = "schema:prepared:"
    shape_ref_prefix: Literal["shape:prepared:"] = "shape:prepared:"
    require_policy_binding: bool = True


class DataPrepRuntimeDeclaration(_RuntimeModel):
    """Complete declarative process configuration for data preparation."""

    schema_version: Literal["data-prep-runtime.v1"] = "data-prep-runtime.v1"
    connector_version: str = Field(
        default="connector:data-prep:v1", pattern=_CONNECTOR_VERSION
    )
    models: tuple[DataPrepModelDeclaration, ...] = Field(default=(), max_length=256)
    policy: DataPrepPolicyDeclaration = Field(
        default_factory=lambda: DataPrepPolicyDeclaration(ref="policy:data-prep:v1")
    )
    icv_shape: DataPrepICVShapeDeclaration = Field(
        default_factory=DataPrepICVShapeDeclaration
    )

    @field_validator("models", mode="before")
    @classmethod
    def _tuple_models(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def model_refs_are_unique(self) -> DataPrepRuntimeDeclaration:
        refs = [model.ref for model in self.models]
        if len(refs) != len(set(refs)):
            raise ValueError("data-prep model references must be unique")
        return self


def _reject_unsafe_keys(value: Any) -> None:
    """Reject authority/code/storage controls before Pydantic interpretation."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().casefold().replace("-", "_")
            if key in _FORBIDDEN_CONFIGURATION_KEYS:
                raise DataPrepRuntimeConfigError(
                    "data-prep runtime declaration contains a forbidden control"
                )
            _reject_unsafe_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_unsafe_keys(child)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataPrepRuntimeConfigError(
            "data-prep runtime declaration is not canonical JSON"
        ) from exc


def runtime_declaration_digest(declaration: DataPrepRuntimeDeclaration) -> str:
    """Return the owner-config digest pinned to model and policy loading."""

    payload = declaration.model_dump(mode="json")
    return f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


def _field_annotation(field: DataPrepFieldDeclaration) -> Any:
    scalar: dict[str, Any] = {
        "bool": StrictBool,
        "int8": StrictInt,
        "int16": StrictInt,
        "int32": StrictInt,
        "int64": StrictInt,
        "uint8": StrictInt,
        "uint16": StrictInt,
        "uint32": StrictInt,
        "uint64": StrictInt,
        "float32": StrictFloat,
        "float64": StrictFloat,
        "string": StrictStr,
    }
    annotation = scalar[field.arrow_type]
    return annotation | None if field.nullable else annotation


def build_approved_row_model_registry(
    declaration: DataPrepRuntimeDeclaration,
) -> RowModelRegistry:
    """Compile only declarative strict scalar models into the kernel registry."""

    models: dict[str, type[BaseModel]] = {}
    for ordinal, model_declaration in enumerate(declaration.models):
        class_name = "DataPrepRowModel" + str(ordinal)
        fields: dict[str, Any] = {
            field.name: (_field_annotation(field), None if field.nullable else ...)
            for field in model_declaration.fields
        }
        model = create_model(
            class_name,
            __config__=ConfigDict(strict=True, extra="forbid", frozen=True),
            **fields,
        )
        models[model_declaration.ref] = model
    return RowModelRegistry(models)


@dataclass(frozen=True, slots=True)
class DataPrepPolicyLoader:
    """Immutable policy loader built solely from a validated declaration."""

    declaration: DataPrepPolicyDeclaration

    @property
    def policy_ref(self) -> str:
        return self.declaration.ref

    def allows_inline_records(self, *, session: Any) -> bool:
        if not self.declaration.allow_inline_records:
            return False
        if (
            self.declaration.tenant_id is not None
            and str(session.tenant) != self.declaration.tenant_id
        ):
            return False
        policy_version = str(getattr(session, "policy_version", "") or "")
        actor = getattr(session, "actor", None)
        if not policy_version or actor is None:
            return False
        actor_id = str(getattr(actor, "actor_id", "") or "")
        roles = {str(role) for role in getattr(actor, "roles", ()) or ()}
        return bool(
            actor_id
            and (
                actor_id == self.declaration.owner_id
                or roles.intersection(self.declaration.read_roles)
            )
        )

    def inline_metadata(self, *, session: Any) -> DataPrepInlineMetadata:
        if not self.allows_inline_records(session=session):
            raise PermissionError("inline records policy is unavailable")
        actor_id = str(getattr(getattr(session, "actor", None), "actor_id", "") or "")
        owner_id = self.declaration.owner_id or actor_id
        return MappingProxyType(
            {
                "enabled": True,
                "tenant_id": str(session.tenant),
                "policy_version": str(session.policy_version),
                "classification": self.declaration.classification,
                "owner_id": owner_id,
                "acl": {
                    "is_public": False,
                    "principal_ids": [owner_id] if owner_id else [],
                    "principal_emails": [],
                    "group_ids": [],
                    "read_roles": list(self.declaration.read_roles),
                    "markings": [],
                },
                "retention": self.declaration.retention,
                "legal_hold": self.declaration.legal_hold,
            }
        )

    def authorize_output(self, source: Any, *, session: Any) -> None:
        if str(getattr(source, "tenant_id", "")) != str(session.tenant):
            raise PermissionError("output policy tenant does not match session")
        if str(getattr(source, "policy_version", "")) != str(
            getattr(session, "policy_version", "") or ""
        ):
            raise DataPrepRuntimeConfigError("output policy version is stale")


@dataclass(frozen=True, slots=True)
class DataPrepICVShapeLoader:
    """Immutable loader/probe for the public native commit and shape contract."""

    declaration: DataPrepICVShapeDeclaration

    def supports(self, engine: Any) -> bool:
        compute = getattr(engine, "graph_compute", None)
        client = getattr(compute, "client", None)
        probe = getattr(client, "supports", None)
        if not callable(probe):
            return False
        try:
            return bool(probe(self.declaration.required_capability))
        except Exception:  # noqa: BLE001 - capability probes fail closed
            return False

    def available(self, engine: Any, *, session: Any | None = None) -> bool:
        if session is not None and self.declaration.require_policy_binding:
            if not str(getattr(session, "policy_version", "") or ""):
                return False
        return self.supports(engine)

    def validate_prepared_refs(
        self,
        *,
        schema_ref: str,
        shape_ref: str,
        schema_digest: str,
        shape_digest: str,
    ) -> None:
        if not schema_ref.startswith(self.declaration.schema_ref_prefix):
            raise DataPrepRuntimeConfigError(
                "prepared schema reference is not approved"
            )
        if not shape_ref.startswith(self.declaration.shape_ref_prefix):
            raise DataPrepRuntimeConfigError("prepared shape reference is not approved")
        digest_pattern = r"^sha256:[0-9a-f]{64}$"
        if not re.fullmatch(digest_pattern, schema_digest) or not re.fullmatch(
            digest_pattern, shape_digest
        ):
            raise DataPrepRuntimeConfigError("prepared shape digest is invalid")


def load_data_prep_runtime_declaration(value: Any = None) -> DataPrepRuntimeDeclaration:
    """Validate an owner-provided declaration; reject all dynamic controls."""

    if value is None:
        return DataPrepRuntimeDeclaration()
    if isinstance(value, DataPrepRuntimeDeclaration):
        return value
    if not isinstance(value, Mapping):
        raise DataPrepRuntimeConfigError(
            "data-prep runtime configuration must be a declarative object"
        )
    _reject_unsafe_keys(value)
    try:
        return DataPrepRuntimeDeclaration.model_validate(value)
    except Exception as exc:  # noqa: BLE001 - public config error is bounded
        raise DataPrepRuntimeConfigError(
            "data-prep runtime declaration is invalid"
        ) from exc


def process_data_prep_declaration(engine: Any = None) -> DataPrepRuntimeDeclaration:
    """Resolve the declaration from the process-owned startup configuration."""

    if engine is not None:
        value = getattr(engine, "_data_prep_runtime_declaration", None)
        if value is not None:
            return load_data_prep_runtime_declaration(value)
    try:
        from agent_utilities.core.config import config

        return load_data_prep_runtime_declaration(
            getattr(config, "data_prep_runtime", None)
        )
    except DataPrepRuntimeConfigError:
        raise
    except Exception as exc:  # pragma: no cover - process config unavailable
        raise DataPrepRuntimeConfigError(
            "process data-prep runtime configuration is unavailable"
        ) from exc


__all__ = [
    "DataPrepFieldDeclaration",
    "DataPrepICVShapeDeclaration",
    "DataPrepICVShapeLoader",
    "DataPrepModelDeclaration",
    "DataPrepPolicyDeclaration",
    "DataPrepPolicyLoader",
    "DataPrepRuntimeConfigError",
    "DataPrepRuntimeDeclaration",
    "build_approved_row_model_registry",
    "load_data_prep_runtime_declaration",
    "process_data_prep_declaration",
    "runtime_declaration_digest",
]
