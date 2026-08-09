"""Typed, bounded execution payloads for repository WorkItems.

The payload is deliberately a small value object.  It is not a command
envelope, a shell script, an artifact reference, or a generic mapping.  The
authority persists the canonical value and workers obtain it through the
owner/tenant-scoped exact-input read in :mod:`repository_work_item`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

PAYLOAD_KIND_BUILD: Literal["repository.build-execution/v1"] = (
    "repository.build-execution/v1"
)
PAYLOAD_SCHEMA_VERSION: Literal["1"] = "1"

MAX_OPERATION_PAYLOAD_BYTES = 12_288
MAX_IDENTIFIER_BYTES = 256
MAX_PATH_BYTES = 512
MAX_ARG_COUNT = 128
MAX_ARG_BYTES = 1_024
MAX_ARGV_BYTES = 8_192
MAX_ARTIFACT_PATTERN_COUNT = 128
MAX_ARTIFACT_PATTERN_BYTES = 512
MAX_ENVIRONMENT_REFERENCE_COUNT = 64
MAX_CACHE_KEY_COMPONENTS = 128
MAX_FEATURE_COUNT = 128
MAX_SEQUENCE_ITEM_BYTES = 256

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_DRIVE_OR_UNC_RE = re.compile(r"^(?:[A-Za-z]:|//|\\\\)")
_ENV_REFERENCE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")
_COMPONENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_CACHE_KEY_DIGEST_RE = re.compile(r"^v2:[0-9a-f]{32}$")
_SHELL_RE = re.compile(r"[;&|`$<>\n\r\x00]")
_CONNECTION_RE = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]{1,31}://|-----begin|(?:password|passwd|secret|token|api[_-]?key)\s*=|bearer\s+|[^\s:@]+:[^\s@]+@|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})"
)
_CACHE_KEY_COMPONENT_NAMES = frozenset(
    {
        "key_version",
        "repo",
        "spec",
        "tree_sha",
        "feature_set",
        "toolchain_fingerprint",
        "target_triple",
        "config_digest",
        "spec_digest",
        "generation_id",
        "generation_digest",
    }
)
_CACHE_KEY_DIGEST_COMPONENT_NAMES = (
    "key_version",
    "repo",
    "spec",
    "tree_sha",
    "feature_set",
    "toolchain_fingerprint",
    "target_triple",
    "config_digest",
    "spec_digest",
    "generation_digest",
)
_APPROVED_DEGRADED_REASONS = frozenset(
    {"dirty-tree", "tree-sha-unresolvable", "toolchain-unfingerprintable"}
)


def _canonical_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="json", exclude_none=False))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    return value


def canonical_payload_json(value: object) -> str:
    """Serialize one payload using the cross-language canonical form."""

    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def payload_digest(value: object) -> str:
    """Return SHA-256 over a payload body without its self-digest field."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, Mapping):
        value = {key: item for key, item in value.items() if key != "payload_digest"}
    return hashlib.sha256(canonical_payload_json(value).encode("utf-8")).hexdigest()


def cache_key_digest_from_components(
    components: Mapping[str, str],
) -> str:
    """Reproduce RMDD-10 ``CacheKey.digest`` exactly.

    The build queue intentionally uses the v2 prefix plus the first 32 hex
    characters of SHA-256 over its ten digest-participating components.  This
    helper keeps the operation payload bound to that identity instead of
    inventing a second cache address.
    """

    if set(components) != _CACHE_KEY_COMPONENT_NAMES:
        raise ValueError("cache-key components do not match the C-05 contract")
    if components["key_version"] != "v2":
        raise ValueError("cache-key components must use C-05 key version v2")
    payload = {name: components[name] for name in _CACHE_KEY_DIGEST_COMPONENT_NAMES}
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return "v2:" + hashlib.sha256(encoded).hexdigest()[:32]


def _bounded_text(
    value: object,
    field_name: str,
    *,
    limit: int = MAX_IDENTIFIER_BYTES,
    connection_safe: bool = True,
) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-blank string")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ValueError(f"{field_name} contains control characters")
    if len(value.encode("utf-8")) > limit:
        raise ValueError(f"{field_name} exceeds its size bound")
    if connection_safe and _CONNECTION_RE.search(value):
        raise ValueError(f"{field_name} contains connection or credential material")
    return value


def _sha(value: str, field_name: str, size: int) -> str:
    if not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{size}}}", value):
        raise ValueError(f"{field_name} must be lowercase hexadecimal")
    return value


def _sequence(value: object, field_name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a bounded sequence")
    return tuple(value)


def _string_sequence(
    value: object,
    field_name: str,
    *,
    limit: int,
    item_limit: int = MAX_SEQUENCE_ITEM_BYTES,
    sort: bool = False,
    unique: bool = False,
) -> tuple[str, ...]:
    items = tuple(
        _bounded_text(item, field_name, limit=item_limit)
        for item in _sequence(value, field_name)
    )
    if len(items) > limit:
        raise ValueError(f"{field_name} exceeds its count bound")
    if unique and len(set(items)) != len(items):
        items = tuple(dict.fromkeys(items))
    return tuple(sorted(items) if sort else items)


class RepositoryCacheKeyComponent(BaseModel):
    """One named, canonical C-05 cache-key component."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: StrictStr
    value: StrictStr

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _COMPONENT_NAME_RE.fullmatch(value):
            raise ValueError("cache-key component name is invalid")
        return value

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        if not value:
            return ""
        return _bounded_text(value, "cache-key component value")


def _components(value: object) -> tuple[RepositoryCacheKeyComponent, ...]:
    if isinstance(value, Mapping):
        value = [{"name": str(name), "value": item} for name, item in value.items()]
    items = _sequence(value, "cache_key_components")
    if len(items) > MAX_CACHE_KEY_COMPONENTS:
        raise ValueError("cache_key_components exceeds its count bound")
    parsed: list[RepositoryCacheKeyComponent] = []
    for item in items:
        if isinstance(item, RepositoryCacheKeyComponent):
            parsed.append(item)
        elif isinstance(item, Mapping):
            parsed.append(RepositoryCacheKeyComponent.model_validate(item))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            parsed.append(RepositoryCacheKeyComponent(name=item[0], value=item[1]))
        else:
            raise ValueError("cache_key_components must contain typed name/value pairs")
    names = [item.name for item in parsed]
    if len(set(names)) != len(names):
        raise ValueError("cache_key_components must have unique names")
    return tuple(sorted(parsed, key=lambda item: item.name))


def _relative_path(value: object, field_name: str, *, allow_glob: bool = False) -> str:
    value = _bounded_text(value, field_name, limit=MAX_PATH_BYTES)
    if _DRIVE_OR_UNC_RE.match(value) or value.startswith("/") or "\\" in value:
        raise ValueError(f"{field_name} must be repository-relative")
    parts = value.split("/")
    if any(part in {"..", ""} for part in parts):
        raise ValueError(f"{field_name} contains an unsafe path component")
    if not allow_glob and any(char in value for char in "*?[]{}"):
        raise ValueError(f"{field_name} must not contain glob syntax")
    if _SHELL_RE.search(value):
        raise ValueError(f"{field_name} contains shell syntax")
    if allow_glob and any(char in value for char in "{}"):
        raise ValueError(f"{field_name} contains unsupported glob syntax")
    return value


def _argv(value: object) -> tuple[str, ...]:
    args = _string_sequence(
        value,
        "argv",
        limit=MAX_ARG_COUNT,
        item_limit=MAX_ARG_BYTES,
    )
    if not args:
        raise ValueError("argv must be non-empty")
    if len("\0".join(args).encode("utf-8")) > MAX_ARGV_BYTES:
        raise ValueError("argv exceeds its byte bound")
    if any(_SHELL_RE.search(arg) for arg in args):
        raise ValueError("argv must contain argv data, not shell syntax")
    if any(char.isspace() for char in args[0]):
        raise ValueError("argv executable must be unambiguous")
    executable = args[0].rsplit("/", 1)[-1].lower()
    if executable in {"sh", "bash", "dash", "zsh", "fish", "cmd", "powershell", "pwsh"}:
        raise ValueError("shell interpreters are not valid executable argv")
    if any(arg in {"-c", "/c", "-command", "--command"} for arg in args[1:]):
        raise ValueError("argv must not encode a shell command string")
    return args


def _artifact_patterns(value: object) -> tuple[str, ...]:
    items = tuple(
        _relative_path(item, "artifact_patterns", allow_glob=True)
        for item in _sequence(value, "artifact_patterns")
    )
    if not items:
        raise ValueError("artifact_patterns must be non-empty")
    if len(items) > MAX_ARTIFACT_PATTERN_COUNT:
        raise ValueError("artifact_patterns exceeds its count bound")
    if sum(len(item.encode("utf-8")) for item in items) > MAX_ARTIFACT_PATTERN_BYTES:
        raise ValueError("artifact_patterns exceeds its byte bound")
    return tuple(sorted(dict.fromkeys(items)))


def _environment_refs(value: object) -> tuple[str, ...]:
    items = _string_sequence(
        value,
        "environment_refs",
        limit=MAX_ENVIRONMENT_REFERENCE_COUNT,
        item_limit=MAX_SEQUENCE_ITEM_BYTES,
        sort=True,
        unique=True,
    )
    for item in items:
        if not _ENV_REFERENCE_RE.fullmatch(item) or "=" in item:
            raise ValueError("environment_refs must be approved reference names")
    return items


class RepositoryBuildExecutionPayloadV1(BaseModel):
    """The closed v1 repository build execution payload."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["repository.build-execution/v1"] = PAYLOAD_KIND_BUILD
    schema_version: Literal["1"] = PAYLOAD_SCHEMA_VERSION
    repository_id: StrictStr
    base_sha: StrictStr
    tree_sha: StrictStr
    generation_id: StrictStr | None = None
    build_spec_name: StrictStr
    spec_digest: StrictStr
    config_digest: StrictStr
    toolchain_digest: StrictStr
    artifact_contract_digest: StrictStr
    feature_set: StrictStr
    target_triple: StrictStr
    cache_key_components: tuple[RepositoryCacheKeyComponent, ...]
    cache_key_digest: StrictStr | None = None
    argv: tuple[StrictStr, ...]
    workdir: StrictStr
    timeout_seconds: StrictInt = Field(ge=1, le=86_400)
    artifact_patterns: tuple[StrictStr, ...]
    environment_refs: tuple[StrictStr, ...] = ()
    execution_policy_ref: StrictStr
    profile_ref: StrictStr
    cacheable: StrictBool
    degraded_reason: StrictStr = ""
    payload_digest: StrictStr | None = None

    @field_validator(
        "repository_id",
        "build_spec_name",
        "target_triple",
        "execution_policy_ref",
        "profile_ref",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _bounded_text(value, info.field_name)

    @field_validator("generation_id")
    @classmethod
    def validate_generation(cls, value: str | None) -> str | None:
        return None if value is None else _bounded_text(value, "generation_id")

    @field_validator("base_sha", "tree_sha")
    @classmethod
    def validate_shas(cls, value: str, info: Any) -> str:
        return _sha(value, info.field_name, 40)

    @field_validator(
        "spec_digest",
        "config_digest",
        "toolchain_digest",
        "artifact_contract_digest",
        "cache_key_digest",
        "payload_digest",
    )
    @classmethod
    def validate_digests(cls, value: str | None, info: Any) -> str | None:
        if info.field_name == "cache_key_digest":
            if value is None:
                return None
            if not _CACHE_KEY_DIGEST_RE.fullmatch(value):
                raise ValueError(
                    "cache_key_digest must be v2 followed by 32 lowercase hex characters"
                )
            return value
        return None if value is None else _sha(value, info.field_name, 64)

    @field_validator("feature_set")
    @classmethod
    def validate_features(cls, value: str) -> str:
        return _bounded_text(value, "feature_set")

    @field_validator("cache_key_components", mode="before")
    @classmethod
    def validate_components(
        cls, value: object
    ) -> tuple[RepositoryCacheKeyComponent, ...]:
        return _components(value)

    @field_validator("argv", mode="before")
    @classmethod
    def validate_command(cls, value: object) -> tuple[str, ...]:
        return _argv(value)

    @field_validator("workdir")
    @classmethod
    def validate_workdir(cls, value: str) -> str:
        return _relative_path(value, "workdir")

    @field_validator("artifact_patterns", mode="before")
    @classmethod
    def validate_artifacts(cls, value: object) -> tuple[str, ...]:
        return _artifact_patterns(value)

    @field_validator("environment_refs", mode="before")
    @classmethod
    def validate_environment(cls, value: object) -> tuple[str, ...]:
        return _environment_refs(value)

    @field_validator("degraded_reason")
    @classmethod
    def validate_degraded_reason(cls, value: str) -> str:
        if value and _SHELL_RE.search(value):
            raise ValueError("degraded_reason contains shell syntax")
        return _bounded_text(value, "degraded_reason") if value else ""

    @model_validator(mode="after")
    def validate_digest_and_size(self) -> RepositoryBuildExecutionPayloadV1:
        components = {item.name: item.value for item in self.cache_key_components}
        if set(components) != _CACHE_KEY_COMPONENT_NAMES:
            raise ValueError("cache-key components do not match the C-05 contract")
        if components["key_version"] != "v2":
            raise ValueError("cache-key components must use C-05 key version v2")
        if components["repo"] != self.repository_id:
            raise ValueError("cache-key repository component disagrees with payload")
        if components["spec"] != self.build_spec_name:
            raise ValueError("cache-key spec component disagrees with payload")
        if self.cacheable:
            if self.degraded_reason:
                raise ValueError("cacheable payload must not carry degraded_reason")
            expected = {
                "feature_set": self.feature_set,
                "target_triple": self.target_triple,
                "config_digest": self.config_digest,
                "spec_digest": self.spec_digest,
                "generation_digest": (
                    hashlib.sha256(self.generation_id.encode("utf-8")).hexdigest()
                    if self.generation_id
                    else ""
                ),
            }
            if components["generation_id"] != (self.generation_id or ""):
                raise ValueError(
                    "cache-key generation component disagrees with payload"
                )
            for name, value in expected.items():
                if components[name] != value:
                    raise ValueError(
                        f"cache-key {name} component disagrees with payload"
                    )
            if self.cache_key_digest != cache_key_digest_from_components(components):
                raise ValueError("cache_key_digest does not match C-05 components")
        else:
            if self.degraded_reason not in _APPROVED_DEGRADED_REASONS:
                raise ValueError("uncacheable payload has an unknown degraded reason")
            if self.cache_key_digest is not None:
                raise ValueError("uncacheable payload must not carry a cache key")
            if any(
                components[name]
                for name in _CACHE_KEY_COMPONENT_NAMES - {"key_version", "repo", "spec"}
            ):
                raise ValueError(
                    "uncacheable payload must not carry cache-key components"
                )
        computed = payload_digest(self)
        if self.payload_digest not in (None, computed):
            raise ValueError("payload_digest does not match the canonical payload")
        sized = self.model_dump(mode="json", exclude_none=False)
        sized["payload_digest"] = computed
        if (
            len(canonical_payload_json(sized).encode("utf-8"))
            > MAX_OPERATION_PAYLOAD_BYTES
        ):
            raise ValueError("operation payload exceeds its encoded size bound")
        if self.payload_digest is None:
            # Pydantic's frozen model validator cannot return a replacement
            # instance when called through ``__init__``.  Set the derived
            # value exactly once after all fields have been normalized; the
            # public model remains immutable thereafter.
            object.__setattr__(self, "payload_digest", computed)
        return self

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> RepositoryBuildExecutionPayloadV1:
        """Copy only with a digest derived from the copied canonical body."""

        del deep
        values = self.model_dump(mode="python", exclude_none=False)
        if update:
            values.update(update)
            if "payload_digest" not in update:
                values.pop("payload_digest", None)
        return type(self).model_validate(values)


RepositoryOperationPayload: TypeAlias = Annotated[
    RepositoryBuildExecutionPayloadV1,
    Field(discriminator="kind"),
]

# This is an additive extension registry, not the shared WorkItem metadata
# mapping.  RMDD-28 can register its lane/resource sibling independently; no
# operation-payload entry replaces or mutates that sibling's key.
OPERATION_PAYLOAD_EXTENSION_KEY: Literal["operation_payload"] = "operation_payload"
OPERATION_PAYLOAD_VARIANTS: Mapping[str, type[BaseModel]] = MappingProxyType(
    {PAYLOAD_KIND_BUILD: RepositoryBuildExecutionPayloadV1}
)
REPOSITORY_WORK_ITEM_EXTENSION_REGISTRY = MappingProxyType(
    {OPERATION_PAYLOAD_EXTENSION_KEY: OPERATION_PAYLOAD_VARIANTS}
)


def compose_operation_payload_extension_registry(
    extensions: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Add the payload sibling without replacing existing RMDD extensions."""

    composed = dict(extensions or {})
    existing = composed.get(OPERATION_PAYLOAD_EXTENSION_KEY)
    if existing is not None and existing != OPERATION_PAYLOAD_VARIANTS:
        raise ValueError("operation_payload extension is already bound differently")
    composed[OPERATION_PAYLOAD_EXTENSION_KEY] = OPERATION_PAYLOAD_VARIANTS
    return MappingProxyType(composed)


def operation_payload_variant(kind: object) -> type[BaseModel]:
    """Resolve a closed discriminator without accepting future mappings."""

    if not isinstance(kind, str):
        raise ValueError("operation payload kind must be a string discriminator")
    variant = OPERATION_PAYLOAD_VARIANTS.get(kind)
    if variant is None:
        raise ValueError("unknown operation payload discriminator")
    return variant


def operation_payload_from_mapping(value: object) -> RepositoryBuildExecutionPayloadV1:
    """Validate one closed payload at an authority boundary."""

    if isinstance(value, RepositoryBuildExecutionPayloadV1):
        return RepositoryBuildExecutionPayloadV1.model_validate(
            value.model_dump(mode="python", exclude_none=False)
        )
    if not isinstance(value, Mapping):
        raise TypeError("operation_payload must be a typed mapping")
    return RepositoryBuildExecutionPayloadV1.model_validate(dict(value))


__all__ = [
    "MAX_OPERATION_PAYLOAD_BYTES",
    "OPERATION_PAYLOAD_EXTENSION_KEY",
    "OPERATION_PAYLOAD_VARIANTS",
    "REPOSITORY_WORK_ITEM_EXTENSION_REGISTRY",
    "PAYLOAD_KIND_BUILD",
    "PAYLOAD_SCHEMA_VERSION",
    "RepositoryBuildExecutionPayloadV1",
    "RepositoryCacheKeyComponent",
    "RepositoryOperationPayload",
    "canonical_payload_json",
    "operation_payload_from_mapping",
    "compose_operation_payload_extension_registry",
    "operation_payload_variant",
    "payload_digest",
]
