"""``FrontendContribution.v1`` provider contract, discovery, and validation.

GOC-24: lets any installed fleet package describe navigation-adjacent read
models, governed actions, panels, and documentation for Agent WebUI
**without a WebUI core edit**. This module is the Agent Utilities side of
the contract: it is the authority for a contribution's declared
identity/version/entry-point and for schema/provenance validation. The
Epistemic Graph catalog projection and the WebUI typed client
(``agent-webui/src/lib/frontend-contributions.ts``) are downstream
consumers, not a second source of truth.

Descriptors are **metadata only**. Discovery never imports the registering
package's Python code -- it resolves ownership the same way
``core.providers`` already does for skill/prompt/ontology providers (via
distribution metadata, never ``importlib.import_module``), then reads one
bounded JSON file (``contribution.json``) with ``json.loads``. A descriptor
cannot execute browser code, grant a capability, or widen egress: this
module rejects executable-looking content outright, and every
``capability``/``schema`` reference stays a *reference* -- policy/preflight
authority remains at the consuming surface (see the ``capability_exists``
callback below), never granted here.

Reused from ``core.providers``/``core.provider_materialization`` on purpose:
the entry-point ownership proof (no ambiguous/unowned source root, no
symlink escape, bounded file count/size) is exactly the same hardened
machinery the skill/prompt/ontology legs already depend on -- this module
adds no second ownership-resolution path.

CONCEPT:AU-ECO.ui.frontend-contribution
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from agent_utilities.core.provider_materialization import (
    ProviderAssetError,
    build_asset_manifest,
)
from agent_utilities.core.providers import ProviderRegistration, provider_registrations

FRONTEND_PROVIDER_GROUP = "agent_utilities.frontend_providers"
DESCRIPTOR_FILENAME = "contribution.json"
SCHEMA_VERSION = "frontend-contribution.v1"

#: Algorithmic/resource budget (lane doc "Algorithmic and resource budget").
MAX_DESCRIPTOR_BYTES = 256 * 1024
MAX_READ_MODELS = 64
MAX_PANELS = 32
MAX_ACTIONS = 64
MAX_SCOPES = 64
MAX_TOPICS = 64
MAX_STRING = 4096
MAX_EXTENSIONS_BYTES = 32 * 1024
MAX_COLUMNS = 64

ContributionStatus = Literal["OK", "DEGRADED", "BLOCKED", "MISSING"]

_RENDERERS = frozenset(
    {
        "data-table",
        "metric-cards",
        "timeseries",
        "trace",
        "map",
        "code",
        "evidence",
        "graph",
        "json",
        "list",
        "detail",
    }
)
_APPROVAL_CLASSES = frozenset({"none", "read", "change", "sensitive", "destructive"})
_CONFIRM_MODES = frozenset({"none", "preflight", "double"})
_PLACEMENTS = frozenset({"row", "toolbar", "panel", "bulk"})

_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SCOPE_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
#: Deliberately broad -- this is a *reject list*, not a sanitizer. Any hit
#: blocks the whole descriptor; there is no "safe" executable content in a
#: metadata-only contract, so false positives are the correct failure mode.
_UNSAFE_CONTENT_RE = re.compile(
    r"javascript:|data:text/html|data:application/|<script|</script"
    r"|on[a-z]+\s*=\s*[\"']|vbscript:|<iframe|file://",
    re.IGNORECASE,
)


class _StrictModel(BaseModel):
    """Extra fields are rejected -- unknown fields are a TCK failure, not a
    silently-ignored extension (lane invariant: "unknown fields are rejected
    at TCK time unless the schema explicitly marks an extension bag")."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _valid_id(value: str) -> str:
    if not _ID_RE.fullmatch(value):
        raise ValueError("must be a lowercase dotted/underscored identifier")
    return value


def _valid_scope(value: str) -> str:
    if not _SCOPE_RE.fullmatch(value):
        raise ValueError("must be a lowercase dotted/colon-delimited scope")
    return value


class NavRef(_StrictModel):
    section: str = Field(min_length=1, max_length=64)
    order: int = Field(ge=0, le=10_000)


class RefreshPolicy(_StrictModel):
    mode: Literal["event", "poll", "manual"]
    fallback_seconds: int = Field(ge=1, le=86_400)


class ReadModelRef(_StrictModel):
    id: str
    schema_id: str = Field(alias="schema", min_length=1, max_length=MAX_STRING)
    capability: str
    renderer: str
    refresh: RefreshPolicy | None = None
    columns: tuple[str, ...] = Field(default=(), max_length=MAX_COLUMNS)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _valid_id(value)

    @field_validator("capability")
    @classmethod
    def _check_capability(cls, value: str) -> str:
        return _valid_id(value)

    @field_validator("renderer")
    @classmethod
    def _check_renderer(cls, value: str) -> str:
        if value not in _RENDERERS:
            raise ValueError(f"renderer {value!r} is not in the allowlist")
        return value

    @field_validator("columns")
    @classmethod
    def _check_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if not item or len(item) > 128:
                raise ValueError("column name is empty or oversized")
        return value


class ActionRef(_StrictModel):
    id: str
    capability: str
    placement: str
    confirm: str
    approval_class: str

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _valid_id(value)

    @field_validator("capability")
    @classmethod
    def _check_capability(cls, value: str) -> str:
        return _valid_id(value)

    @field_validator("placement")
    @classmethod
    def _check_placement(cls, value: str) -> str:
        if value not in _PLACEMENTS:
            raise ValueError(f"placement {value!r} is not in the allowlist")
        return value

    @field_validator("confirm")
    @classmethod
    def _check_confirm(cls, value: str) -> str:
        if value not in _CONFIRM_MODES:
            raise ValueError(f"confirm {value!r} is not in the allowlist")
        return value

    @field_validator("approval_class")
    @classmethod
    def _check_class(cls, value: str) -> str:
        if value not in _APPROVAL_CLASSES:
            raise ValueError(f"approval_class {value!r} is not in the allowlist")
        return value


class PanelRef(_StrictModel):
    id: str
    renderer: str

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _valid_id(value)

    @field_validator("renderer")
    @classmethod
    def _check_renderer(cls, value: str) -> str:
        if value not in _RENDERERS:
            raise ValueError(f"renderer {value!r} is not in the allowlist")
        return value


class Provenance(_StrictModel):
    source: Literal["package-entry-point"]
    signer_key_id: str = Field(min_length=1, max_length=256)
    artifact_digest: str

    @field_validator("artifact_digest")
    @classmethod
    def _check_digest(cls, value: str) -> str:
        if not _DIGEST_RE.fullmatch(value):
            raise ValueError("artifact_digest must be sha256:<hex64>")
        return value


class FrontendContributionV1(_StrictModel):
    """The exact v1 shape from the GOC-24 lane doc's canonical example.

    Compatibility is additive-only within v1 -- a new *optional* field is a
    non-breaking change; anything else is a v2 with a migration adapter
    (lane invariant), so this model deliberately has no version-negotiation
    escape hatch of its own.
    """

    schema_version: Literal["frontend-contribution.v1"]
    package_id: str
    package_version: str = Field(min_length=1, max_length=128)
    descriptor_version: int = Field(ge=1, le=1000)
    descriptor_digest: str
    title: str = Field(min_length=1, max_length=128)
    icon: str = Field(min_length=1, max_length=64)
    nav: NavRef
    required_scopes: tuple[str, ...] = Field(default=(), max_length=MAX_SCOPES)
    read_models: tuple[ReadModelRef, ...] = Field(
        min_length=1, max_length=MAX_READ_MODELS
    )
    actions: tuple[ActionRef, ...] = Field(default=(), max_length=MAX_ACTIONS)
    panels: tuple[PanelRef, ...] = Field(default=(), max_length=MAX_PANELS)
    realtime_topics: tuple[str, ...] = Field(default=(), max_length=MAX_TOPICS)
    empty_state: str = Field(min_length=1, max_length=MAX_STRING)
    docs_ref: str
    provenance: Provenance
    extensions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("package_id")
    @classmethod
    def _check_package_id(cls, value: str) -> str:
        return _valid_id(value)

    @field_validator("descriptor_digest")
    @classmethod
    def _check_descriptor_digest(cls, value: str) -> str:
        if not _DIGEST_RE.fullmatch(value):
            raise ValueError("descriptor_digest must be sha256:<hex64>")
        return value

    @field_validator("docs_ref")
    @classmethod
    def _check_docs_ref(cls, value: str) -> str:
        # Lane invariant: "URI/docs schemes (pkg:/approved local docs only)".
        # A remote/credentialed origin in docs_ref is an egress vector, not
        # documentation -- reject anything but the pkg: scheme outright.
        if not value.startswith("pkg:") or len(value) > MAX_STRING:
            raise ValueError("docs_ref must use the pkg: scheme")
        return value

    @field_validator("required_scopes", "realtime_topics")
    @classmethod
    def _check_scope_lists(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            _valid_scope(item)
        return value

    @model_validator(mode="after")
    def _check_minimum_read_model(self) -> FrontendContributionV1:
        # Lane invariant: "Required minimum descriptor must include one read
        # model (health or inventory)".
        if not any(model.id in {"health", "inventory"} for model in self.read_models):
            raise ValueError(
                "descriptor must include a 'health' or 'inventory' read model"
            )
        return self

    @model_validator(mode="after")
    def _check_extensions_bounded(self) -> FrontendContributionV1:
        encoded = json.dumps(
            self.extensions, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        if len(encoded) > MAX_EXTENSIONS_BYTES:
            raise ValueError("extensions exceeds its byte bound")
        return self


def compute_descriptor_digest(payload: dict[str, Any]) -> str:
    """Deterministic digest over a descriptor payload.

    Computed over the canonical (sorted-key, compact) JSON encoding of
    ``payload`` with its own ``descriptor_digest`` field blanked out first
    (the field cannot include a hash of itself). Package authors and this
    module's discovery path both use this function, so a descriptor's
    declared digest is directly re-derivable and checkable -- never trusted
    as an opaque claim.
    """

    normalized = dict(payload)
    normalized["descriptor_digest"] = ""
    canonical = json.dumps(normalized, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def scan_unsafe_content(raw_text: str) -> str | None:
    """Return the first unsafe-content match in ``raw_text``, or ``None``.

    Run over the *raw* descriptor bytes before JSON parsing, so it also
    catches unsafe content hidden in a field the schema would otherwise
    accept as an unremarkable string (title, empty_state, icon, extensions).
    """

    match = _UNSAFE_CONTENT_RE.search(raw_text)
    return match.group(0) if match else None


@dataclass(frozen=True, slots=True)
class FrontendContributionRecord:
    """One provider's contribution, resolved to an explicit terminal status.

    Every installed ``agent_utilities.frontend_providers`` entry point
    produces exactly one record -- there is no silent skip (lane invariant:
    "no malformed provider is silently skipped").
    """

    package_id: str
    provider_name: str
    status: ContributionStatus
    reason: str | None
    descriptor: FrontendContributionV1 | None
    descriptor_digest: str | None
    registration_digest: str
    source_digest: str | None


def _blocked(
    registration: ProviderRegistration, *, reason: str, package_id: str | None = None
) -> FrontendContributionRecord:
    return FrontendContributionRecord(
        package_id=package_id or registration.name,
        provider_name=registration.name,
        status="BLOCKED",
        reason=reason,
        descriptor=None,
        descriptor_digest=None,
        registration_digest=registration.digest,
        source_digest=None,
    )


def _load_one(
    registration: ProviderRegistration,
    *,
    capability_exists: Callable[[str], bool] | None,
    trusted_signers: frozenset[str],
) -> FrontendContributionRecord:
    if registration.source_root is None:
        return FrontendContributionRecord(
            package_id=registration.name,
            provider_name=registration.name,
            status="MISSING",
            reason="source_unresolved",
            descriptor=None,
            descriptor_digest=None,
            registration_digest=registration.digest,
            source_digest=None,
        )

    try:
        manifest = build_asset_manifest(
            registration.source_root,
            leg="data",
            allowed_relative_paths=registration.owned_paths,
        )
    except (OSError, ProviderAssetError, ValueError) as exc:
        return _blocked(registration, reason=f"asset_manifest_invalid:{exc}")

    matches = [
        entry
        for entry in manifest.entries
        if entry.relative_path == DESCRIPTOR_FILENAME
    ]
    if len(matches) != 1:
        return _blocked(registration, reason="descriptor_file_missing_or_ambiguous")
    entry = matches[0]
    if entry.size > MAX_DESCRIPTOR_BYTES:
        return _blocked(registration, reason="descriptor_oversized")

    try:
        raw_text = entry.source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return _blocked(registration, reason=f"descriptor_unreadable:{exc}")
    if len(raw_text.encode("utf-8")) > MAX_DESCRIPTOR_BYTES:
        return _blocked(registration, reason="descriptor_oversized")

    unsafe = scan_unsafe_content(raw_text)
    if unsafe is not None:
        return _blocked(registration, reason="unsafe_content")

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return _blocked(registration, reason=f"descriptor_invalid_json:{exc}")
    if not isinstance(payload, dict):
        return _blocked(registration, reason="descriptor_not_object")

    try:
        descriptor = FrontendContributionV1.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        field_path = ".".join(str(part) for part in first.get("loc", ())) or "<root>"
        return _blocked(registration, reason=f"schema_violation:{field_path}")

    if descriptor.package_id.casefold() != registration.name.casefold():
        return _blocked(
            registration, reason="package_id_mismatch", package_id=descriptor.package_id
        )

    expected_digest = compute_descriptor_digest(payload)
    if descriptor.descriptor_digest != expected_digest:
        return _blocked(
            registration,
            reason="descriptor_digest_mismatch",
            package_id=descriptor.package_id,
        )

    # Fail-closed provenance: an unconfigured/empty trust allowlist blocks
    # EVERY descriptor, including a genuinely well-formed one (lane
    # invariant + repo-wide "fail closed" rule: a degraded read must never
    # grant permission -- absence of trust configuration is not "trust
    # everyone").
    if descriptor.provenance.signer_key_id not in trusted_signers:
        return _blocked(
            registration, reason="signer_untrusted", package_id=descriptor.package_id
        )

    status: ContributionStatus = "OK"
    reason: str | None = None
    if capability_exists is not None:
        referenced = {ref.capability for ref in descriptor.read_models}
        referenced.update(ref.capability for ref in descriptor.actions)
        missing = sorted(cap for cap in referenced if not capability_exists(cap))
        if missing:
            status = "DEGRADED"
            reason = "capability_unresolved:" + ",".join(missing)

    return FrontendContributionRecord(
        package_id=descriptor.package_id,
        provider_name=registration.name,
        status=status,
        reason=reason,
        descriptor=descriptor,
        descriptor_digest=descriptor.descriptor_digest,
        registration_digest=registration.digest,
        source_digest=entry.digest,
    )


def discover_frontend_contributions(
    *,
    capability_exists: Callable[[str], bool] | None = None,
    trusted_signers: frozenset[str] = frozenset(),
) -> tuple[FrontendContributionRecord, ...]:
    """Discover, parse, and validate every installed frontend contribution.

    Bounded ``O(P)`` over installed ``agent_utilities.frontend_providers``
    entry points (lane budget); no provider code is imported and no network
    call is made. One malformed package quarantines to its own ``BLOCKED``
    row and never hides or mutates another package's record.

    ``capability_exists`` is an injected reference check (the real check --
    against the live capability catalog -- belongs to the catalog
    materialization layer that calls this function in production; passing
    ``None`` here skips the cross-check rather than fabricating a verdict).
    ``trusted_signers`` is the configured signer-key-id allowlist; an empty
    set fails every descriptor closed, per provenance policy above.
    """

    records = [
        _load_one(
            registration,
            capability_exists=capability_exists,
            trusted_signers=trusted_signers,
        )
        for registration in provider_registrations(FRONTEND_PROVIDER_GROUP)
    ]
    return tuple(sorted(records, key=lambda record: record.package_id.casefold()))


def catalog_digest(records: tuple[FrontendContributionRecord, ...]) -> str:
    """Opaque digest identifying one full-catalog discovery snapshot (``catalog_epoch``)."""

    payload = [
        (
            record.package_id,
            record.status,
            record.descriptor_digest or "",
            record.registration_digest,
        )
        for record in records
    ]
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
