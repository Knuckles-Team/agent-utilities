#!/usr/bin/python
"""Named, explicitly-reviewable genesis k8s deployment-input profiles.

CONCEPT:AU-OS.deployment.genesis-environment-profiles — see
``.specify/design/genesis-environment-profiles/design.md`` for the full decision
record. Summary:

* A profile is ten typed, CLOSED-schema sections — ``environment``, ``target``,
  ``release``, ``runtime``, ``filesystem``, ``configuration``, ``secrets``,
  ``network``, ``identity``, ``validation`` — assembled into one
  :class:`EnvironmentProfile`. Every field a section declares must be present in the
  YAML file (an omission or an unrecognized extra key is a load error); nothing is
  silently filled in by a Python-side default. That is what makes a profile
  *reviewable*: everything the deployment will do is visible in the file, not
  resolved at apply time.
* The named profile SET is data, not a closed enum. ``dev`` / ``test`` / ``prod``
  ship as ``deploy/environments/*.yaml``; an operator adds ``uat``/``sit``/anything
  else by dropping a YAML file in that directory or in their own
  ``~/.config/agent-utilities/environments/`` — no code change.
* ``secrets.required`` holds references only (``env://``, ``vault://``,
  ``secret://``, or ``k8s-secret://<namespace>/<name>``) — never a value. A required
  secret with no reference is a load-time :class:`MissingSecretReferenceError`,
  never a default, an empty value, or an inferred lookup — "we do not infer
  credentials."

This module only defines, discovers, loads, and validates the schema. Rendering a
profile into live Kubernetes objects remains ``agent-os-genesis``'s job (Phase 4,
``deploy/k8s/production-cell/`` + ``scripts/release/render_production_cell.py``) —
see the design doc's "Scope note".
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent_utilities.core.paths import config_dir

from .config_generator import _is_secret

#: Repo-shipped default profiles (dev/test/prod). Never written to at runtime.
BUILTIN_ENVIRONMENTS_DIR = (
    Path(__file__).resolve().parents[2] / "deploy" / "environments"
)

#: ``genesis.yaml`` — read as data (not imported) so this module has no dependency
#: on the generator and no circular-import risk; see design doc "reuses genesis.yaml
#: instead of re-declaring its enums".
_GENESIS_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "genesis.yaml"

_SECRET_REF_RE = re.compile(
    r"^(?:"
    r"env://[A-Za-z_][A-Za-z0-9_]{0,127}"
    r"|vault://\S+"
    r"|secret://\S+"
    r"|k8s-secret://[a-z0-9]([a-z0-9.-]{0,251}[a-z0-9])?/[a-z0-9]([a-z0-9.-]{0,251}[a-z0-9])?"
    r")$"
)

_WRITABLE_MEDIA = frozenset({"emptyDir", "hostPath", "pvc"})


class EnvironmentProfileError(Exception):
    """A profile is missing, malformed, or fails validation.

    Always names the exact file, section, and key at fault — never a generic
    "invalid profile" message. Fail loud, fail specific.
    """


class MissingSecretReferenceError(EnvironmentProfileError):
    """A required secret has no reference. We do not infer credentials.

    Raised in place of proceeding with a default, an empty value, or a guessed
    lookup path — see AGENTS.md "Secrets & credential retrieval" and the design
    doc's second principle.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Schema — ten closed sections
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EnvironmentInputs:
    """Which named environment this is, and how strict its deploy gates are."""

    name: str
    tier: str  # "non-prod" | "prod"
    description: str
    labels: Mapping[str, str]


@dataclass(frozen=True)
class TargetInputs:
    """Where it deploys: orchestrator, namespace, permission boundary."""

    orchestrator: str
    namespace: str
    authority: str  # genesis.yaml run_plan.substrate_authority
    cluster_context_ref: str | None  # a kubeconfig CONTEXT NAME, never credentials
    node_selector: Mapping[str, str]


@dataclass(frozen=True)
class ReleaseInputs:
    """What ships: image, pin discipline, rollout shape."""

    image_repository: str
    tag_policy: str  # "digest-pinned" | "floating-tag"
    image_pull_policy: str
    revision: str | None  # digest/tag when tag_policy == digest-pinned
    rollout_strategy: str  # "RollingUpdate" | "Recreate"


@dataclass(frozen=True)
class RuntimeInputs:
    """How much: replicas, resource requests/limits, restart policy."""

    replicas: int
    requests_cpu: str
    requests_memory: str
    limits_cpu: str
    limits_memory: str
    restart_policy: str


@dataclass(frozen=True)
class WritablePath:
    """One writable mount a read-only-root-filesystem container still needs."""

    mount_path: str
    medium: str  # "emptyDir" | "hostPath" | "pvc"
    reason: str
    size_limit: str | None


@dataclass(frozen=True)
class FilesystemInputs:
    """Root-fs posture + the explicit writable-path exceptions to it."""

    read_only_root_filesystem: bool
    writable_paths: tuple[WritablePath, ...]
    read_only_mounts: tuple[str, ...]


@dataclass(frozen=True)
class ConfigurationInputs:
    """Non-secret configuration only — checked against the secret-suffix heuristic."""

    config_map_refs: tuple[str, ...]
    env: Mapping[str, str]


@dataclass(frozen=True)
class SecretReference:
    """A named secret's SOURCE, never its value.

    ``ref`` is one of ``env://VAR``, ``vault://path``, ``secret://path``, or
    ``k8s-secret://<namespace>/<name>``. ``keys`` documents which fields the
    referenced secret is expected to contain (for review), not their values.
    """

    name: str
    ref: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class SecretsInputs:
    """Every secret this profile's deployment requires, by reference."""

    required: tuple[SecretReference, ...]


@dataclass(frozen=True)
class NetworkInputs:
    """Ingress, service exposure, and namespace-level traffic policy."""

    ingress_host: str | None
    service_port: int
    network_policy_enabled: bool
    allowed_namespaces: tuple[str, ...]


@dataclass(frozen=True)
class IdentityInputs:
    """IdP wiring + the workload identity/service-account posture."""

    idp: str  # genesis.yaml run_plan.idp
    service_account: str
    automount_service_account_token: bool
    client_secret_ref: str | None  # a NAME in secrets.required, not a literal ref


@dataclass(frozen=True)
class FunctionalCheck:
    """A proof obligation stronger than liveness — e.g. a real MCP ``tools/list``."""

    kind: str  # e.g. "mcp-tools-list", "graph-query-roundtrip"
    target: str
    expected: str


@dataclass(frozen=True)
class ValidationInputs:
    """What must be PROVEN post-deploy, not merely observed as "not crashing"."""

    readiness_checks: tuple[str, ...]
    liveness_checks: tuple[str, ...]
    functional_checks: tuple[FunctionalCheck, ...]


@dataclass(frozen=True)
class EnvironmentProfile:
    """The ten-category genesis k8s deployment-input profile for one named environment."""

    environment: EnvironmentInputs
    target: TargetInputs
    release: ReleaseInputs
    runtime: RuntimeInputs
    filesystem: FilesystemInputs
    configuration: ConfigurationInputs
    secrets: SecretsInputs
    network: NetworkInputs
    identity: IdentityInputs
    validation: ValidationInputs
    source: Path = field(compare=False)


_TOP_LEVEL_SECTIONS = (
    "environment",
    "target",
    "release",
    "runtime",
    "filesystem",
    "configuration",
    "secrets",
    "network",
    "identity",
    "validation",
)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing — fail loud, name the exact section/key
# ─────────────────────────────────────────────────────────────────────────────


def _require_keys(
    mapping: Any, keys: Sequence[str], *, section: str, source: Path
) -> Mapping[str, Any]:
    if not isinstance(mapping, Mapping):
        raise EnvironmentProfileError(
            f"{source}: section {section!r} must be a mapping, got "
            f"{type(mapping).__name__}."
        )
    missing = [k for k in keys if k not in mapping]
    extra = [k for k in mapping if k not in keys]
    problems = []
    if missing:
        problems.append(f"missing required key(s) {missing}")
    if extra:
        problems.append(f"unrecognized key(s) {extra}")
    if problems:
        raise EnvironmentProfileError(
            f"{source}: section {section!r} " + "; ".join(problems) + ". "
            f"Expected exactly: {list(keys)}. Profiles are explicitly reviewable — "
            "every value must be present in the file; nothing is filled in silently."
        )
    return mapping


def _str(mapping: Mapping[str, Any], key: str, *, section: str, source: Path) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise EnvironmentProfileError(
            f"{source}: {section}.{key} must be a string, got {type(value).__name__}."
        )
    return value


def _opt_str(
    mapping: Mapping[str, Any], key: str, *, section: str, source: Path
) -> str | None:
    value = mapping[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise EnvironmentProfileError(
            f"{source}: {section}.{key} must be a string or null, got "
            f"{type(value).__name__}."
        )
    return value


def _bool(mapping: Mapping[str, Any], key: str, *, section: str, source: Path) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise EnvironmentProfileError(
            f"{source}: {section}.{key} must be true/false, got {type(value).__name__}."
        )
    return value


def _int(mapping: Mapping[str, Any], key: str, *, section: str, source: Path) -> int:
    value = mapping[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise EnvironmentProfileError(
            f"{source}: {section}.{key} must be an integer, got {type(value).__name__}."
        )
    return value


def _str_tuple(
    mapping: Mapping[str, Any], key: str, *, section: str, source: Path
) -> tuple[str, ...]:
    value = mapping[key]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise EnvironmentProfileError(
            f"{source}: {section}.{key} must be a list of strings, got {value!r}."
        )
    return tuple(value)


def _str_mapping(
    mapping: Mapping[str, Any], key: str, *, section: str, source: Path
) -> Mapping[str, str]:
    value = mapping[key]
    if not isinstance(value, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise EnvironmentProfileError(
            f"{source}: {section}.{key} must be a mapping of string to string, got "
            f"{value!r}."
        )
    return dict(value)


def _environment_from_mapping(raw: Any, source: Path) -> EnvironmentInputs:
    m = _require_keys(
        raw,
        ("name", "tier", "description", "labels"),
        section="environment",
        source=source,
    )
    return EnvironmentInputs(
        name=_str(m, "name", section="environment", source=source),
        tier=_str(m, "tier", section="environment", source=source),
        description=_str(m, "description", section="environment", source=source),
        labels=_str_mapping(m, "labels", section="environment", source=source),
    )


def _target_from_mapping(raw: Any, source: Path) -> TargetInputs:
    m = _require_keys(
        raw,
        (
            "orchestrator",
            "namespace",
            "authority",
            "cluster_context_ref",
            "node_selector",
        ),
        section="target",
        source=source,
    )
    return TargetInputs(
        orchestrator=_str(m, "orchestrator", section="target", source=source),
        namespace=_str(m, "namespace", section="target", source=source),
        authority=_str(m, "authority", section="target", source=source),
        cluster_context_ref=_opt_str(
            m, "cluster_context_ref", section="target", source=source
        ),
        node_selector=_str_mapping(m, "node_selector", section="target", source=source),
    )


def _release_from_mapping(raw: Any, source: Path) -> ReleaseInputs:
    m = _require_keys(
        raw,
        (
            "image_repository",
            "tag_policy",
            "image_pull_policy",
            "revision",
            "rollout_strategy",
        ),
        section="release",
        source=source,
    )
    return ReleaseInputs(
        image_repository=_str(m, "image_repository", section="release", source=source),
        tag_policy=_str(m, "tag_policy", section="release", source=source),
        image_pull_policy=_str(
            m, "image_pull_policy", section="release", source=source
        ),
        revision=_opt_str(m, "revision", section="release", source=source),
        rollout_strategy=_str(m, "rollout_strategy", section="release", source=source),
    )


def _runtime_from_mapping(raw: Any, source: Path) -> RuntimeInputs:
    m = _require_keys(
        raw,
        (
            "replicas",
            "requests_cpu",
            "requests_memory",
            "limits_cpu",
            "limits_memory",
            "restart_policy",
        ),
        section="runtime",
        source=source,
    )
    return RuntimeInputs(
        replicas=_int(m, "replicas", section="runtime", source=source),
        requests_cpu=_str(m, "requests_cpu", section="runtime", source=source),
        requests_memory=_str(m, "requests_memory", section="runtime", source=source),
        limits_cpu=_str(m, "limits_cpu", section="runtime", source=source),
        limits_memory=_str(m, "limits_memory", section="runtime", source=source),
        restart_policy=_str(m, "restart_policy", section="runtime", source=source),
    )


def _writable_path_from_mapping(raw: Any, *, index: int, source: Path) -> WritablePath:
    section = f"filesystem.writable_paths[{index}]"
    m = _require_keys(
        raw,
        ("mount_path", "medium", "reason", "size_limit"),
        section=section,
        source=source,
    )
    return WritablePath(
        mount_path=_str(m, "mount_path", section=section, source=source),
        medium=_str(m, "medium", section=section, source=source),
        reason=_str(m, "reason", section=section, source=source),
        size_limit=_opt_str(m, "size_limit", section=section, source=source),
    )


def _filesystem_from_mapping(raw: Any, source: Path) -> FilesystemInputs:
    m = _require_keys(
        raw,
        ("read_only_root_filesystem", "writable_paths", "read_only_mounts"),
        section="filesystem",
        source=source,
    )
    writable_raw = m["writable_paths"]
    if not isinstance(writable_raw, list):
        raise EnvironmentProfileError(
            f"{source}: filesystem.writable_paths must be a list, got {writable_raw!r}."
        )
    writable = tuple(
        _writable_path_from_mapping(item, index=i, source=source)
        for i, item in enumerate(writable_raw)
    )
    return FilesystemInputs(
        read_only_root_filesystem=_bool(
            m, "read_only_root_filesystem", section="filesystem", source=source
        ),
        writable_paths=writable,
        read_only_mounts=_str_tuple(
            m, "read_only_mounts", section="filesystem", source=source
        ),
    )


def _configuration_from_mapping(raw: Any, source: Path) -> ConfigurationInputs:
    m = _require_keys(
        raw, ("config_map_refs", "env"), section="configuration", source=source
    )
    return ConfigurationInputs(
        config_map_refs=_str_tuple(
            m, "config_map_refs", section="configuration", source=source
        ),
        env=_str_mapping(m, "env", section="configuration", source=source),
    )


def _secret_reference_from_mapping(
    raw: Any, *, index: int, source: Path
) -> SecretReference:
    section = f"secrets.required[{index}]"
    if not isinstance(raw, Mapping):
        raise EnvironmentProfileError(
            f"{source}: {section} must be a mapping, got {type(raw).__name__}."
        )
    missing = [k for k in ("name", "ref") if k not in raw]
    if missing:
        raise EnvironmentProfileError(
            f"{source}: {section} missing required key(s) {missing}."
        )
    extra = [k for k in raw if k not in ("name", "ref", "keys")]
    if extra:
        raise EnvironmentProfileError(
            f"{source}: {section} unrecognized key(s) {extra}. Expected: "
            "['name', 'ref', 'keys']."
        )
    name = _str(raw, "name", section=section, source=source)
    ref = raw.get("ref")
    if not isinstance(ref, str) or not ref.strip():
        raise MissingSecretReferenceError(
            f"{source}: {section} (name={name!r}) has no ref. We do not infer "
            "credentials — declare env://VAR, vault://<path>, secret://<path>, or "
            "k8s-secret://<namespace>/<name>."
        )
    keys = raw.get("keys", [])
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        raise EnvironmentProfileError(
            f"{source}: {section}.keys must be a list of strings, got {keys!r}."
        )
    return SecretReference(name=name, ref=ref, keys=tuple(keys))


def _secrets_from_mapping(raw: Any, source: Path) -> SecretsInputs:
    m = _require_keys(raw, ("required",), section="secrets", source=source)
    required_raw = m["required"]
    if not isinstance(required_raw, list):
        raise EnvironmentProfileError(
            f"{source}: secrets.required must be a list, got {required_raw!r}."
        )
    return SecretsInputs(
        required=tuple(
            _secret_reference_from_mapping(item, index=i, source=source)
            for i, item in enumerate(required_raw)
        )
    )


def _network_from_mapping(raw: Any, source: Path) -> NetworkInputs:
    m = _require_keys(
        raw,
        (
            "ingress_host",
            "service_port",
            "network_policy_enabled",
            "allowed_namespaces",
        ),
        section="network",
        source=source,
    )
    return NetworkInputs(
        ingress_host=_opt_str(m, "ingress_host", section="network", source=source),
        service_port=_int(m, "service_port", section="network", source=source),
        network_policy_enabled=_bool(
            m, "network_policy_enabled", section="network", source=source
        ),
        allowed_namespaces=_str_tuple(
            m, "allowed_namespaces", section="network", source=source
        ),
    )


def _identity_from_mapping(raw: Any, source: Path) -> IdentityInputs:
    m = _require_keys(
        raw,
        (
            "idp",
            "service_account",
            "automount_service_account_token",
            "client_secret_ref",
        ),
        section="identity",
        source=source,
    )
    return IdentityInputs(
        idp=_str(m, "idp", section="identity", source=source),
        service_account=_str(m, "service_account", section="identity", source=source),
        automount_service_account_token=_bool(
            m, "automount_service_account_token", section="identity", source=source
        ),
        client_secret_ref=_opt_str(
            m, "client_secret_ref", section="identity", source=source
        ),
    )


def _functional_check_from_mapping(
    raw: Any, *, index: int, source: Path
) -> FunctionalCheck:
    section = f"validation.functional_checks[{index}]"
    m = _require_keys(
        raw, ("kind", "target", "expected"), section=section, source=source
    )
    return FunctionalCheck(
        kind=_str(m, "kind", section=section, source=source),
        target=_str(m, "target", section=section, source=source),
        expected=_str(m, "expected", section=section, source=source),
    )


def _validation_from_mapping(raw: Any, source: Path) -> ValidationInputs:
    m = _require_keys(
        raw,
        ("readiness_checks", "liveness_checks", "functional_checks"),
        section="validation",
        source=source,
    )
    functional_raw = m["functional_checks"]
    if not isinstance(functional_raw, list):
        raise EnvironmentProfileError(
            f"{source}: validation.functional_checks must be a list, got "
            f"{functional_raw!r}."
        )
    return ValidationInputs(
        readiness_checks=_str_tuple(
            m, "readiness_checks", section="validation", source=source
        ),
        liveness_checks=_str_tuple(
            m, "liveness_checks", section="validation", source=source
        ),
        functional_checks=tuple(
            _functional_check_from_mapping(item, index=i, source=source)
            for i, item in enumerate(functional_raw)
        ),
    )


def _profile_from_mapping(
    raw: Mapping[str, Any], *, source: Path
) -> EnvironmentProfile:
    _require_keys(raw, _TOP_LEVEL_SECTIONS, section="<top level>", source=source)
    return EnvironmentProfile(
        environment=_environment_from_mapping(raw["environment"], source),
        target=_target_from_mapping(raw["target"], source),
        release=_release_from_mapping(raw["release"], source),
        runtime=_runtime_from_mapping(raw["runtime"], source),
        filesystem=_filesystem_from_mapping(raw["filesystem"], source),
        configuration=_configuration_from_mapping(raw["configuration"], source),
        secrets=_secrets_from_mapping(raw["secrets"], source),
        network=_network_from_mapping(raw["network"], source),
        identity=_identity_from_mapping(raw["identity"], source),
        validation=_validation_from_mapping(raw["validation"], source),
        source=source,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Discovery + extension
# ─────────────────────────────────────────────────────────────────────────────


def _extension_dir() -> Path:
    """Operator-owned extension directory — add a profile here with zero code change."""
    return config_dir() / "environments"


def _environments_search_dirs() -> tuple[Path, ...]:
    return (BUILTIN_ENVIRONMENTS_DIR, _extension_dir())


def list_environment_profiles() -> dict[str, Path]:
    """Every discovered profile name -> the file that defines it.

    Scans the built-in ``deploy/environments/`` directory first, then the
    operator's extension directory (:func:`_extension_dir`); a later directory's
    file of the same stem OVERRIDES an earlier one (an operator can locally
    customize a shipped ``prod.yaml`` by placing their own), and a name that
    appears only in the extension directory is a genuinely new profile — the set
    is open by construction, never a closed enum.
    """
    found: dict[str, Path] = {}
    for directory in _environments_search_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.yaml")):
            found[path.stem] = path
    return found


def load_environment_profile(name: str) -> EnvironmentProfile:
    """Load and fully validate the named profile.

    Raises :class:`EnvironmentProfileError` (or the more specific
    :class:`MissingSecretReferenceError`) naming exactly what is wrong — an unknown
    profile name lists what WAS found and where to add a new one; a malformed or
    incomplete file names the section/key at fault. Never falls back to a nearby
    name or a default.
    """
    catalog = list_environment_profiles()
    if name not in catalog:
        available = ", ".join(sorted(catalog)) or "(none found)"
        raise EnvironmentProfileError(
            f"Unknown environment profile {name!r}. Available: {available}. "
            f"Add {BUILTIN_ENVIRONMENTS_DIR / f'{name}.yaml'} (repo-shipped) or "
            f"{_extension_dir() / f'{name}.yaml'} (operator extension) to define "
            "it — profile names are never inferred or guessed."
        )
    path = catalog[name]
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EnvironmentProfileError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise EnvironmentProfileError(
            f"{path}: must be a YAML mapping at the top level, got {type(raw).__name__}."
        )
    profile = _profile_from_mapping(raw, source=path)
    validate_environment_profile(profile)
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Cross-section + genesis.yaml-composed validation
# ─────────────────────────────────────────────────────────────────────────────


def _genesis_run_plan(key: str) -> list[str]:
    """Read one ``run_plan.<key>`` enum straight out of ``genesis.yaml``.

    Composes the existing manifest instead of re-declaring a second copy of its
    enums (design doc: "reuses genesis.yaml instead of re-declaring its enums").
    """
    if not _GENESIS_MANIFEST_PATH.is_file():
        raise EnvironmentProfileError(
            f"{_GENESIS_MANIFEST_PATH} not found — environment-profile validation "
            "cross-checks target/identity fields against genesis.yaml's run_plan "
            "enums and cannot proceed without it."
        )
    manifest = yaml.safe_load(_GENESIS_MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    run_plan = manifest.get("run_plan") or {}
    value = run_plan.get(key)
    if not isinstance(value, list):
        raise EnvironmentProfileError(
            f"{_GENESIS_MANIFEST_PATH}: run_plan.{key} is missing or not a list; "
            "cannot validate against it."
        )
    return value


def validate_environment_profile(profile: EnvironmentProfile) -> None:
    """Fail loudly on anything this schema promises never to allow through silently.

    Raises :class:`EnvironmentProfileError` (or :class:`MissingSecretReferenceError`
    for the credential-specific case) naming the exact problem. Returns ``None`` on
    success — never a partial-success value; see AGENTS.md "Fail closed".
    """
    source = profile.source

    # -- secrets: every ref must use a recognized reference scheme, never a value.
    secret_names = {s.name for s in profile.secrets.required}
    for secret in profile.secrets.required:
        if not _SECRET_REF_RE.match(secret.ref):
            raise EnvironmentProfileError(
                f"{source}: secrets.required[name={secret.name!r}].ref "
                f"{secret.ref!r} is not a recognized reference scheme "
                "(env://VAR, vault://<path>, secret://<path>, "
                "k8s-secret://<namespace>/<name>). A literal value is never "
                "accepted here."
            )
    if len(secret_names) != len(profile.secrets.required):
        raise EnvironmentProfileError(
            f"{source}: secrets.required has duplicate secret name(s)."
        )

    # -- identity: a declared client_secret_ref must name a real secrets.required entry.
    if (
        profile.identity.client_secret_ref is not None
        and profile.identity.client_secret_ref not in secret_names
    ):
        raise MissingSecretReferenceError(
            f"{source}: identity.client_secret_ref="
            f"{profile.identity.client_secret_ref!r} does not name any entry in "
            f"secrets.required ({sorted(secret_names)}). Declare the secret "
            "reference there first — identity never carries its own credential ref."
        )

    # -- configuration: non-secret section must not carry secret-shaped keys.
    for key in profile.configuration.env:
        if _is_secret(key):
            raise EnvironmentProfileError(
                f"{source}: configuration.env has secret-shaped key {key!r}. "
                "configuration is for non-secret settings only — declare this in "
                "secrets.required as a reference instead."
            )

    # -- filesystem: every writable-path exception must be justified and typed.
    for wp in profile.filesystem.writable_paths:
        if wp.medium not in _WRITABLE_MEDIA:
            raise EnvironmentProfileError(
                f"{source}: filesystem writable path {wp.mount_path!r} has "
                f"unrecognized medium {wp.medium!r}; expected one of "
                f"{sorted(_WRITABLE_MEDIA)}."
            )
        if not wp.reason.strip():
            raise EnvironmentProfileError(
                f"{source}: filesystem writable path {wp.mount_path!r} has no "
                "reason — every writable exception to read-only-root must be "
                "justified in the file so a reviewer can see why it exists."
            )

    # -- target/identity: cross-check against genesis.yaml's own run_plan enums.
    orchestrators = _genesis_run_plan("orchestrators")
    if profile.target.orchestrator not in orchestrators:
        raise EnvironmentProfileError(
            f"{source}: target.orchestrator {profile.target.orchestrator!r} is not "
            f"one of genesis.yaml's run_plan.orchestrators {orchestrators}."
        )
    authorities = _genesis_run_plan("substrate_authority")
    if profile.target.authority not in authorities:
        raise EnvironmentProfileError(
            f"{source}: target.authority {profile.target.authority!r} is not one "
            f"of genesis.yaml's run_plan.substrate_authority {authorities}."
        )
    idps = _genesis_run_plan("idp")
    if profile.identity.idp not in idps:
        raise EnvironmentProfileError(
            f"{source}: identity.idp {profile.identity.idp!r} is not one of "
            f"genesis.yaml's run_plan.idp {idps}."
        )

    # -- release: prod tier must pin by digest, never a floating tag.
    if profile.environment.tier == "prod":
        if profile.release.tag_policy != "digest-pinned":
            raise EnvironmentProfileError(
                f"{source}: environment.tier is 'prod' but release.tag_policy is "
                f"{profile.release.tag_policy!r} — prod requires 'digest-pinned' "
                "(floating tags are development-only)."
            )
        if not profile.release.revision:
            raise EnvironmentProfileError(
                f"{source}: environment.tier is 'prod' with tag_policy "
                "'digest-pinned' but release.revision is empty — a digest-pinned "
                "release must name the digest."
            )

    # -- validation: every profile must prove more than liveness — a real MCP
    #    tools/list, not merely /health (standing rule, not prod-only).
    if not any(
        fc.kind == "mcp-tools-list" for fc in profile.validation.functional_checks
    ):
        raise EnvironmentProfileError(
            f"{source}: validation.functional_checks has no 'mcp-tools-list' "
            "entry. A real MCP tools/list call must be proven post-deploy — "
            "checking only readiness/liveness (/health) is not sufficient."
        )


def profile_summary(profile: EnvironmentProfile) -> dict[str, Any]:
    """A reviewable, JSON-safe view of a loaded profile (no secret values — refs only)."""
    return {
        "source": str(profile.source),
        "environment": {
            "name": profile.environment.name,
            "tier": profile.environment.tier,
            "description": profile.environment.description,
            "labels": dict(profile.environment.labels),
        },
        "target": {
            "orchestrator": profile.target.orchestrator,
            "namespace": profile.target.namespace,
            "authority": profile.target.authority,
            "cluster_context_ref": profile.target.cluster_context_ref,
            "node_selector": dict(profile.target.node_selector),
        },
        "release": {
            "image_repository": profile.release.image_repository,
            "tag_policy": profile.release.tag_policy,
            "image_pull_policy": profile.release.image_pull_policy,
            "revision": profile.release.revision,
            "rollout_strategy": profile.release.rollout_strategy,
        },
        "runtime": {
            "replicas": profile.runtime.replicas,
            "requests_cpu": profile.runtime.requests_cpu,
            "requests_memory": profile.runtime.requests_memory,
            "limits_cpu": profile.runtime.limits_cpu,
            "limits_memory": profile.runtime.limits_memory,
            "restart_policy": profile.runtime.restart_policy,
        },
        "filesystem": {
            "read_only_root_filesystem": profile.filesystem.read_only_root_filesystem,
            "writable_paths": [
                {
                    "mount_path": wp.mount_path,
                    "medium": wp.medium,
                    "reason": wp.reason,
                    "size_limit": wp.size_limit,
                }
                for wp in profile.filesystem.writable_paths
            ],
            "read_only_mounts": list(profile.filesystem.read_only_mounts),
        },
        "configuration": {
            "config_map_refs": list(profile.configuration.config_map_refs),
            "env": dict(profile.configuration.env),
        },
        "secrets": {
            "required": [
                {"name": s.name, "ref": s.ref, "keys": list(s.keys)}
                for s in profile.secrets.required
            ]
        },
        "network": {
            "ingress_host": profile.network.ingress_host,
            "service_port": profile.network.service_port,
            "network_policy_enabled": profile.network.network_policy_enabled,
            "allowed_namespaces": list(profile.network.allowed_namespaces),
        },
        "identity": {
            "idp": profile.identity.idp,
            "service_account": profile.identity.service_account,
            "automount_service_account_token": profile.identity.automount_service_account_token,
            "client_secret_ref": profile.identity.client_secret_ref,
        },
        "validation": {
            "readiness_checks": list(profile.validation.readiness_checks),
            "liveness_checks": list(profile.validation.liveness_checks),
            "functional_checks": [
                {"kind": fc.kind, "target": fc.target, "expected": fc.expected}
                for fc in profile.validation.functional_checks
            ],
        },
    }
