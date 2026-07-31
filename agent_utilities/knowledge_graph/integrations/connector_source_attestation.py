"""Current-only signed source attestation for connector capability bundles.

This contract proves only repository-owned bundle structure. It deliberately has
no representation for a passing live tool or executed fixture check; those belong
to :mod:`connector_certification` and its external-live evidence record.

CONCEPT:AU-KG.ontology.derived-compatibility-band — the compatibility band an
attestation carries is **derived** from the one pinned version of each component
(``agent_utilities._version.__version__`` for the runtime, the release
compatibility matrix for the engine) and is validated by *containment*, never by
string equality. Two consequences, both deliberate:

* The band is a **minor floor** (``>=MAJOR.MINOR.0,<MAJOR+1``). A patch release
  therefore recomputes to the identical string, so a patch bump structurally
  cannot change what an attestation must declare.
* Validation asks "does the declared band still contain the released version?",
  so an attestation signed under 2.1.0 stays admissible under 2.1.1 and 2.9.9 and
  is rejected only by the major boundary it already declared.

Before this contract was derived, ``.bumpversion.cfg`` rewrote the literal band on
every bump and the gate compared it for exact equality: a routine patch release
invalidated the entire signed connector fleet. Nothing may reintroduce that
coupling — ``scripts/check_version_consistency.py`` fails closed if any
``bumpversion`` file section targets this module.
"""

from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from agent_utilities.knowledge_graph.ontology import ontology_integrity
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
)

SOURCE_ATTESTATION_KEYS = frozenset(
    {
        "api_version",
        "kind",
        "schema_version",
        "connector",
        "validated_at",
        "mode",
        "status",
        "live_certified",
        "checks",
        "compatibility",
        "artifacts",
        "signer",
        "signature_algorithm",
        "signing_public_key",
        "signature",
    }
)

#: The bundle layout revision. A real schema number, not a package version — it
#: moves only when the artifact set itself changes, so it is compared exactly.
BUNDLE_SCHEMA_VERSION = "2"

#: The exact component bands an attestation declares. Both are derived; neither
#: is duplicated anywhere a release tool could rewrite it.
COMPATIBILITY_KEYS = ("agent_utilities", "bundle_schema", "epistemic_graph")

_UTC_TIMESTAMP = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _minor_band(version: str) -> str:
    """Render the minor-floor band a release of ``version`` is published under."""

    try:
        parsed = Version(version)
    except InvalidVersion as exc:
        raise ValueError("pinned component version is invalid") from exc
    return f">={parsed.major}.{parsed.minor}.0,<{parsed.major + 1}"


def _released_runtime_version() -> str:
    """The single agent-utilities version authority."""

    from agent_utilities._version import __version__

    return str(__version__)


def _released_engine_version() -> str:
    """The engine version the release compatibility matrix pins, exactly once."""

    from importlib.resources import files

    import yaml

    resource = files("deploy.release").joinpath("compatibility-matrix.yml")
    try:
        document = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
        pinned = str(document["components"]["epistemic-graph"]["version"])
    except (KeyError, OSError, TypeError, yaml.YAMLError) as exc:
        raise ValueError("release compatibility matrix is unreadable") from exc
    if not pinned.startswith("=="):
        raise ValueError("release compatibility matrix engine pin is not exact")
    return pinned.removeprefix("==")


@lru_cache(maxsize=1)
def _pinned_component_versions() -> tuple[str, str]:
    return _released_runtime_version(), _released_engine_version()


def reset_compatibility_cache() -> None:
    """Drop the memoized pinned versions (tests, and after a release bump)."""

    _pinned_component_versions.cache_clear()


def source_compatibility() -> dict[str, str]:
    """The compatibility band a freshly generated attestation must declare."""

    runtime_version, engine_version = _pinned_component_versions()
    return {
        "agent_utilities": _minor_band(runtime_version),
        "bundle_schema": BUNDLE_SCHEMA_VERSION,
        "epistemic_graph": _minor_band(engine_version),
    }


def _band_violations(key: str, declared: Any, released: str) -> list[str]:
    """A declared band is admissible when it is bounded and still contains us."""

    try:
        specifier = SpecifierSet(str(declared))
    except InvalidSpecifier:
        return [f"source attestation {key} band is not a version specifier"]
    operators = {clause.operator for clause in specifier}
    if ">=" not in operators or "<" not in operators:
        return [f"source attestation {key} band is not bounded on both sides"]
    if not specifier.contains(Version(released)):
        return [f"source attestation {key} band excludes the released version"]
    return []


def compatibility_violations(declared: Any) -> list[str]:
    """Validate a declared band by containment, never by textual equality."""

    if not isinstance(declared, dict) or set(declared) != set(COMPATIBILITY_KEYS):
        return ["source attestation compatibility shape differs"]
    if declared.get("bundle_schema") != BUNDLE_SCHEMA_VERSION:
        return ["source attestation bundle schema differs"]
    runtime_version, engine_version = _pinned_component_versions()
    return [
        *_band_violations(
            "agent_utilities", declared.get("agent_utilities"), runtime_version
        ),
        *_band_violations(
            "epistemic_graph", declared.get("epistemic_graph"), engine_version
        ),
    ]


def expected_source_checks(manifest: ConnectorManifest) -> dict[str, str]:
    """Return the exact structural checks represented by the source attestation."""

    sync_result = "passed" if manifest.sync else "not-applicable"
    return {
        "declared_tool_schema": sync_result,
        "synthetic_fixture_contract": sync_result,
        "shacl_parse": "passed",
        "privacy_contract": "passed",
        "manifest_integrity": "passed",
    }


def source_attestation_violations(
    attestation: dict[str, Any], manifest: ConnectorManifest
) -> list[str]:
    """Validate the exact offline contract without inferring live execution."""

    violations: list[str] = []
    if set(attestation) != SOURCE_ATTESTATION_KEYS:
        violations.append("source attestation shape differs")
    expected_scalars = {
        "api_version": "graphos.io/v1",
        "kind": "ConnectorSourceAttestation",
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "mode": "offline-source",
        "status": "source-validated",
        "live_certified": False,
    }
    if any(attestation.get(key) != value for key, value in expected_scalars.items()):
        violations.append("source attestation identity differs")
    if attestation.get("connector") != manifest.connector:
        violations.append("source attestation connector differs")
    if attestation.get("checks") != expected_source_checks(manifest):
        violations.append("source attestation checks differ")
    if any(
        not isinstance(sync.tool_schema_sha256, str)
        or not _SHA256.fullmatch(sync.tool_schema_sha256)
        for sync in manifest.sync
    ):
        violations.append("declared tool schema fingerprint is invalid")
    violations.extend(compatibility_violations(attestation.get("compatibility")))
    validated_at = attestation.get("validated_at")
    if not isinstance(validated_at, str) or not _UTC_TIMESTAMP.fullmatch(validated_at):
        violations.append("source attestation timestamp is invalid")
    artifacts = attestation.get("artifacts")
    if (
        not isinstance(artifacts, dict)
        or not artifacts
        or any(
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            for path, digest in artifacts.items()
        )
    ):
        violations.append("source attestation artifact ledger is invalid")
    return violations


def build_source_attestation(
    manifest: ConnectorManifest,
    *,
    artifacts: dict[str, str],
    now: datetime,
    release_signer: ontology_integrity.ReleaseSigner,
) -> dict[str, Any]:
    """Build and sign a source-only attestation with no live-pass state."""

    attestation: dict[str, Any] = {
        "api_version": "graphos.io/v1",
        "kind": "ConnectorSourceAttestation",
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "connector": manifest.connector,
        "validated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "offline-source",
        "status": "source-validated",
        "live_certified": False,
        "checks": expected_source_checks(manifest),
        "compatibility": source_compatibility(),
        "artifacts": dict(sorted(artifacts.items())),
        "signer": release_signer.signer_id,
        "signature_algorithm": release_signer.algorithm,
        "signing_public_key": release_signer.public_key,
        "signature": None,
    }
    violations = source_attestation_violations(attestation, manifest)
    if violations:
        raise ValueError("source attestation inputs are invalid")
    attestation["signature"] = release_signer.sign(
        ontology_integrity.canonical_signed_document_hash(attestation)
    )
    return attestation


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "COMPATIBILITY_KEYS",
    "SOURCE_ATTESTATION_KEYS",
    "build_source_attestation",
    "compatibility_violations",
    "expected_source_checks",
    "reset_compatibility_cache",
    "source_attestation_violations",
    "source_compatibility",
]
