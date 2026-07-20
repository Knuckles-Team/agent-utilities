#!/usr/bin/env python3
"""Fail-closed certification gate for connector-owned capability bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.integrations.connector_source_attestation import (  # noqa: E402
    source_attestation_violations,
)
from agent_utilities.knowledge_graph.ontology import ontology_integrity  # noqa: E402
from agent_utilities.knowledge_graph.ontology.connector_manifest import (  # noqa: E402
    ConnectorManifest,
)
from agent_utilities.knowledge_graph.ontology.connector_manifest_gate import (  # noqa: E402
    check_manifest_bytes,
)
from agent_utilities.security.persistence_privacy import (  # noqa: E402
    PersistencePrivacyGuard,
)

_PROVIDER_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON artifact is not an object")
    return value


def _module(repo: Path) -> Path:
    matches = sorted(
        path.parent.parent for path in repo.glob("*/ontology/certification.json")
    )
    if len(matches) != 1:
        raise ValueError("provider has no unique certification record")
    return matches[0]


def _manifest(path: Path) -> ConnectorManifest:
    return ConnectorManifest.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(path.parts)
        and value == path.as_posix()
        and "\\" not in value
        and not path.is_absolute()
        and ".." not in path.parts
    )


def _is_regular_contained_file(root: Path, path: Path) -> bool:
    """Return whether ``path`` is a non-symlink regular file below ``root``."""
    try:
        root_metadata = root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
            root_metadata.st_mode
        ):
            return False
        relative = path.relative_to(root)
        if not relative.parts or ".." in relative.parts:
            return False

        current = root
        for index, part in enumerate(relative.parts):
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                return False
            if index < len(relative.parts) - 1:
                if not stat.S_ISDIR(metadata.st_mode):
                    return False
            elif not stat.S_ISREG(metadata.st_mode):
                return False

        resolved_root = root.resolve(strict=True)
        current.resolve(strict=True).relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _privacy_scan_text(path: Path) -> str:
    """Mask signed digest material before applying PII pattern detection.

    Base64 signatures and hashes are high-entropy evidence, not user content,
    and can randomly contain strings matching formats such as IBANs. The gate
    still scans every semantic field and the complete TTL text; it masks only
    values under explicit cryptographic field names in structured artifacts.
    """
    text = path.read_text(encoding="utf-8")
    try:
        document = (
            json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        )
    except (ValueError, yaml.YAMLError):
        return text

    evidence: list[str] = []

    def _collect(value: Any, key: str = "") -> None:
        normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        if isinstance(value, dict):
            for child_key, child in value.items():
                _collect(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                _collect(child, key)
        elif isinstance(value, str) and (
            normalized in {"hash", "signature", "signing_public_key"}
            or normalized.endswith(("_hash", "_sha256", "_signature", "_public_key"))
        ):
            evidence.append(value)

    _collect(document)
    for value in sorted(set(evidence), key=len, reverse=True):
        if value:
            text = text.replace(value, "[CRYPTOGRAPHIC_EVIDENCE]")
    return text


def _lock_entry(lock_path: Path, connector: str) -> dict[str, Any]:
    lock = ontology_integrity.load_lock(lock_path)
    return dict(lock.get(f"agents/{connector}/connector_manifest.yml") or {})


def _configured_provider_names(workspace_path: Path) -> tuple[str, ...]:
    workspace = yaml.safe_load(workspace_path.read_text(encoding="utf-8")) or {}
    try:
        repositories = workspace["subdirectories"]["agent-packages"]["subdirectories"][
            "agents"
        ]["repositories"]
    except (KeyError, TypeError) as exc:
        raise ValueError("workspace is missing the provider repository branch") from exc
    if not isinstance(repositories, list):
        raise ValueError("provider repositories must be a list")
    if not repositories:
        raise ValueError("provider repository membership must not be empty")

    providers: list[str] = []
    for entry in repositories:
        if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
            raise ValueError("every provider repository needs a URL")
        name = entry["url"].rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".git"):
            name = name[:-4]
        if not _PROVIDER_NAME.fullmatch(name):
            raise ValueError("provider URL has an invalid repository name")
        providers.append(name)
    if len(providers) != len(set(providers)):
        raise ValueError("workspace contains duplicate provider repositories")
    return tuple(providers)


def _provider_owned_names(agents_root: Path) -> tuple[str, ...]:
    """Return every immediate checkout that claims a provider-owned manifest."""

    try:
        metadata = agents_root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError
        agents_root.resolve(strict=True)
        candidates = tuple(agents_root.iterdir())
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("provider checkout root is invalid") from exc

    providers: list[str] = []
    for candidate in candidates:
        try:
            (candidate / "connector_manifest.yml").lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("provider-owned manifest cannot be inspected") from exc
        if not _PROVIDER_NAME.fullmatch(candidate.name):
            raise ValueError("provider-owned manifest has an invalid repository name")
        providers.append(candidate.name)
    return tuple(sorted(providers))


def _default_workspace() -> Path:
    return (
        ROOT.parent
        / "agents"
        / "repository-manager"
        / "repository_manager"
        / "workspace.yml"
    )


def check_one(
    repo: Path,
    *,
    bundled_root: Path,
    lock_path: Path,
    privacy: PersistencePrivacyGuard,
) -> list[str]:
    violations: list[str] = []
    if repo.is_symlink() or not repo.is_dir():
        return ["provider checkout is not a contained directory"]
    module = _module(repo)
    manifest_path = repo / "connector_manifest.yml"
    certification_path = module / "ontology" / "certification.json"
    if not _is_regular_contained_file(repo, manifest_path):
        return ["provider manifest is not a regular contained file"]
    if not _is_regular_contained_file(repo, certification_path):
        return ["certification record is not a regular contained file"]

    manifest = _manifest(manifest_path)
    certification = _load_json(certification_path)

    if manifest.connector != repo.name or certification.get("connector") != repo.name:
        violations.append("provider identity differs across its bundle")
    violations.extend(source_attestation_violations(certification, manifest))

    violations.extend(check_manifest_bytes(manifest_path, require_signature=True))
    bundled = bundled_root / repo.name / "connector_manifest.yml"
    if not _is_regular_contained_file(bundled_root, bundled):
        violations.append("bundled manifest is not a regular contained file")
    elif bundled.read_bytes() != manifest_path.read_bytes():
        violations.append("bundled manifest differs from provider-owned manifest")

    artifacts = certification.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        violations.append("certification has no artifact ledger")
        artifacts = {}
    certified_paths: dict[str, Path] = {}
    for relative, expected in sorted(artifacts.items(), key=lambda item: str(item[0])):
        if not isinstance(relative, str) or not _safe_relative(relative):
            violations.append("certification contains a non-relative artifact name")
            continue
        path = repo.joinpath(*PurePosixPath(relative).parts)
        if not _is_regular_contained_file(repo, path):
            violations.append("certified artifact is not a regular contained file")
            continue
        certified_paths[relative] = path
        if _sha256(path) != expected:
            violations.append("certified artifact hash differs")

    required = {
        (module / "ontology" / "shapes" / "connector.shacl.ttl")
        .relative_to(repo)
        .as_posix(),
        (module / "ontology" / "mappings" / "source.yaml").relative_to(repo).as_posix(),
        (module / "ontology" / "fixtures" / "records.json")
        .relative_to(repo)
        .as_posix(),
        (module / "ontology" / "migrations" / "manifest.json")
        .relative_to(repo)
        .as_posix(),
        "connector_manifest.yml",
    }
    if not required.issubset(set(artifacts)):
        violations.append("certification artifact ledger is incomplete")

    privacy_violation = False
    privacy_unscannable = False
    privacy_targets = dict(certified_paths)
    privacy_targets[certification_path.relative_to(repo).as_posix()] = (
        certification_path
    )
    for path in privacy_targets.values():
        try:
            scan_text = _privacy_scan_text(path)
            clean, report = privacy.sanitize_text(scan_text)
        except (OSError, UnicodeError, ValueError, yaml.YAMLError):
            privacy_unscannable = True
            continue
        if report.changed or clean != scan_text:
            privacy_violation = True
    if privacy_violation:
        violations.append("bundle artifact contains a persistence privacy violation")
    if privacy_unscannable:
        violations.append("bundle artifact is not privacy-scannable")

    digest = ontology_integrity.canonical_signed_document_hash(certification)
    entry = _lock_entry(lock_path, repo.name)
    pinned_certification_key = str(entry.get("certification_signing_public_key") or "")
    if not ontology_integrity.verify_release_signature(
        digest,
        certification.get("signature"),
        signer_id=certification.get("signer"),
        algorithm=certification.get("signature_algorithm"),
        public_key=certification.get("signing_public_key"),
        trusted_public_keys=(pinned_certification_key,)
        if pinned_certification_key
        else (),
    ):
        violations.append("certification release signature is invalid")
    expected_pin = {
        "certification_file_sha256": _sha256(certification_path),
        "certification_hash": digest,
        "certification_signer": certification.get("signer"),
        "certification_signature_algorithm": certification.get("signature_algorithm"),
        "certification_signing_public_key": certification.get("signing_public_key"),
        "certification_signature": certification.get("signature"),
    }
    if any(entry.get(key) != value for key, value in expected_pin.items()):
        violations.append("certification differs from its signed release pin")

    if not required.issubset(certified_paths):
        return violations

    shapes = module / "ontology" / "shapes" / "connector.shacl.ttl"
    try:
        import rdflib

        graph = rdflib.Graph()
        graph.parse(str(shapes), format="turtle")
        target_class = rdflib.URIRef("http://www.w3.org/ns/shacl#targetClass")
        targets = {str(value) for value in graph.objects(predicate=target_class)}
        for resource in manifest.resources:
            if f"http://knuckles.team/kg#{resource.name}" not in targets:
                violations.append("SHACL coverage is incomplete")
                break
    except Exception:  # noqa: BLE001
        violations.append("SHACL artifact does not parse")

    mapping_path = module / "ontology" / "mappings" / "source.yaml"
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    mapped_sync = mapping.get("sync", []) if isinstance(mapping, dict) else []
    if not isinstance(mapped_sync, list) or len(mapped_sync) != len(manifest.sync):
        violations.append("source mapping does not cover every signed preset")
    else:
        for signed, mapped in zip(manifest.sync, mapped_sync, strict=True):
            if not isinstance(mapped, dict) or any(
                mapped.get(field) != getattr(signed, field)
                for field in ("preset", "server", "tool", "tool_schema_sha256")
            ):
                violations.append("source mapping differs from signed preset")
                break

    fixtures = _load_json(module / "ontology" / "fixtures" / "records.json")
    rows = fixtures.get("fixtures")
    if not isinstance(rows, list) or len(rows) != len(manifest.sync):
        violations.append("synthetic fixture coverage is incomplete")
    else:
        for row in rows:
            if (
                not isinstance(row, dict)
                or row.get("expected", {}).get("acl_state") != "quarantine"
            ):
                violations.append("synthetic fixture does not fail closed")
                break

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents-root", type=Path, required=True)
    parser.add_argument("--bundled-root", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=_default_workspace(),
        help="repository-manager workspace whose provider membership is authoritative",
    )
    parser.add_argument(
        "--connector",
        action="append",
        default=[],
        help="Check only this connector (repeatable); defaults to the complete fleet.",
    )
    args = parser.parse_args()
    privacy = PersistencePrivacyGuard()
    selected = set(args.connector)
    try:
        configured = set(_configured_provider_names(args.workspace))
        provider_owned = set(_provider_owned_names(args.agents_root))
    except (OSError, ValueError, yaml.YAMLError):
        print("capability bundle gate could not resolve configured providers")
        return 1
    unknown = selected - configured
    if unknown:
        print(
            "capability bundle gate could not resolve "
            f"{len(unknown)} configured provider(s)"
        )
        return 1

    target_names = sorted(selected or configured)
    repos: list[Path] = []
    failed: dict[str, list[str]] = {
        name: ["provider-owned manifest is outside configured workspace membership"]
        for name in sorted(provider_owned - configured)
    }
    for name in target_names:
        repo = args.agents_root / name
        if repo.is_symlink() or not repo.is_dir():
            failed[name] = ["configured provider checkout is missing"]
        elif not (repo / "connector_manifest.yml").is_file():
            failed[name] = ["configured provider-owned manifest is missing"]
        else:
            repos.append(repo)

    for repo in repos:
        try:
            violations = check_one(
                repo,
                bundled_root=args.bundled_root,
                lock_path=args.lock_path,
                privacy=privacy,
            )
        except Exception as exc:  # noqa: BLE001 - type only, never sensitive details
            violations = [f"bundle check raised {type(exc).__name__}"]
        if violations:
            failed[repo.name] = violations

    if failed:
        print(f"capability bundle gate failed for {len(failed)} provider(s)")
        for provider, violations in sorted(failed.items()):
            print(f"- {provider}: {len(violations)} violation(s)")
        return 1
    print(f"capability bundle gate passed for {len(target_names)} provider(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
