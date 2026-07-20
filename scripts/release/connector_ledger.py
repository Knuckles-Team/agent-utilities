#!/usr/bin/env python3
"""Assemble, externally sign, and verify the live connector release ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.integrations.connector_certification import (  # noqa: E402
    CertificationBundle,
    load_certification_record,
    safe_record_name,
    verify_certification_record,
)
from scripts.release import check_compatibility as compatibility  # noqa: E402
from scripts.release.assemble_manifest import (  # noqa: E402
    AssemblyError,
    _exact,
    _external_command,
    _mapping,
    _write,
)

_SIGNATURE_SCHEME = re.compile(r"^[a-z0-9][a-z0-9+._-]{1,63}$")
_SIGNATURE_VALUE = re.compile(r"^[A-Za-z0-9+/_=-]{16,16384}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _bundles(agents_root: Path) -> tuple[CertificationBundle, ...]:
    if agents_root.is_symlink() or not agents_root.is_dir():
        raise AssemblyError("connector package root is unavailable")
    bundles: list[CertificationBundle] = []
    for path in sorted(agents_root.iterdir()):
        if path.is_symlink() or not path.is_dir():
            continue
        manifest_path = path / "connector_manifest.yml"
        if not manifest_path.is_file():
            continue
        bundle = CertificationBundle.load(path)
        if bundle.manifest.sync:
            bundles.append(bundle)
    if not bundles:
        raise AssemblyError("no certifiable connector bundles were discovered")
    return tuple(bundles)


def assemble_unsigned(
    bundles: Iterable[CertificationBundle], records_root: Path
) -> dict[str, Any]:
    """Bind every signed live record to its current connector capability bundle."""

    if records_root.is_symlink() or not records_root.is_dir():
        raise AssemblyError("connector certification records are unavailable")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bundle in sorted(bundles, key=lambda item: item.manifest.connector):
        connector = bundle.manifest.connector
        if connector in seen:
            raise AssemblyError("connector certification identity is duplicated")
        seen.add(connector)
        record = load_certification_record(records_root / safe_record_name(connector))
        public_key = str(bundle.manifest.provenance.signing_public_key or "")
        violations = verify_certification_record(
            record,
            trusted_public_keys=(public_key,) if public_key else (),
            require_live=True,
        )
        expected_bundle = {
            "manifest_sha256": bundle.manifest_sha256,
            "fixtures_sha256": bundle.fixtures_sha256,
            "shapes_sha256": bundle.shapes_sha256,
            "schema_version": bundle.manifest.schema_version,
        }
        if violations or record.get("connector") != connector:
            raise AssemblyError("connector live certification is invalid")
        if record.get("bundle") != expected_bundle:
            raise AssemblyError("connector certification is stale")
        entries.append(
            {
                "connector": connector,
                "certifiedAt": str(record["certified_at"]),
                "recordDigest": _digest(_canonical(record)),
                "bundleDigest": _digest(_canonical(expected_bundle)),
            }
        )
    return {
        "apiVersion": "graphos.io/v1",
        "kind": "ConnectorLiveCertificationLedger",
        "ledgerVersion": 1,
        "entryCount": len(entries),
        "entries": entries,
    }


def sign_ledger(
    unsigned: dict[str, Any], *, signer_env: str, verifier_env: str
) -> dict[str, Any]:
    if "signature" in unsigned:
        raise AssemblyError("connector ledger is already signed")
    _validate_unsigned(unsigned)
    subject_digest = compatibility.canonical_digest(unsigned)
    response = _external_command(signer_env, _canonical(unsigned))
    _exact(
        response,
        {"scheme", "subjectDigest", "bundleDigest", "signerIdentityDigest", "signature"},
        field="external signer response",
    )
    if response.get("subjectDigest") != subject_digest:
        raise AssemblyError("external signer did not bind the connector ledger")
    signed = {
        **unsigned,
        "signature": {
            "scheme": str(response["scheme"]),
            "subjectDigest": subject_digest,
            "bundleDigest": str(response["bundleDigest"]),
            "signerIdentityDigest": str(response["signerIdentityDigest"]),
            "value": str(response["signature"]),
            "verifierEnv": verifier_env,
        },
    }
    _validate_signed(signed)
    return signed


def _validate_unsigned(value: dict[str, Any]) -> None:
    _exact(
        value,
        {"apiVersion", "kind", "ledgerVersion", "entryCount", "entries"},
        field="connector ledger",
    )
    if (
        value.get("apiVersion") != "graphos.io/v1"
        or value.get("kind") != "ConnectorLiveCertificationLedger"
        or value.get("ledgerVersion") != 1
    ):
        raise AssemblyError("unsupported connector ledger apiVersion/kind")
    entries = value.get("entries")
    if (
        not isinstance(entries, list)
        or not entries
        or value.get("entryCount") != len(entries)
        or len(entries) != compatibility._CURRENT_CONNECTOR_ENTRIES
    ):
        raise AssemblyError("connector ledger entry count is invalid")
    connectors: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise AssemblyError("connector ledger entry must be a mapping")
        _exact(
            entry,
            {"connector", "certifiedAt", "recordDigest", "bundleDigest"},
            field="connector ledger entry",
        )
        connector = str(entry.get("connector") or "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,127}", connector):
            raise AssemblyError("connector ledger identity is invalid")
        connectors.append(connector)
        compatibility._digest(entry.get("recordDigest"), "recordDigest")
        compatibility._digest(entry.get("bundleDigest"), "bundleDigest")
        if not re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
            str(entry.get("certifiedAt") or ""),
        ):
            raise AssemblyError("connector certification timestamp is invalid")
    if connectors != sorted(set(connectors)):
        raise AssemblyError("connector ledger entries are not unique and ordered")


def _validate_signed(value: dict[str, Any]) -> None:
    if not isinstance(value.get("signature"), dict):
        raise AssemblyError("connector ledger signature is required")
    unsigned = {key: child for key, child in value.items() if key != "signature"}
    _validate_unsigned(unsigned)
    signature = value["signature"]
    _exact(
        signature,
        {
            "scheme",
            "subjectDigest",
            "bundleDigest",
            "signerIdentityDigest",
            "value",
            "verifierEnv",
        },
        field="connector ledger signature",
    )
    for field in ("subjectDigest", "bundleDigest", "signerIdentityDigest"):
        compatibility._digest(signature.get(field), f"connector ledger signature.{field}")
    if signature["subjectDigest"] != compatibility.canonical_digest(unsigned):
        raise AssemblyError("connector ledger signature does not bind the ledger")
    if not _SIGNATURE_SCHEME.fullmatch(str(signature.get("scheme") or "")):
        raise AssemblyError("connector ledger signature scheme is invalid")
    if not _SIGNATURE_VALUE.fullmatch(str(signature.get("value") or "")):
        raise AssemblyError("connector ledger signature value is invalid")
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", str(signature.get("verifierEnv") or "")):
        raise AssemblyError("connector ledger verifier environment name is invalid")


def verify_ledger(
    signed: dict[str, Any], expected_unsigned: dict[str, Any]
) -> None:
    _validate_signed(signed)
    unsigned = {key: value for key, value in signed.items() if key != "signature"}
    if unsigned != expected_unsigned:
        raise AssemblyError("connector ledger differs from current live records")
    verifier_env = signed["signature"]["verifierEnv"]
    raw = os.environ.get(verifier_env, "")
    if not raw:
        raise AssemblyError("connector ledger verifier command is absent")
    try:
        command = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssemblyError("connector ledger verifier must be a JSON argv array") from exc
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise AssemblyError("connector ledger verifier must be a JSON argv array")
    result = subprocess.run(
        command, input=_canonical(signed), capture_output=True, check=False, timeout=120
    )
    if result.returncode != 0:
        raise AssemblyError(
            "connector ledger verification failed; output_digest="
            + hashlib.sha256(result.stdout + result.stderr).hexdigest()
        )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssemblyError("connector ledger verifier returned non-JSON") from exc
    if (
        not isinstance(response, dict)
        or response.get("verified") is not True
        or response.get("subjectDigest") != signed["signature"]["subjectDigest"]
    ):
        raise AssemblyError("connector ledger verifier did not bind its subject")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graphos-connector-release-ledger")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--agents-root", type=Path, required=True)
    assemble_parser.add_argument("--records-root", type=Path, required=True)
    assemble_parser.add_argument("--output", type=Path, required=True)
    assemble_parser.add_argument("--signer-env", required=True)
    assemble_parser.add_argument("--verifier-env", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--agents-root", type=Path, required=True)
    verify_parser.add_argument("--records-root", type=Path, required=True)
    verify_parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        expected = assemble_unsigned(_bundles(args.agents_root), args.records_root)
        if args.operation == "assemble":
            _write(
                args.output,
                sign_ledger(
                    expected,
                    signer_env=args.signer_env,
                    verifier_env=args.verifier_env,
                ),
            )
        else:
            verify_ledger(_mapping(args.ledger), expected)
    except Exception as exc:  # noqa: BLE001 - privacy-safe release boundary
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps({"ok": True, "operation": args.operation}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
