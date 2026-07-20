#!/usr/bin/env python3
"""Fail-closed release gate for signed connector lifecycle certifications."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.integrations.connector_certification import (  # noqa: E402
    REQUIRED_CHECKS,
    CertificationBundle,
    load_certification_record,
    safe_record_name,
    verify_certification_record,
)


def _source_contract() -> int:
    expected = {
        "bundle_integrity",
        "live_tool_schema",
        "fixture_ingest",
        "update",
        "delete",
        "replay_idempotency",
        "governance_preservation",
        "semantic_validation",
        "count_reconciliation",
        "cleanup",
    }
    if set(REQUIRED_CHECKS) != expected:
        print("connector live-certification source contract failed")
        return 1
    print("connector live-certification source contract passed")
    return 0


def _check_one(
    bundle: CertificationBundle,
    records_root: Path,
    *,
    require_live: bool,
) -> list[str]:
    violations: list[str] = []
    record_path = records_root / safe_record_name(bundle.manifest.connector)
    if not record_path.is_file():
        return ["signed certification record is absent"]
    record = load_certification_record(record_path)
    public_key = str(bundle.manifest.provenance.signing_public_key or "")
    violations.extend(
        verify_certification_record(
            record,
            trusted_public_keys=(public_key,) if public_key else (),
            require_live=require_live,
        )
    )
    if record.get("connector") != bundle.manifest.connector:
        violations.append("certification connector identity differs")
    expected = {
        "manifest_sha256": bundle.manifest_sha256,
        "fixtures_sha256": bundle.fixtures_sha256,
        "shapes_sha256": bundle.shapes_sha256,
        "schema_version": bundle.manifest.schema_version,
    }
    if record.get("bundle") != expected:
        violations.append("certification does not bind the current capability bundle")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents-root", type=Path)
    parser.add_argument("--records-root", type=Path)
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        if args.agents_root or args.records_root:
            parser.error("--self-check cannot be combined with fleet paths")
        return _source_contract()
    if args.agents_root is None or args.records_root is None:
        parser.error("fleet validation requires --agents-root and --records-root")

    connectors: list[CertificationBundle] = []
    failures: dict[str, list[str]] = {}
    for path in sorted(args.agents_root.iterdir()):
        manifest_path = path / "connector_manifest.yml"
        if path.is_symlink() or not path.is_dir() or not manifest_path.is_file():
            continue
        try:
            bundle = CertificationBundle.load(path)
        except Exception as exc:
            failures[path.name] = [f"bundle verification raised {type(exc).__name__}"]
            continue
        if bundle.manifest.sync:
            connectors.append(bundle)
    for bundle in connectors:
        try:
            violations = _check_one(
                bundle,
                args.records_root,
                require_live=args.require_live,
            )
        except Exception as exc:  # type only; never print path/record/source content
            violations = [f"certification gate raised {type(exc).__name__}"]
        if violations:
            failures[bundle.manifest.connector] = violations
    if failures:
        print(f"connector live-certification gate failed for {len(failures)} provider(s)")
        for connector, violations in sorted(failures.items()):
            print(f"- {connector}: {len(violations)} violation(s)")
        return 1
    print(f"connector live-certification gate passed for {len(connectors)} provider(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
