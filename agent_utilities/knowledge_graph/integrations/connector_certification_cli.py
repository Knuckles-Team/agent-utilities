"""Installed CLI for governed connector lifecycle certification."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from ..ontology.ontology_integrity import ReleaseSigner
from .connector_certification import (
    CertificationBundle,
    CertificationLimits,
    CertificationPolicy,
    ReferenceCertificationDriver,
    RuntimeCommandCertificationDriver,
    certify_connector,
    load_live_profile,
    write_certification_record,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one governed connector certification and sign aggregate evidence."
    )
    parser.add_argument("--connector-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("offline-fixture", "external-live"),
        default="offline-fixture",
    )
    parser.add_argument(
        "--profile-ref",
        help="required externalized profile reference for external-live mode",
    )
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    limits = CertificationLimits(
        max_records=args.max_records,
        timeout_seconds=args.timeout_seconds,
    )
    bundle = CertificationBundle.load(args.connector_root)
    signer = ReleaseSigner.from_runtime()
    if args.mode == "external-live":
        if not args.profile_ref:
            raise SystemExit("external-live mode requires --profile-ref")
        profile = load_live_profile(args.profile_ref)
        driver = RuntimeCommandCertificationDriver(profile=profile, limits=limits)
        policy = CertificationPolicy(
            tenant=profile.tenant,
            retention=profile.retention,
        )
    else:
        if args.profile_ref:
            raise SystemExit("offline-fixture mode does not accept --profile-ref")
        driver = ReferenceCertificationDriver()
        policy = CertificationPolicy()
    record = asyncio.run(
        certify_connector(
            bundle,
            mode=args.mode,
            signer=signer,
            driver=driver,
            policy=policy,
            limits=limits,
        )
    )
    write_certification_record(args.output, record)
    print(
        "connector certification "
        f"status={record['status']} mode={record['mode']} "
        f"checks={len(record['checks'])}"
    )
    return 0 if record["status"] in {"certified", "offline-validated"} else 1
