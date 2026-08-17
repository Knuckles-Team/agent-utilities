#!/usr/bin/env python3
"""Controlled-release orchestrator for connector-manifest regeneration + signing.

GOC-84 (connector manifest / implementation parity) is hard-blocked by BUG-234: the
release signing key (``ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY``) is not reachable from
any interactive agent session, and regenerating/signing manifests without it would
either fail (correctly) or, worse, produce a manifest signed by some *other* key that
nothing trusts — a signed-but-stale (or signed-by-the-wrong-key) manifest is strictly
worse than an unsigned one, because it manufactures trust nothing reviewed.

This script is the "job" GOC-16 was asked to build: the mechanism an operator runs,
inside a controlled release job that holds the real key via the existing OpenBao
``ClusterSecretStore`` custody path (see ``docs/release/connector-manifest-signing-
custody.md``), never inside an interactive agent session. It performs the exact
sequence named for BUG-234's resolution, and refuses to skip a step:

    1. freeze      -- verify the working tree is exactly the reviewed commit + lock
    2. regenerate   -- run the repo's OWN generators (never reimplemented here)
    3. sign         -- (only with --sign) generators resolve the release key
                       themselves via ``ontology_integrity.release_signer_for_publication``
                       -- this script never touches key material directly
    4. verify       -- re-run the compile-before-sync gate against what was written,
                       and (with --require-built-artifact) refuse to certify unless
                       running against an installed, non-editable build

Every mode short of ``--sign`` needs no key and is safe to run anywhere, including
this session, to produce a reviewable diff. ``--sign`` requires
``ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF`` to be a versioned ``vault://``/``secret://``
reference (``release_signer_for_publication`` refuses anything else) and is intended to
run ONLY inside the k8s Job in ``deploy/release/connector-manifest-signing-job.yaml``.

Exit codes: 0 = clean regeneration (and, with --sign, a verified signed result);
1 = a freeze/regeneration/verification violation (see stdout for the bounded reason);
2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NATIVE_OUTPUT_DIR = (
    ROOT
    / "agent_utilities"
    / "knowledge_graph"
    / "ontology"
    / "connector_manifests"
    / "native-source-connectors"
)


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *argv], cwd=ROOT, capture_output=True, text=True, check=False
    )


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def verify_freeze(*, frozen_sha: str | None, expected_lock_digest: str | None) -> list[str]:
    """Step 1 — the working tree must be EXACTLY the reviewed commit and lock.

    No manifest may ever be generated against a moving target (GOC-84's own explicit
    hard rule) — a manifest generated against unfrozen code goes stale the moment
    anything else lands.
    """
    from agent_utilities.knowledge_graph.ontology import ontology_integrity

    problems: list[str] = []
    dirty = _git("status", "--porcelain")
    if dirty:
        problems.append(
            "working tree is not clean (uncommitted changes) — freeze requires an "
            "exact, committed commit"
        )
    head = _git("rev-parse", "HEAD")
    if frozen_sha and head != frozen_sha:
        problems.append(f"HEAD {head} != expected frozen commit {frozen_sha}")
    if expected_lock_digest:
        try:
            live_digest = ontology_integrity.dependency_lock_digest()
        except ontology_integrity.ReleaseSigningError as exc:
            problems.append(f"dependency lock could not be read: {exc}")
        else:
            if live_digest != expected_lock_digest:
                problems.append(
                    f"live uv.lock digest {live_digest} != expected frozen digest "
                    f"{expected_lock_digest} — the dependency lock moved since the "
                    "commit this run was told to freeze against"
                )
    return problems


def verify_built_artifact() -> list[str]:
    """Refuse to certify against a source-tree-only (editable) install.

    GOC-84 names "built-wheel vs source-tree mismatch" as an adversarial case a
    source-tree-only check cannot see: a stale build cache or a partial rebuild can
    leave the *shipped* artifact diverged from a source tree that looks clean. This
    checks the installed ``agent-utilities`` distribution's own metadata for an
    editable/direct-reference install marker and refuses to proceed if found.
    """
    import importlib.metadata as metadata

    try:
        dist = metadata.distribution("agent-utilities")
    except metadata.PackageNotFoundError:
        return [
            "agent-utilities is not installed as a distribution in this interpreter "
            "-- built-artifact verification requires running against an installed "
            "wheel, not a bare source checkout"
        ]
    try:
        direct_url_text = dist.read_text("direct_url.json")
    except Exception:  # noqa: BLE001 - absence is the expected non-editable case
        direct_url_text = None
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except ValueError:
            direct_url = {}
        if isinstance(direct_url, dict) and direct_url.get("dir_info", {}).get(
            "editable"
        ):
            return [
                "agent-utilities is installed EDITABLE (pip install -e / uv pip "
                "install -e) -- this is a source-tree install, not a built artifact; "
                "install the built wheel into a clean interpreter to certify"
            ]
    return []


def regenerate(*, agents_root: Path | None, dry_run: bool, sign: bool) -> dict:
    """Step 2/3 — run the repo's OWN generators; this script never reimplements them."""

    report: dict[str, object] = {"dry_run": dry_run, "signed": sign, "results": []}

    native_argv = ["scripts/generate_native_connector_manifest.py"]
    if dry_run:
        native_argv += ["--output-dir", str(Path("/tmp") / "goc16-native-dry-run")]  # noqa: S108
    proc = _run(native_argv)
    report["results"].append(
        {
            "generator": "generate_native_connector_manifest.py",
            "ok": proc.returncode == 0,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    )

    if agents_root is not None:
        fleet_argv = [
            "scripts/generate_connector_manifests.py",
            "--all",
            "--agents-root",
            str(agents_root),
        ]
        if dry_run:
            fleet_argv.append("--dry-run")
        proc = _run(fleet_argv)
        report["results"].append(
            {
                "generator": "generate_connector_manifests.py",
                "ok": proc.returncode == 0,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        )

    report["ok"] = all(r["ok"] for r in report["results"])  # type: ignore[index]
    return report


def verify(*, agents_root: Path | None) -> dict:
    """Step 4 — re-run the compile-before-sync gate against what generation wrote."""

    argv = ["scripts/check_connector_manifests.py"]
    if agents_root is not None:
        argv += ["--agents-root", str(agents_root)]
    proc = _run(argv)
    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--frozen-sha",
        help="the exact reviewed commit SHA this run must be checked out at",
    )
    ap.add_argument(
        "--expected-lock-digest",
        help="the exact uv.lock digest (ontology_integrity.dependency_lock_digest) "
        "this run must match",
    )
    ap.add_argument("--agents-root", type=Path, help="the agents/ fleet root")
    ap.add_argument(
        "--sign",
        action="store_true",
        help="write real signed manifests (requires a vault://-backed "
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF; run only inside the controlled "
        "release job, never interactively)",
    )
    ap.add_argument(
        "--require-built-artifact",
        action="store_true",
        help="refuse to proceed unless running against a non-editable, installed "
        "build (see docs/release/connector-manifest-signing-custody.md)",
    )
    args = ap.parse_args()

    report: dict[str, object] = {}

    freeze_problems = verify_freeze(
        frozen_sha=args.frozen_sha, expected_lock_digest=args.expected_lock_digest
    )
    report["freeze"] = {"ok": not freeze_problems, "problems": freeze_problems}
    if freeze_problems:
        print(json.dumps(report, indent=2))
        return 1

    if args.require_built_artifact:
        artifact_problems = verify_built_artifact()
        report["built_artifact"] = {
            "ok": not artifact_problems,
            "problems": artifact_problems,
        }
        if artifact_problems:
            print(json.dumps(report, indent=2))
            return 1

    regen = regenerate(
        agents_root=args.agents_root, dry_run=not args.sign, sign=args.sign
    )
    report["regenerate"] = regen
    if not regen["ok"]:
        print(json.dumps(report, indent=2))
        return 1

    if args.sign:
        verification = verify(agents_root=args.agents_root)
        report["verify"] = verification
        if not verification["ok"]:
            print(json.dumps(report, indent=2))
            return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
