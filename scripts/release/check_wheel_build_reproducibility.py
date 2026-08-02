#!/usr/bin/env python3
"""Replay pipeline.yml's `build-wheel` job locally, in full.

CONCEPT:AU-OS.governance.tiered-merge-gate — D-CIP-7.

**The vector this gate defends.** ``pipeline.yml``'s ``build-wheel`` job
builds the release wheel **twice** from the same source revision, requires
the two artifacts to be byte-for-byte identical (a non-deterministic build —
embedded timestamps, unordered zip members, a stray absolute path — would
otherwise ship a wheel that cannot be reproduced or audited), and runs
``scripts/release/check_release_wheel.py`` (exact release-surface contract)
and ``scripts/check_wheel_privacy.py`` (no local/machine-specific content)
against **both** candidates before selecting one to publish. Before this
gate, the local ``wheel-build`` pre-commit hook only ran a single
``uv build``, so a packaging regression in any of these four checks
(non-reproducibility, release-surface drift, or a privacy leak baked into
the wheel) was invisible until the release push.

**What it does.** Builds ``dist-primary`` and ``dist-reproduction`` with
``uv build --wheel --no-build-isolation`` (mirroring ``pipeline.yml``
exactly, including ``SOURCE_DATE_EPOCH=0`` for reproducibility), runs both
downstream checks against each wheel, and requires the two wheels' sha256
digests to match. Reports a structured result; exits non-zero on the first
failure category, in the same order the CI job would surface it.

Run it via::

    python3 scripts/release/check_wheel_build_reproducibility.py

``--self-check`` proves the digest-mismatch path is a hard failure (not
silently accepted) by comparing two byte-different files as a stand-in for
two wheels, without paying for a real double build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_RELEASE_WHEEL = REPO_ROOT / "scripts" / "release" / "check_release_wheel.py"
CHECK_WHEEL_PRIVACY = REPO_ROOT / "scripts" / "check_wheel_privacy.py"


def _run(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, check=False
    )
    ok = proc.returncode == 0
    tail = (proc.stderr or proc.stdout).strip()
    return ok, ("ok" if ok else "\n".join(tail.splitlines()[-15:]))


def _build_wheel(repo_root: Path, out_dir: Path) -> tuple[bool, str, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = "0"
    proc = subprocess.run(
        ["uv", "build", "--wheel", "--no-build-isolation", "--out-dir", str(out_dir)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip()
        return False, "\n".join(tail.splitlines()[-15:]), None
    candidates = sorted(out_dir.glob("agent_utilities-*.whl"))
    if len(candidates) != 1:
        return False, f"expected exactly 1 wheel in {out_dir}, found {len(candidates)}", None
    return True, "ok", candidates[0]


def check(repo_root: Path = REPO_ROOT) -> tuple[int, dict]:
    with tempfile.TemporaryDirectory(prefix="wheel-repro-primary-") as primary_dir, \
         tempfile.TemporaryDirectory(prefix="wheel-repro-reproduction-") as repro_dir:
        result: dict = {"ok": False, "steps": {}}

        ok, detail, primary_wheel = _build_wheel(repo_root, Path(primary_dir))
        result["steps"]["buildPrimary"] = {"ok": ok, "detail": detail}
        if not ok or primary_wheel is None:
            return 1, result

        ok, detail, repro_wheel = _build_wheel(repo_root, Path(repro_dir))
        result["steps"]["buildReproduction"] = {"ok": ok, "detail": detail}
        if not ok or repro_wheel is None:
            return 1, result

        for label, wheel in (("primary", primary_wheel), ("reproduction", repro_wheel)):
            ok, detail = _run(
                ["python3", str(CHECK_RELEASE_WHEEL), str(wheel)], repo_root
            )
            result["steps"][f"releaseContract_{label}"] = {"ok": ok, "detail": detail}
            if not ok:
                return 1, result

            ok, detail = _run(
                ["python3", str(CHECK_WHEEL_PRIVACY), str(wheel)], repo_root
            )
            result["steps"][f"privacy_{label}"] = {"ok": ok, "detail": detail}
            if not ok:
                return 1, result

        digest_ok, digest = digests_match(primary_wheel, repro_wheel)
        result["steps"]["byteIdenticalDigests"] = {
            "ok": digest_ok,
            "detail": digest if digest_ok else f"digest mismatch: {digest}",
        }
        if not digest_ok:
            return 1, result

        result["ok"] = True
        return 0, result


def digests_match(a: Path, b: Path) -> tuple[bool, str]:
    if a.name != b.name:
        return False, f"filename mismatch: {a.name} != {b.name}"
    digest_a = hashlib.sha256(a.read_bytes()).hexdigest()
    digest_b = hashlib.sha256(b.read_bytes()).hexdigest()
    if digest_a != digest_b:
        return False, f"{digest_a} != {digest_b}"
    return True, digest_a


def _self_check() -> tuple[int, dict]:
    """Prove a digest mismatch is a hard failure, without a real double build."""
    tmp = Path(tempfile.mkdtemp(prefix="wheel-repro-selfcheck-"))
    try:
        same_a = tmp / "agent_utilities-1.0.0-py3-none-any.whl"
        same_b_dir = tmp / "b"
        same_b_dir.mkdir()
        same_b = same_b_dir / "agent_utilities-1.0.0-py3-none-any.whl"
        same_a.write_bytes(b"identical content")
        same_b.write_bytes(b"identical content")
        matched_ok, _ = digests_match(same_a, same_b)

        diff_a = tmp / "c" / "agent_utilities-1.0.0-py3-none-any.whl"
        diff_a.parent.mkdir()
        diff_a.write_bytes(b"identical content")
        diff_b = tmp / "d" / "agent_utilities-1.0.0-py3-none-any.whl"
        diff_b.parent.mkdir()
        diff_b.write_bytes(b"embedded timestamp makes this different")
        mismatched_ok, mismatch_detail = digests_match(diff_a, diff_b)
        caught_mismatch = not mismatched_ok

        ok = matched_ok and caught_mismatch
        return (0 if ok else 1), {
            "ok": ok,
            "selfCheck": True,
            "acceptsByteIdenticalPair": matched_ok,
            "catchesDigestMismatch": caught_mismatch,
            "mismatchDetail": mismatch_detail,
        }
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-check", action="store_true", help="prove the digest-mismatch path fails closed"
    )
    args = parser.parse_args()

    if args.self_check:
        rc, result = _self_check()
    else:
        rc, result = check()

    print(json.dumps(result, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
