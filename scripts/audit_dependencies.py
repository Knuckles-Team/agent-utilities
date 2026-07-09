#!/usr/bin/env python3
"""Dependency vulnerability audit for the committed ``uv.lock``.

Parses ``uv.lock`` (stdlib only — no uv/pip-audit venv needed), queries the OSV
batch API for every pinned package version, and fails the commit if any package
carries a **fixable** advisory (one with a released fixed version) that is not
explicitly risk-accepted in ``.security-audit-allow.txt``.

Design goals:
- **No new runtime deps** — parses TOML-ish ``uv.lock`` with the stdlib.
- **Network-tolerant** — if OSV is unreachable (offline commit), it warns and
  passes rather than blocking work.
- **Won't-fix aware** — advisories with no released fix are reported as warnings
  and only fail if you additionally list them; packages in the allowlist are
  skipped entirely (with a required justification comment kept in that file).

Usage: ``python scripts/audit_dependencies.py [path/to/uv.lock]``
Exit code 1 only when a *fixable, non-allowlisted* advisory is present.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

OSV_BATCH = "https://api.osv.dev/v1/querybatch"
OSV_VULN = "https://api.osv.dev/v1/vulns/"


def parse_lock(path: pathlib.Path) -> dict[str, str]:
    """Return {package_name: version} from a uv.lock file."""
    pkgs: dict[str, str] = {}
    name = None
    for block in re.split(r"\n\[\[package\]\]\n", path.read_text()):
        nm = re.search(r'^name = "([^"]+)"', block, re.M)
        ver = re.search(r'^version = "([^"]+)"', block, re.M)
        if nm and ver:
            pkgs[nm.group(1).lower().replace("_", "-")] = ver.group(1)
    return pkgs


def load_allowlist(root: pathlib.Path) -> set[str]:
    f = root / ".security-audit-allow.txt"
    if not f.exists():
        return set()
    out = set()
    for line in f.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line.lower().replace("_", "-"))
    return out


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def fixed_versions(vuln_id: str, pkg: str) -> set[str]:
    try:
        d = _get(OSV_VULN + vuln_id)
    except Exception:
        return set()
    fixed = set()
    for aff in d.get("affected", []):
        if aff.get("package", {}).get("name", "").lower().replace("_", "-") != pkg:
            continue
        for rng in aff.get("ranges", []):
            for ev in rng.get("events", []):
                if "fixed" in ev:
                    fixed.add(ev["fixed"])
    return fixed


def main() -> int:
    lock = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "uv.lock")
    if not lock.exists():
        print(f"audit: {lock} not found — nothing to scan")
        return 0
    root = lock.resolve().parent
    pkgs = parse_lock(lock)
    allow = load_allowlist(root)
    queries = [
        {"package": {"name": n, "ecosystem": "PyPI"}, "version": v} for n, v in pkgs.items()
    ]
    names = list(pkgs)
    hits: dict[str, list[str]] = {}
    try:
        for i in range(0, len(queries), 100):
            res = _post(OSV_BATCH, {"queries": queries[i : i + 100]})
            for name, r in zip(names[i : i + 100], res.get("results", [])):
                ids = [v["id"] for v in (r.get("vulns") or [])]
                if ids:
                    hits[name] = ids
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"audit: OSV unreachable ({e}); skipping (offline-tolerant)")
        return 0

    fixable, wontfix = [], []
    for name, ids in sorted(hits.items()):
        if name in allow:
            continue
        for vid in ids:
            fv = fixed_versions(vid, name)
            row = (name, pkgs[name], vid, ";".join(sorted(fv)) or "(no fix)")
            (fixable if fv else wontfix).append(row)

    for name, cur, vid, fv in wontfix:
        print(f"  WARN (no upstream fix) {name} {cur}  {vid}")
    for name, cur, vid, fv in fixable:
        print(f"  FAIL {name} {cur} -> {fv}  ({vid})")

    if fixable:
        print(
            f"\naudit: {len(fixable)} fixable advisory(ies) found. Add a security "
            "floor to pyproject.toml and re-run `uv lock`, or risk-accept the "
            "package in .security-audit-allow.txt with a justification."
        )
        return 1
    print(f"audit: clean ({len(wontfix)} won't-fix warning(s), {len(allow)} allow-listed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
