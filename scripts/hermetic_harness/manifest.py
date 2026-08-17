"""GOC-38 run manifest: build, digest, and validate.

CONCEPT:AU-GOC.harness.manifest

The manifest is frozen by ``schemas/manifest.schema.json`` (see that file's
``description`` for the freeze statement). This module never invents optional
fields -- ``jsonschema.validate`` with ``additionalProperties: false`` is the
enforcement point, not a docstring promise.

Interpreter identity is decided from the venv's ``pyvenv.cfg`` content, never
from ``os.path.realpath()`` on the interpreter path. ``realpath()`` on a uv
``.venv/bin/python3`` always resolves *out* of the venv (it is usually a
symlink to a base interpreter), so a guard that inferred identity that way
previously ``os.execve()``'d out of the venv mid-run and killed a whole pytest
session silently (exit 75). See ``venv_identity_digest`` below.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import jsonschema

from . import SCHEMA_DIR

MANIFEST_SCHEMA = json.loads((SCHEMA_DIR / "manifest.schema.json").read_text())

# Env vars a child test process is allowed to inherit. Everything else is
# stripped before exec. UV_PROJECT_ENVIRONMENT is deliberately NOT here:
# an inherited copy previously let a test rebuild the lane's real venv as
# its own fixture (349 packages -> 1), voiding every verdict from that run.
DEFAULT_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TERM",
    "SHELL",
    "TZ",
    "PYTHONHASHSEED",
    "PYTHONDONTWRITEBYTECODE",
)

DEFAULT_TEMP_ROOT = "/var/tmp/l9/hermetic-harness"


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def venv_identity_digest(venv_path: Path) -> str | None:
    """Digest the venv's own ``pyvenv.cfg`` -- the identity check.

    Deliberately does NOT call ``realpath()`` anywhere in this function or on
    its result; a caller who wants to compare "is this the venv I asked for"
    must compare THIS digest, not a resolved path.
    """
    cfg = venv_path / "pyvenv.cfg"
    if not cfg.is_file():
        return None
    return sha256_bytes(cfg.read_bytes())


def installed_package_count(venv_path: Path) -> int:
    """Best-effort count of installed distributions via importlib.metadata
    against the venv's site-packages, without invoking the interpreter
    (invoking it would itself depend on the identity we are trying to
    verify)."""
    candidates = list(venv_path.glob("lib/python*/site-packages")) + list(
        venv_path.glob("Lib/site-packages")
    )
    if not candidates:
        return 0
    site_packages = candidates[0]
    count = 0
    for entry in site_packages.iterdir():
        if entry.name.endswith((".dist-info", ".egg-info")):
            count += 1
    return count


@dataclasses.dataclass(frozen=True)
class Manifest:
    manifest_version: str
    candidate_sha: str
    repo: str
    test_selection: dict[str, list[str]]
    lock_digest: str
    interpreter_digest: str
    expected_venv_path: str
    env_allowlist: list[str]
    resource_limits: dict[str, int]
    timeout_seconds: float
    grace_seconds: float
    lease_scope: list[str]
    redaction_policy: str
    temp_root: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def digest(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256_bytes(canonical.encode("utf-8"))


def validate_manifest(data: dict[str, Any]) -> None:
    """Raises jsonschema.ValidationError on any missing/malformed field."""
    jsonschema.validate(instance=data, schema=MANIFEST_SCHEMA)


def git_sha(repo_path: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def build_manifest(
    *,
    repo: str,
    repo_path: Path,
    test_paths: list[str],
    lockfile_path: Path,
    venv_path: Path,
    timeout_seconds: float,
    grace_seconds: float = 15.0,
    markers: list[str] | None = None,
    deselect: list[str] | None = None,
    env_allowlist: list[str] | None = None,
    resource_limits: dict[str, int] | None = None,
    lease_scope: list[str] | None = None,
    redaction_policy: str = "default-v1",
    temp_root: str = DEFAULT_TEMP_ROOT,
) -> Manifest:
    interpreter_digest = venv_identity_digest(venv_path)
    if interpreter_digest is None:
        # Fail loud rather than silently substituting sys.executable's own
        # digest -- that is exactly the "looked like the right interpreter"
        # failure mode this manifest exists to prevent.
        raise FileNotFoundError(
            f"no pyvenv.cfg under {venv_path} -- refusing to synthesize an "
            "interpreter identity for the manifest"
        )
    manifest = Manifest(
        manifest_version="1.0.0",
        candidate_sha=git_sha(repo_path),
        repo=repo,
        test_selection={
            "paths": test_paths,
            "markers": markers or [],
            "deselect": deselect or [],
        },
        lock_digest=sha256_file(lockfile_path),
        interpreter_digest=interpreter_digest,
        expected_venv_path=str(venv_path),
        env_allowlist=list(env_allowlist or DEFAULT_ENV_ALLOWLIST),
        resource_limits=resource_limits
        or {"rss_mb": 4096, "fd": 1024, "processes": 64, "disk_mb": 2048},
        timeout_seconds=timeout_seconds,
        grace_seconds=grace_seconds,
        lease_scope=list(lease_scope or []),
        redaction_policy=redaction_policy,
        temp_root=temp_root,
    )
    validate_manifest(manifest.to_dict())
    return manifest


def filtered_env(allowlist: list[str], source_env: dict[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    """Returns (child_env, rejected_vars). ``source_env`` defaults to the
    current process environment; passed explicitly so callers/tests can
    simulate an inherited UV_PROJECT_ENVIRONMENT without mutating os.environ.
    """
    source = source_env if source_env is not None else dict(os.environ)
    allow = set(allowlist)
    child = {k: v for k, v in source.items() if k in allow}
    rejected = sorted(k for k in source if k not in allow)
    return child, rejected


def current_interpreter_identity() -> dict[str, Any]:
    """Never uses realpath(). Reports sys.executable/sys.prefix verbatim."""
    return {
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "interpreter_version": ".".join(map(str, sys.version_info[:3])),
    }
