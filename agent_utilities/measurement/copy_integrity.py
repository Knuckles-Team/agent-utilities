"""Copy integrity for measurement (measurement harness, capability C).

CONCEPT:AU-OS.measurement.copy-integrity

Direct response to the incident where ``rsync -a`` (no ``-H``) copied a
package to a test host and silently dropped one hardlinked source file
(``distributed_state_manager.py``, 2 links) — 76 of 77 files copied, and the
test suite then reported 14,159 errors, read as a catastrophic regression
against a repo that was actually fine. The copy was never verified; the
error count was.

This module provides (1) :func:`copy_tree` — a copy helper that defaults to
``rsync -aH --delete`` (the ``-H`` incident-3 was missing) with a
``git archive`` alternative for git-tracked trees, and (2)
:func:`verify_copy` — a manifest-hash comparison that proves the copy is a
faithful reproduction of the source *before* anything is measured against
it. ``copy_tree`` calls ``verify_copy`` itself and raises
:class:`CopyIntegrityError` on any mismatch, so a caller cannot use a
silently-incomplete copy by simply forgetting to check.
"""

from __future__ import annotations

import dataclasses
import hashlib
import shutil
import subprocess
from pathlib import Path


class CopyIntegrityError(Exception):
    """Raised when a copy does not manifest-match its source."""


@dataclasses.dataclass(frozen=True)
class CopyIntegrityResult:
    ok: bool
    source_file_count: int
    dest_file_count: int
    missing_in_dest: tuple[
        str, ...
    ]  # present in source, absent (or hash-mismatched-absent) in dest
    extra_in_dest: tuple[str, ...]  # present in dest, absent in source
    mismatched: tuple[str, ...]  # present in both, different content hash

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise CopyIntegrityError(
                "copy integrity check FAILED: "
                f"source has {self.source_file_count} files, dest has {self.dest_file_count}; "
                f"missing_in_dest={list(self.missing_in_dest)} "
                f"extra_in_dest={list(self.extra_in_dest)} "
                f"mismatched={list(self.mismatched)}"
            )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_for(root: Path, *, git_tracked_only: bool = False) -> dict[str, str]:
    """Build a ``{relative_path: sha256}`` manifest for every regular file under ``root``.

    With ``git_tracked_only=True``, the file list comes from
    ``git ls-files`` (scoped to files git actually tracks) instead of a
    filesystem walk — useful when comparing a working tree against a
    ``git archive`` extraction that intentionally excludes ignored files.
    """
    root = Path(root)
    if git_tracked_only:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=True,
        )
        rel_paths = [p for p in out.stdout.split(b"\0") if p]
        return {
            p.decode(): _sha256_file(root / p.decode())
            for p in rel_paths
            if (root / p.decode()).is_file()
        }

    manifest: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            manifest[str(path.relative_to(root))] = _sha256_file(path)
    return manifest


def verify_copy(
    source: Path, dest: Path, *, git_tracked_only: bool = False
) -> CopyIntegrityResult:
    """Prove (or disprove) that ``dest`` is a faithful copy of ``source``.

    Compares a full manifest (relative path -> content hash), not just a
    file count — a count-only check would miss a copy that dropped one file
    and duplicated another. This is the check that must fail on the exact
    incident-3 shape: a copy silently missing one file out of many.
    """
    src_manifest = manifest_for(source, git_tracked_only=git_tracked_only)
    dst_manifest = manifest_for(dest, git_tracked_only=git_tracked_only)

    missing = tuple(sorted(p for p in src_manifest if p not in dst_manifest))
    extra = tuple(sorted(p for p in dst_manifest if p not in src_manifest))
    mismatched = tuple(
        sorted(
            p
            for p in src_manifest
            if p in dst_manifest and src_manifest[p] != dst_manifest[p]
        )
    )
    ok = not missing and not extra and not mismatched
    return CopyIntegrityResult(
        ok=ok,
        source_file_count=len(src_manifest),
        dest_file_count=len(dst_manifest),
        missing_in_dest=missing,
        extra_in_dest=extra,
        mismatched=mismatched,
    )


def copy_tree(
    source: Path, dest: Path, *, method: str = "rsync", delete: bool = True
) -> CopyIntegrityResult:
    """Copy ``source`` to ``dest`` for measurement, then PROVE the copy is complete.

    ``method="rsync"`` (default) runs ``rsync -aH [--delete]`` — the ``-H``
    (preserve hard links) is the flag incident 3 was missing; without it,
    rsync can silently fail to reproduce every hardlinked file it should.
    ``method="git_archive"`` instead extracts ``git archive HEAD`` into
    ``dest``, appropriate when the source is a clean git work tree and only
    tracked content should be measured.

    Raises :class:`CopyIntegrityError` if the resulting copy does not
    manifest-match the source — the caller cannot forget to check, because
    the check happens here rather than being left as a separate optional
    step.
    """
    source = Path(source)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if method == "rsync":
        rsync = shutil.which("rsync")
        if rsync is None:
            raise RuntimeError(
                "rsync not found on PATH; cannot copy_tree(method='rsync')"
            )
        cmd = [rsync, "-aH"]
        if delete:
            cmd.append("--delete")
        cmd += [f"{source}/", f"{dest}/"]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        result = verify_copy(source, dest)
    elif method == "git_archive":
        proc = subprocess.run(
            ["git", "-C", str(source), "archive", "HEAD"],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["tar", "-x", "-C", str(dest)],
            input=proc.stdout,
            check=True,
        )
        result = verify_copy(source, dest, git_tracked_only=True)
    else:
        raise ValueError(
            f"unknown method {method!r}; expected 'rsync' or 'git_archive'"
        )

    result.raise_if_bad()
    return result
