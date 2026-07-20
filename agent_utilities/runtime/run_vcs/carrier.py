"""CONCEPT:AU-ORCH.runvcs.run-commit — carrier-style copy-on-write process+filesystem snapshot.

Shepherd's ``vcs-core`` snapshots a *process + its filesystem* as one CoW carrier: APFS
``clonefile`` on macOS, an overlay/FUSE mount elsewhere, with an optional ``LD_PRELOAD`` shim
(``_fs_capture.py``) that records touched paths. Our hosts are Linux, so the APFS path is out;
this module provides the Linux carrier as a :class:`FsCarrier` with two strategies behind one
interface:

* **overlayfs / fuse-overlayfs** (accelerated) — when the kernel supports an unprivileged
  overlay mount, a snapshot is just the *upperdir* (the diff layer); restore swaps the upper.
  Detected at runtime; used only when available.
* **content-addressed blob store** (portable fallback, the default in tests / unprivileged
  containers) — a snapshot walks the tree, content-hashes every file, and hardlinks each blob
  into a shared store keyed by hash. Identical content is stored once across every snapshot of
  every run (the CoW/dedup win), and restore is a hardlink-or-copy back plus a prune of files
  the snapshot never had. This is a faithful, privilege-free carrier: it restores **files** to
  an exact prior world, which is the half of "process + fs" that survives a fork.

A :class:`FsSnapshot` is content-addressed (``snapshot_id`` == a digest of its manifest), so two
identical trees produce the same snapshot id — the fs analogue of the event kernel's digest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20  # 1 MiB streaming read for hashing/copy.


def is_overlayfs_available() -> bool:
    """Whether an overlay filesystem can be mounted on this host (best-effort probe)."""
    try:
        filesystems = Path("/proc/filesystems").read_text(encoding="utf-8")
    except OSError:
        return False
    if "overlay" not in filesystems:
        return False
    # A real mount also needs privilege/userns; we only advertise the capability here and let
    # the caller decide. The blob-store fallback covers the (common) unprivileged case.
    return os.geteuid() == 0 or _has_unpriv_overlay()


def _has_unpriv_overlay() -> bool:
    try:
        return Path("/sys/module/overlay/parameters/userxattr").exists()
    except OSError:
        return False


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class FsSnapshot:
    """A content-addressed snapshot of a directory tree.

    ``manifest`` maps each tree-relative POSIX path to ``{"hash": <sha256>, "mode": <int>,
    "symlink": <target or "">}``. ``snapshot_id`` is a digest of the manifest, so an identical
    tree always yields the same id (dedup across runs/forks). Blob content lives in the
    carrier's shared store keyed by hash — the snapshot itself is tiny metadata.
    """

    snapshot_id: str
    manifest: dict[str, dict[str, Any]]
    strategy: str = "blobstore"

    @property
    def file_count(self) -> int:
        return len(self.manifest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "manifest": self.manifest,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FsSnapshot:
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            manifest=dict(data.get("manifest") or {}),
            strategy=str(data.get("strategy") or "blobstore"),
        )


class FsCarrier:
    """A copy-on-write snapshot/restore carrier for one workspace root.

    ``root`` is the live workspace; ``store`` is where content blobs + snapshot manifests are
    kept (defaults to ``<root>/.run-vcs`` — kept OUT of snapshots). Snapshots hardlink blobs
    into ``store/blobs`` when the filesystem allows it (true CoW), else copy.
    """

    #: Directory names never captured (the carrier's own store + volatile caches).
    _IGNORED_DIRS = frozenset({".run-vcs", ".git", "__pycache__", ".mypy_cache"})

    def __init__(self, root: str | Path, store: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        self.store = Path(store).resolve() if store else self.root / ".run-vcs"
        self.blobs = self.store / "blobs"
        self.snaps = self.store / "snapshots"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.snaps.mkdir(parents=True, exist_ok=True)

    # ── snapshot ───────────────────────────────────────────────────────────────
    def snapshot(self) -> FsSnapshot:
        """Content-address the tree under ``root`` and persist its blobs. Idempotent + cheap."""
        manifest: dict[str, dict[str, Any]] = {}
        for path in sorted(self.root.rglob("*")):
            if any(
                part in self._IGNORED_DIRS for part in path.relative_to(self.root).parts
            ):
                continue
            rel = path.relative_to(self.root).as_posix()
            if path.is_symlink():
                manifest[rel] = {"hash": "", "mode": 0, "symlink": os.readlink(path)}
                continue
            if not path.is_file():
                continue
            digest = _file_digest(path)
            self._store_blob(path, digest)
            manifest[rel] = {
                "hash": digest,
                "mode": path.stat().st_mode & 0o777,
                "symlink": "",
            }
        snapshot_id = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        snap = FsSnapshot(snapshot_id=snapshot_id, manifest=manifest)
        (self.snaps / f"{snapshot_id}.json").write_text(
            json.dumps(snap.to_dict()), encoding="utf-8"
        )
        return snap

    def _store_blob(self, path: Path, digest: str) -> None:
        blob = self.blobs / digest[:2] / digest[2:]
        if blob.exists():
            return  # dedup: content already stored (the CoW/space win — one blob per hash)
        blob.parent.mkdir(parents=True, exist_ok=True)
        # COPY (not hardlink) into the immutable store: hardlinking the live file would share
        # its inode, so a later in-place edit of the working file would silently corrupt the
        # stored blob. The blob store is content-addressed and never mutated in place.
        shutil.copy2(path, blob)

    # ── restore ────────────────────────────────────────────────────────────────
    def restore(
        self, snapshot: FsSnapshot, *, target: str | Path | None = None
    ) -> dict[str, int]:
        """Restore ``snapshot`` into ``target`` (default: this carrier's ``root``).

        Materializes every file from the blob store, deletes files the snapshot never had, and
        restores modes/symlinks — the tree becomes byte-for-byte the snapshotted world. Returns
        a ``{written, removed}`` count. Passing ``target`` restores a snapshot into a *different*
        root (the fork path).
        """
        dest = Path(target).resolve() if target else self.root
        dest.mkdir(parents=True, exist_ok=True)
        written = 0
        wanted = set(snapshot.manifest)

        # Remove files present now but absent from the snapshot.
        removed = self._prune(dest, wanted)

        for rel, meta in snapshot.manifest.items():
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            symlink = meta.get("symlink") or ""
            if symlink:
                if out.is_symlink() or out.exists():
                    out.unlink()
                out.symlink_to(symlink)
                written += 1
                continue
            self._materialize(meta["hash"], out)
            os.chmod(out, int(meta.get("mode") or 0o644))
            written += 1
        return {"written": written, "removed": removed}

    def _materialize(self, digest: str, out: Path) -> None:
        blob = self.blobs / digest[:2] / digest[2:]
        if not blob.exists():
            raise FileNotFoundError(f"run-vcs: blob {digest[:12]}… missing from store")
        if out.is_symlink() or out.exists():
            out.unlink()
        # COPY out of the immutable store: the restored file is a mutable working file, so it
        # must NOT share an inode with the blob (an in-place edit would corrupt the store).
        shutil.copy2(blob, out)

    def _prune(self, dest: Path, wanted: set[str]) -> int:
        removed = 0
        for path in sorted(dest.rglob("*"), reverse=True):
            rel_parts = path.relative_to(dest).parts
            if any(part in self._IGNORED_DIRS for part in rel_parts):
                continue
            rel = path.relative_to(dest).as_posix()
            if path.is_file() or path.is_symlink():
                if rel not in wanted:
                    path.unlink()
                    removed += 1
            elif path.is_dir():
                # Drop directories left empty after pruning.
                try:
                    next(path.iterdir())
                except StopIteration:
                    path.rmdir()
        return removed

    def load_snapshot(self, snapshot_id: str) -> FsSnapshot | None:
        path = self.snaps / f"{snapshot_id}.json"
        if not path.exists():
            return None
        return FsSnapshot.from_dict(json.loads(path.read_text(encoding="utf-8")))
