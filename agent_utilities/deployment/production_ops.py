"""Fail-closed production backup and restore-validation commands.

The commands deliberately emit only opaque bundle digests and aggregate counts.
They never print engine endpoints, volume locations, principals, credentials, or
the contents of a restored graph.  Runtime locations and authority arrive through
the pod environment/volume mounts and are not written into evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from filelock import FileLock

from agent_utilities.core.config import AgentConfig, setting
from agent_utilities.knowledge_graph.core.engine_transport import (
    engine_client_transport_kwargs,
    native_endpoint_address,
)


class ProductionOperationError(RuntimeError):
    """A production operation could not satisfy its safety contract."""


def _required_env(name: str) -> str:
    value = str(setting(name, "") or "").strip()
    if not value:
        raise ProductionOperationError(f"required runtime setting {name} is absent")
    return value


def _verified_context() -> dict[str, Any]:
    """Build the engine v2 context from workload-identity-projected settings."""
    return {
        "principal": _required_env("GRAPH_OS_BACKUP_PRINCIPAL"),
        "tenant": _required_env("GRAPH_OS_BACKUP_TENANT"),
        "audience": _required_env("AUTH_JWT_AUDIENCE"),
        "agent_id": _required_env("GRAPH_OS_BACKUP_PRINCIPAL"),
        "roles": ["backup-operator"],
        "scopes": ["kg:admin"],
        "delegation": [],
        "policy_version": _required_env("KG_POLICY_VERSION"),
    }


def _coordinator_transport() -> tuple[str, dict[str, Any]]:
    """Resolve the sole configured coordinator through the shared TLS policy."""
    config = AgentConfig()
    endpoints = config.graph_service_endpoints or []
    if len(endpoints) != 1:
        raise ProductionOperationError(
            "production operations require exactly one GRAPH_SERVICE_ENDPOINTS coordinator"
        )
    endpoint = str(endpoints[0]).strip()
    try:
        address = native_endpoint_address(endpoint)[0]
        transport = engine_client_transport_kwargs(endpoint, config=config)
    except Exception as exc:
        raise ProductionOperationError(
            "the configured coordinator transport is invalid"
        ) from exc
    return address, transport


def _inside(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    if root != candidate and root not in candidate.parents:
        raise ProductionOperationError("operation target escapes its mounted root")
    return candidate


def _tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = 0
    size = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        files += 1
    return "sha256:" + digest.hexdigest(), files, size


def _archive_lock(archive_root: Path, *, timeout_seconds: int = 7500) -> FileLock:
    return FileLock(
        str(archive_root / ".graphos-operations.lock"), timeout=timeout_seconds
    )


def _retention_count() -> int:
    value = int(setting("GRAPHOS_BACKUP_RETENTION_COUNT", 2))
    if not 2 <= value <= 1440:
        raise ProductionOperationError(
            "GRAPHOS_BACKUP_RETENTION_COUNT must be inside 2..1440"
        )
    return value


def _prune_bundles(archive_root: Path, *, keep: int) -> int:
    """Bound mounted full bundles; object-store versioning owns long retention."""
    candidates = sorted(
        (
            path
            for path in archive_root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and path.name.startswith("bundle-")
            and (path / "MANIFEST.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    removed = 0
    for candidate in candidates[keep:]:
        shutil.rmtree(_inside(archive_root, candidate))
        removed += 1
    cutoff = time.time() - 86400
    for candidate in archive_root.iterdir():
        if (
            candidate.is_dir()
            and not candidate.is_symlink()
            and candidate.name.startswith("bundle-")
            and not (candidate / "MANIFEST.json").exists()
            and candidate.stat().st_mtime < cutoff
        ):
            shutil.rmtree(_inside(archive_root, candidate))
            removed += 1
    return removed


def _recovery_manifest(bundle: Path) -> dict[str, int]:
    """Require the portable coordinator-aware backup format and aggregate proofs."""
    manifest_path = bundle / "MANIFEST.json"
    coordinator_path = bundle / "admin-mutations.redb"
    if not manifest_path.is_file() or not coordinator_path.is_file():
        raise ProductionOperationError("backup omits portable recovery state")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        admin = value["admin_mutations"]
        counts = {
            "admin_batches": int(admin["batches"]),
            "prepared_parents": int(admin["prepared"]),
            "encrypted_recovery_plans": int(admin["encrypted_private_payloads"]),
            "xshard_prepares": int(value["xshard_prepares"]),
            "xshard_decisions": int(value["xshard_decisions"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProductionOperationError("backup recovery manifest is malformed") from exc
    if int(value.get("format_version", 0)) != 3 or any(
        count < 0 for count in counts.values()
    ):
        raise ProductionOperationError("backup recovery manifest is not supported")
    if counts["prepared_parents"] < counts["encrypted_recovery_plans"]:
        raise ProductionOperationError(
            "encrypted recovery plans have no prepared parents"
        )
    return counts


async def _backup(archive_root: Path) -> dict[str, Any]:
    from epistemic_graph import EpistemicGraphClient

    archive_root.mkdir(parents=True, exist_ok=True)
    archive_root = archive_root.resolve()
    bundle = _inside(
        archive_root,
        archive_root / f"bundle-{int(time.time())}-{secrets.token_hex(6)}",
    )
    address, transport = _coordinator_transport()
    client = await EpistemicGraphClient.connect(
        tcp_addr=address,
        auth_secret=_required_env("GRAPH_SERVICE_AUTH_SECRET"),
        verified_context=_verified_context(),
        **transport,
    )
    try:
        try:
            await client.admin.backup(str(bundle), label="scheduled")
        except Exception:
            # Boundary-change retries deliberately leave no MANIFEST. Remove that
            # unpublished full copy immediately so a busy cell cannot accumulate
            # a day of failed, multi-hundred-GiB attempts. A manifest-bearing
            # bundle is retained because this may only be an acknowledgement loss.
            with _archive_lock(archive_root):
                if (
                    bundle.is_dir()
                    and not bundle.is_symlink()
                    and not (bundle / "MANIFEST.json").is_file()
                ):
                    shutil.rmtree(_inside(archive_root, bundle))
            raise
    finally:
        await client.close()
    try:
        recovery = _recovery_manifest(bundle)
        bundle_digest, file_count, byte_count = _tree_digest(bundle)
    except Exception:
        with _archive_lock(archive_root):
            if bundle.is_dir() and not bundle.is_symlink():
                shutil.rmtree(_inside(archive_root, bundle))
        raise
    with _archive_lock(archive_root):
        pruned_bundles = _prune_bundles(archive_root, keep=_retention_count())
    return {
        "operation": "backup",
        "ok": True,
        "bundle_digest": bundle_digest,
        "file_count": file_count,
        "byte_count": byte_count,
        "pruned_bundle_count": pruned_bundles,
        **recovery,
    }


def _latest_bundle(archive_root: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in archive_root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and (path / "MANIFEST.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise ProductionOperationError("archive contains no complete backup bundle")
    return _inside(archive_root, candidates[0])


def _wait_for_port(
    process: subprocess.Popen[bytes], port: int, timeout_s: float
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ProductionOperationError(
                "restored engine exited before becoming ready"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    raise ProductionOperationError("restored engine did not become ready in time")


async def _probe_restored_engine(port: int) -> None:
    from epistemic_graph import EpistemicGraphClient

    client = await EpistemicGraphClient.connect(
        tcp_addr=f"127.0.0.1:{port}",
        auth_secret=_required_env("GRAPH_SERVICE_AUTH_SECRET"),
        verified_context=_verified_context(),
    )
    try:
        health = await client.health()
        if not isinstance(health, dict):
            raise ProductionOperationError(
                "restored engine returned malformed health data"
            )
    finally:
        await client.close()


async def _restore_validate(archive_root: Path, scratch_root: Path) -> dict[str, Any]:
    archive_root = archive_root.resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch_root = scratch_root.resolve()
    destination = _inside(
        scratch_root,
        scratch_root / f"restore-{int(time.time())}-{secrets.token_hex(6)}",
    )
    destination.mkdir(mode=0o700)
    restore_bin = str(setting("EPISTEMIC_GRAPH_RESTORE_BIN", "restore"))
    server_bin = str(setting("EPISTEMIC_GRAPH_SERVER_BIN", "epistemic-graph-server"))
    port = int(setting("RESTORE_VALIDATION_PORT", 19_100))
    if not 1024 <= port <= 65535:
        raise ProductionOperationError("RESTORE_VALIDATION_PORT is outside 1024..65535")
    try:
        # Hold the archive lock only while the selected bundle is read. Backup may
        # create a newer bundle concurrently, but its retention pass waits, so the
        # source cannot disappear midway through the offline restore.
        with _archive_lock(archive_root, timeout_seconds=60):
            bundle = _latest_bundle(archive_root)
            restored = subprocess.run(
                [
                    restore_bin,
                    "--bundle",
                    str(bundle),
                    "--persist-dir",
                    str(destination),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )
            if restored.returncode != 0:
                raise ProductionOperationError(
                    "offline restore failed; output_digest="
                    + hashlib.sha256(restored.stdout).hexdigest()
                )
            recovery = _recovery_manifest(bundle)
            if not (destination / "admin-mutations.redb").is_file():
                raise ProductionOperationError(
                    "restore omitted the admin mutation coordinator"
                )
            bundle_digest, file_count, byte_count = _tree_digest(bundle)
        env = dict(os.environ)
        process = subprocess.Popen(
            [
                server_bin,
                "--tcp-addr",
                f"127.0.0.1:{port}",
                "--persist-dir",
                str(destination),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_port(process, port, timeout_s=120.0)
            await _probe_restored_engine(port)
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        return {
            "operation": "restore_validation",
            "ok": True,
            "bundle_digest": bundle_digest,
            "file_count": file_count,
            "byte_count": byte_count,
            "health_probe": "passed",
            **recovery,
        }
    finally:
        shutil.rmtree(destination, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graph-os-production-ops")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    backup = subparsers.add_parser("backup")
    backup.add_argument("--archive-root", type=Path, required=True)
    restore = subparsers.add_parser("restore-validate")
    restore.add_argument("--archive-root", type=Path, required=True)
    restore.add_argument("--scratch-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "backup":
            report = asyncio.run(_backup(args.archive_root))
        else:
            report = asyncio.run(
                _restore_validate(args.archive_root, args.scratch_root)
            )
    except Exception as exc:  # noqa: BLE001 - CLI returns one privacy-safe failure
        report = {
            "operation": args.operation,
            "ok": False,
            "error_type": type(exc).__name__,
            "error_digest": "sha256:"
            + hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
        }
        print(json.dumps(report, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
