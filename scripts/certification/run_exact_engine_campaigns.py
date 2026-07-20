#!/usr/bin/env python3
"""Run all non-local exact engine campaigns against one immutable artifact.

The orchestrator never discovers or builds an engine.  Deployment resolves the
explicit AgentConfig and release inputs, then supplies one engine, one release
Python, the source-frozen Epistemic Graph producer tree, and private runtime
locations.  Six existing producer scripts run serially with bounded output and
timeouts.  Only their closure-validated, path-free JSON evidence is published.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Final, NoReturn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import source_freeze_gate  # noqa: E402
from scripts.release import exact_artifact_closure as closure  # noqa: E402
from scripts.release import generate_component_evidence  # noqa: E402

CAMPAIGN_ORDER: Final = (
    "performance",
    "fault-restart",
    "protocol-authorization",
    "multimodal",
    "knowledge-batch",
    "reasoning-repair",
)
PRODUCER_SCRIPTS: Final = {
    "performance": "certify_exact_performance.py",
    "fault-restart": "certify_exact_fault_restart.py",
    "protocol-authorization": "certify_exact_protocol_authorization.py",
    "multimodal": "certify_exact_multimodal.py",
    "knowledge-batch": "certify_exact_knowledge_batch.py",
    "reasoning-repair": "certify_exact_reasoning_repair.py",
}
OUTPUT_FILES: Final = {
    "performance": "performance.json",
    "fault-restart": "fault-restart.json",
    "protocol-authorization": "protocol-authorization.json",
    "multimodal": "multimodal.json",
    "knowledge-batch": "knowledge-batch.json",
    "reasoning-repair": "reasoning-repair.json",
}
CAMPAIGN_TIMEOUT_SECONDS: Final = {
    "performance": 14_400,
    "fault-restart": 7_200,
    "protocol-authorization": 3_600,
    "multimodal": 7_200,
    "knowledge-batch": 3_600,
    "reasoning-repair": 3_600,
}

_HEX_64 = re.compile(r"^[a-f0-9]{64}$")
_RELEASE_ID = re.compile(r"^release-[a-z0-9][a-z0-9.-]{2,63}$")
_MAX_EVIDENCE_BYTES: Final = 16 * 1024 * 1024
_MAX_AUTHORITY_BYTES: Final = 64 * 1024
_MAX_SCRIPT_BYTES: Final = 4 * 1024 * 1024
_MAX_EXECUTABLE_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_CHILD_OUTPUT_BYTES: Final = 1_048_576


class CampaignOrchestrationError(RuntimeError):
    """One stable, privacy-safe orchestration failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> NoReturn:
    raise CampaignOrchestrationError(code)


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_regular(
    path: Path,
    *,
    maximum: int,
    executable: bool = False,
    private: bool = False,
) -> tuple[int, os.stat_result]:
    if not path.is_absolute() or path.is_symlink():
        _fail("input_file_invalid")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
    except OSError:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        _fail("input_file_invalid")
    assert descriptor is not None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or not 0 < metadata.st_size <= maximum
        or (executable and metadata.st_mode & 0o111 == 0)
        or (private and metadata.st_mode & 0o077 != 0)
        or (private and hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        os.close(descriptor)
        _fail("input_file_invalid")
    return descriptor, metadata


def _read_regular(path: Path, *, maximum: int, private: bool = False) -> bytes:
    descriptor, metadata = _open_regular(path, maximum=maximum, private=private)
    before = _identity(metadata)
    payload = bytearray()
    try:
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
    except OSError:
        _fail("input_file_changed")
    finally:
        os.close(descriptor)
    try:
        path_metadata = path.stat(follow_symlinks=False)
    except OSError:
        _fail("input_file_changed")
    if (
        len(payload) != metadata.st_size
        or _identity(after) != before
        or (path_metadata.st_dev, path_metadata.st_ino)
        != (metadata.st_dev, metadata.st_ino)
    ):
        _fail("input_file_changed")
    return bytes(payload)


def _hash_regular(
    path: Path,
    *,
    maximum: int,
    executable: bool = False,
) -> str:
    descriptor, metadata = _open_regular(
        path,
        maximum=maximum,
        executable=executable,
    )
    before = _identity(metadata)
    digest = hashlib.sha256()
    observed = 0
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            observed += len(chunk)
            if observed > maximum:
                _fail("input_file_invalid")
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError:
        _fail("input_file_changed")
    finally:
        os.close(descriptor)
    try:
        path_metadata = path.stat(follow_symlinks=False)
    except OSError:
        _fail("input_file_changed")
    if (
        observed != metadata.st_size
        or _identity(after) != before
        or (path_metadata.st_dev, path_metadata.st_ino)
        != (metadata.st_dev, metadata.st_ino)
    ):
        _fail("input_file_changed")
    return digest.hexdigest()


def _private_directory(path: Path, *, writable: bool) -> Path:
    if not path.is_absolute():
        _fail("private_directory_invalid")
    try:
        resolved = path.resolve(strict=True)
        metadata = path.stat(follow_symlinks=False)
    except OSError:
        _fail("private_directory_invalid")
    if (
        resolved != path.absolute()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_mode & 0o077 != 0
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        or (writable and not os.access(resolved, os.W_OK | os.X_OK))
    ):
        _fail("private_directory_invalid")
    return resolved


def _producer_root(path: Path) -> Path:
    if not path.is_absolute():
        _fail("producer_root_invalid")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        _fail("producer_root_invalid")
    scripts = resolved / "scripts"
    if not resolved.is_dir() or not scripts.is_dir() or scripts.is_symlink():
        _fail("producer_root_invalid")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except ValueError:
        return False
    return True


def _source_freeze_binding(path: Path, expected_digest: str, root: Path) -> str:
    if _HEX_64.fullmatch(expected_digest) is None:
        _fail("source_freeze_digest_invalid")
    raw = _read_regular(path, maximum=_MAX_EVIDENCE_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        _fail("source_freeze_digest_mismatch")
    try:
        generate_component_evidence._source_freeze(path)
        value = json.loads(raw)
    except Exception:
        _fail("source_freeze_evidence_invalid")
    repositories = value.get("repositories") if isinstance(value, dict) else None
    if not isinstance(repositories, list):
        _fail("source_freeze_evidence_invalid")
    matching = [
        item
        for item in repositories
        if isinstance(item, dict) and item.get("id") == "epistemic-graph"
    ]
    if len(matching) != 1:
        _fail("source_freeze_evidence_invalid")
    expected_root_digest = str(matching[0].get("sha256_after") or "")
    if (
        _HEX_64.fullmatch(expected_root_digest) is None
        or source_freeze_gate.source_tree_digest(root) != expected_root_digest
    ):
        _fail("producer_source_digest_mismatch")
    return expected_root_digest


def _stage_engine(source: Path, expected_digest: str, destination: Path) -> None:
    if _HEX_64.fullmatch(expected_digest) is None:
        _fail("engine_digest_invalid")
    descriptor, metadata = _open_regular(
        source,
        maximum=_MAX_EXECUTABLE_BYTES,
        executable=True,
    )
    before = _identity(metadata)
    output: int | None = None
    digest = hashlib.sha256()
    observed = 0
    try:
        output = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0o500,
        )
        while chunk := os.read(descriptor, 1024 * 1024):
            observed += len(chunk)
            if observed > _MAX_EXECUTABLE_BYTES:
                _fail("engine_file_invalid")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(output, view)
                if written <= 0:
                    _fail("engine_stage_failed")
                view = view[written:]
        os.fsync(output)
        os.fchmod(output, 0o500)
        after = os.fstat(descriptor)
    except CampaignOrchestrationError:
        raise
    except OSError:
        _fail("engine_stage_failed")
    finally:
        if output is not None:
            os.close(output)
        os.close(descriptor)
    try:
        path_metadata = source.stat(follow_symlinks=False)
    except OSError:
        _fail("input_file_changed")
    if (
        observed != metadata.st_size
        or _identity(after) != before
        or (path_metadata.st_dev, path_metadata.st_ino)
        != (metadata.st_dev, metadata.st_ino)
        or digest.hexdigest() != expected_digest
        or _hash_regular(
            destination,
            maximum=_MAX_EXECUTABLE_BYTES,
            executable=True,
        )
        != expected_digest
    ):
        _fail("engine_digest_mismatch")


def _minimal_environment(root: Path) -> dict[str, str]:
    root.mkdir(mode=0o700)
    directories = {
        "HOME": root / "home",
        "TMPDIR": root / "tmp",
        "XDG_CACHE_HOME": root / "cache",
        "XDG_CONFIG_HOME": root / "config",
        "XDG_DATA_HOME": root / "data",
        "XDG_RUNTIME_DIR": root / "runtime",
        "XDG_STATE_HOME": root / "state",
    }
    for path in directories.values():
        path.mkdir(mode=0o700)
    return {
        **{name: str(path) for name, path in directories.items()},
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.defpath,
        "RUST_BACKTRACE": "0",
        "TZ": "UTC",
    }


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)
    if process.poll() is None:
        process.kill()
        process.wait()


def _run_bounded(
    argv: list[str],
    *,
    campaign: str,
    cwd: Path,
    environment: dict[str, str],
) -> None:
    if (
        campaign not in CAMPAIGN_TIMEOUT_SECONDS
        or not argv
        or any(not item for item in argv)
    ):
        _fail("campaign_argv_invalid")
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        _fail(f"{campaign}_start_failed")
    overflow = threading.Event()

    def drain(stream: Any) -> None:
        observed = 0
        try:
            while chunk := stream.read(64 * 1024):
                observed += len(chunk)
                if observed > _MAX_CHILD_OUTPUT_BYTES:
                    overflow.set()
        finally:
            stream.close()

    assert process.stdout is not None
    assert process.stderr is not None
    threads = [
        threading.Thread(target=drain, args=(process.stdout,), daemon=True),
        threading.Thread(target=drain, args=(process.stderr,), daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        try:
            returncode = process.wait(timeout=CAMPAIGN_TIMEOUT_SECONDS[campaign])
        except subprocess.TimeoutExpired:
            _terminate(process)
            _fail(f"{campaign}_timeout")
    finally:
        for thread in threads:
            thread.join(timeout=10)
        if process.poll() is None:
            _terminate(process)
    if overflow.is_set():
        _fail(f"{campaign}_output_overflow")
    if returncode != 0:
        _fail(f"{campaign}_failed")


def _campaign_argv(
    campaign: str,
    *,
    python: Path,
    script: Path,
    engine: Path,
    engine_sha256: str,
    output: Path,
    authority_config: Path,
    work_root: Path,
    performance_evidence: Path | None,
    performance_digest: str | None,
    markdown_output: Path,
) -> list[str]:
    prefix = [str(python), "-E", "-s", "-B", str(script)]
    if campaign == "performance":
        return [
            *prefix,
            "--engine-binary",
            str(engine),
            "--engine-sha256",
            engine_sha256,
            "--authority-config",
            str(authority_config),
            "--work-root",
            str(work_root),
            "--json-output",
            str(output),
            "--markdown-output",
            str(markdown_output),
        ]
    argv = [
        *prefix,
        "--binary",
        str(engine),
        "--binary-sha256",
        engine_sha256,
    ]
    if campaign == "multimodal":
        if performance_evidence is None or performance_digest is None:
            _fail("performance_binding_missing")
        argv.extend(
            [
                "--performance-evidence",
                str(performance_evidence),
                "--performance-evidence-sha256",
                performance_digest,
            ]
        )
    argv.extend(["--output", str(output)])
    return argv


def _json_without_duplicates(payload: bytes) -> dict[str, Any]:
    def convert(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                _fail("campaign_evidence_duplicate_key")
            value[key] = item
        return value

    try:
        value = json.loads(payload, object_pairs_hook=convert)
    except (UnicodeError, json.JSONDecodeError):
        _fail("campaign_evidence_invalid")
    if not isinstance(value, dict):
        _fail("campaign_evidence_invalid")
    return value


def _validate_campaign_evidence(
    campaign: str,
    path: Path,
    *,
    engine_sha256: str,
    performance_digest: str | None,
) -> str:
    payload = _read_regular(path, maximum=_MAX_EVIDENCE_BYTES, private=True)
    digest = hashlib.sha256(payload).hexdigest()
    value = _json_without_duplicates(payload)
    try:
        if campaign == "performance":
            closure._validate_performance(value, engine_sha256)
        elif campaign == "fault-restart":
            closure._validate_fault_restart(value, engine_sha256)
        elif campaign == "protocol-authorization":
            closure._validate_protocol_authorization(value, engine_sha256)
        elif campaign == "multimodal":
            if performance_digest is None:
                _fail("performance_binding_missing")
            closure._validate_multimodal(value, engine_sha256, performance_digest)
        elif campaign == "knowledge-batch":
            closure._validate_knowledge_batch(value, engine_sha256)
        elif campaign == "reasoning-repair":
            closure._validate_reasoning_repair(value, engine_sha256)
        else:
            _fail("campaign_inventory_invalid")
    except closure.ClosureError:
        _fail(f"{campaign}_evidence_invalid")
    return digest


def _script_inventory(root: Path) -> dict[str, tuple[Path, str]]:
    inventory: dict[str, tuple[Path, str]] = {}
    for campaign in CAMPAIGN_ORDER:
        path = root / "scripts" / PRODUCER_SCRIPTS[campaign]
        digest = hashlib.sha256(
            _read_regular(path, maximum=_MAX_SCRIPT_BYTES)
        ).hexdigest()
        inventory[campaign] = (path, digest)
    return inventory


def _fsync_tree(path: Path) -> None:
    for name in OUTPUT_FILES.values():
        descriptor = os.open(
            path / name,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_campaigns(
    *,
    release_id: str,
    engine_binary: Path,
    engine_sha256: str,
    campaign_python: Path,
    campaign_python_sha256: str,
    epistemic_graph_root: Path,
    source_freeze_evidence: Path,
    source_freeze_sha256: str,
    authority_config: Path,
    work_root: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Run and publish the six exact campaign documents or fail with no output."""

    if _RELEASE_ID.fullmatch(release_id) is None:
        _fail("release_id_invalid")
    if (
        _HEX_64.fullmatch(engine_sha256) is None
        or _HEX_64.fullmatch(campaign_python_sha256) is None
    ):
        _fail("release_digest_invalid")
    producer_root = _producer_root(epistemic_graph_root)
    private_work_root = _private_directory(work_root, writable=True)
    output_parent = _private_directory(output_dir.parent, writable=True)
    if not output_dir.is_absolute() or output_dir.parent.resolve() != output_parent:
        _fail("output_directory_invalid")
    if output_dir.exists() or output_dir.is_symlink():
        _fail("output_directory_exists")
    if _is_within(output_dir, producer_root) or _is_within(
        private_work_root, producer_root
    ):
        _fail("runtime_location_overlaps_producer_source")
    if _is_within(authority_config, producer_root):
        _fail("authority_config_overlaps_producer_source")
    _read_regular(
        authority_config,
        maximum=_MAX_AUTHORITY_BYTES,
        private=True,
    )
    expected_source_digest = _source_freeze_binding(
        source_freeze_evidence,
        source_freeze_sha256,
        producer_root,
    )
    python_path = campaign_python.absolute()
    if (
        _hash_regular(
            python_path,
            maximum=_MAX_EXECUTABLE_BYTES,
            executable=True,
        )
        != campaign_python_sha256
    ):
        _fail("campaign_python_digest_mismatch")
    scripts = _script_inventory(producer_root)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_parent,
        )
    )
    staging.chmod(0o700)
    published = False
    try:
        with tempfile.TemporaryDirectory(
            prefix="exact-engine-campaigns-",
            dir=private_work_root,
        ) as scratch_text:
            scratch = Path(scratch_text)
            scratch.chmod(0o700)
            staged_engine = scratch / "epistemic-graph-server"
            _stage_engine(engine_binary.absolute(), engine_sha256, staged_engine)
            environment = _minimal_environment(scratch / "environment")
            performance_path: Path | None = None
            performance_digest: str | None = None
            for campaign in CAMPAIGN_ORDER:
                if (
                    _hash_regular(
                        python_path,
                        maximum=_MAX_EXECUTABLE_BYTES,
                        executable=True,
                    )
                    != campaign_python_sha256
                    or _hash_regular(
                        staged_engine,
                        maximum=_MAX_EXECUTABLE_BYTES,
                        executable=True,
                    )
                    != engine_sha256
                ):
                    _fail("release_artifact_changed")
                output = staging / OUTPUT_FILES[campaign]
                argv = _campaign_argv(
                    campaign,
                    python=python_path,
                    script=scripts[campaign][0],
                    engine=staged_engine,
                    engine_sha256=engine_sha256,
                    output=output,
                    authority_config=authority_config,
                    work_root=private_work_root,
                    performance_evidence=performance_path,
                    performance_digest=performance_digest,
                    markdown_output=scratch / "performance.md",
                )
                _run_bounded(
                    argv,
                    campaign=campaign,
                    cwd=producer_root / "scripts",
                    environment=environment,
                )
                digest = _validate_campaign_evidence(
                    campaign,
                    output,
                    engine_sha256=engine_sha256,
                    performance_digest=performance_digest,
                )
                if campaign == "performance":
                    performance_path = output
                    performance_digest = digest

        if (
            source_freeze_gate.source_tree_digest(producer_root)
            != expected_source_digest
        ):
            _fail("producer_source_changed")
        if _hash_regular(
            python_path,
            maximum=_MAX_EXECUTABLE_BYTES,
            executable=True,
        ) != campaign_python_sha256 or any(
            hashlib.sha256(_read_regular(path, maximum=_MAX_SCRIPT_BYTES)).hexdigest()
            != digest
            for path, digest in scripts.values()
        ):
            _fail("release_input_changed")
        if {entry.name for entry in staging.iterdir()} != set(OUTPUT_FILES.values()):
            _fail("campaign_output_inventory_invalid")
        _fsync_tree(staging)
        if output_dir.exists() or output_dir.is_symlink():
            _fail("output_directory_exists")
        os.rename(staging, output_dir)
        directory = os.open(
            output_parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        published = True
        return {
            campaign: output_dir / OUTPUT_FILES[campaign] for campaign in CAMPAIGN_ORDER
        }
    except CampaignOrchestrationError:
        raise
    except OSError:
        _fail("campaign_publication_failed")
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the six closure-ready exact engine campaigns serially."
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--engine-binary", type=Path, required=True)
    parser.add_argument("--engine-sha256", required=True)
    parser.add_argument("--campaign-python", type=Path, required=True)
    parser.add_argument("--campaign-python-sha256", required=True)
    parser.add_argument("--epistemic-graph-root", type=Path, required=True)
    parser.add_argument("--source-freeze-evidence", type=Path, required=True)
    parser.add_argument("--source-freeze-sha256", required=True)
    parser.add_argument("--authority-config", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_campaigns(
            release_id=args.release_id,
            engine_binary=args.engine_binary,
            engine_sha256=args.engine_sha256,
            campaign_python=args.campaign_python,
            campaign_python_sha256=args.campaign_python_sha256,
            epistemic_graph_root=args.epistemic_graph_root,
            source_freeze_evidence=args.source_freeze_evidence,
            source_freeze_sha256=args.source_freeze_sha256,
            authority_config=args.authority_config,
            work_root=args.work_root,
            output_dir=args.output_dir,
        )
    except CampaignOrchestrationError as exc:
        print(f"exact engine campaigns: FAIL ({exc.code})", file=sys.stderr)
        return 1
    print(f"exact engine campaigns: PASS ({len(result)} campaigns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
