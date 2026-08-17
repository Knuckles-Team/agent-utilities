"""GOC-38 evidence envelope: build, validate, and compute a falsifiable verdict.

CONCEPT:AU-GOC.harness.evidence-envelope

``compute_verdict`` is the single authority for ``verdict.green``. No adapter
and no consumer (GOC-39, GOC-40) may declare green by any other path -- that
is the point of freezing this module's contract rather than each lane
re-deriving "did this pass".

A run is flagged non-green, with an explicit reason, whenever any of:

* ``collection.collection_count == 0`` (vacuous run -- nothing was actually
  measured, e.g. the ``git -C <subdir> ls-files`` root-relative-path defect
  that made ~20 gate helpers silently measure an empty universe).
* ``environment.venv_identity_match is False`` (wrong interpreter -- e.g. a
  bare ``uv run pytest`` silently running the system pytest).
* ``resources.survivor_check.clean is False`` (a process, thread, or cargo
  binary is still alive after the deadline/grace/kill sequence).
* ``contamination.invalidated is True`` (venv overwrite, cache-lock race,
  daemon-lease bleed observed before or after the run).
* a stream is truncated with no verified digest (can't prove what was
  actually captured).
* the exit-status provenance is not a direct wait status (i.e., anything
  that went through a shell pipeline, where ``$?`` is only the last stage).

This makes the envelope falsifiable: a reader can tell a real pass from a
vacuous one without re-running anything.
"""

from __future__ import annotations

import json
import platform
import socket
import time
import uuid
from pathlib import Path
from typing import Any

import jsonschema

from . import SCHEMA_DIR
from .launcher import LaunchResult
from .manifest import Manifest, current_interpreter_identity, sha256_bytes

ENVELOPE_SCHEMA = json.loads((SCHEMA_DIR / "envelope.schema.json").read_text())


def validate_envelope(data: dict[str, Any]) -> None:
    jsonschema.validate(instance=data, schema=ENVELOPE_SCHEMA)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _digest_bytes(data: bytes) -> str:
    return sha256_bytes(data)


def build_stream_record(
    data: bytes, *, truncated: bool, truncation_marker: str | None, path: str, retention_class: str = "raw-30d"
) -> dict[str, Any]:
    return {
        "byte_count": len(data),
        "truncated": truncated,
        "truncation_marker": truncation_marker,
        "digest": _digest_bytes(data),
        "retention_class": retention_class,
        "path": path,
    }


def compute_verdict(envelope_body: dict[str, Any]) -> dict[str, Any]:
    """Pure function: envelope fields in, verdict out. No I/O, no re-run."""
    reasons: list[str] = []

    collection_count = envelope_body["collection"]["collection_count"]
    if collection_count == 0:
        reasons.append("zero tests collected (vacuous run)")

    venv_identity_match = envelope_body["environment"]["venv_identity_match"]
    if not venv_identity_match:
        reasons.append("interpreter identity does not match the manifest's expected venv")

    survivor_clean = envelope_body["resources"]["survivor_check"]["clean"]
    if not survivor_clean:
        reasons.append("a process in the launched group survived teardown")

    invalidated = envelope_body["contamination"]["invalidated"]
    if invalidated:
        reasons.append("contamination invalidated the attempt")

    for stream_name in ("stdout", "stderr"):
        stream = envelope_body["streams"][stream_name]
        if stream["truncated"] and not stream["digest"]:
            reasons.append(f"{stream_name} truncated without a verified digest")

    provenance = envelope_body["exit"]["provenance"]
    if provenance not in ("direct_waitpid", "direct_wait4"):
        reasons.append(f"exit-status provenance is not a direct wait status ({provenance!r})")

    normalized_outcome = envelope_body["exit"]["normalized_outcome"]

    if reasons:
        if collection_count == 0:
            outcome = "VACUOUS"
        elif not survivor_clean or normalized_outcome == "TIMEOUT":
            outcome = "BLOCKED" if normalized_outcome != "TIMEOUT" else "TIMEOUT"
        else:
            outcome = "FAILED"
        green = False
    else:
        outcome = normalized_outcome if normalized_outcome in ("PASSED", "FAILED") else normalized_outcome
        green = normalized_outcome == "PASSED"

    falsifiable = (
        collection_count > 0
        and venv_identity_match
        and not invalidated
        and all(
            not envelope_body["streams"][s]["truncated"] or envelope_body["streams"][s]["digest"]
            for s in ("stdout", "stderr")
        )
    )

    return {
        "outcome": outcome,
        "green": green,
        "reasons": reasons,
        "falsifiable": falsifiable,
    }


def build_envelope(
    *,
    manifest: Manifest,
    repo_path: Path,
    branch: str,
    dirty: bool,
    launch: LaunchResult,
    adapter: str,
    venv_identity_match: bool,
    package_count: int,
    env_rejected_vars: list[str],
    collection_count: int,
    deselection_count: int,
    quarantine_count: int,
    selection_digest: str,
    collection_source: str,
    stdout_path: str,
    stderr_path: str,
    stdout_truncation_marker: str | None,
    stderr_truncation_marker: str | None,
    assertion_summary: dict[str, int],
    normalized_outcome: str,
    exit_provenance: str,
    leases: list[dict[str, Any]] | None = None,
    resource_samples: dict[str, Any] | None = None,
    contamination_pre: dict[str, Any] | None = None,
    contamination_post: dict[str, Any] | None = None,
    contamination_findings: list[str] | None = None,
    contamination_invalidated: bool = False,
    lane: str = "GOC-38",
    actor: str = "goc-38-harness",
    reviewer: str | None = None,
) -> dict[str, Any]:
    """Assembles a schema-valid envelope body and computes its verdict.
    Raises jsonschema.ValidationError if the assembled body does not match
    ``envelope.schema.json`` -- callers must not catch this to "degrade
    gracefully"; a malformed envelope is not evidence.
    """
    identity = current_interpreter_identity()
    resource_samples = resource_samples or {}

    body: dict[str, Any] = {
        "envelope_version": "1.0.0",
        "envelope_id": str(uuid.uuid4()),
        "manifest_digest": manifest.digest(),
        "state": "EVIDENCE-FLUSHED",
        "state_history": [
            {"state": "DECLARED", "at": _now(), "reason": None},
            {"state": "EVIDENCE-FLUSHED", "at": _now(), "reason": None},
        ],
        "candidate": {
            "repo": manifest.repo,
            "git_sha": manifest.candidate_sha,
            "branch": branch,
            "dirty": dirty,
        },
        "environment": {
            "sys_executable": identity["sys_executable"],
            "sys_prefix": identity["sys_prefix"],
            "venv_pyvenv_cfg_digest": manifest.interpreter_digest,
            "interpreter_version": identity["interpreter_version"],
            "lock_digest": manifest.lock_digest,
            "lockfile_path": "uv.lock",
            "package_count": package_count,
            "expected_venv_path": manifest.expected_venv_path,
            "venv_identity_match": venv_identity_match,
            "env_allowlist": manifest.env_allowlist,
            "env_rejected_vars": env_rejected_vars,
            "temp_root": manifest.temp_root,
            "host": socket.gethostname(),
            "os": platform.platform(),
        },
        "command": {
            "argv": launch.argv,
            "shell": False,
            "cwd": launch.cwd,
            "adapter": adapter,
        },
        "resources": {
            "process_group_leader_pid": launch.process_group_leader_pid,
            "process_group_start_time": _now(),
            "descendant_pids_observed": launch.survivors_before_kill,
            "deadline_seconds": manifest.timeout_seconds,
            "grace_seconds": manifest.grace_seconds,
            "cancellation": {
                "fired": launch.timed_out,
                "signal_sequence": launch.signal_sequence,
                "escalated": launch.escalated,
            },
            "survivor_check": {
                "performed": True,
                "survivors_before_kill": launch.survivors_before_kill,
                "survivors_after_kill": launch.survivors_after_kill,
                "clean": len(launch.survivors_after_kill) == 0,
            },
            "rss_peak_bytes": resource_samples.get("rss_peak_bytes"),
            "cpu_time_seconds": resource_samples.get("cpu_time_seconds"),
            "io_wait_seconds": resource_samples.get("io_wait_seconds"),
            "open_files_peak": resource_samples.get("open_files_peak"),
            "disk_delta_bytes": resource_samples.get("disk_delta_bytes"),
        },
        "leases": leases or [],
        "collection": {
            "collection_count": collection_count,
            "deselection_count": deselection_count,
            "quarantine_count": quarantine_count,
            "selection_digest": selection_digest,
            "collection_source": collection_source,
        },
        "streams": {
            "stdout": build_stream_record(
                launch.stdout,
                truncated=launch.stdout_truncated,
                truncation_marker=stdout_truncation_marker,
                path=stdout_path,
            ),
            "stderr": build_stream_record(
                launch.stderr,
                truncated=launch.stderr_truncated,
                truncation_marker=stderr_truncation_marker,
                path=stderr_path,
            ),
        },
        "exit": {
            "adapter_native_exit_code": launch.exit_code,
            "signal": launch.signal_name,
            "normalized_outcome": normalized_outcome,
            "assertion_summary": assertion_summary,
            "provenance": exit_provenance,
        },
        "contamination": {
            "pre_snapshot": contamination_pre or {},
            "post_snapshot": contamination_post or {},
            "findings": contamination_findings or [],
            "invalidated": contamination_invalidated,
        },
        "redaction": {
            "policy": manifest.redaction_policy,
            "redacted_patterns": ["token", "cookie", "tenant_payload", "private_path"],
            "applied": True,
        },
        "audit": {
            "actor": actor,
            "lane": lane,
            "reviewer": reviewer,
            "created_at": _now(),
            "flushed_at": _now(),
        },
    }

    body["verdict"] = compute_verdict(body)
    validate_envelope(body)
    return body
