"""Provenance headers for measurements (measurement harness, capability A).

CONCEPT:AU-OS.measurement.provenance-header

Built in direct response to a cluster of eight false alarms raised in one
session (documented in ``plans/graph-os-completion-program/BUG-LEDGER.md``,
BUG-220..BUG-229) that shared one root cause: the measuring INSTRUMENT
differed from the thing being claimed about, and nothing recorded which
instrument had actually been used. A verdict with no record of its own
interpreter, host, tree identity, or system load cannot be told apart from a
verdict measured on a stale copy, a different Python, or an overloaded box.

This module makes "which instrument measured this" a first-class, mandatory
part of every result. :class:`ProvenanceHeader` is captured at the start of a
measurement and closed at the end; :func:`require_provenance` is the
verifier that rejects a result carrying no header (or a malformed one) —
callers that consume measurement results (gates, reports, dashboards) are
expected to call it before trusting a verdict.

Three further incidents in the same session were all "the SAME command
passed here and failed there" — and in every case the header fields below
would have shown why, had anyone compared them:

* ``bash scripts/check_ontology.py`` passed while the pre-commit HOOK of the
  same name failed, because the hook's venv had extra fleet packages
  installed that changed what the check discovered — a different
  ``interpreter`` path (``sys.executable``, different venv).
* A local gate pinned to 2 cores via ``taskset -c 0,1`` on a 64-core host
  PASSED, while the real 2-vCPU CI runner FAILED the same test (tokio sizes
  its worker pool from ``available_parallelism``, and scheduling differs
  even at the same nominal core count) — a different ``cpu_affinity``/
  ``cpu_count``.
* ``rustc-wrapper = "sccache"`` passed every local gate (this host HAS
  sccache) and then killed every CI job in 24s (runners do not) — a
  different ``env_fingerprint`` (``RUSTC_WRAPPER`` et al.).

:func:`environment_mismatches` / :func:`require_same_environment` compare
two headers on exactly these fields, so "did these two runs actually use
the same instrument" is a function call instead of a guess.

Nothing here prevents a caller from lying about its own command line -- the
header is only as honest as the code that fills it in. What it *does*
guarantee is that a verdict with no header, or a header missing a required
field, is mechanically rejected rather than silently trusted.
"""

from __future__ import annotations

import dataclasses
import getpass
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

#: Fields a header MUST carry to be admissible. Kept as a tuple (not derived
#: from the dataclass) so `require_provenance` can validate a plain dict
#: (e.g. one round-tripped through JSON) without importing the dataclass.
REQUIRED_FIELDS = (
    "schema_version",
    "interpreter",
    "hostname",
    "user",
    "cwd",
    "command",
    "git_sha",
    "git_dirty",
    "is_copy",
    "load_avg_start",
    "load_avg_end",
    "timestamp_start",
    "timestamp_end",
    "cpu_count",
    "cpu_affinity",
    "env_fingerprint",
)


class ProvenanceError(Exception):
    """Base class for provenance-header defects."""


class MissingProvenanceError(ProvenanceError):
    """Raised by :func:`require_provenance` when a result has no admissible header."""


#: Environment variables whose presence/value has, historically, been the
#: entire difference between a passing local gate and a failing CI run (or
#: vice versa). Extend this tuple, don't hardcode a new one-off check,
#: when the next such incident is found.
WATCHED_ENV_VARS = (
    "VIRTUAL_ENV",
    "UV_PROJECT_ENVIRONMENT",
    "PYTHONPATH",
    "RUSTC_WRAPPER",
    "CARGO_BUILD_RUSTC_WRAPPER",
    "CARGO_TARGET_DIR",
)

#: Header fields compared by :func:`environment_mismatches` — the set that,
#: if it differs between two runs, means the two runs are not actually
#: measuring with the same instrument even if the command line was
#: identical.
DEFAULT_COMPARISON_FIELDS = (
    "interpreter",
    "cpu_count",
    "cpu_affinity",
    "env_fingerprint",
)


def _load_avg() -> tuple[float, float, float] | None:
    try:
        return tuple(os.getloadavg())  # type: ignore[return-value]
    except (OSError, AttributeError):
        # getloadavg() is POSIX-only; on a platform without it we record
        # None rather than fabricate a number.
        return None


def _cpu_affinity() -> list[int] | None:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        # Not available on this platform (e.g. macOS) -- record None, not a
        # guessed full-core-set.
        return None
    try:
        return sorted(getter(0))
    except OSError:
        return None


def _env_fingerprint(
    watch: tuple[str, ...] = WATCHED_ENV_VARS,
) -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in watch}


def git_identity(tree: Path | str) -> tuple[str | None, bool | None]:
    """Return ``(sha, dirty)`` for the git tree rooted at ``tree``.

    Both are ``None`` (never a fabricated ``"unknown"`` SHA or a guessed
    ``False``) when ``tree`` is not inside a git work tree at all, so a
    caller can tell "not git" apart from "git, but clean" without a magic
    string.
    """
    tree = str(tree)
    try:
        sha = subprocess.run(
            ["git", "-C", tree, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if sha.returncode != 0:
            return None, None
        dirty = subprocess.run(
            ["git", "-C", tree, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        is_dirty = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
        return sha.stdout.strip(), is_dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


@dataclasses.dataclass
class ProvenanceHeader:
    """Everything needed to tell one measurement's instrument apart from another's.

    Construct with :meth:`start`, run the measurement, then call
    :meth:`finish` to fill in the end-of-run fields (load average at end —
    incident 4 — and the wall timestamp).
    """

    schema_version: int
    interpreter: str
    hostname: str
    user: str
    cwd: str
    command: list[str]
    git_sha: str | None
    git_dirty: bool | None
    tree: str
    is_copy: bool
    copy_integrity: dict[str, Any] | None
    load_avg_start: tuple[float, float, float] | None
    load_avg_end: tuple[float, float, float] | None
    timestamp_start: float
    timestamp_end: float | None
    platform: str
    cpu_count: int | None
    cpu_affinity: list[int] | None
    env_fingerprint: dict[str, str | None]

    @classmethod
    def start(
        cls,
        command: list[str],
        *,
        tree: str | Path | None = None,
        is_copy: bool = False,
        copy_integrity: dict[str, Any] | None = None,
        watched_env_vars: tuple[str, ...] = WATCHED_ENV_VARS,
    ) -> ProvenanceHeader:
        tree = str(tree) if tree is not None else os.getcwd()
        sha, dirty = git_identity(tree)
        return cls(
            schema_version=SCHEMA_VERSION,
            interpreter=sys.executable,
            hostname=socket.gethostname(),
            user=getpass.getuser(),
            cwd=os.getcwd(),
            command=list(command),
            cpu_count=os.cpu_count(),
            cpu_affinity=_cpu_affinity(),
            env_fingerprint=_env_fingerprint(watched_env_vars),
            git_sha=sha,
            git_dirty=dirty,
            tree=tree,
            is_copy=is_copy,
            copy_integrity=copy_integrity,
            load_avg_start=_load_avg(),
            load_avg_end=None,
            timestamp_start=time.time(),
            timestamp_end=None,
            platform=platform.platform(),
        )

    def finish(self) -> ProvenanceHeader:
        """Return a copy with the end-of-run fields filled in.

        Immutable-update rather than mutate-in-place so a caller cannot
        accidentally hand out a header that looks "finished" (has
        ``load_avg_end``) while a measurement is still running elsewhere on
        the same object.
        """
        return dataclasses.replace(
            self,
            load_avg_end=_load_avg(),
            timestamp_end=time.time(),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def require_provenance(result: Any) -> dict[str, Any]:
    """Verifier: reject a measurement result that carries no admissible header.

    Accepts either a ``ProvenanceHeader`` or a plain ``dict`` with a
    ``"provenance"`` key (the shape a JSON-serialized result takes), or a
    dict that *is itself* a provenance header. Raises
    :class:`MissingProvenanceError` — never returns a falsy/"unknown" value
    — when the header is absent or missing a required field, so a caller
    cannot mistake a rejected result for a passing one.

    Returns the header dict on success, so callers can chain
    ``header = require_provenance(result)``.
    """
    if isinstance(result, ProvenanceHeader):
        header: Any = result.to_dict()
    elif isinstance(result, dict) and "provenance" in result:
        header = result["provenance"]
    elif isinstance(result, dict):
        header = result
    else:
        raise MissingProvenanceError(
            f"result of type {type(result).__name__} carries no provenance header at all"
        )

    if not isinstance(header, dict):
        raise MissingProvenanceError(
            f"provenance header must be a dict, got {type(header).__name__}"
        )

    missing = [f for f in REQUIRED_FIELDS if f not in header]
    if missing:
        raise MissingProvenanceError(
            f"provenance header missing required field(s): {missing!r} — "
            "a result without a complete header is inadmissible, not a pass"
        )
    if header.get("timestamp_end") is None or header.get("load_avg_end") is None:
        raise MissingProvenanceError(
            "provenance header was never finish()ed (timestamp_end/load_avg_end "
            "unset) — an in-flight or abandoned measurement is not admissible"
        )
    return header


class EnvironmentMismatchError(ProvenanceError):
    """Raised by :func:`require_same_environment` when two headers used different instruments."""


def _field(header: ProvenanceHeader | dict[str, Any], name: str) -> Any:
    if isinstance(header, ProvenanceHeader):
        return getattr(header, name)
    return header.get(name)


def environment_mismatches(
    header_a: ProvenanceHeader | dict[str, Any],
    header_b: ProvenanceHeader | dict[str, Any],
    fields: tuple[str, ...] = DEFAULT_COMPARISON_FIELDS,
) -> dict[str, tuple[Any, Any]]:
    """Return every field in ``fields`` on which ``header_a`` and ``header_b`` disagree.

    Empty dict means the two measurements ran under the same interpreter,
    core count/affinity, and watched environment variables — i.e. they are
    actually comparable. A non-empty dict names exactly the incident-5/6/7
    shape: two runs of "the same" command that were not, in fact, run with
    the same instrument.
    """
    diffs: dict[str, tuple[Any, Any]] = {}
    for name in fields:
        val_a = _field(header_a, name)
        val_b = _field(header_b, name)
        if val_a != val_b:
            diffs[name] = (val_a, val_b)
    return diffs


def require_same_environment(
    header_a: ProvenanceHeader | dict[str, Any],
    header_b: ProvenanceHeader | dict[str, Any],
    fields: tuple[str, ...] = DEFAULT_COMPARISON_FIELDS,
) -> None:
    """Raise :class:`EnvironmentMismatchError` if two measurement headers used different instruments.

    Call this before treating "gate A passed, gate B failed" as a real
    behavioral difference (e.g. a genuine regression) rather than an
    artifact of running under a different venv, core-affinity, or build
    environment variable — exactly what incidents 5 (venv drift), 6
    (taskset vs real CI core count), and 7 (sccache present locally, absent
    in CI) turned out to be.
    """
    diffs = environment_mismatches(header_a, header_b, fields)
    if diffs:
        raise EnvironmentMismatchError(
            "two measurements were not taken with the same instrument — "
            f"mismatched field(s): {diffs!r}. A pass/fail delta between them "
            "is not evidence of a real behavioral difference until this is "
            "resolved (same interpreter, same core affinity, same watched "
            "env vars)."
        )
