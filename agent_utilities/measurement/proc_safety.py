"""Safe process targeting (measurement harness, capability F).

CONCEPT:AU-OS.measurement.safe-process-targeting

Direct response to the incident where ``pgrep -f "<pattern>"`` matched the
INVOKING shell's own command line (the pattern string was present in the
caller's own argv/heredoc, e.g. because the calling script echoed or built
the pattern it was about to search for) and killed the caller mid-run.
``pgrep -f`` / ``pkill -f`` match against the full command line of every
process, with no built-in concept of "not me" or "not my parent shell".

:func:`kill_by_pattern` fixes this by construction: it (1) enumerates
candidates via ``/proc`` (or an injected process table, for deterministic
tests), (2) excludes the caller's own PID and every ancestor PID up the
parent chain, (3) re-verifies each surviving candidate's
``/proc/<pid>/cmdline`` actually names the intended program as a real
token — not merely a substring that could appear inside an unrelated shell
invocation that happens to mention it — and (4) refuses to act at all when
more than one non-excluded candidate remains, unless the caller explicitly
opts into killing multiple matches.
"""

from __future__ import annotations

import dataclasses
import os
import signal as signal_module
from collections.abc import Callable, Iterable
from pathlib import Path


class AmbiguousMatchError(Exception):
    """Raised when more than one candidate matches and multi-kill was not requested."""


@dataclasses.dataclass(frozen=True)
class ProcCandidate:
    pid: int
    ppid: int
    argv: tuple[str, ...]  # the REAL NUL-separated argv, boundaries preserved

    @property
    def cmdline(self) -> str:
        """Space-joined argv, for substring matching/display only.

        Joining loses argv boundaries (an argument containing a space is
        indistinguishable from two arguments) -- that is exactly why token
        matching (:func:`_cmdline_names_program`) must use ``argv``
        directly instead of re-splitting this string. Keep using
        ``cmdline`` only for plain substring checks.
        """
        return " ".join(self.argv)


@dataclasses.dataclass(frozen=True)
class KillResult:
    pattern: str
    excluded_pids: tuple[int, ...]  # caller + ancestors, removed before matching
    matched: tuple[ProcCandidate, ...]  # candidates that WOULD be / were targeted
    killed: tuple[int, ...]  # pids actually signaled (empty if dry_run)
    dry_run: bool


# --- process table access, overridable for deterministic tests --------------

ProcTableFn = Callable[[], Iterable[ProcCandidate]]


def _read_argv(pid: int, proc_root: Path) -> tuple[str, ...] | None:
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    argv = tuple(p for p in raw.decode(errors="replace").split("\0") if p)
    return argv or None


def _read_ppid(pid: int, proc_root: Path) -> int | None:
    try:
        stat = (proc_root / str(pid) / "stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    # Field 4 is ppid, but field 2 (comm) is parenthesized and may itself
    # contain spaces/parens, so split on the LAST ')' before tokenizing.
    after_comm = stat.rsplit(")", 1)[-1].split()
    if len(after_comm) < 2:
        return None
    try:
        return int(after_comm[1])
    except ValueError:
        return None


def live_proc_table(proc_root: Path = Path("/proc")) -> list[ProcCandidate]:
    """Enumerate real processes visible under ``proc_root`` (default ``/proc``)."""
    candidates: list[ProcCandidate] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        argv = _read_argv(pid, proc_root)
        if not argv:
            continue
        ppid = _read_ppid(pid, proc_root)
        if ppid is None:
            continue
        candidates.append(ProcCandidate(pid=pid, ppid=ppid, argv=argv))
    return candidates


def ancestor_pids(
    start_pid: int,
    table: Iterable[ProcCandidate],
    *,
    max_depth: int = 128,
) -> set[int]:
    """Walk the parent chain from ``start_pid`` up to pid 1 (or table exhaustion).

    Returns the ancestor PIDs (not including ``start_pid`` itself — callers
    that also want to exclude the starting process should add it
    separately). ``max_depth`` bounds a pathological ppid cycle.
    """
    by_pid = {c.pid: c for c in table}
    ancestors: set[int] = set()
    current = start_pid
    for _ in range(max_depth):
        cand = by_pid.get(current)
        if cand is None or cand.ppid == current or cand.ppid <= 0:
            break
        ancestors.add(cand.ppid)
        current = cand.ppid
        if current == 1:
            break
    return ancestors


def _cmdline_names_program(argv: tuple[str, ...], program: str) -> bool:
    """True if ``program`` is a real argv ELEMENT of ``argv``, not a bare substring.

    Operates on the REAL argv tuple (kernel-delivered NUL-separated
    boundaries), not a re-split of a space-joined string — re-splitting a
    joined ``cmdline`` cannot tell "one argument containing a space" apart
    from "two arguments", which is exactly how a shell invocation like
    ``bash -c "echo 'looking for foo.py in logs'"`` (a SINGLE argv element
    that happens to contain the program name inside a sentence) would be
    misread as naming the program. Matching against real argv elements
    guards against precisely that: that whole sentence is one argv token,
    equal to none of ``foo.py``/``.../foo.py``/basename ``foo.py``, so it
    correctly does not match.
    """
    return any(
        tok == program or tok.endswith("/" + program) or Path(tok).name == program
        for tok in argv
    )


def find_candidates(
    pattern: str,
    *,
    proc_table: Iterable[ProcCandidate] | None = None,
    self_pid: int | None = None,
    require_token_match: bool = False,
) -> tuple[list[ProcCandidate], tuple[int, ...]]:
    """Find processes matching ``pattern``, with the caller + its ancestors pre-excluded.

    Returns ``(matched_candidates, excluded_pids)``. ``proc_table`` and
    ``self_pid`` are injectable so tests can supply a synthetic process
    table instead of depending on real ``/proc`` contents (which are
    inherently racy and environment-specific) — this is what makes the
    "does not kill its own caller" property mechanically provable rather
    than merely likely.

    With ``require_token_match=True``, matching additionally requires
    ``pattern`` to appear as a whole argv token (see
    :func:`_cmdline_names_program`) rather than any substring match — use
    this when ``pattern`` is a program/script name, to rule out a process
    whose command line merely *mentions* the name in an unrelated argument.
    """
    table = list(proc_table) if proc_table is not None else live_proc_table()
    me = self_pid if self_pid is not None else os.getpid()
    ancestors = ancestor_pids(me, table)
    excluded = frozenset({me}) | ancestors

    matched = []
    for cand in table:
        if cand.pid in excluded:
            continue
        if pattern not in cand.cmdline:
            continue
        if require_token_match and not _cmdline_names_program(cand.argv, pattern):
            continue
        matched.append(cand)
    return matched, tuple(sorted(excluded))


def kill_by_pattern(
    pattern: str,
    *,
    signal: int = signal_module.SIGTERM,
    dry_run: bool = True,
    allow_multiple: bool = False,
    proc_table: Iterable[ProcCandidate] | None = None,
    self_pid: int | None = None,
    require_token_match: bool = True,
    kill_fn: Callable[[int, int], None] = os.kill,
) -> KillResult:
    """Signal every process matching ``pattern`` — safely.

    Safety properties, all enforced here rather than left to the caller:

    * The calling process, and every one of its ancestors up the parent
      chain, is excluded from matching BEFORE the pattern is even applied —
      this is what prevents the incident-8 shape (a pattern that happens to
      appear in the caller's own command line) from ever reaching the kill
      step.
    * ``require_token_match`` (default ``True``) additionally requires the
      pattern to name a real argv token, not merely appear as a substring
      of an unrelated argument.
    * Unless ``allow_multiple=True``, more than one surviving candidate
      raises :class:`AmbiguousMatchError` instead of killing all of them.
    * ``dry_run=True`` by default: callers must opt in to actually signal
      anything. The returned :class:`KillResult` always reports what WOULD
      be (or was) matched, so a dry run is fully informative on its own.
    """
    matched, excluded = find_candidates(
        pattern,
        proc_table=proc_table,
        self_pid=self_pid,
        require_token_match=require_token_match,
    )

    if len(matched) > 1 and not allow_multiple:
        raise AmbiguousMatchError(
            f"pattern {pattern!r} matched {len(matched)} processes "
            f"({[c.pid for c in matched]}) after excluding self+ancestors "
            f"{excluded}; pass allow_multiple=True to signal all of them"
        )

    killed: list[int] = []
    if not dry_run:
        for cand in matched:
            kill_fn(cand.pid, signal)
            killed.append(cand.pid)

    return KillResult(
        pattern=pattern,
        excluded_pids=excluded,
        matched=tuple(matched),
        killed=tuple(killed),
        dry_run=dry_run,
    )
