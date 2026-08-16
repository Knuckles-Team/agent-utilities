"""Capability F proof: safe process targeting (incident 8).

Incident: `pgrep -f "<pattern>"` matched the invoking shell's own command
line (the pattern string was present in the caller's own argv/heredoc) and
killed the caller. Proves, with a synthetic (injected) process table so the
result is deterministic rather than depending on real /proc contents:
(1) a pattern that matches BOTH the caller and an unrelated target process
excludes the caller (and its ancestor chain) and only ever targets the
other process; (2) a caller-only match (nothing else running) raises
instead of self-selecting; (3) require_token_match rejects a pattern that
only appears as a substring of a single (real, unsplit) argv element.

``ProcCandidate.argv`` is a tuple of the REAL argv elements, exactly as the
kernel delivers them via ``/proc/<pid>/cmdline`` (NUL-separated, boundaries
preserved) — not a re-split of a flattened string. That distinction matters
for the last case: a shell invocation like
``bash -c "echo 'looking for measure_target.py in logs'"`` has that entire
sentence as ONE argv element, not several words worth of separate
arguments.
"""

from __future__ import annotations

import pytest

from agent_utilities.measurement.proc_safety import (
    AmbiguousMatchError,
    ProcCandidate,
    find_candidates,
    kill_by_pattern,
)

PATTERN = "measure_target.py"


def _table_with_self_and_target(self_pid: int = 100, shell_pid: int = 1) -> list[ProcCandidate]:
    return [
        # init
        ProcCandidate(pid=1, ppid=0, argv=("/sbin/init",)),
        # the invoking shell (ancestor of self)
        ProcCandidate(pid=shell_pid, ppid=1, argv=("/bin/bash",)),
        # the CALLING process itself -- its own command line happens to
        # MENTION the pattern (e.g. it built the pattern string into its own
        # argv), exactly like incident 8.
        ProcCandidate(pid=self_pid, ppid=shell_pid, argv=("python3", "orchestrator.py", "--watch", PATTERN)),
        # the actual intended target
        ProcCandidate(pid=200, ppid=1, argv=("python3", PATTERN)),
    ]


def test_incident_8_self_is_excluded_even_though_pattern_matches_self():
    table = _table_with_self_and_target()
    matched, excluded = find_candidates(
        PATTERN, proc_table=table, self_pid=100, require_token_match=False
    )
    assert 100 not in [c.pid for c in matched]
    assert 100 in excluded
    assert [c.pid for c in matched] == [200]


def test_incident_8_ancestor_shell_is_also_excluded():
    """If the pattern also happened to appear in the parent shell's own
    argv, the ancestor chain must be excluded too, not just the direct
    caller pid."""
    shell_pid = 1
    table_with_shell_match = [
        ProcCandidate(pid=1, ppid=0, argv=("/sbin/init",)),
        ProcCandidate(pid=shell_pid, ppid=1, argv=("/bin/bash", "-c", f"...{PATTERN}...")),
        ProcCandidate(pid=100, ppid=shell_pid, argv=("python3", "orchestrator.py", "--watch", PATTERN)),
        ProcCandidate(pid=200, ppid=1, argv=("python3", PATTERN)),
    ]
    matched, excluded = find_candidates(
        PATTERN, proc_table=table_with_shell_match, self_pid=100, require_token_match=False
    )
    assert shell_pid in excluded
    assert [c.pid for c in matched] == [200]


def test_kill_by_pattern_never_targets_the_caller(monkeypatch):
    table = _table_with_self_and_target()
    killed_pids: list[int] = []

    def fake_kill(pid: int, sig: int) -> None:
        killed_pids.append(pid)

    result = kill_by_pattern(
        PATTERN,
        proc_table=table,
        self_pid=100,
        dry_run=False,
        require_token_match=False,
        kill_fn=fake_kill,
    )
    assert killed_pids == [200]
    assert 100 not in killed_pids
    assert result.excluded_pids and 100 in result.excluded_pids


def test_kill_by_pattern_refuses_when_only_the_caller_matches(monkeypatch):
    """If the ONLY process mentioning the pattern is the caller itself, the
    correct behavior is zero targets -- not falling back to killing self."""
    table = [
        ProcCandidate(pid=1, ppid=0, argv=("/sbin/init",)),
        ProcCandidate(pid=100, ppid=1, argv=("python3", "orchestrator.py", "--watch", PATTERN)),
    ]
    killed_pids: list[int] = []
    result = kill_by_pattern(
        PATTERN,
        proc_table=table,
        self_pid=100,
        dry_run=False,
        require_token_match=False,
        kill_fn=lambda pid, sig: killed_pids.append(pid),
    )
    assert killed_pids == []
    assert result.matched == ()


def test_ambiguous_match_refused_without_allow_multiple():
    table = [
        ProcCandidate(pid=1, ppid=0, argv=("/sbin/init",)),
        ProcCandidate(pid=200, ppid=1, argv=("python3", PATTERN)),
        ProcCandidate(pid=201, ppid=1, argv=("python3", PATTERN, "--replica")),
    ]
    with pytest.raises(AmbiguousMatchError):
        kill_by_pattern(PATTERN, proc_table=table, self_pid=999, require_token_match=False)


def test_require_token_match_rejects_a_bare_substring_mention():
    """A process whose SOLE argv element merely MENTIONS the pattern inside
    a sentence (the real /proc/<pid>/cmdline shape of `bash -c "..."`) must
    not match when require_token_match=True."""
    table = [
        ProcCandidate(pid=1, ppid=0, argv=("/sbin/init",)),
        ProcCandidate(
            pid=300,
            ppid=1,
            argv=("/bin/bash", "-c", f"echo 'looking for {PATTERN} in logs'"),
        ),
    ]
    matched, _ = find_candidates(PATTERN, proc_table=table, self_pid=999, require_token_match=True)
    assert matched == []


def test_require_token_match_accepts_a_real_invocation():
    table = [
        ProcCandidate(pid=1, ppid=0, argv=("/sbin/init",)),
        ProcCandidate(pid=300, ppid=1, argv=("python3", PATTERN, "--flag")),
    ]
    matched, _ = find_candidates(PATTERN, proc_table=table, self_pid=999, require_token_match=True)
    assert [c.pid for c in matched] == [300]


def test_dry_run_is_the_default_and_signals_nothing():
    table = _table_with_self_and_target()
    calls = []
    result = kill_by_pattern(
        PATTERN,
        proc_table=table,
        self_pid=100,
        require_token_match=False,
        kill_fn=lambda pid, sig: calls.append(pid),
    )
    assert result.dry_run is True
    assert calls == []
    assert result.killed == ()
    assert [c.pid for c in result.matched] == [200]


def test_cmdline_property_joins_argv_for_display():
    cand = ProcCandidate(pid=1, ppid=0, argv=("python3", "foo.py", "--flag"))
    assert cand.cmdline == "python3 foo.py --flag"
