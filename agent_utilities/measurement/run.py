"""Exit-code-correct process execution (measurement harness, capability D).

CONCEPT:AU-OS.measurement.exit-code-correctness

Direct response to the incident where a gate was run as
``python3 script.py | tail -25`` followed by ``echo "EXIT=$?"`` in a shell —
``$?`` there is ``tail``'s exit status, not the script's. ``tail`` almost
always exits 0, so the gate was reported "exit 0" (pass) when the thing
actually being measured had never had its exit status observed at all.

:func:`run` captures the exit status of the process it launches directly
(``subprocess.run``, no shell, no pipeline) — there is no intermediate
pipeline stage whose exit status could be substituted for the measured
process's own. :func:`run.result` also distinguishes a normal exit code
from termination by signal (negative ``returncode`` from Python's
``subprocess``) and refuses to let a killed process read as code 0.

The companion linter (``scripts/check_measurement_exit_code_antipattern.py``,
built on :func:`scan_for_pipeline_exit_antipattern`) statically flags shell
snippets that reproduce this incident's exact shape.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
import time


class KilledBySignalError(Exception):
    """Raised when a measured process was terminated by a signal, not a normal exit.

    A caller that only wants a pass/fail boolean must not be able to read
    "process was killed" as returncode-0-equivalent; this exception makes
    that case impossible to miss silently. Use :func:`run` with
    ``raise_on_signal=False`` to get the raw (negative) returncode instead.
    """

    def __init__(self, signal_num: int, cmd: list[str]):
        self.signal_num = signal_num
        self.cmd = cmd
        super().__init__(
            f"process killed by signal {signal_num} ({cmd!r}) — "
            "this is NOT a pass; do not coerce it to exit code 0"
        )


@dataclasses.dataclass(frozen=True)
class RunResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    killed_by_signal: int | None  # positive signal number, or None for a normal exit

    @property
    def ok(self) -> bool:
        return self.killed_by_signal is None and self.returncode == 0


def run(
    cmd: list[str],
    *,
    raise_on_signal: bool = True,
    timeout: float | None = None,
    **subprocess_kwargs,
) -> RunResult:
    """Run ``cmd`` and capture ITS exit status — never a pipeline stage's.

    ``cmd`` must be an argv list, not a shell string — this is what
    structurally rules out the ``a | b`` shape whose last-stage exit status
    silently substitutes for the real one. If a caller genuinely needs a
    shell pipeline measured, they must capture ``PIPESTATUS`` (bash) or
    ``pipefail`` themselves; this function does not support ``shell=True``
    at all, on purpose.
    """
    if isinstance(cmd, str):
        raise TypeError(
            "run() requires an argv list, not a shell string — a string "
            "invites exactly the `cmd | tail` pipeline shape this module "
            "exists to make impossible. Pass e.g. ['python3', 'script.py']."
        )
    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        **subprocess_kwargs,
    )
    duration = time.monotonic() - start

    killed_by_signal = -proc.returncode if proc.returncode < 0 else None
    if killed_by_signal is not None and raise_on_signal:
        raise KilledBySignalError(killed_by_signal, cmd)

    return RunResult(
        cmd=list(cmd),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_s=duration,
        killed_by_signal=killed_by_signal,
    )


# --- D linter: statically flag the `cmd | tail` + `$?` shape -----------------

# A pipeline whose LAST stage is a filter command commonly (and wrongly)
# assumed to be exit-status-transparent. This is a heuristic allow-list of
# the incident's own shape (`tail`) plus its obvious siblings — it is not a
# full shell parser, deliberately: a real parser is not dependency-light and
# the incident shape is simple text to match.
_FILTER_TAIL_CMDS = (
    r"(?:tail|head|grep|egrep|fgrep|sed|awk|sort|uniq|wc|cat|column|cut|tr)"
)
_PIPE_TO_FILTER_RE = re.compile(r"\|\s*" + _FILTER_TAIL_CMDS + r"\b")
_DOLLAR_QUESTION_RE = re.compile(r"\$\?")
_PIPESTATUS_RE = re.compile(r"PIPESTATUS|pipefail")

# How many lines after a flagged pipe to look for a `$?` read before giving up.
_LOOKAHEAD_LINES = 5


@dataclasses.dataclass(frozen=True)
class AntipatternHit:
    line_no: int  # 1-based, of the `| tail`-shaped pipeline
    pipe_line: str
    dollar_question_line_no: int
    dollar_question_line: str


def scan_for_pipeline_exit_antipattern(text: str) -> list[AntipatternHit]:
    """Return every ``cmd | tail`` (or sibling) + later bare ``$?`` shape found in ``text``.

    A hit requires BOTH: a pipeline ending in a known filter command, and a
    ``$?`` read within :data:`_LOOKAHEAD_LINES` lines after it with no
    intervening ``PIPESTATUS``/``pipefail`` guard. Lines that guard
    correctly (``set -o pipefail`` before the pipe, or reading
    ``${PIPESTATUS[0]}`` instead of ``$?``) are not flagged.
    """
    lines = text.splitlines()
    hits: list[AntipatternHit] = []
    for i, line in enumerate(lines):
        if not _PIPE_TO_FILTER_RE.search(line):
            continue
        window = lines[i : i + 1 + _LOOKAHEAD_LINES]
        window_text = "\n".join(window)
        if _PIPESTATUS_RE.search(window_text):
            continue
        for j, wline in enumerate(window):
            if j == 0:
                continue  # the pipe line itself
            if _DOLLAR_QUESTION_RE.search(wline) and not _PIPESTATUS_RE.search(wline):
                hits.append(
                    AntipatternHit(
                        line_no=i + 1,
                        pipe_line=line,
                        dollar_question_line_no=i + 1 + j,
                        dollar_question_line=wline,
                    )
                )
                break
    return hits
