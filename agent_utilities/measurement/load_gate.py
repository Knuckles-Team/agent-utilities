"""Load gating for measurements (measurement harness, capability B).

CONCEPT:AU-OS.measurement.load-gate

Two incidents motivate this module directly:

* A test suite measured at load average 15.82 reported "2 failed / 10
  errors"; the *same* suite at load 4.03 reported "1 failed / 382 passed"
  and ran 7x faster. Load was never asserted, so the first number was
  reported as a real regression.
* Separately, this host's own incident history records a load average of
  ~62 with swap exhausted on a 24-core box (ratio ~2.6x cores) — the
  documented danger zone this module exists to keep a measurement out of.

The fix is not "measure load and report it" (the 15.82 run already *could*
have reported its load — nothing stopped that number from being generated
and believed). The fix is refusing to emit a pass/fail verdict at all above
a threshold, replacing it with a distinct status that cannot be mistaken for
a number. ``TOO_LOADED_TO_MEASURE`` is a sentinel string, not a score — code
that compares it against a numeric threshold or truthy-checks it should
fail loudly, not quietly pass.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Literal

TOO_LOADED_TO_MEASURE = "TOO_LOADED_TO_MEASURE"

Status = Literal["OK", "TOO_LOADED_TO_MEASURE", "UNKNOWN_LOAD"]


class TooLoadedToMeasureError(Exception):
    """Raised by :func:`gate_or_raise` when load exceeds the configured threshold.

    Deliberately a distinct exception type (not a return value, not a
    number) so a caller cannot accidentally coerce an abort into a falsy
    "0 failures" pass. See module docstring: a killed/aborted run must
    never read as a pass.
    """

    def __init__(self, load1: float, threshold: float):
        self.load1 = load1
        self.threshold = threshold
        super().__init__(
            f"{TOO_LOADED_TO_MEASURE}: load average {load1:.2f} exceeds "
            f"threshold {threshold:.2f} — refusing to emit a verdict"
        )


def default_threshold(cpu_count: int | None = None) -> float:
    """Derive a default load-average threshold from core count.

    One times the core count (load == fully busy, no queueing) is the
    textbook "saturated" line; this uses a slightly looser 1.5x to avoid
    false-flagging ordinary CI parallelism, while staying an order of
    magnitude below the ~2.6x-cores/swap-exhausted incident this module
    exists to prevent. Override with ``MEASUREMENT_LOAD_THRESHOLD`` (an
    absolute load-average number) when a host's normal operating load
    differs.
    """
    from agent_utilities.core._env import setting

    override = setting("MEASUREMENT_LOAD_THRESHOLD")
    if override is not None:
        return float(override)
    n = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    return 1.5 * n


@dataclasses.dataclass(frozen=True)
class LoadGateResult:
    status: Status
    load1: float | None
    load5: float | None
    load15: float | None
    threshold: float

    @property
    def ok(self) -> bool:
        return self.status == "OK"


def _read_loadavg() -> tuple[float, float, float] | None:
    try:
        return tuple(os.getloadavg())  # type: ignore[return-value]
    except (OSError, AttributeError):
        return None


def check_load(threshold: float | None = None) -> LoadGateResult:
    """Read current load average and classify it against ``threshold``.

    Never raises. Returns a :class:`LoadGateResult` whose ``status`` is one
    of the three ``Status`` literals — in particular ``UNKNOWN_LOAD`` (not
    ``"OK"``) on a platform where load average cannot be read, so "we
    couldn't tell" is never silently treated as "load is fine".
    """
    t = threshold if threshold is not None else default_threshold()
    avg = _read_loadavg()
    if avg is None:
        return LoadGateResult(
            status="UNKNOWN_LOAD", load1=None, load5=None, load15=None, threshold=t
        )
    load1, load5, load15 = avg
    status: Status = "OK" if load1 <= t else "TOO_LOADED_TO_MEASURE"
    return LoadGateResult(
        status=status, load1=load1, load5=load5, load15=load15, threshold=t
    )


def gate_or_raise(threshold: float | None = None) -> LoadGateResult:
    """Like :func:`check_load`, but raise :class:`TooLoadedToMeasureError` on overload.

    This is the entry point measurement callers should use immediately
    before emitting a verdict — it makes "too loaded" a control-flow event
    (an exception a caller must handle or propagate) rather than a field a
    caller can forget to check.
    """
    result = check_load(threshold)
    if result.status == "TOO_LOADED_TO_MEASURE":
        raise TooLoadedToMeasureError(result.load1, result.threshold)  # type: ignore[arg-type]
    return result
