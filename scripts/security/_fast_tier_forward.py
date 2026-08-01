"""Shared subprocess-forwarding helper for merge-queue fast-tier forwarder gates.

CONCEPT:AU-OS.governance.tiered-merge-gate — D-ML-1.

**Why forwarders exist.** ``.github/workflows/guardrails.yml`` and
``.github/workflows/security.yml`` trigger on ``push`` (and matching-path
``pull_request``) — both dark since 2026-07-28 (D-ML-1: nothing has been
pushed; ``origin/main`` is 415+ commits behind local ``main``, so every
guardrail those workflows run has been silently un-gating every merge since).
The operator's decision (D-ML-1) is to port the checks those workflows run
into the merge queue's fast tier instead of restoring push CI.

The fast tier (``agent_utilities/governance/merge_queue.py``) discovers gates
by globbing ``scripts/security/check_*.py`` **only** — a directory-scoped
glob, deliberately, so it can't accidentally pick up an unrelated top-level
script. Several of the dark workflows' checks are healthy, already-reviewed
scripts that simply live *outside* that directory (``scripts/deployment/``,
``scripts/release/``, or bare ``scripts/``), so the queue never sees them.

Two ways to fix that: move the canonical scripts into ``scripts/security/``
(but ``guardrails.yml`` still references their current paths, and moving
them would need updating that dark-but-not-dead workflow too, for zero
present benefit), or teach the queue a second discovery path (the module's
own docstring: "prefer adding checks there over modifying the queue's own
code"). Both are worse than the third option this module implements: a tiny
forwarder file *in* ``scripts/security/`` that runs the existing, unmodified
canonical script as a subprocess and relays its verdict byte-for-byte. The
canonical script stays exactly where ``guardrails.yml`` expects it — so
restoring push CI later needs no further change — and the fast tier gets a
``check_*.py`` it can discover.

**Fail-closed vs. an honest absence.** If the canonical target script is
missing, or the interpreter can't even be launched, that is a *degraded
read* (this repo's codified distinction — see
``agent_utilities/governance/merge_queue.py``'s ``run_contract_checks``
docstring) and :func:`forward` raises :class:`ForwardError` rather than
reporting a clean pass. A target that runs and exits 0 is a genuine pass;
a target that exits nonzero is relayed as nonzero, never swallowed.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


class ForwardError(RuntimeError):
    """The forwarder could not even launch the canonical target script."""


def forward(
    *,
    repository_root: Path,
    target_relative: str,
    extra_args: list[str] | None = None,
    timeout: int = 55,
) -> int:
    """Run ``target_relative`` (a path relative to *repository_root*) as a
    subprocess with *extra_args*, and relay its exit code unchanged.

    ``timeout`` defaults just under the merge queue's own
    ``CONTRACT_CHECK_BUDGET_SECONDS`` (60s, see ``merge_queue.py``) so a hang
    here is reported as *this forwarder* timing out — naming the target it
    was waiting on — rather than as an opaque budget-exceeded from the
    caller.
    """
    target = (repository_root / target_relative).resolve()
    if not target.is_file():
        raise ForwardError(
            f"canonical target script is missing: {target} "
            f"(repository_root={repository_root}) — a fast-tier gate that "
            "cannot find what it delegates to must refuse, not pass"
        )
    argv = [sys.executable, str(target), *(extra_args or [])]
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv,
            cwd=str(repository_root),
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ForwardError(f"could not run canonical target {target}: {exc}") from exc
    return proc.returncode


def self_check(repository_root: Path, target_relative: str) -> None:
    """Prove the forwarding contract this module gives every ``check_*.py``
    forwarder in this directory:

    1. a real target that exits 0 is relayed as 0;
    2. a real target that exits 1 is relayed as 1 — **not swallowed**, which
       is exactly the masking this gate must never do;
    3. a missing target fails closed (raises :class:`ForwardError`), rather
       than reporting a clean pass because it found nothing to run; and
    4. the caller's declared canonical target actually exists in
       *repository_root* — so a forwarder can't silently point at a typo'd
       or since-moved path forever and always "pass" via case 3 without
       anyone noticing it stopped checking anything.

    Raises :class:`AssertionError` (not :class:`ForwardError`) on failure —
    a self-check failure is a defect in *this* module, distinct from the
    fail-closed behavior it is verifying.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        (tmp_root / "good.py").write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
        (tmp_root / "bad.py").write_text("import sys\nsys.exit(1)\n", encoding="utf-8")

        rc_good = forward(repository_root=tmp_root, target_relative="good.py")
        if rc_good != 0:
            raise AssertionError(
                f"self-check: a passing target was not relayed as 0 (got {rc_good})"
            )

        rc_bad = forward(repository_root=tmp_root, target_relative="bad.py")
        if rc_bad == 0:
            raise AssertionError(
                "self-check: a FAILING target was relayed as 0 — a forwarder "
                "that swallows a real failure is exactly the masking this "
                "gate must never do"
            )

        try:
            forward(repository_root=tmp_root, target_relative="does_not_exist.py")
        except ForwardError:
            pass
        else:
            raise AssertionError(
                "self-check: forwarding to a missing target did not fail closed"
            )

    target = (repository_root / target_relative).resolve()
    if not target.is_file():
        raise AssertionError(
            f"self-check: the declared canonical target does not exist: "
            f"{target} — this forwarder's TARGET is stale"
        )
