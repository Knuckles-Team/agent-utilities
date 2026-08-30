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

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


class ForwardError(RuntimeError):
    """The forwarder could not even launch the canonical target script."""


@dataclass(frozen=True)
class GateSpec:
    """The immutable forwarding contract for one security gate wrapper."""

    prog: str
    target_relative: str
    extra_args: tuple[str, ...] = ()


_GATE_SPECS: dict[str, GateSpec] = {
    "check_compatibility_matrix_gate.py": GateSpec(
        prog="check-compatibility-matrix-gate",
        target_relative="scripts/release/check_compatibility.py",
        extra_args=("--matrix-only",),
    ),
    "check_connector_live_certification_gate.py": GateSpec(
        prog="check-connector-live-certification-gate",
        target_relative="scripts/check_connector_live_certification.py",
        extra_args=("--self-check",),
    ),
    "check_context_compiler_boundary_gate.py": GateSpec(
        prog="check-context-compiler-boundary-gate",
        target_relative="scripts/check_context_compiler_boundary.py",
    ),
    "check_current_only_contract_gate.py": GateSpec(
        prog="check-current-only-contract-gate",
        target_relative="scripts/check_current_only_contract.py",
        extra_args=("--new-only",),
    ),
    "check_exact_artifact_closure_gate.py": GateSpec(
        prog="check-exact-artifact-closure-gate",
        target_relative="scripts/check_exact_artifact_closure.py",
    ),
    "check_exact_local_gates_harness_gate.py": GateSpec(
        prog="check-exact-local-gates-harness-gate",
        target_relative="scripts/check_exact_local_gates_harness.py",
    ),
    "check_external_graph_contract_gate.py": GateSpec(
        prog="check-external-graph-contract-gate",
        target_relative="scripts/check_external_graph_contract.py",
    ),
    "check_http_egress_boundary_gate.py": GateSpec(
        prog="check-http-egress-boundary-gate",
        target_relative="scripts/check_http_egress_boundary.py",
    ),
    "check_native_change_envelope_boundary_gate.py": GateSpec(
        prog="check-native-change-envelope-boundary-gate",
        target_relative="scripts/check_native_change_envelope_boundary.py",
    ),
    "check_native_work_item_boundary_gate.py": GateSpec(
        prog="check-native-work-item-boundary-gate",
        target_relative="scripts/check_native_work_item_boundary.py",
    ),
    "check_no_legacy_markers_gate.py": GateSpec(
        prog="check-no-legacy-markers-gate",
        target_relative="scripts/check_no_legacy_markers.py",
    ),
    "check_production_cell_gate.py": GateSpec(
        prog="check-production-cell-gate",
        target_relative="scripts/deployment/check_production_assets.py",
    ),
    "check_public_graph_boundary_gate.py": GateSpec(
        prog="check-public-graph-boundary-gate",
        target_relative="scripts/check_public_graph_boundary.py",
    ),
    "check_skill_validation_certification_gate.py": GateSpec(
        prog="check-skill-validation-certification-gate",
        target_relative="scripts/check_skill_validation_certification.py",
    ),
    "check_swallowed_errors_gate.py": GateSpec(
        prog="check-swallowed-errors-gate",
        target_relative="scripts/check_swallowed_errors.py",
    ),
    "check_swarm_assets_gate.py": GateSpec(
        prog="check-swarm-assets-gate",
        target_relative="scripts/deployment/check_swarm_assets.py",
        extra_args=("--self-check",),
    ),
    "check_tool_refs_gate.py": GateSpec(
        prog="check-tool-refs-gate",
        target_relative="scripts/check_tool_refs.py",
    ),
}


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


def run_gate(
    *,
    argv: Sequence[str] | None = None,
    prog: str,
    target_relative: str,
    extra_args: Sequence[str] = (),
    failure_target: str | None = None,
) -> int:
    """Run one of the thin fast-tier gate entrypoints.

    The security forwarders all expose the same CLI contract: optionally run
    the forwarding self-check, fail closed when the canonical target cannot be
    launched, relay a target failure unchanged, and emit a JSON verdict.  Keep
    that contract here so each gate only declares its target and any fixed
    arguments. ``failure_target`` preserves a forwarder's established error
    label when it includes fixed arguments in that label.
    """
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    repo_root = args.repository_root.resolve()

    if args.self_check:
        try:
            self_check(repo_root, target_relative)
        except AssertionError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
            return 1

    try:
        rc = forward(
            repository_root=repo_root,
            target_relative=target_relative,
            extra_args=list(extra_args),
        )
    except ForwardError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1

    if rc != 0:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{failure_target or target_relative} exited {rc}",
                    "forwardedTo": target_relative,
                },
                sort_keys=True,
            )
        )
        return rc
    print(
        json.dumps(
            {"ok": True, "forwardedTo": target_relative, "selfCheck": args.self_check},
            sort_keys=True,
        )
    )
    return 0


def bind_gate(
    module_path: str | Path,
    module_name: str,
) -> Callable[[Sequence[str] | None], int]:
    """Bind a wrapper module to its shared CLI entrypoint.

    The returned callable preserves the wrapper's import-time ``main`` API.
    When a wrapper is executed directly, this function invokes that callable
    and exits with its result; importing a wrapper only binds ``main`` and does
    not run a subprocess.
    """
    try:
        spec = _GATE_SPECS[Path(module_path).name]
    except KeyError as exc:
        raise ValueError(
            f"no fast-tier gate specification for {Path(module_path).name!r}"
        ) from exc

    def main(argv: Sequence[str] | None = None) -> int:
        return run_gate(
            argv=argv,
            prog=spec.prog,
            target_relative=spec.target_relative,
            extra_args=spec.extra_args,
        )

    if module_name == "__main__":
        raise SystemExit(main())
    return main
