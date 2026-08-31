#!/usr/bin/env python3
"""Run the repository's governance contract scripts as one bounded gate.

The merge queue and contributor hook share this entry point so discovery,
parallelism, diagnostics, and failure behavior cannot drift.  Child processes
run concurrently, but their failure reports are emitted in stable script-name
order.  Each diagnostic line is attributed to its script so a line-based queue
comparison can see a new finding from a check that was already failing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

CHECK_GLOB = "check_*.py"
MAX_WORKERS = 4
RUN_BUDGET_SECONDS = 55.0


class ContractRunnerError(RuntimeError):
    """Contract discovery could not produce a trustworthy run set."""


@dataclass(frozen=True)
class ContractResult:
    """Captured outcome for one governance contract script."""

    relative_path: str
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None and not self.timed_out


def _captured_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _discover(
    repository_root: Path, excluded: Collection[str]
) -> list[tuple[str, Path]]:
    contract_dir = repository_root / "scripts" / "security"
    if not contract_dir.is_dir():
        raise ContractRunnerError(f"contract directory is missing: {contract_dir}")

    scripts = sorted(contract_dir.glob(CHECK_GLOB), key=lambda path: path.name)
    if not scripts:
        raise ContractRunnerError(
            f"no governance contract scripts matched {contract_dir / CHECK_GLOB}"
        )

    discovered = {path.name for path in scripts}
    unknown = sorted(set(excluded) - discovered)
    if unknown:
        raise ContractRunnerError(
            "excluded contract script(s) were not discovered: " + ", ".join(unknown)
        )

    return [
        (path.relative_to(repository_root).as_posix(), path)
        for path in scripts
        if path.name not in excluded
    ]


def _run_one(
    relative_path: str,
    *,
    repository_root: Path,
    interpreter: str,
    deadline: float,
    environment: dict[str, str],
) -> ContractResult:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return ContractResult(relative_path=relative_path, timed_out=True)

    try:
        completed = subprocess.run(
            [interpreter, relative_path],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=remaining,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        return ContractResult(
            relative_path=relative_path,
            stdout=_captured_text(exc.stdout),
            stderr=_captured_text(exc.stderr),
            timed_out=True,
        )
    except OSError as exc:
        return ContractResult(relative_path=relative_path, error=str(exc))

    return ContractResult(
        relative_path=relative_path,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_contract_checks(
    repository_root: Path,
    *,
    excluded: Collection[str] = (),
    interpreter: str = sys.executable,
    max_workers: int = MAX_WORKERS,
    timeout_seconds: float = RUN_BUDGET_SECONDS,
) -> list[ContractResult]:
    """Run discovered contracts concurrently and return them in stable order."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    root = repository_root.resolve()
    scripts = _discover(root, excluded)
    deadline = time.monotonic() + timeout_seconds
    child_environment = os.environ.copy()
    child_environment["AGENT_UTILITIES_TESTING"] = "true"
    with ThreadPoolExecutor(max_workers=min(max_workers, len(scripts))) as pool:
        futures = {
            relative_path: pool.submit(
                _run_one,
                relative_path,
                repository_root=root,
                interpreter=interpreter,
                deadline=deadline,
                environment=child_environment,
            )
            for relative_path, _path in scripts
        }
        return [futures[relative_path].result() for relative_path, _path in scripts]


def format_failures(results: Sequence[ContractResult]) -> str:
    """Render complete child diagnostics in deterministic per-script blocks."""
    lines: list[str] = []
    for result in results:
        if result.ok:
            continue
        if result.timed_out:
            lines.append(f"TIMEOUT {result.relative_path}")
        elif result.error is not None:
            lines.append(f"ERROR {result.relative_path}: {result.error}")
        else:
            lines.append(f"FAIL {result.relative_path} exit={result.returncode}")

        diagnostics = False
        for stream_name, content in (
            ("stdout", result.stdout),
            ("stderr", result.stderr),
        ):
            for line in content.splitlines():
                diagnostics = True
                lines.append(f"{result.relative_path} {stream_name}: {line}")
        if not diagnostics and result.error is None:
            lines.append(f"{result.relative_path} diagnostic: child produced no output")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SCRIPT_NAME",
        help="Exclude one discovered check_*.py basename (repeatable).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        results = run_contract_checks(
            args.repository_root,
            excluded=args.exclude,
        )
    except (ContractRunnerError, ValueError) as exc:
        print(f"ERROR contract-checks: {exc}")
        return 1

    report = format_failures(results)
    if report:
        print(report)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
