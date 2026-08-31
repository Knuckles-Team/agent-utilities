"""Focused tests for the shared governance contract aggregate runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPOSITORY_ROOT / "scripts" / "security" / "run_contract_checks.py"
SPEC = importlib.util.spec_from_file_location("run_security_contract_checks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _contract(repository_root: Path, name: str, source: str) -> None:
    directory = repository_root / "scripts" / "security"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(source, encoding="utf-8")


def test_known_good_contracts_pass(tmp_path: Path) -> None:
    _contract(tmp_path, "check_zulu.py", "print('zulu ok')\n")
    _contract(tmp_path, "check_alpha.py", "print('alpha ok')\n")

    results = RUNNER.run_contract_checks(tmp_path, max_workers=2)

    assert [result.relative_path for result in results] == [
        "scripts/security/check_alpha.py",
        "scripts/security/check_zulu.py",
    ]
    assert all(result.ok for result in results)
    assert RUNNER.format_failures(results) == ""
    assert RUNNER.main(["--repository-root", str(tmp_path)]) == 0


def test_children_use_the_hermetic_test_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")
    _contract(
        tmp_path,
        "check_environment.py",
        "import os\nprint(os.environ['AGENT_UTILITIES_TESTING'])\n",
    )

    result = RUNNER.run_contract_checks(tmp_path)[0]

    assert result.ok
    assert result.stdout.strip() == "true"


def test_known_bad_contract_preserves_stdout_and_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _contract(
        tmp_path,
        "check_bad.py",
        "import sys\nprint('finding: bad edge')\n"
        "print('repair: reconnect edge', file=sys.stderr)\nsys.exit(3)\n",
    )

    assert RUNNER.main(["--repository-root", str(tmp_path)]) == 1
    assert capsys.readouterr().out.splitlines() == [
        "FAIL scripts/security/check_bad.py exit=3",
        "scripts/security/check_bad.py stdout: finding: bad edge",
        "scripts/security/check_bad.py stderr: repair: reconnect edge",
    ]


def test_parallel_completion_is_rendered_deterministically(tmp_path: Path) -> None:
    _contract(
        tmp_path,
        "check_alpha.py",
        "import sys, time\ntime.sleep(0.08)\nprint('alpha')\nsys.exit(1)\n",
    )
    _contract(
        tmp_path,
        "check_beta.py",
        "import sys\nprint('beta')\nsys.exit(1)\n",
    )

    reports = [
        RUNNER.format_failures(RUNNER.run_contract_checks(tmp_path, max_workers=2))
        for _attempt in range(2)
    ]

    assert reports[0] == reports[1]
    assert reports[0].splitlines() == [
        "FAIL scripts/security/check_alpha.py exit=1",
        "scripts/security/check_alpha.py stdout: alpha",
        "FAIL scripts/security/check_beta.py exit=1",
        "scripts/security/check_beta.py stdout: beta",
    ]


def test_timeout_and_launch_error_fail_closed(tmp_path: Path) -> None:
    _contract(
        tmp_path,
        "check_slow.py",
        "import time\ntime.sleep(10)\n",
    )
    timed_out = RUNNER.run_contract_checks(
        tmp_path, max_workers=1, timeout_seconds=0.05
    )
    assert timed_out[0].timed_out is True
    assert timed_out[0].ok is False
    assert RUNNER.format_failures(timed_out).splitlines() == [
        "TIMEOUT scripts/security/check_slow.py",
        "scripts/security/check_slow.py diagnostic: child produced no output",
    ]

    launch_error = RUNNER.run_contract_checks(
        tmp_path,
        interpreter=str(tmp_path / "missing-python"),
        max_workers=1,
    )
    assert launch_error[0].error is not None
    assert launch_error[0].ok is False
    assert RUNNER.format_failures(launch_error).startswith(
        "ERROR scripts/security/check_slow.py:"
    )


def test_unknown_exclusion_fails_closed(tmp_path: Path) -> None:
    _contract(tmp_path, "check_present.py", "pass\n")

    with pytest.raises(RUNNER.ContractRunnerError, match="were not discovered"):
        RUNNER.run_contract_checks(tmp_path, excluded={"check_missing.py"})


def test_queue_and_contributor_hook_use_the_shared_runner() -> None:
    queue_config = (REPOSITORY_ROOT / ".mergequeue.yaml").read_text(encoding="utf-8")
    hook_config = (REPOSITORY_ROOT / ".pre-commit-config.yaml").read_text(
        encoding="utf-8"
    )

    assert (
        'command: [".venv/bin/python", "scripts/security/run_contract_checks.py"]'
        in queue_config
    )
    hook = hook_config.split("- id: contract-checks", 1)[1].split("- id:", 1)[0]
    assert "entry: python3 scripts/security/run_contract_checks.py" in hook
    assert "--exclude check_current_only_contract_gate.py" in hook
    assert "--exclude check_sbom_licenses.py" in hook
    assert "python3 -c" not in hook
