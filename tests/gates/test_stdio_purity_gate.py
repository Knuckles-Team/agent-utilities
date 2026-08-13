"""Meta-proof for the served-MCP-surface stdout-write gate (B-19).

Both directions, per the program's own standing rule ("a gate never
demonstrated against a known-bad input is not evidence"): the gate must FAIL
on a deliberately reintroduced ``print()``/``sys.stdout.write()`` and PASS once
it is removed.
"""

from __future__ import annotations

from pathlib import Path

from scripts.check_no_stdout_writes import validate


def test_bare_print_in_served_module_is_rejected(tmp_path: Path) -> None:
    package = tmp_path / "agent_utilities" / "mcp"
    package.mkdir(parents=True)
    (package / "leaky.py").write_text(
        'def handle_request() -> None:\n    print("this corrupts the JSON-RPC frame stream")\n',
        encoding="utf-8",
    )

    errors = validate(package)

    assert len(errors) == 1
    assert "leaky.py:2" in errors[0]
    assert "print(...)" in errors[0]


def test_explicit_sys_stdout_write_is_rejected(tmp_path: Path) -> None:
    package = tmp_path / "agent_utilities" / "mcp"
    package.mkdir(parents=True)
    (package / "leaky.py").write_text(
        "import sys\n\n\ndef handle_request() -> None:\n    sys.stdout.write('leak')\n",
        encoding="utf-8",
    )

    errors = validate(package)

    assert len(errors) == 1
    assert "sys.stdout.write(...)" in errors[0]


def test_print_removed_or_routed_to_stderr_passes(tmp_path: Path) -> None:
    package = tmp_path / "agent_utilities" / "mcp"
    package.mkdir(parents=True)
    (package / "clean.py").write_text(
        "import logging\nimport sys\n\n"
        "logger = logging.getLogger(__name__)\n\n\n"
        "def handle_request() -> None:\n"
        '    logger.info("this goes to stderr via logging, not print")\n'
        '    print("explicitly routed to stderr, not stdout", file=sys.stderr)\n',
        encoding="utf-8",
    )

    assert validate(package) == []


def test_the_real_served_mcp_package_is_currently_clean() -> None:
    """The gate must actually run clean against the real repo — no baseline,
    no ratchet, zero print()/sys.stdout.write() in the served surface today."""
    root = Path(__file__).resolve().parents[2]
    package = root / "agent_utilities" / "mcp"

    assert validate(package) == []
