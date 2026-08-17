"""GOC-38-W02/acceptance-gate tests: the envelope must be falsifiable.

The central known-bad proof for this lane: a deliberately vacuous run (zero
tests collected) or a wrong-interpreter run must NOT be reported green, even
when every other field looks like a pass (exit code 0, no exception, clean
process teardown). ``compute_verdict`` is exercised directly against
schema-valid envelope bodies so these checks do not depend on spawning real
subprocesses.
"""

from __future__ import annotations

import copy

import jsonschema
import pytest

from scripts.hermetic_harness.envelope import compute_verdict, validate_envelope


def _minimal_valid_body() -> dict:
    return {
        "envelope_version": "1.0.0",
        "envelope_id": "11111111-1111-1111-1111-111111111111",
        "manifest_digest": "sha256:" + "a" * 64,
        "state": "EVIDENCE-FLUSHED",
        "state_history": [{"state": "DECLARED", "at": "2026-08-16T00:00:00Z", "reason": None}],
        "candidate": {"repo": "agent-utilities", "git_sha": "deadbeef", "branch": "goc/goc-38", "dirty": False},
        "environment": {
            "sys_executable": "/venv/bin/python3",
            "sys_prefix": "/venv",
            "venv_pyvenv_cfg_digest": "sha256:" + "b" * 64,
            "interpreter_version": "3.14.4",
            "lock_digest": "sha256:" + "c" * 64,
            "lockfile_path": "uv.lock",
            "package_count": 700,
            "expected_venv_path": "/venv",
            "venv_identity_match": True,
            "env_allowlist": ["PATH", "HOME"],
            "env_rejected_vars": ["UV_PROJECT_ENVIRONMENT"],
            "temp_root": "/var/tmp/l9/hermetic-harness",
            "host": "r710",
            "os": "Linux",
        },
        "command": {
            "argv": ["/venv/bin/python3", "-m", "pytest", "-q", "tests/gates/test_cpd_gate.py"],
            "shell": False,
            "cwd": "/repo",
            "adapter": "pytest",
        },
        "resources": {
            "process_group_leader_pid": 4242,
            "process_group_start_time": "2026-08-16T00:00:00Z",
            "descendant_pids_observed": [],
            "deadline_seconds": 300,
            "grace_seconds": 15,
            "cancellation": {"fired": False, "signal_sequence": [], "escalated": False},
            "survivor_check": {
                "performed": True,
                "survivors_before_kill": [],
                "survivors_after_kill": [],
                "clean": True,
            },
            "rss_peak_bytes": 1024,
            "cpu_time_seconds": 1.0,
            "io_wait_seconds": 0.0,
            "open_files_peak": 10,
            "disk_delta_bytes": 0,
        },
        "leases": [],
        "collection": {
            "collection_count": 214,
            "deselection_count": 6,
            "quarantine_count": 0,
            "selection_digest": "sha256:" + "d" * 64,
            "collection_source": "pytest --collect-only -q",
        },
        "streams": {
            "stdout": {
                "byte_count": 100,
                "truncated": False,
                "truncation_marker": None,
                "digest": "sha256:" + "e" * 64,
                "retention_class": "raw-30d",
                "path": "/evidence/stdout.raw",
            },
            "stderr": {
                "byte_count": 0,
                "truncated": False,
                "truncation_marker": None,
                "digest": "sha256:" + "f" * 64,
                "retention_class": "raw-30d",
                "path": "/evidence/stderr.raw",
            },
        },
        "exit": {
            "adapter_native_exit_code": 0,
            "signal": None,
            "normalized_outcome": "PASSED",
            "assertion_summary": {"passed": 214, "failed": 0, "errors": 0, "skipped": 0},
            "provenance": "direct_waitpid",
        },
        "contamination": {
            "pre_snapshot": {},
            "post_snapshot": {},
            "findings": [],
            "invalidated": False,
        },
        "redaction": {"policy": "default-v1", "redacted_patterns": ["token"], "applied": True},
        "audit": {
            "actor": "goc-38-harness",
            "lane": "GOC-38",
            "reviewer": None,
            "created_at": "2026-08-16T00:00:00Z",
            "flushed_at": "2026-08-16T00:00:00Z",
        },
    }


def _with_verdict(body: dict) -> dict:
    body = copy.deepcopy(body)
    body["verdict"] = compute_verdict(body)
    return body


def test_genuine_pass_is_green_and_falsifiable():
    body = _with_verdict(_minimal_valid_body())
    validate_envelope(body)  # schema-valid
    assert body["verdict"]["green"] is True
    assert body["verdict"]["outcome"] == "PASSED"
    assert body["verdict"]["falsifiable"] is True
    assert body["verdict"]["reasons"] == []


def test_known_bad_zero_collection_is_never_green():
    """The central acceptance-gate proof: a deliberately vacuous run (zero
    tests collected) must FAIL/flag rather than report green, even with a
    clean exit code and clean teardown recorded elsewhere in the envelope."""
    base = _minimal_valid_body()
    base["collection"]["collection_count"] = 0
    base["collection"]["deselection_count"] = 0
    verdict = compute_verdict(base)
    assert verdict["green"] is False
    assert verdict["outcome"] == "VACUOUS"
    assert any("zero tests collected" in r for r in verdict["reasons"])
    assert verdict["falsifiable"] is False


def test_known_bad_wrong_interpreter_is_never_green():
    """A bare `uv run pytest` silently running the system pytest instead of
    the project's -- ~80 false verdicts across 5 lanes previously. The
    envelope must flag this explicitly rather than trust a passing exit
    code from the wrong interpreter."""
    base = _minimal_valid_body()
    base["environment"]["venv_identity_match"] = False
    verdict = compute_verdict(base)
    assert verdict["green"] is False
    assert any("interpreter identity" in r for r in verdict["reasons"])
    assert verdict["falsifiable"] is False


def test_known_bad_survivor_after_kill_is_never_green():
    base = _minimal_valid_body()
    base["resources"]["survivor_check"]["survivors_after_kill"] = [9999]
    base["resources"]["survivor_check"]["clean"] = False
    verdict = compute_verdict(base)
    assert verdict["green"] is False
    assert any("survived teardown" in r for r in verdict["reasons"])


def test_known_bad_contamination_invalidates_even_a_passing_run():
    base = _minimal_valid_body()
    base["contamination"]["invalidated"] = True
    base["contamination"]["findings"] = ["venv package_count dropped 700 -> 1 mid-run"]
    verdict = compute_verdict(base)
    assert verdict["green"] is False
    assert any("contamination" in r for r in verdict["reasons"])


def test_known_bad_truncated_stream_without_digest_is_never_green():
    """Never infer success from an empty/truncated capture without a
    verified digest -- 23 CLI/script tests previously asserted on
    capsys.readouterr().out after a 'successful' run and got empty stdout."""
    base = _minimal_valid_body()
    base["streams"]["stdout"]["truncated"] = True
    base["streams"]["stdout"]["digest"] = None
    verdict = compute_verdict(base)
    assert verdict["green"] is False
    assert any("stdout truncated" in r for r in verdict["reasons"])
    assert verdict["falsifiable"] is False


def test_known_bad_non_direct_exit_provenance_is_never_green():
    """$? after a pipeline is the last command's status only -- exit-status
    provenance must be an explicit direct wait status, never inferred."""
    base = _minimal_valid_body()
    base["exit"]["provenance"] = "shell_dollar_question_after_pipeline"
    verdict = compute_verdict(base)
    assert verdict["green"] is False
    assert any("provenance" in r for r in verdict["reasons"])


def test_schema_rejects_additional_properties():
    body = _with_verdict(_minimal_valid_body())
    body["unexpected_field"] = "should not be accepted silently"
    with pytest.raises(jsonschema.ValidationError):
        validate_envelope(body)


def test_schema_rejects_missing_required_top_level_field():
    body = _with_verdict(_minimal_valid_body())
    del body["contamination"]
    with pytest.raises(jsonschema.ValidationError):
        validate_envelope(body)


def test_schema_rejects_shell_true():
    body = _with_verdict(_minimal_valid_body())
    body["command"]["shell"] = True
    with pytest.raises(jsonschema.ValidationError):
        validate_envelope(body)
