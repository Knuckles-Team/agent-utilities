"""Capability A proof: mandatory provenance header (incidents 5, 6, 7 + the general verifier).

* Incident 5: `bash scripts/check_ontology.py` passed while the pre-commit
  HOOK of the same name failed -- different venv (different `interpreter`).
* Incident 6: a local "2-core" gate via `taskset -c 0,1` on a 64-core host
  passed while the real 2-vCPU CI runner failed the same test -- different
  `cpu_affinity`/`cpu_count`.
* Incident 7: `rustc-wrapper = "sccache"` passed every local gate (host has
  sccache) then killed every CI job -- different `env_fingerprint`.

Each is proven by constructing two ProvenanceHeaders that differ on exactly
the field the incident turned on, and showing `require_same_environment`
raises. Plus the umbrella requirement: a result with NO header at all must
be mechanically rejected by `require_provenance`.
"""

from __future__ import annotations

import time

import pytest

from agent_utilities.measurement.provenance import (
    EnvironmentMismatchError,
    MissingProvenanceError,
    ProvenanceHeader,
    environment_mismatches,
    require_provenance,
    require_same_environment,
)


def _header(**overrides) -> ProvenanceHeader:
    base = dict(
        schema_version=1,
        interpreter="/usr/bin/python3",
        hostname="host",
        user="u",
        cwd="/tmp",
        command=["true"],
        git_sha="deadbeef",
        git_dirty=False,
        tree="/repo",
        is_copy=False,
        copy_integrity=None,
        load_avg_start=(1.0, 1.0, 1.0),
        load_avg_end=(1.0, 1.0, 1.0),
        timestamp_start=time.time(),
        timestamp_end=time.time(),
        platform="Linux",
        cpu_count=24,
        cpu_affinity=list(range(24)),
        env_fingerprint={"VIRTUAL_ENV": None},
    )
    base.update(overrides)
    return ProvenanceHeader(**base)


def test_incident_5_different_venv_interpreter_is_caught():
    hook_run = _header(interpreter="/home/u/repo/.git/hooks/.venv/bin/python3")
    bash_run = _header(interpreter="/home/u/repo/.venv/bin/python3")
    diffs = environment_mismatches(hook_run, bash_run)
    assert "interpreter" in diffs
    with pytest.raises(EnvironmentMismatchError):
        require_same_environment(hook_run, bash_run)


def test_incident_6_taskset_vs_real_ci_core_count_is_caught():
    local_taskset_2_of_64 = _header(cpu_count=64, cpu_affinity=[0, 1])
    real_2vcpu_ci_runner = _header(cpu_count=2, cpu_affinity=[0, 1])
    diffs = environment_mismatches(local_taskset_2_of_64, real_2vcpu_ci_runner)
    assert "cpu_count" in diffs
    with pytest.raises(EnvironmentMismatchError):
        require_same_environment(local_taskset_2_of_64, real_2vcpu_ci_runner)


def test_incident_7_sccache_env_fingerprint_is_caught():
    local_with_sccache = _header(env_fingerprint={"RUSTC_WRAPPER": "sccache", "VIRTUAL_ENV": None})
    ci_without_sccache = _header(env_fingerprint={"RUSTC_WRAPPER": None, "VIRTUAL_ENV": None})
    diffs = environment_mismatches(local_with_sccache, ci_without_sccache)
    assert "env_fingerprint" in diffs
    with pytest.raises(EnvironmentMismatchError):
        require_same_environment(local_with_sccache, ci_without_sccache)


def test_identical_environments_produce_no_mismatch():
    a = _header()
    b = _header()
    assert environment_mismatches(a, b) == {}
    require_same_environment(a, b)  # must not raise


def test_require_provenance_rejects_a_headerless_result():
    """The umbrella rule: a result without a header is inadmissible, not a pass."""
    with pytest.raises(MissingProvenanceError):
        require_provenance({"passed": True, "failures": 0})  # no provenance at all

    with pytest.raises(MissingProvenanceError):
        require_provenance("exit 0")  # not even a dict/header type

    with pytest.raises(MissingProvenanceError):
        require_provenance(object())


def test_require_provenance_rejects_an_unfinished_header():
    header = ProvenanceHeader.start(["true"])  # never .finish()ed
    with pytest.raises(MissingProvenanceError):
        require_provenance(header)


def test_require_provenance_accepts_a_complete_finished_header():
    header = ProvenanceHeader.start(["true"]).finish()
    admitted = require_provenance(header)
    assert admitted["interpreter"] == header.interpreter


def test_provenance_header_start_captures_real_interpreter_and_cpu_facts():
    import os
    import sys

    header = ProvenanceHeader.start(["true"])
    assert header.interpreter == sys.executable
    assert header.cpu_count == os.cpu_count()
