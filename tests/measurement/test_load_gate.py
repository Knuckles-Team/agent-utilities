"""Capability B proof: load gating (incident 4, and the 24-core/load-62 danger note).

Incident: a test suite at load average 15.82 reported "2 failed / 10
errors"; the SAME suite at load 4.03 reported "1 failed / 382 passed" and
ran 7x faster. Load was never asserted, so the first (noise-corrupted)
result was reported as a real regression. Separately, the danger zone this
module exists to keep a measurement out of is documented as load ~62 with
swap exhausted on a 24-core box (ratio ~2.6x cores).

Proves: (1) the incident-4 high-load reading is refused with the distinct
TOO_LOADED_TO_MEASURE status/exception, never a number; (2) the low-load
reading is allowed through; (3) an abort is a raised exception, not a
falsy/zero return that could be mistaken for a pass.
"""

from __future__ import annotations

import pytest

from agent_utilities.measurement import load_gate


def test_incident_4_high_load_reading_is_refused_not_reported(monkeypatch):
    """load average 15.82 on a host where the configured threshold is lower
    (simulating a smaller/more conservative box than this 24-core dev host)
    must come back as TOO_LOADED_TO_MEASURE, never as a pass/fail number."""
    monkeypatch.setattr(load_gate, "_read_loadavg", lambda: (15.82, 12.0, 9.0))
    result = load_gate.check_load(threshold=10.0)
    assert result.status == load_gate.TOO_LOADED_TO_MEASURE
    assert result.status != "OK"
    assert not result.ok


def test_incident_4_low_load_reading_is_allowed(monkeypatch):
    monkeypatch.setattr(load_gate, "_read_loadavg", lambda: (4.03, 3.5, 3.0))
    result = load_gate.check_load(threshold=10.0)
    assert result.status == "OK"
    assert result.ok
    assert result.load1 == 4.03


def test_gate_or_raise_aborts_with_an_exception_never_a_falsy_pass(monkeypatch):
    """The core safety property: a killed/aborted measurement must never
    read as a pass. gate_or_raise() must raise, not return 0/False/None."""
    monkeypatch.setattr(load_gate, "_read_loadavg", lambda: (15.82, 12.0, 9.0))
    with pytest.raises(load_gate.TooLoadedToMeasureError) as excinfo:
        load_gate.gate_or_raise(threshold=10.0)
    # The exception message names the sentinel string, not a bare number.
    assert load_gate.TOO_LOADED_TO_MEASURE in str(excinfo.value)


def test_danger_zone_load_62_on_24_core_box_is_refused(monkeypatch):
    """The documented danger note: load ~62, swap exhausted, 24-core box
    (ratio ~2.6x cores) — must be refused under the derived default
    threshold (1.5x cores)."""
    monkeypatch.setattr(load_gate, "_read_loadavg", lambda: (62.0, 58.0, 40.0))
    threshold = load_gate.default_threshold(cpu_count=24)
    assert threshold == 36.0
    result = load_gate.check_load(threshold=threshold)
    assert result.status == load_gate.TOO_LOADED_TO_MEASURE


def test_unknown_load_is_not_silently_treated_as_ok(monkeypatch):
    monkeypatch.setattr(load_gate, "_read_loadavg", lambda: None)
    result = load_gate.check_load(threshold=10.0)
    assert result.status == "UNKNOWN_LOAD"
    assert not result.ok


def test_default_threshold_respects_env_override(monkeypatch):
    monkeypatch.setenv("MEASUREMENT_LOAD_THRESHOLD", "5.5")
    assert load_gate.default_threshold(cpu_count=24) == 5.5
