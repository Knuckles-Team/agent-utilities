"""Meta-tests for the current-only API and configuration contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_current_only_contract as current_only_contract
from scripts.check_current_only_contract import (
    DATED_HISTORICAL_RECORD_MARKER,
    check,
)


def test_gate_rejects_retired_configuration_switch(tmp_path: Path) -> None:
    source = tmp_path / "deployment.md"
    source.write_text("Set " + "KG_" + "SERVED_PROFILE=0\n", encoding="utf-8")

    violations = check(tmp_path, paths=[source])

    assert len(violations) == 1
    assert "retired surface" in violations[0]


def test_gate_accepts_current_reference_only_configuration(tmp_path: Path) -> None:
    source = tmp_path / "deployment.md"
    source.write_text(
        "Use OIDC_CLIENT_SECRET_REF with the configured secret backend.\n",
        encoding="utf-8",
    )

    assert check(tmp_path, paths=[source]) == []


def test_gate_rejects_retired_graphos_launcher_keys(tmp_path: Path) -> None:
    source = tmp_path / "mcp_config.json"
    source.write_text(
        '{"env": {"' + "ENGINE_" + 'MODE": "remote"}}\n',
        encoding="utf-8",
    )

    violations = check(tmp_path, paths=[source])

    assert len(violations) == 1
    assert "retired surface" in violations[0]


@pytest.mark.parametrize(
    "retired",
    [
        "GRAPH_" + "BACKEND",
        "GRAPH_" + "AUTHORITY",
    ],
)
def test_gate_rejects_retired_authority_surfaces(tmp_path: Path, retired: str) -> None:
    source = tmp_path / "authority.md"
    source.write_text(retired + "\n", encoding="utf-8")

    violations = check(tmp_path, paths=[source])

    assert len(violations) == 1
    assert retired in violations[0]


def test_gate_reports_multiple_needles_once_in_declaration_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "multiple.md"
    source.write_text(
        "AGENT_" + "API_KEY GRAPH_" + "BACKEND AGENT_" + "API_KEY\n",
        encoding="utf-8",
    )

    violations = check(tmp_path, paths=[source])

    assert len(violations) == 2
    assert "GRAPH_" + "BACKEND" in violations[0]
    assert "AGENT_" + "API_KEY" in violations[1]


def test_gate_does_not_match_a_retired_identifier_inside_a_current_name(
    tmp_path: Path,
) -> None:
    source = tmp_path / "current.py"
    source.write_text("GRAPH_" + "BACKEND_L1 = 'live'\n", encoding="utf-8")

    violations = check(tmp_path, paths=[source])

    assert all("GRAPH_" + "BACKEND'" not in violation for violation in violations)


def test_combined_matcher_preserves_legacy_matching_semantics() -> None:
    needles = (
        current_only_contract.RETIRED_IDENTIFIERS
        + current_only_contract.RAW_ROUTE_FRAGMENTS
    )
    samples = [
        candidate
        for needle in needles
        for candidate in (needle, f"before {needle} after", f"{needle} {needle}")
    ]
    samples.extend(
        [
            "GRAPH_" + "BACKEND_L1",
            "prefix_GRAPH_" + "BACKEND",
            "AGENT_" + "API_KEY GRAPH_" + "BACKEND AGENT_" + "API_KEY",
        ]
    )

    for line in samples:
        expected = [
            needle
            for needle in needles
            if current_only_contract._needle_matches(needle, line)
        ]
        assert current_only_contract._matching_needles(line) == expected


# NOTE: there is deliberately no "retired checkpoint module"/"retired durable
# execution authority" test here. `agent_utilities/orchestration/
# durable_execution.py` (`DurableExecutionManager`, `SQLiteCheckpointStore`,
# `PostgresCheckpointStore`, the `DURABLE_EXECUTION_DB` setting) is the LIVE,
# current, exactly-once durable-execution backend -- imported by
# `knowledge_graph/durable_execution_kg.py`, `knowledge_graph/research/
# loop_controller.py`, `harness/agentic_evolution_engine.py`, and
# `orchestration/durable_tool_surface.py`, and documented as authoritative by
# durable_execution_kg.py's own module docstring ("a PROVENANCE MIRROR of an
# already-real, already-durable backend row" -- the backend it mirrors IS this
# module). A prior version of this test suite asserted these names/this path
# were retired; verified false against the live tree (all four names are
# read/imported by current, non-test code) and removed rather than encoding a
# retirement that was never made -- adding them to
# scripts/check_current_only_contract.py's RETIRED_IDENTIFIERS/RETIRED_PATHS
# to satisfy the old assertions would have made that gate flag this repo's
# own current infrastructure as retired debt.


@pytest.mark.parametrize(
    "retired",
    [
        "agent-utilities-" + "kg",
        "KG_SERVER_" + "HOST",
        "KG_SERVER_" + "PORT",
        "KG" + "Coordinator",
        "kg_" + "coordinator",
    ],
)
def test_gate_rejects_retired_kg_sidecar_surfaces(tmp_path: Path, retired: str) -> None:
    source = tmp_path / "graphos.md"
    source.write_text(retired + "\n", encoding="utf-8")

    violations = check(tmp_path, paths=[source])

    assert len(violations) == 1
    assert retired in violations[0]


def test_gate_rejects_retired_graph_mirror_api(tmp_path: Path) -> None:
    source = tmp_path / "architecture.md"
    source.write_text(
        "Call reconcile_" + "to_durable after each checkout.\n",
        encoding="utf-8",
    )

    violations = check(tmp_path, paths=[source])

    assert len(violations) == 1
    assert "retired surface" in violations[0]


def test_gate_rejects_retired_numeric_introspection(tmp_path: Path) -> None:
    source = tmp_path / "numeric.py"
    source.write_text(
        "from agent_utilities.numeric import " + "HAVE_" + "KERNEL\n",
        encoding="utf-8",
    )

    violations = check(tmp_path, paths=[source])

    assert len(violations) == 1
    assert "retired surface" in violations[0]


@pytest.mark.parametrize(
    "retired",
    [
        "AGENT_API_" + "KEY",
        "DEVELOPER_HOST_TOOLS_" + "ENABLED",
        "legacy_observations_v1_" + "get_many",
        "parse_concept_" + "id",
    ],
)
def test_gate_rejects_newly_retired_surfaces(tmp_path: Path, retired: str) -> None:
    source = tmp_path / "surface.md"
    source.write_text(retired + "\n", encoding="utf-8")

    violations = check(tmp_path, paths=[source])

    assert len(violations) == 1
    assert retired in violations[0]


def test_gate_scopes_retired_chat_compactor_to_its_old_owner(
    tmp_path: Path,
) -> None:
    retired = tmp_path / "agent_utilities" / "core" / "chat_persistence.py"
    retired.parent.mkdir(parents=True)
    retired.write_text(
        "async def compact_" + "messages():\n    pass\n", encoding="utf-8"
    )
    current = tmp_path / "agent_utilities" / "core" / "contextual_model.py"
    current.write_text("async def compact_messages():\n    pass\n", encoding="utf-8")

    violations = check(tmp_path, paths=[retired, current])

    assert len(violations) == 1
    assert "chat_persistence.py" in violations[0]


def test_gate_rejects_retired_path_without_a_self_reference(tmp_path: Path) -> None:
    retired_name = "agent_" + "launcher.py"
    source = tmp_path / "agent_utilities" / "core" / retired_name
    source.parent.mkdir(parents=True)
    source.write_text("pass\n", encoding="utf-8")

    violations = check(tmp_path, paths=[source])

    assert violations == [
        "agent_utilities/core/" + retired_name + ": retired path exists"
    ]


# WD10-R-RESIDZERO: the ``ACCEPTED_RESIDUALS`` allowlist mechanism (D-MQR-11,
# BUG-032/GOC-59 shape -- a typed registry of specific relative-path/needle
# pairs, each exempted individually and printed as carried, non-blocking
# INFO) is retired. Every remaining entry on it pointed at exactly one file,
# ``docs/operations/phase10-cutover-runbook.md``, a dated incident runbook
# that intentionally names retired configuration keys as evidence. Rather
# than re-enumerate that file (or any other) by path, the gate now exempts a
# CATEGORY: a ``docs/`` file that declares itself a dated historical record
# via ``DATED_HISTORICAL_RECORD_MARKER`` near its own top (see
# ``_is_dated_historical_record`` in the module and its docstring). The four
# tests below are the required known-bad-input proof for that category rule:
# a marked docs/ file is exempt (the intended, chosen behaviour); an
# UNmarked docs/ file with the exact same retired text still fails; a marked
# file OUTSIDE docs/ still fails (the marker cannot buy an exemption for
# runtime code); and the real, shipped runbook -- not a synthetic fixture --
# produces zero violations for the three keys its own banner names.


def test_gate_exempts_a_marked_docs_file_from_retired_surface_scanning(
    tmp_path: Path,
) -> None:
    """Known-bad input, chosen behaviour: the SAME retired needle that fails
    everywhere else in this file does not fail here, because the containing
    file is a marked dated historical record under docs/."""
    source = tmp_path / "docs" / "operations" / "some-incident-runbook.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        f"# Incident runbook\n\n<!-- {DATED_HISTORICAL_RECORD_MARKER} -->\n"
        "Found " + "GRAPH_" + "BACKEND" + " live on the drifted host.\n",
        encoding="utf-8",
    )

    assert check(tmp_path, paths=[source]) == []


def test_gate_still_rejects_the_same_needle_in_an_unmarked_docs_file(
    tmp_path: Path,
) -> None:
    """The exemption is the marker, not the ``docs/`` directory by itself --
    an otherwise-identical docs/ file with no marker still fails."""
    source = tmp_path / "docs" / "operations" / "some-incident-runbook.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# Incident runbook\n\nFound " + "GRAPH_" + "BACKEND" + " live.\n",
        encoding="utf-8",
    )

    violations = check(tmp_path, paths=[source])

    assert len(violations) == 1
    assert "retired surface" in violations[0]


def test_gate_ignores_the_marker_outside_docs(tmp_path: Path) -> None:
    """The category requires BOTH conditions: the marker cannot exempt a
    file outside docs/, so runtime/test/deploy code cannot buy its way out
    of this gate by pasting the marker in a comment."""
    source = tmp_path / "agent_utilities" / "core" / "notes.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        f"# {DATED_HISTORICAL_RECORD_MARKER}\n"
        "# Found " + "GRAPH_" + "BACKEND" + " live.\n",
        encoding="utf-8",
    )

    violations = check(tmp_path, paths=[source])

    assert len(violations) == 1
    assert "retired surface" in violations[0]


def test_gate_passes_clean_on_the_real_shipped_cutover_runbook() -> None:
    """Integration proof against the real file this lane's brief measured:
    the runbook still names its three retired keys as of GOC-59's own
    2026-08-09 banner, and the gate must produce zero violations for it
    under the real repository root -- not a synthetic fixture standing in
    for it."""
    from scripts.check_current_only_contract import ROOT

    runbook = ROOT / "docs" / "operations" / "phase10-cutover-runbook.md"
    assert runbook.is_file()
    text = runbook.read_text(encoding="utf-8")
    assert DATED_HISTORICAL_RECORD_MARKER in text
    for retired in ("GRAPH_" + "BACKEND", "ENGINE_" + "MODE", "ENGINE_" + "ENDPOINT"):
        assert retired in text  # still names them -- the record is intact

    assert check(ROOT, paths=[runbook]) == []
