"""Meta-tests for the current-only API and configuration contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_current_only_contract import AcceptedResidual, check, check_report


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


# D-MQR-11 (2026-08-16): the gate used to exit 1 on findings its own source
# marked as accepted residuals (BUG-032/GOC-59 -- documented, cross-repo-
# blocked, intentionally carried), with no way for a reader to tell "carried
# debt" from "new regression." ACCEPTED_RESIDUALS + check_report() fix that:
# an accepted finding is reported as INFO and does not fail the gate; anything
# NOT on that documented, rationale-required list still does. These two tests
# are the required proof against known-bad input -- run them against the
# pre-fix module (no ``AcceptedResidual``/``check_report`` symbols existed) and
# the import itself fails; after the fix, both pass.


def test_accepted_residual_requires_rationale_and_owner() -> None:
    """The registry data format makes an unexplained entry impossible to add."""
    AcceptedResidual(
        relative="some/path.py", needle=None, owner="OWNER-1", reason="a real reason"
    )  # sanity: a fully-populated entry is fine

    with pytest.raises(ValueError):
        AcceptedResidual(
            relative="some/path.py", needle=None, owner="OWNER-1", reason=""
        )

    with pytest.raises(ValueError):
        AcceptedResidual(
            relative="some/path.py", needle=None, owner="   ", reason="a real reason"
        )

    with pytest.raises(ValueError):
        AcceptedResidual(
            relative="  ", needle=None, owner="OWNER-1", reason="a real reason"
        )


def test_gate_carries_a_documented_residual_but_still_rejects_a_new_one(
    tmp_path: Path,
) -> None:
    """Known-bad input: one documented accepted residual, one genuinely new,
    undocumented retired path in the same run. The accepted one must be
    reported as carried (and never drive a non-zero exit); the new one must
    still fail exactly as before."""
    accepted_residual_path = tmp_path / "agent_utilities" / "exceptions.py"
    accepted_residual_path.parent.mkdir(parents=True)
    accepted_residual_path.write_text("# back-compat shim\n", encoding="utf-8")

    new_retired_name = "agent_" + "launcher.py"
    new_violation_path = tmp_path / "agent_utilities" / "core" / new_retired_name
    new_violation_path.parent.mkdir(parents=True)
    new_violation_path.write_text("pass\n", encoding="utf-8")

    report = check_report(tmp_path, paths=[accepted_residual_path, new_violation_path])

    # The accepted residual is carried, not a failure -- and it does not leak
    # into the "new" (failing) bucket at all. (Needle split the same way
    # scripts/check_current_only_contract.py splits its own spellings, so
    # this assertion does not trip the gate on its own source.)
    assert len(report.accepted) == 1
    assert "agent_utilities/exceptions" + ".py" in report.accepted[0]
    assert "agent_utilities/exceptions" + ".py" not in "".join(report.new)

    # The new, undocumented retired path still fails, unchanged from before,
    # and does not get absorbed into the accepted bucket.
    assert report.new == [
        "agent_utilities/core/" + new_retired_name + ": retired path exists"
    ]
    assert new_retired_name not in "".join(report.accepted)

    # check() is the exit-code-driving surface main() uses: it must mirror
    # report.new exactly (accepted residuals never leak into it).
    assert (
        check(tmp_path, paths=[accepted_residual_path, new_violation_path])
        == report.new
    )
