"""Meta-tests for the current-only API and configuration contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_current_only_contract import check


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
