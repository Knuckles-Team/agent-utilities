"""Meta-test: the CPD drift gate passes clean and trips on a broken fixture.

A gate that can't fail is not a gate. CONCEPT:AU-KG.retrieval.capability-power-descriptor
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_utilities.mcp.tool_specs import (
    INTENT_VERBS,
    SUPPORTED_FEATURES,
    TOOL_VERBS,
    canonical_tool_names,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
MD_PATH = ROOT / "docs" / "capabilities-power.md"
JSON_PATH = ROOT / "docs" / "capabilities-power.json"
PACKAGE_JSON_PATH = (
    ROOT
    / "agent_utilities"
    / "knowledge_graph"
    / "retrieval"
    / "capabilities-power.json"
)

def _numeric_kernel_available() -> bool:
    try:
        import agent_utilities.numeric  # noqa: F401
    except ImportError:
        return False
    return True


# The CPD gate builds the FULL MCP tool registry (`kg_server`) to prove every
# tool has exactly one CPD — that pulls BOTH the serving stack (starlette/fastmcp)
# AND the numeric kernel. CI's guardrails job installs only the package core
# (deliberately lean, and the compiled kernel isn't pip-installable), so the
# registry can't be built there and the coverage gate would be meaningless against
# a partial registry anyway. It runs in the FULL-env pre-commit (the
# `guardrail-cpd-drift` hook) instead. Skip when either piece is absent.
_needs_server_stack = pytest.mark.skipif(
    importlib.util.find_spec("starlette") is None or not _numeric_kernel_available(),
    reason="CPD gate needs the full MCP tool registry (serving stack "
    "starlette/fastmcp + numeric kernel); runs in the full-env pre-commit, "
    "not the lean CI job",
)


def _run_check_cpd() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "check_cpd.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


@_needs_server_stack
def test_check_cpd_passes_on_the_checked_in_set():
    """The committed docs/capabilities-power.{md,json} must be in sync right now."""
    result = _run_check_cpd()
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def _restore_md():
    original = MD_PATH.read_text(encoding="utf-8")
    try:
        yield
    finally:
        MD_PATH.write_text(original, encoding="utf-8")


@_needs_server_stack
def test_check_cpd_trips_when_the_checked_in_doc_is_stale(_restore_md):
    """Appending content the live generator would never produce must fail the gate."""
    with MD_PATH.open("a", encoding="utf-8") as fh:
        fh.write("\n<!-- hand-edited, never regenerated -->\n")
    result = _run_check_cpd()
    assert result.returncode == 1
    assert "DRIFT" in result.stdout or "stale" in result.stdout


@pytest.fixture
def _restore_json():
    original = JSON_PATH.read_text(encoding="utf-8")
    try:
        yield
    finally:
        JSON_PATH.write_text(original, encoding="utf-8")


@_needs_server_stack
def test_check_cpd_trips_when_a_cpd_is_deleted_from_the_json(_restore_json):
    """Removing one capability from the checked-in JSON must fail coverage/drift."""
    import json

    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert data["capabilities"], "fixture precondition: at least one CPD present"
    data["capabilities"].pop()
    data["count"] = len(data["capabilities"])
    JSON_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    result = _run_check_cpd()
    assert result.returncode == 1


@pytest.fixture
def _restore_package_json():
    original = PACKAGE_JSON_PATH.read_text(encoding="utf-8")
    try:
        yield
    finally:
        PACKAGE_JSON_PATH.write_text(original, encoding="utf-8")


@_needs_server_stack
def test_check_cpd_trips_when_the_packaged_catalog_diverges(_restore_package_json):
    """The runtime catalog is a generated mirror, never an independent copy."""
    with PACKAGE_JSON_PATH.open("a", encoding="utf-8") as fh:
        fh.write("\n")
    result = _run_check_cpd()
    assert result.returncode == 1
    assert "byte-identical" in result.stdout or "stale" in result.stdout


def test_packaged_cpd_catalog_matches_documentation_catalog() -> None:
    """The wheel carries every supported feature profile byte-for-byte."""
    import json

    assert PACKAGE_JSON_PATH.read_bytes() == JSON_PATH.read_bytes()
    data = json.loads(PACKAGE_JSON_PATH.read_text(encoding="utf-8"))
    capability_ids = {item["id"] for item in data["capabilities"]}
    expected_ids = set(canonical_tool_names(features=SUPPORTED_FEATURES))
    assert data["count"] == len(expected_ids)
    assert capability_ids == expected_ids
    for item in data["capabilities"]:
        expected_verbs = (
            [item["id"]] if item["id"] in INTENT_VERBS else list(TOOL_VERBS[item["id"]])
        )
        assert item["intent_verbs"] == expected_verbs


def test_load_cpds_reads_the_full_packaged_catalog() -> None:
    """Installed Graph-OS resolution must not silently fall back to zero CPDs."""
    from agent_utilities.knowledge_graph.retrieval import capability_context

    capability_context._load_raw.cache_clear()
    try:
        assert set(capability_context.load_cpds()) == set(
            canonical_tool_names(features=SUPPORTED_FEATURES)
        )
    finally:
        capability_context._load_raw.cache_clear()


def test_checked_in_catalog_includes_intent_and_optional_descriptors() -> None:
    import json

    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    capability_ids = {item["id"] for item in data["capabilities"]}
    assert set(INTENT_VERBS) <= capability_ids
    assert "quant" in capability_ids


def test_action_inventory_comes_from_the_generated_manifest() -> None:
    """Ambient client schemas cannot change a checked-in catalog contract."""
    sys.path.insert(0, str(SCRIPTS))
    import gen_capability_power as generator

    actions = generator.get_actions_for_tool(
        "engine_graph",
        {
            "properties": {
                "action": {
                    "default": "ambient_action",
                    "description": "ambient_action | another_ambient_action",
                }
            }
        },
        {"engine_graph": ["manifest_action", "another_manifest_action"]},
        {},  # engine_domains — empty so the manifest-precedence path (under test) is hit
        (),  # mining_actions
        (),  # graphlearn_actions
        (),  # deep_mining_actions
    )

    assert actions == ["another_manifest_action", "manifest_action"]


def test_generation_timestamp_honors_source_date_epoch(monkeypatch) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import gen_capability_power as generator

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")

    assert generator.generation_timestamp() == "1970-01-01T00:00:00Z"


@_needs_server_stack
def test_generation_restores_environment_and_runtime_registries(monkeypatch):
    sys.path.insert(0, str(SCRIPTS))
    import gen_capability_power as generator

    from agent_utilities.mcp import kg_server

    async def _probe() -> str:
        return "probe"

    monkeypatch.setenv("MCP_TOOL_MODE", "verbose")
    original_registered = dict(kg_server.REGISTERED_TOOLS)
    original_routes = dict(kg_server.ACTION_TOOL_ROUTES)
    try:
        kg_server.REGISTERED_TOOLS["state_probe"] = _probe
        kg_server.ACTION_TOOL_ROUTES["state_probe"] = "/state-probe"
        registered_before = dict(kg_server.REGISTERED_TOOLS)
        routes_before = dict(kg_server.ACTION_TOOL_ROUTES)

        cpds, _generated_at = generator.generate(
            None, refresh_cache=False, prefer_cache=True
        )

        assert {cpd.id for cpd in cpds} >= set(INTENT_VERBS)
        assert "quant" in {cpd.id for cpd in cpds}
        for cpd in cpds:
            expected_verbs = (
                [cpd.id] if cpd.id in INTENT_VERBS else list(TOOL_VERBS[cpd.id])
            )
            assert cpd.intent_verbs == expected_verbs
        assert os.environ["MCP_TOOL_MODE"] == "verbose"
        assert kg_server.REGISTERED_TOOLS == registered_before
        assert kg_server.ACTION_TOOL_ROUTES == routes_before
    finally:
        kg_server.REGISTERED_TOOLS.clear()
        kg_server.REGISTERED_TOOLS.update(original_registered)
        kg_server.ACTION_TOOL_ROUTES.clear()
        kg_server.ACTION_TOOL_ROUTES.update(original_routes)


@_needs_server_stack
def test_every_registered_non_intent_verb_tool_has_a_cpd() -> None:
    """Regression guard for the ``engine_placement`` incident.

    graph-os fails EVERY intent-verb call (ask/find/act/why/write/manage)
    closed the instant one live, registered granular tool has no packaged
    CPD (``intent_tools._build_candidates`` — CONCEPT:
    AU-ECO.mcp.intent-surface-cpd-ranking). ``engine_placement`` registered
    live (the "placement" engine domain became reachable) while the
    packaged, checked-in ``capabilities-power.json`` stayed one entry short
    — this asserts that specific invariant directly, against the actual
    live tool registry (not merely the static ``tool_specs`` universe), so
    the exact class of drift that broke production cannot silently reship.

    See ``tests/unit/test_intent_surface.py`` for the companion end-to-end
    regression test that exercises ``resolve_intent`` itself.
    """
    from agent_utilities.knowledge_graph.retrieval.capability_context import (
        load_cpds,
    )
    from agent_utilities.mcp import kg_server
    from agent_utilities.mcp.tools.intent_tools import INTENT_VERBS as _VERBS

    kg_server.ensure_tools_registered()
    live_tools = set(kg_server.REGISTERED_TOOLS) - set(_VERBS)
    assert live_tools, "fixture precondition: at least one granular tool registered"
    packaged_ids = set(load_cpds())
    missing = live_tools - packaged_ids
    assert not missing, f"Tools with no packaged CPD: {sorted(missing)}"
