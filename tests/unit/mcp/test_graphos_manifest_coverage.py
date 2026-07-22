"""Every live-registered graph-os tool family must appear in the generated
action manifest (``agent_utilities/mcp/_graphos_action_manifest.py``).

The manifest is produced by ``scripts/gen_graphos_manifest.py`` (see that
module's docstring: "Regenerate after changing the graph-os tool surface"),
and ``tool_specs.py`` treats it as authoritative for the canonical tool
universe (``TOOL_SPECS`` / ``canonical_tool_names``). Regeneration is a manual
step, not a pre-commit-enforced one — nothing previously caught a registrar
being added to ``kg_server._build_server``'s registrar list (or a new
action-routed tool module being wired in) without the manifest being
regenerated to match. That let ``graph_engineering`` and ``graph_argument``
ship as live, callable MCP tools that were entirely absent from the manifest
skill-validation harness checks skills against, and (downstream, since the
CPD catalog is built by walking the SAME manifest) absent from the packaged
Capability Power Descriptor catalog — which made the whole default
``MCP_TOOL_MODE=intent`` surface fail closed with a ``RuntimeError`` the
moment either tool was live-registered, since ``_build_candidates`` requires
every registered tool to have a CPD.

This is the general form of
``tests/unit/test_engine_api_coverage.py::test_every_verbose_engine_op_exists_in_manifest``
(which only covers ``engine_*`` domain tools) — it covers the full
condensed/granular tool surface, one family at a time, so this class of drift
fails a fast unit test instead of surfacing later as a runtime import error
or a skill-validation-harness FAIL.
"""

from __future__ import annotations

import os


def _manifest_tool_families() -> set[str]:
    from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS

    return {entry["tool"] for entry in GRAPHOS_ACTIONS}


def _live_condensed_tool_families() -> set[str]:
    """Rebuild the condensed/canonical action-routed tool surface in isolation
    (mirrors ``scripts/gen_graphos_manifest.py::build_manifest``) and return
    every tool family name it registers, restoring global registries after.
    """
    from agent_utilities.mcp import kg_server

    registered_before = dict(kg_server.REGISTERED_TOOLS)
    routes_before = dict(kg_server.ACTION_TOOL_ROUTES)
    prior_mode = os.environ.get("MCP_TOOL_MODE")
    try:
        kg_server.REGISTERED_TOOLS.clear()
        kg_server.ACTION_TOOL_ROUTES.clear()
        kg_server.ACTION_TOOL_ROUTES.update(kg_server.BASE_ACTION_TOOL_ROUTES)
        kg_server._build_server(
            bootstrap=False,
            tool_profile="intent",
            canonical_surface=True,
        )
        return set(kg_server.ACTION_TOOL_ROUTES)
    finally:
        if prior_mode is None:
            os.environ.pop("MCP_TOOL_MODE", None)
        else:
            os.environ["MCP_TOOL_MODE"] = prior_mode
        kg_server.REGISTERED_TOOLS.clear()
        kg_server.REGISTERED_TOOLS.update(registered_before)
        kg_server.ACTION_TOOL_ROUTES.clear()
        kg_server.ACTION_TOOL_ROUTES.update(routes_before)


def test_every_registered_tool_family_is_in_the_generated_manifest():
    live_families = _live_condensed_tool_families()
    manifest_families = _manifest_tool_families()
    missing = sorted(live_families - manifest_families)
    assert not missing, (
        "tool(s) registered in kg_server.ACTION_TOOL_ROUTES but absent from "
        "the generated agent_utilities/mcp/_graphos_action_manifest.py -- "
        "run `python scripts/gen_graphos_manifest.py` and `ruff format` the "
        f"output, then add a TOOL_VERBS entry in tool_specs.py: {missing}"
    )


def test_manifest_has_no_phantom_tool_families():
    """The inverse check: a manifest entry naming a tool that is not (or no
    longer) registered anywhere is a phantom — the ``graph_hydrate`` bug this
    repo already hit once (see commit ``74cfe632``, which hand-removed one
    phantom entry). Regenerating catches every phantom at once instead of one
    hand-discovered case at a time.
    """
    live_families = _live_condensed_tool_families()
    manifest_families = _manifest_tool_families()
    phantom = sorted(manifest_families - live_families)
    assert not phantom, (
        "tool family/families in the generated manifest have no matching "
        "live registration -- regenerate with "
        f"`python scripts/gen_graphos_manifest.py`: {phantom}"
    )
