"""Tests for the ``ActionSpec``/``actions:`` typed-action schema extension
(CA-32, DEC-CA-07, CONCEPT:AU-KG.ontology.connector-typed-actions).

Covers:

  * the rich ``actions:`` block (``parameters``/``target_resource``/
    ``conflict_policy``/``requires_approval``/``approval_class``/``effects``)
    round-trips through :class:`ConnectorManifest`,
  * every field is additive with a backward-compatible default — the old
    ``{id, name, description}``-only shape still validates, and ``extra="forbid"``
    still rejects an unknown field,
  * ``target_resource`` must resolve to a ``resources[].name`` in the SAME
    manifest,
  * ``requires_approval: false`` is refused when the action id/name looks
    destructive (DEC-CA-07 "Security and privacy"),
  * every manifest actually on disk under ``ontology/connector_manifests/``
    still loads, unchanged (CA-32-W01's bundled-manifest compatibility proof),
  * the gate's :func:`undeclared_mutating_tools` — fail-open with no explicit
    mutating signal, fail-closed once one is present and undeclared, and
    satisfied once the matching ``ActionSpec.id`` is declared.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agent_utilities.knowledge_graph.ontology import connector_manifest_gate as gate
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ActionParameterSpec,
    ActionSpec,
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    ResourceSpec,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MANIFESTS_ROOT = (
    _REPO_ROOT
    / "agent_utilities"
    / "knowledge_graph"
    / "ontology"
    / "connector_manifests"
)


def _manifest(actions: list[ActionSpec], *, resources=None) -> ConnectorManifest:
    return ConnectorManifest(
        connector="widget-mcp",
        resources=resources or [ResourceSpec(name="Widget", id_prefix="widget")],
        actions=actions,
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="0" * 64)),
    )


# ── schema round-trip + backward compatibility ─────────────────────────────


def test_rich_action_spec_round_trips():
    action = ActionSpec(
        id="update_widget",
        name="Update Widget",
        description="Update a widget's fields.",
        label="Update Widget",
        parameters=[
            ActionParameterSpec(name="widget_id", type="string", required=True),
            ActionParameterSpec(
                name="title", type="string", required=False, description="New title"
            ),
        ],
        target_resource="Widget",
        conflict_policy="manual_review",
        requires_approval=True,
        approval_class="sensitive",
        effects=["mutate:Widget"],
    )
    manifest = _manifest([action])
    dumped = manifest.model_dump(mode="json")
    again = ConnectorManifest.model_validate(dumped)
    restored = again.actions[0]
    assert restored.id == "update_widget"
    assert restored.parameters[0].name == "widget_id"
    assert restored.target_resource == "Widget"
    assert restored.conflict_policy == "manual_review"
    assert restored.requires_approval is True
    assert restored.approval_class == "sensitive"
    assert restored.effects == ["mutate:Widget"]


def test_old_three_field_shape_still_validates_with_new_defaults():
    """The exact shape all 71/72 shipped manifests use today."""
    action = ActionSpec(id="epistemic-answer", name="Epistemic Answer", description="x")
    assert action.label == ""
    assert action.parameters == []
    assert action.target_resource is None
    assert action.conflict_policy is None
    assert action.requires_approval is True
    assert action.approval_class == "unclassified"
    assert action.effects == []
    # A manifest built purely from the old shape validates with no new fields set.
    manifest = _manifest([action], resources=[])
    assert manifest.actions[0].id == "epistemic-answer"


def test_extra_forbid_still_enforced_on_action_spec():
    with pytest.raises(ValidationError):
        ActionSpec.model_validate({"id": "x", "not_a_real_field": True})


def test_extra_forbid_still_enforced_on_action_parameter_spec():
    with pytest.raises(ValidationError):
        ActionParameterSpec.model_validate({"name": "x", "bogus": 1})


def test_conflict_policy_rejects_value_outside_the_enum():
    with pytest.raises(ValidationError):
        _manifest([ActionSpec(id="x", conflict_policy="always_win")])  # type: ignore[arg-type]


# ── target_resource cross-field validation ─────────────────────────────────


def test_target_resource_must_resolve_to_a_declared_resource():
    _manifest([ActionSpec(id="list_widgets", target_resource="Widget")])  # passes


def test_target_resource_rejected_when_unresolved():
    with pytest.raises(ValidationError, match="does not name a resources"):
        _manifest([ActionSpec(id="list_widgets", target_resource="NoSuchResource")])


def test_target_resource_none_is_always_accepted():
    """Every existing manifest's a2a-capability actions target no resource."""
    _manifest([ActionSpec(id="run_graph_flow")], resources=[])


# ── requires_approval / destructive-name guard (DEC-CA-07 security) ───────


def test_requires_approval_false_on_destructive_action_is_rejected():
    with pytest.raises(ValidationError, match="destructive-looking"):
        _manifest([ActionSpec(id="delete_widget", requires_approval=False)])


def test_requires_approval_false_on_destructive_name_is_rejected_via_name_field():
    with pytest.raises(ValidationError, match="destructive-looking"):
        _manifest(
            [
                ActionSpec(
                    id="widget_op", name="Purge Widget Cache", requires_approval=False
                )
            ]
        )


def test_requires_approval_false_on_non_destructive_action_is_accepted():
    _manifest([ActionSpec(id="list_widgets", requires_approval=False)])


def test_requires_approval_defaults_true():
    assert ActionSpec(id="x").requires_approval is True


# ── CA-32-W01: all on-disk manifests still load, byte-compatible ──────────


def _on_disk_manifest_paths() -> list[Path]:
    return sorted(
        Path(p)
        for p in glob.glob(str(_MANIFESTS_ROOT / "*" / "connector_manifest.yml"))
    )


def test_bundled_manifest_count_matches_measured_fleet():
    """CA-32-W01's original baseline measured 69; three fleet lanes (CA-40
    lakekeeper-mcp, CA-42 spark-mcp, CA-43 opensearch-mcp) landed on main
    ahead of this lane and brought it to 72 — re-measured at rebase time
    (2026-08-26). This assertion is a tripwire, not a magic number: if it
    fails, the fleet moved again and the count below needs re-verifying
    against reality, not bumping blindly."""
    paths = _on_disk_manifest_paths()
    assert len(paths) == 72, (
        f"expected 72 bundled connector_manifest.yml files (CA-32 lane baseline, "
        f"re-measured 2026-08-26), found {len(paths)} — the fleet inventory moved; "
        f"re-verify the lane's premise."
    )


@pytest.mark.parametrize("path", _on_disk_manifest_paths(), ids=lambda p: p.parent.name)
def test_bundled_manifest_loads_unchanged(path: Path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest = ConnectorManifest.model_validate(raw)
    # Re-dumping and re-validating must reproduce the same actions (proves the
    # new optional fields didn't silently coerce/alter anything on load).
    again = ConnectorManifest.model_validate(manifest.model_dump(mode="json"))
    assert [a.model_dump() for a in again.actions] == [
        a.model_dump() for a in manifest.actions
    ]


# ── gate: undeclared_mutating_tools / require_declared_actions ────────────


def _write_pkg_with_tool(
    tmp_path: Path,
    pkg: str,
    *,
    tags: str = "",
    annotations: str = "",
    func_name: str = "delete_widget",
) -> Path:
    pkg_root = tmp_path / pkg
    pkg_root.mkdir(parents=True)
    module = pkg_root / "mcp_server.py"
    kwargs = []
    if annotations:
        kwargs.append(f"annotations={annotations}")
    if tags:
        kwargs.append(f"tags={tags}")
    decorator_args = ", ".join(kwargs)
    module.write_text(
        "from fastmcp import FastMCP\n"
        "mcp = FastMCP('x')\n\n"
        f"@mcp.tool({decorator_args})\n"
        f"async def {func_name}(widget_id: str) -> dict:\n"
        "    return {}\n\n"
        "@mcp.tool(tags={'widgets'})\n"
        "async def list_widgets() -> dict:\n"
        "    return {}\n",
        encoding="utf-8",
    )
    return pkg_root


def test_no_mutating_signal_is_fail_open(tmp_path: Path):
    _write_pkg_with_tool(tmp_path, "widget-mcp", tags="{'widgets'}")
    manifest = _manifest([])
    assert (
        gate.undeclared_mutating_tools(
            "widget-mcp", agents_root=tmp_path, manifest=manifest
        )
        == []
    )


def test_mutating_tag_undeclared_is_caught(tmp_path: Path):
    _write_pkg_with_tool(tmp_path, "widget-mcp", tags="{'widgets', 'mutating'}")
    manifest = _manifest([])
    undeclared = gate.undeclared_mutating_tools(
        "widget-mcp", agents_root=tmp_path, manifest=manifest
    )
    assert undeclared == ["delete_widget"]
    assert "list_widgets" not in undeclared


def test_destructive_hint_annotation_undeclared_is_caught(tmp_path: Path):
    _write_pkg_with_tool(
        tmp_path,
        "widget-mcp",
        annotations="{'destructiveHint': True}",
    )
    manifest = _manifest([])
    assert gate.undeclared_mutating_tools(
        "widget-mcp", agents_root=tmp_path, manifest=manifest
    ) == ["delete_widget"]


def test_read_only_hint_false_annotation_undeclared_is_caught(tmp_path: Path):
    _write_pkg_with_tool(
        tmp_path,
        "widget-mcp",
        annotations="{'readOnlyHint': False}",
    )
    manifest = _manifest([])
    assert gate.undeclared_mutating_tools(
        "widget-mcp", agents_root=tmp_path, manifest=manifest
    ) == ["delete_widget"]


def test_read_only_hint_true_is_not_mutating(tmp_path: Path):
    _write_pkg_with_tool(
        tmp_path,
        "widget-mcp",
        annotations="{'readOnlyHint': True}",
    )
    manifest = _manifest([])
    assert (
        gate.undeclared_mutating_tools(
            "widget-mcp", agents_root=tmp_path, manifest=manifest
        )
        == []
    )


def test_declared_action_satisfies_the_gate(tmp_path: Path):
    _write_pkg_with_tool(tmp_path, "widget-mcp", tags="{'widgets', 'mutating'}")
    manifest = _manifest([ActionSpec(id="delete_widget", target_resource="Widget")])
    assert (
        gate.undeclared_mutating_tools(
            "widget-mcp", agents_root=tmp_path, manifest=manifest
        )
        == []
    )


def test_unknown_package_is_fail_open_not_a_crash(tmp_path: Path):
    manifest = _manifest([])
    assert (
        gate.undeclared_mutating_tools(
            "no-such-package", agents_root=tmp_path, manifest=manifest
        )
        == []
    )


def test_check_manifest_bytes_known_bad_fixture_fails_named(tmp_path: Path):
    """CA-32 acceptance gate #3: the CLI/runtime gate itself, not just the
    bare helper, fails closed and names the undeclared tool."""
    pkg = "widget-mcp"
    _write_pkg_with_tool(tmp_path, pkg, tags="{'widgets', 'mutating'}")
    manifest = _manifest([])
    manifest_path = tmp_path / pkg / "connector_manifest.yml"
    manifest_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    violations = gate.check_manifest_bytes(
        manifest_path, require_declared_actions=True, agents_root=tmp_path
    )
    assert any("delete_widget" in v and "[actions]" in v for v in violations)


def test_check_manifest_bytes_known_bad_fixture_passes_when_flag_off(tmp_path: Path):
    """Default gate behavior (as used by source_sync/precheck_source/the CLI
    sweep today) is unaffected — proves the new rule is opt-in, not a
    regression to any existing call site."""
    pkg = "widget-mcp"
    _write_pkg_with_tool(tmp_path, pkg, tags="{'widgets', 'mutating'}")
    manifest = _manifest([])
    manifest_path = tmp_path / pkg / "connector_manifest.yml"
    manifest_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    violations = gate.check_manifest_bytes(manifest_path)
    assert not any("[actions]" in v for v in violations)


@pytest.mark.skipif(
    not gate.resolve_agents_root().is_dir(),
    reason="live agent-packages/agents fleet checkout not present",
)
@pytest.mark.parametrize(
    "pkg", sorted(p.name for p in _MANIFESTS_ROOT.iterdir() if p.is_dir())
)
def test_undeclared_mutating_tools_against_live_fleet_source(pkg: str):
    """Runs the real gate function against the REAL sibling ``agents/<pkg>``
    source (when this checkout has it) to prove the detector inspects live
    code, not a synthetic fixture only.

    CA-32-W01 measured 8/72 packages (audio-transcriber, container-manager-mcp,
    lakekeeper-mcp, microsoft-agent, opensearch-mcp, spark-mcp,
    systems-manager, tunnel-manager — re-measured 2026-08-26 against the live
    fleet after rebase, which added three of these eight: lakekeeper-mcp,
    opensearch-mcp, spark-mcp) already tag a tool mutating without declaring
    it — a real, correct, PRE-EXISTING gap this lane's Non-goals explicitly
    leave to the packages themselves to close, so those 8 are the
    expected/known exceptions here, not a test bug. This is a floor, not a
    ceiling: the scan only recognizes ``@mcp.tool(...)`` decorator syntax, so
    a package registering tools via ``mcp.tool(...)(func)`` call syntax
    (e.g. ``genius-agent``) has an undetected mutating tool and is correctly
    absent from this set — a known scan limitation, not a clean bill of
    health.
    """
    known_undeclared = {
        "audio-transcriber",
        "container-manager-mcp",
        "lakekeeper-mcp",
        "microsoft-agent",
        "opensearch-mcp",
        "spark-mcp",
        "systems-manager",
        "tunnel-manager",
    }
    agents_root = gate.resolve_agents_root()
    raw = yaml.safe_load((_MANIFESTS_ROOT / pkg / "connector_manifest.yml").read_text())
    manifest = ConnectorManifest.model_validate(raw)
    undeclared = gate.undeclared_mutating_tools(
        pkg, agents_root=agents_root, manifest=manifest
    )
    if pkg in known_undeclared:
        assert undeclared, f"{pkg} was expected to still have undeclared mutating tools"
    else:
        assert undeclared == []
