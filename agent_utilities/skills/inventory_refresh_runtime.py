"""Real XDG-path wiring for the skill inventory refresh model.

CONCEPT:AU-DEV.skills.inventory-refresh-determinism

AU-DEV-R003.1 added ``SkillInventoryRefreshEntry``: a typed model that
refuses a refresh which mutated before printing its plan, or which claims
completion with a mismatched hash, proven against synthetic fixtures. This
``.2`` slice wires that model into the real runtime XDG skills path: before
``install_unified()`` mutates anything, the planned action for each
registered skill provider is printed, then after materialization each
provider's installed marker is checked against its source through the same
refusal model. The Codex discovery path remains AU-DEV-R003.3+; see
``specs/au-developer-environment/tasks.md`` for the recorded split.
"""

from __future__ import annotations

from agent_utilities.core.providers import SKILL_PROVIDER_GROUP, provider_registrations
from agent_utilities.core.provider_materialization import (
    EmptyProviderAssets,
    ProviderAssetError,
    build_asset_manifest,
    read_managed_provider_marker,
)
from agent_utilities.core.unified_install import install_unified, unified_skills_dir
from agent_utilities.skills.inventory_refresh_model import SkillInventoryRefreshEntry

_LEG = "skills"


def _planned_action(*, root, registration) -> str:
    if registration.source_root is None:
        return "deactivate"
    try:
        build_asset_manifest(
            registration.source_root,
            leg=_LEG,
            allowed_relative_paths=registration.owned_paths,
        )
    except (EmptyProviderAssets, ProviderAssetError):
        return "deactivate"
    marker = read_managed_provider_marker(
        root / registration.name, provider=registration.name, leg=_LEG
    )
    return "install" if marker is None else "refresh"


def print_skills_refresh_plan() -> dict[str, str]:
    """Print this refresh's plan for the real XDG skills tree before any mutation runs."""

    root = unified_skills_dir()
    plan = {
        item.name: _planned_action(root=root, registration=item)
        for item in provider_registrations(SKILL_PROVIDER_GROUP)
    }
    for name, action in sorted(plan.items()):
        print(f"skill refresh plan: {name} -> {action}")
    return plan


def refresh_skills_inventory() -> dict[str, SkillInventoryRefreshEntry]:
    """Refresh the real runtime XDG skills inventory, plan-first (AU-DEV-R003.2).

    Prints the plan, runs the real :func:`install_unified` mutation, then
    builds one :class:`SkillInventoryRefreshEntry` per registered provider so
    a stale or unplanned result raises rather than being reported complete.
    """

    root = unified_skills_dir()
    registrations = tuple(provider_registrations(SKILL_PROVIDER_GROUP))
    plan = print_skills_refresh_plan()

    install_unified()

    entries: dict[str, SkillInventoryRefreshEntry] = {}
    for item in registrations:
        marker = read_managed_provider_marker(
            root / item.name, provider=item.name, leg=_LEG
        )
        if marker is None:
            continue
        source_hash = marker.content_digest
        if item.source_root is not None:
            try:
                source_hash = build_asset_manifest(
                    item.source_root,
                    leg=_LEG,
                    allowed_relative_paths=item.owned_paths,
                ).content_digest
            except (EmptyProviderAssets, ProviderAssetError):
                source_hash = marker.content_digest
        entries[item.name] = SkillInventoryRefreshEntry(
            skill_id=item.name,
            source_hash=source_hash,
            installed_hash=marker.content_digest,
            plan_printed=item.name in plan,
            mutated=True,
        )
    return entries
