#!/usr/bin/python
"""CONCEPT:AU-ECO.mcp.kg-skill-verb-coverage — Graph-OS domain-skill coverage.

The bundled skill suite is intentionally small: one workflow skill owns many
related Graph-OS verbs. Coverage therefore cannot be inferred from a skill slug.
Each participating skill declares its machine-readable contract in
``agents/graph-os.yaml`` while ``SKILL.md`` keeps the portable ``name`` and
``description`` frontmatter required by agent clients.

This module discovers those sidecars across installed skill providers, validates
their schema, and compares their claims with the immutable canonical ToolSpec
universe. There are no naming fallbacks, frontmatter fallbacks, or
intentionally-unskilled waivers: every required core verb and every
feature-qualified optional verb must be claimed explicitly by exactly one valid
domain sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent_utilities.mcp.tool_specs import (
    SUPPORTED_FEATURES,
    TOOL_SPECS_BY_NAME,
    canonical_tool_names,
)

GRAPH_OS_SIDECAR = Path("agents") / "graph-os.yaml"
GRAPH_OS_SCHEMA_VERSION = 2
VALID_TIERS: frozenset[str] = frozenset({"domain", "platform"})
_SIDECAR_KEYS: frozenset[str] = frozenset({"schema_version", "tier", "claims"})
_CLAIMS_KEYS: frozenset[str] = frozenset({"core", "features"})


@dataclass(frozen=True)
class SkillMeta:
    """Coverage metadata discovered from one ``agents/graph-os.yaml`` sidecar."""

    name: str
    tier: str
    core_claims: tuple[str, ...]
    feature_claims: tuple[tuple[str, tuple[str, ...]], ...]
    path: Path
    errors: tuple[str, ...] = ()

    @property
    def wraps(self) -> tuple[str, ...]:
        """All claims, retained as one deterministic validation projection."""
        optional = (tool for _feature, tools in self.feature_claims for tool in tools)
        return tuple(sorted((*self.core_claims, *optional)))

    def claims_for(self, features: frozenset[str]) -> tuple[str, ...]:
        """Claims enabled by an explicit feature profile."""
        enabled = [*self.core_claims]
        for feature, tools in self.feature_claims:
            if feature in features:
                enabled.extend(tools)
        return tuple(sorted(enabled))


@dataclass
class CoverageReport:
    """Difference between valid sidecar claims and a canonical ToolSpec profile."""

    uncovered: list[str] = field(default_factory=list)
    orphans: list[tuple[str, str]] = field(default_factory=list)
    duplicates: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    invalid_sidecars: list[tuple[str, str]] = field(default_factory=list)
    covered: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (
            self.uncovered or self.orphans or self.duplicates or self.invalid_sidecars
        )


def verb_universe(
    *, features: frozenset[str] = frozenset(), include_intent: bool = True
) -> set[str]:
    """Return the canonical Graph-OS surface for an explicit feature profile."""
    return set(canonical_tool_names(features=features, include_intent=include_intent))


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse only enough portable frontmatter to identify a skill."""
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    try:
        data = yaml.safe_load(text[3:end])
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def parse_graph_os_sidecar(path: Path, *, skill_name: str) -> SkillMeta:
    """Load and validate one Graph-OS coverage sidecar.

    The schema is deliberately small and closed:

    ``schema_version: 2``
        Identifies the contract version.
    ``tier: domain|platform``
        A domain skill must claim verbs; a platform workflow must not.
    ``claims.core: [verb, ...]``
        Sorted, unique required ToolSpec names.
    ``claims.features.<feature>: [verb, ...]``
        Sorted, unique optional ToolSpec names enabled by that feature.
    """
    errors: list[str] = []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return SkillMeta(
            skill_name,
            "",
            (),
            (),
            path,
            (f"sidecar is unreadable: {type(exc).__name__}",),
        )

    if not isinstance(raw, dict):
        return SkillMeta(skill_name, "", (), (), path, ("sidecar must be a mapping",))

    extra = sorted(set(raw) - _SIDECAR_KEYS)
    missing = sorted(_SIDECAR_KEYS - set(raw))
    if extra:
        errors.append(f"unsupported keys: {extra}")
    if missing:
        errors.append(f"missing keys: {missing}")

    version = raw.get("schema_version")
    if version != GRAPH_OS_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {GRAPH_OS_SCHEMA_VERSION}, got {version!r}"
        )

    tier = raw.get("tier")
    if not isinstance(tier, str) or tier not in VALID_TIERS:
        errors.append(f"tier must be one of {sorted(VALID_TIERS)}, got {tier!r}")
        tier = str(tier or "")

    claims_raw = raw.get("claims")
    core_claims: tuple[str, ...] = ()
    feature_claims: tuple[tuple[str, tuple[str, ...]], ...] = ()
    if not isinstance(claims_raw, dict):
        errors.append("claims must be a mapping")
    else:
        claim_extra = sorted(set(claims_raw) - _CLAIMS_KEYS)
        claim_missing = sorted(_CLAIMS_KEYS - set(claims_raw))
        if claim_extra:
            errors.append(f"claims has unsupported keys: {claim_extra}")
        if claim_missing:
            errors.append(f"claims is missing keys: {claim_missing}")

        core_raw = claims_raw.get("core")
        if not isinstance(core_raw, list) or not all(
            isinstance(item, str) and item for item in core_raw
        ):
            errors.append("claims.core must be a list of non-empty strings")
        else:
            core_claims = tuple(core_raw)
            if len(core_claims) != len(set(core_claims)):
                errors.append("claims.core must not contain duplicates")
            if list(core_claims) != sorted(core_claims):
                errors.append("claims.core must be sorted")
            for tool in core_claims:
                spec = TOOL_SPECS_BY_NAME.get(tool)
                if spec is None:
                    errors.append(f"claims.core contains unknown tool {tool!r}")
                elif spec.feature is not None:
                    errors.append(
                        f"claims.core tool {tool!r} requires feature {spec.feature!r}"
                    )

        features_raw = claims_raw.get("features")
        parsed_features: list[tuple[str, tuple[str, ...]]] = []
        if not isinstance(features_raw, dict):
            errors.append("claims.features must be a mapping")
        else:
            if list(features_raw) != sorted(features_raw):
                errors.append("claims.features keys must be sorted")
            for feature, tools_raw in features_raw.items():
                if not isinstance(feature, str) or feature not in SUPPORTED_FEATURES:
                    errors.append(
                        f"claims.features contains unsupported feature {feature!r}"
                    )
                    continue
                if not isinstance(tools_raw, list) or not all(
                    isinstance(item, str) and item for item in tools_raw
                ):
                    errors.append(
                        f"claims.features.{feature} must be a list of non-empty strings"
                    )
                    continue
                tools = tuple(tools_raw)
                if len(tools) != len(set(tools)):
                    errors.append(
                        f"claims.features.{feature} must not contain duplicates"
                    )
                if list(tools) != sorted(tools):
                    errors.append(f"claims.features.{feature} must be sorted")
                for tool in tools:
                    spec = TOOL_SPECS_BY_NAME.get(tool)
                    if spec is None:
                        errors.append(
                            f"claims.features.{feature} contains unknown tool {tool!r}"
                        )
                    elif spec.feature != feature:
                        errors.append(
                            f"claims.features.{feature} tool {tool!r} belongs to "
                            f"feature {spec.feature!r}"
                        )
                parsed_features.append((feature, tools))
        feature_claims = tuple(parsed_features)

    has_claims = bool(core_claims) or any(tools for _feature, tools in feature_claims)
    if tier == "domain" and not has_claims:
        errors.append("domain skills must claim at least one verb")
    if tier == "platform" and has_claims:
        errors.append("platform skills must use empty core and feature claims")

    return SkillMeta(
        skill_name,
        tier,
        core_claims,
        feature_claims,
        path,
        tuple(errors),
    )


def _provider_dirs() -> list[Path]:
    """Return the unified resolver's current, validated skill directories."""
    from agent_utilities.core.providers import resolve_skill_provider_dirs

    # Resolution deliberately fails closed on ambiguous global skill identities.
    # There is no direct-import fallback because it could revive retired provider
    # roots or bypass the distribution-ownership and generation checks.
    return [path for _provider, path in resolve_skill_provider_dirs()]


def discover_skills(roots: list[Path] | None = None) -> list[SkillMeta]:
    """Discover every skill that opts into Graph-OS coverage by sidecar."""
    discovered: list[SkillMeta] = []
    seen: set[str] = set()
    for root in roots or _provider_dirs():
        for sidecar in sorted(root.rglob(str(GRAPH_OS_SIDECAR))):
            key = str(sidecar.resolve())
            if key in seen:
                continue
            seen.add(key)
            skill_dir = sidecar.parent.parent
            skill_md = skill_dir / "SKILL.md"
            frontmatter = (
                _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
                if skill_md.is_file()
                else {}
            )
            name = str(frontmatter.get("name") or skill_dir.name)
            meta = parse_graph_os_sidecar(sidecar, skill_name=name)
            if not skill_md.is_file():
                meta = SkillMeta(
                    meta.name,
                    meta.tier,
                    meta.core_claims,
                    meta.feature_claims,
                    meta.path,
                    (*meta.errors, "SKILL.md is missing"),
                )
            discovered.append(meta)
    return discovered


def compute_coverage(
    roots: list[Path] | None = None,
    *,
    features: frozenset[str] | None = None,
) -> CoverageReport:
    """Compare explicit domain-skill claims with one canonical ToolSpec profile.

    ``features=None`` validates the distribution's complete contract, including
    every known optional feature.  Passing an explicit set validates that
    deployment profile without treating disabled optional claims as orphans.
    """
    selected_features = SUPPORTED_FEATURES if features is None else features
    universe = verb_universe(features=selected_features)
    report = CoverageReport()
    covered: dict[str, list[str]] = {}

    for skill in discover_skills(roots):
        for error in skill.errors:
            report.invalid_sidecars.append((skill.name, error))
        if skill.errors or skill.tier != "domain":
            continue
        for verb in skill.claims_for(selected_features):
            if verb not in universe:
                report.orphans.append((skill.name, verb))
            else:
                covered.setdefault(verb, []).append(skill.name)

    report.covered = {verb: sorted(skills) for verb, skills in sorted(covered.items())}
    report.duplicates = [
        (verb, tuple(skills))
        for verb, skills in report.covered.items()
        if len(skills) > 1
    ]
    report.uncovered = sorted(universe - set(covered))
    report.orphans.sort()
    report.invalid_sidecars.sort()
    return report


def main() -> int:
    """Run the coverage gate and print a concise deterministic report."""
    report = compute_coverage()
    if report.ok:
        print(
            "Graph-OS skill coverage OK — "
            f"{len(report.covered)} canonical tools explicitly covered by domain skills."
        )
        return 0
    if report.uncovered:
        print("UNCOVERED GRAPH-OS VERBS:")
        for verb in report.uncovered:
            print(f"  - {verb}")
    if report.orphans:
        print("SIDECAR CLAIMS FOR UNKNOWN VERBS:")
        for skill, verb in report.orphans:
            print(f"  - {skill} -> {verb}")
    if report.duplicates:
        print("GRAPH-OS VERBS WITH MULTIPLE DOMAIN OWNERS:")
        for verb, skills in report.duplicates:
            print(f"  - {verb}: {', '.join(skills)}")
    if report.invalid_sidecars:
        print("INVALID GRAPH-OS SIDECARS:")
        for skill, error in report.invalid_sidecars:
            print(f"  - {skill}: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
