#!/usr/bin/python
"""Distribution-owned provider discovery and current runtime asset resolution.

Provider entry points identify data directories owned by the registering wheel.  The
resolver uses distribution metadata rather than importing provider packages, rejects
ambiguous ownership, validates every source as bounded regular files, and selects an
XDG generation only when its registration and content digests match the installed
source exactly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import Distribution, EntryPoint, entry_points
from pathlib import Path, PurePosixPath

from agent_utilities.core.provider_materialization import (
    MAX_PROVIDER_FILE_BYTES,
    AssetManifest,
    ProviderAssetError,
    ProviderMaterializationError,
    build_asset_manifest,
    is_safe_provider_name,
    marker_path_exists,
    registration_digest,
    resolve_managed_generation,
)

logger = logging.getLogger(__name__)

SKILL_PROVIDER_GROUP = "agent_utilities.skill_providers"
PROMPT_PROVIDER_GROUP = "agent_utilities.prompt_providers"
ONTOLOGY_PROVIDER_GROUP = "agent_utilities.ontology_providers"

_GROUP_LEGS = {
    SKILL_PROVIDER_GROUP: "skills",
    PROMPT_PROVIDER_GROUP: "prompts",
    ONTOLOGY_PROVIDER_GROUP: "ontologies",
}
_MODULE_NAME = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$", re.ASCII
)
_PROVIDER_GROUP = re.compile(r"^agent_utilities\.[A-Za-z0-9_.-]+_providers$", re.ASCII)
_FRONTMATTER_NAME = re.compile(r"(?m)^name:\s*[\"']?([^\r\n#\"']+?)[\"']?\s*$")


class ProviderRegistrationError(ProviderMaterializationError):
    """Provider metadata is invalid or does not prove source ownership."""


class ProviderRegistrationConflict(ProviderRegistrationError):
    """Two registrations claim the same portable provider identity."""


class DuplicateSkillIdentity(ProviderMaterializationError):
    """Two current skill sources declare the same skill identity."""


def _is_linklike(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    """A path-free provider identity tied to one owning distribution."""

    name: str
    group: str
    target: str
    owner_name: str
    owner_version: str
    digest: str
    source_root: Path | None
    owned_paths: frozenset[str]


@dataclass(frozen=True, slots=True)
class ProviderAssets:
    """One registration paired with its verified current source manifest."""

    registration: ProviderRegistration
    source_root: Path
    manifest: AssetManifest


def _distribution_identity(distribution: Distribution) -> tuple[str, str]:
    try:
        name = str(distribution.metadata["Name"]).strip()
        version = str(distribution.version).strip()
    except Exception as exc:  # noqa: BLE001 - converted to a path-free typed failure
        raise ProviderRegistrationError(
            "provider owner metadata is incomplete"
        ) from exc
    if not name or not version:
        raise ProviderRegistrationError("provider owner metadata is incomplete")
    return name, version


def _owned_source_root(
    entry_point: EntryPoint, distribution: Distribution
) -> tuple[Path, frozenset[str]]:
    target = entry_point.value.strip()
    if _MODULE_NAME.fullmatch(target) is None:
        raise ProviderRegistrationError(
            "provider target must be one data package module"
        )
    prefix = PurePosixPath(*target.split("."))
    distribution_files = distribution.files
    if distribution_files is None:
        raise ProviderRegistrationError("provider owner has no auditable file manifest")

    roots: set[Path] = set()
    owned_paths: set[str] = set()
    for package_path in distribution_files:
        relative = PurePosixPath(str(package_path).replace("\\", "/"))
        if len(relative.parts) <= len(prefix.parts):
            continue
        if relative.parts[: len(prefix.parts)] != prefix.parts:
            continue
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise ProviderRegistrationError("provider owner file manifest is unsafe")
        located = Path(distribution.locate_file(package_path))
        root = located
        for _ in relative.parts[len(prefix.parts) :]:
            root = root.parent
        roots.add(root)
        owned_paths.add(PurePosixPath(*relative.parts[len(prefix.parts) :]).as_posix())
    if not owned_paths or len(roots) != 1:
        raise ProviderRegistrationError(
            "provider target is not owned by its distribution"
        )
    root = roots.pop()
    try:
        root = root.resolve(strict=True)
        info = root.lstat()
    except OSError as exc:
        raise ProviderRegistrationError("provider source root is unavailable") from exc
    if _is_linklike(root) or not stat.S_ISDIR(info.st_mode):
        raise ProviderRegistrationError(
            "provider source root is not a regular directory"
        )
    return root, frozenset(owned_paths)


def provider_registrations(group: str) -> tuple[ProviderRegistration, ...]:
    """Return unambiguous, distribution-owned registrations for one known leg."""

    if _PROVIDER_GROUP.fullmatch(group) is None:
        raise ValueError("provider entry-point group is unsupported")
    try:
        discovered = tuple(entry_points(group=group))
    except Exception as exc:  # noqa: BLE001
        raise ProviderRegistrationError(
            "provider metadata cannot be enumerated"
        ) from exc

    registrations: list[ProviderRegistration] = []
    identities: dict[str, str] = {}
    for entry_point in sorted(
        discovered,
        key=lambda item: (
            item.name.casefold(),
            item.name,
            item.value,
            str(getattr(getattr(item, "dist", None), "version", "")),
        ),
    ):
        if not is_safe_provider_name(entry_point.name):
            raise ProviderRegistrationError("provider name is not a portable component")
        portable = entry_point.name.casefold()
        if portable in identities:
            raise ProviderRegistrationConflict("provider name has multiple owners")
        distribution = getattr(entry_point, "dist", None)
        if distribution is None:
            raise ProviderRegistrationError("provider registration has no owner")
        owner_name, owner_version = _distribution_identity(distribution)
        digest = registration_digest(
            {
                "group": group,
                "provider": entry_point.name,
                "target": entry_point.value,
                "owner": owner_name.casefold(),
                "version": owner_version,
            }
        )
        try:
            source_root, owned_paths = _owned_source_root(entry_point, distribution)
        except ProviderRegistrationError:
            source_root = None
            owned_paths = frozenset()
        identities[portable] = digest
        registrations.append(
            ProviderRegistration(
                name=entry_point.name,
                group=group,
                target=entry_point.value,
                owner_name=owner_name,
                owner_version=owner_version,
                digest=digest,
                source_root=source_root,
                owned_paths=owned_paths,
            )
        )
    return tuple(registrations)


def registered_provider_names(group: str) -> tuple[str, ...]:
    """Return safe, unambiguous provider names without importing provider modules."""

    return tuple(item.name for item in provider_registrations(group))


def current_provider_assets(group: str) -> tuple[ProviderAssets, ...]:
    """Resolve and validate all current sources, isolating provider-local failures."""

    if _PROVIDER_GROUP.fullmatch(group) is None:
        raise ValueError("provider entry-point group is unsupported")
    leg = _GROUP_LEGS.get(group, "data")
    assets: list[ProviderAssets] = []
    for item in provider_registrations(group):
        if item.source_root is None:
            logger.warning("Provider assets unavailable (error_code=source_unresolved)")
            continue
        try:
            manifest = build_asset_manifest(
                item.source_root,
                leg=leg,
                allowed_relative_paths=item.owned_paths,
            )
        except (OSError, ProviderAssetError, ValueError) as exc:
            logger.warning(
                "Provider assets unavailable (exception_type=%s)", type(exc).__name__
            )
            continue
        assets.append(
            ProviderAssets(
                registration=item,
                source_root=item.source_root,
                manifest=manifest,
            )
        )
    return tuple(assets)


def iter_provider_dirs(group: str) -> list[tuple[str, Path]]:
    """Return verified current package roots without importing provider code."""

    return [
        (assets.registration.name, assets.source_root)
        for assets in current_provider_assets(group)
    ]


def _managed_or_source(assets: ProviderAssets, *, leg: str, root: Path) -> Path:
    materialized = resolve_managed_generation(
        root / assets.registration.name,
        provider=assets.registration.name,
        leg=leg,
        registration=assets.registration.digest,
        source_manifest=assets.manifest,
    )
    return materialized or assets.source_root


def resolve_base_prompt_dir() -> Path:
    """Resolve the hub's validated prompt generation or its current trusted source."""

    from agent_utilities.core.paths import unified_prompts_dir
    from agent_utilities.core.unified_install import OWN_PROVIDER, own_provider_asset

    source, digest, manifest = own_provider_asset("prompts")
    return (
        resolve_managed_generation(
            unified_prompts_dir() / OWN_PROVIDER,
            provider=OWN_PROVIDER,
            leg="prompts",
            registration=digest,
            source_manifest=manifest,
        )
        or source
    )


def resolve_prompt_provider_dirs() -> list[tuple[str, Path]]:
    """Resolve each current prompt provider to a verified generation or live source."""

    from agent_utilities.core.paths import unified_prompts_dir

    root = unified_prompts_dir()
    resolved = [
        (
            assets.registration.name,
            _managed_or_source(assets, leg="prompts", root=root),
        )
        for assets in current_provider_assets(PROMPT_PROVIDER_GROUP)
        if assets.registration.name != "agent-utilities"
    ]
    return sorted(resolved, key=lambda item: item[0].casefold())


def _skill_dirs(root: Path) -> list[Path]:
    return sorted({path.parent for path in root.rglob("SKILL.md")})


def _skill_identity(skill_root: Path) -> str:
    path = skill_root / "SKILL.md"
    try:
        info = path.lstat()
        if (
            _is_linklike(path)
            or not path.is_file()
            or info.st_size > MAX_PROVIDER_FILE_BYTES
        ):
            raise ProviderAssetError("skill instruction file is invalid")
        with path.open("r", encoding="utf-8") as handle:
            text = handle.read(MAX_PROVIDER_FILE_BYTES + 1)
    except (OSError, UnicodeDecodeError) as exc:
        raise ProviderAssetError("skill instruction file cannot be read") from exc
    if len(text.encode("utf-8")) > MAX_PROVIDER_FILE_BYTES:
        raise ProviderAssetError("skill instruction file exceeds its bound")
    if text.startswith("---"):
        end = text.find("---", 3)
        frontmatter = text[3:end] if end >= 0 else text[:4096]
        match = _FRONTMATTER_NAME.search(frontmatter)
        if match:
            identity = match.group(1).strip()
            if (
                identity
                and len(identity) <= 256
                and all(ord(ch) >= 32 for ch in identity)
            ):
                return identity
    return skill_root.name


def _own_skill_assets() -> tuple[str, str, Path, AssetManifest]:
    from agent_utilities._version import __version__

    root = Path(__file__).resolve().parent.parent / "skills"
    manifest = build_asset_manifest(root, leg="skills")
    digest = registration_digest(
        {
            "group": SKILL_PROVIDER_GROUP,
            "provider": "agent-utilities",
            "target": "agent_utilities.skills",
            "owner": "agent-utilities",
            "version": __version__,
        }
    )
    return "agent-utilities", digest, root, manifest


def _add_skill(
    selected: dict[str, tuple[str, Path]], *, identity: str, provider: str, root: Path
) -> None:
    portable_identity = identity.casefold()
    if portable_identity in selected:
        raise DuplicateSkillIdentity("duplicate current skill identity")
    selected[portable_identity] = (provider, root)


def resolve_skill_provider_dirs() -> list[tuple[str, Path]]:
    """Return exactly one verified source per globally unique current skill."""

    from agent_utilities.core.paths import skills_dir

    xdg_root = skills_dir()
    selected: dict[str, tuple[str, Path]] = {}

    provider_assets = [
        assets
        for assets in current_provider_assets(SKILL_PROVIDER_GROUP)
        if assets.registration.name != "agent-utilities"
    ]
    own_name, own_digest, own_root, own_manifest = _own_skill_assets()
    own_materialized = resolve_managed_generation(
        xdg_root / own_name,
        provider=own_name,
        leg="skills",
        registration=own_digest,
        source_manifest=own_manifest,
    )
    provider_roots: list[tuple[str, Path]] = [(own_name, own_materialized or own_root)]
    provider_roots.extend(
        (
            assets.registration.name,
            _managed_or_source(assets, leg="skills", root=xdg_root),
        )
        for assets in provider_assets
    )
    for provider, root in sorted(provider_roots, key=lambda item: item[0].casefold()):
        for skill_root in _skill_dirs(root):
            _add_skill(
                selected,
                identity=_skill_identity(skill_root),
                provider=provider,
                root=skill_root,
            )

    try:
        children: Iterable[Path] = sorted(
            xdg_root.iterdir(), key=lambda item: item.name.casefold()
        )
    except OSError:
        children = ()
    for child in children:
        try:
            info = child.lstat()
        except OSError:
            continue
        if _is_linklike(child) or not stat.S_ISDIR(info.st_mode):
            continue
        # A marker entry reserves this root forever as provider-owned. Corrupt,
        # inactive, retired, and current provider roots are never reinterpreted as
        # flat operator skills.
        if marker_path_exists(child):
            continue
        if not (child / "SKILL.md").is_file():
            continue
        build_asset_manifest(child, leg="skills")
        _add_skill(
            selected,
            identity=_skill_identity(child),
            provider="xdg-local",
            root=child,
        )
    return [selected[identity] for identity in sorted(selected)]


def provider_registry_digest(group: str) -> str:
    """Return an opaque digest for diagnostics without exposing provider metadata."""

    payload = [item.digest for item in provider_registrations(group)]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()
