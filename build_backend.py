"""Setuptools build backend that embeds a deterministic CycloneDX wheel SBOM."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.metadata import Metadata
from packaging.requirements import InvalidRequirement, Requirement
from setuptools import build_meta as _setuptools

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+!-]{0,127}$")
_MAX_WHEEL_BYTES = 256 * 1024 * 1024
_MAX_LICENSE_CATALOG_BYTES = 256 * 1024
_MAX_LICENSE_EXPRESSION_BYTES = 4_096
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_RUNTIME_SCRIPT_PACKAGES = frozenset({"certification", "release", "scale"})
_LICENSE_CATALOG = (
    Path(__file__)
    .with_name("agent_utilities")
    .joinpath("dependency-license-catalog.json")
)


class WheelSbomError(ValueError):
    """A wheel cannot be converted into a privacy-safe SBOM-bearing artifact."""


def _normalized_name(value: str) -> str:
    if not _NAME.fullmatch(value):
        raise WheelSbomError("wheel metadata contains an invalid package name")
    return re.sub(r"[-_.]+", "-", value).lower()


def _safe_member(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or "\\" in value or path.is_absolute() or ".." in path.parts:
        raise WheelSbomError("wheel contains an unsafe member name")
    return path


def _retain_runtime_member(value: str) -> bool:
    """Keep only installed runtime modules from the source tooling packages."""

    path = _safe_member(value)
    if not path.parts:
        return False
    if path.parts[0] == "scripts":
        return (
            path.parts == ("scripts", "__init__.py")
            or len(path.parts) >= 3
            and path.parts[1] in _RUNTIME_SCRIPT_PACKAGES
        )
    if path.parts[0] == "deploy":
        return path.parts == ("deploy", "__init__.py") or (
            len(path.parts) >= 3 and path.parts[1] == "release"
        )
    return True


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise WheelSbomError("license catalog contains duplicate fields")
        value[key] = item
    return value


def _license_catalog(path: Path) -> dict[str, str]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > _MAX_LICENSE_CATALOG_BYTES
    ):
        raise WheelSbomError("license catalog is not a bounded regular file")
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_unique_object)
    except WheelSbomError:
        raise
    except Exception as exc:
        raise WheelSbomError("license catalog is not valid JSON") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "licenses"}
        or value.get("version") != 1
        or not isinstance(value.get("licenses"), dict)
        or not value["licenses"]
        or len(value["licenses"]) > 2_048
    ):
        raise WheelSbomError("license catalog schema is invalid")
    catalog: dict[str, str] = {}
    for name, expression in value["licenses"].items():
        if (
            not isinstance(name, str)
            or _normalized_name(name) != name
            or not isinstance(expression, str)
            or not expression
            or len(expression.encode("utf-8")) > _MAX_LICENSE_EXPRESSION_BYTES
            or "\x00" in expression
        ):
            raise WheelSbomError("license catalog entry is invalid")
        raw = (
            "Metadata-Version: 2.4\n"
            "Name: license-catalog-entry\n"
            "Version: 1\n"
            f"License-Expression: {expression}\n\n"
        )
        try:
            normalized = Metadata.from_email(raw, validate=True).license_expression
        except Exception as exc:
            raise WheelSbomError("license catalog SPDX expression is invalid") from exc
        if normalized != expression:
            raise WheelSbomError("license catalog SPDX expression is not canonical")
        catalog[name] = expression
    return catalog


def _sbom(metadata: bytes, wheel_tag: str, license_catalog: Path) -> bytes:
    message = BytesParser(policy=compat32).parsebytes(metadata)
    name = str(message.get("Name") or "")
    version = str(message.get("Version") or "")
    normalized = _normalized_name(name)
    if not _VERSION.fullmatch(version):
        raise WheelSbomError("wheel metadata contains an invalid package version")
    requirements = sorted(set(message.get_all("Requires-Dist") or ()))
    grouped: dict[str, list[str]] = {}
    for declaration in requirements:
        try:
            requirement = Requirement(declaration)
        except InvalidRequirement as exc:
            raise WheelSbomError(
                "wheel metadata contains an invalid dependency"
            ) from exc
        if requirement.url is not None:
            raise WheelSbomError("wheel dependency contains a direct reference")
        grouped.setdefault(_normalized_name(requirement.name), []).append(declaration)
    licenses = _license_catalog(license_catalog)
    missing_licenses = {normalized, *grouped} - set(licenses)
    if missing_licenses:
        raise WheelSbomError("license catalog does not cover every wheel component")
    root_ref = f"pkg:pypi/{normalized}@{version}"
    components: list[dict[str, Any]] = []
    dependency_refs: list[str] = []
    for dependency, declarations in sorted(grouped.items()):
        reference = f"pkg:pypi/{dependency}"
        dependency_refs.append(reference)
        components.append(
            {
                "type": "library",
                "bom-ref": reference,
                "name": dependency,
                "purl": reference,
                "licenses": [{"expression": licenses[dependency]}],
                "properties": [
                    {"name": "python:requires-dist", "value": declaration}
                    for declaration in declarations
                ],
            }
        )
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "library",
                "bom-ref": root_ref,
                "name": normalized,
                "version": version,
                "purl": root_ref,
                "licenses": [{"expression": licenses[normalized]}],
                "properties": [{"name": "python:wheel-tag", "value": wheel_tag}],
            }
        },
        "components": components,
        "dependencies": [
            {"ref": root_ref, "dependsOn": dependency_refs},
            *({"ref": value, "dependsOn": []} for value in dependency_refs),
        ],
    }
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _record_row(name: str, payload: bytes) -> tuple[str, str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return name, "sha256=" + digest.decode("ascii"), str(len(payload))


def _new_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, _FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _normalized_info(source: zipfile.ZipInfo) -> zipfile.ZipInfo:
    """Return canonical, timestamp-free metadata for an existing wheel member."""

    info = _new_info(source.filename)
    if source.is_dir():
        info.external_attr = 0o40755 << 16
    else:
        path = _safe_member(source.filename)
        is_data_script = (
            len(path.parts) >= 3
            and path.parts[0].endswith(".data")
            and path.parts[1] == "scripts"
        )
        if is_data_script:
            info.external_attr = 0o100755 << 16
    return info


def embed_wheel_sbom(
    wheel: Path,
    *,
    license_catalog: Path | None = None,
) -> None:
    """Embed a deterministic PEP 770 CycloneDX SBOM and rebuild ``RECORD``."""

    if (
        wheel.is_symlink()
        or not wheel.is_file()
        or wheel.stat().st_size > _MAX_WHEEL_BYTES
    ):
        raise WheelSbomError("wheel input is not a bounded regular file")
    with zipfile.ZipFile(wheel, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise WheelSbomError("wheel contains duplicate members")
        for name in names:
            _safe_member(name)
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        if not (len(metadata_names) == len(wheel_names) == len(record_names) == 1):
            raise WheelSbomError("wheel metadata layout is not exact")
        dist_info = metadata_names[0].removesuffix("METADATA")
        if (
            wheel_names[0] != dist_info + "WHEEL"
            or record_names[0] != dist_info + "RECORD"
        ):
            raise WheelSbomError("wheel metadata directories differ")
        retained_infos = [
            info for info in infos if _retain_runtime_member(info.filename)
        ]
        payloads = {info.filename: archive.read(info) for info in retained_infos}
        info_by_name = {info.filename: info for info in retained_infos}

    wheel_message = BytesParser(policy=compat32).parsebytes(payloads[wheel_names[0]])
    tags = sorted(set(wheel_message.get_all("Tag") or ()))
    if not tags:
        raise WheelSbomError("wheel has no compatibility tag")
    sbom_name = dist_info + "sboms/package.cyclonedx.json"
    payloads[sbom_name] = _sbom(
        payloads[metadata_names[0]],
        ",".join(tags),
        license_catalog or _LICENSE_CATALOG,
    )
    record_name = record_names[0]
    payloads.pop(record_name, None)
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    writer.writerows(_record_row(name, payloads[name]) for name in sorted(payloads))
    writer.writerow((record_name, "", ""))
    payloads[record_name] = record.getvalue().encode("utf-8")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".wheel-sbom-", dir=wheel.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for name in sorted(payloads):
                if name in {record_name, sbom_name}:
                    continue
                archive.writestr(_normalized_info(info_by_name[name]), payloads[name])
            archive.writestr(_new_info(sbom_name), payloads[sbom_name])
            archive.writestr(_new_info(record_name), payloads[record_name])
        os.replace(temporary, wheel)
    finally:
        temporary.unlink(missing_ok=True)


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build with setuptools, then make the returned wheel SBOM-bearing."""

    filename = _setuptools.build_wheel(
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )
    embed_wheel_sbom(Path(wheel_directory) / filename)
    return filename


build_sdist = _setuptools.build_sdist
get_requires_for_build_sdist = _setuptools.get_requires_for_build_sdist
get_requires_for_build_wheel = _setuptools.get_requires_for_build_wheel
prepare_metadata_for_build_wheel = _setuptools.prepare_metadata_for_build_wheel

if hasattr(_setuptools, "build_editable"):
    build_editable = _setuptools.build_editable
    get_requires_for_build_editable = _setuptools.get_requires_for_build_editable
    prepare_metadata_for_build_editable = (
        _setuptools.prepare_metadata_for_build_editable
    )
