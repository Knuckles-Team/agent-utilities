#!/usr/bin/env python3
"""Generate a privacy-safe CycloneDX SBOM and enforce license policy.

The inventory is derived from the frozen environment created by ``uv sync``.
It contains only normalized package names, versions, PURLs, and SPDX license
identifiers: installation paths, URLs, users, hosts, and environment values are
never serialized.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from collections import deque
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


class LicenseAuditError(ValueError):
    """Dependency license metadata is absent or violates policy."""


_ALLOWED_SPDX = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "BlueOak-1.0.0",
        "CC0-1.0",
        "ISC",
        "MIT",
        "MIT-CMU",
        "MPL-2.0",
        "PSF-2.0",
        "Python-2.0",
        "Unicode-3.0",
        "Unicode-DFS-2016",
        "Unlicense",
        "W3C-20150513",
        "Zlib",
    }
)
_DENIED_LICENSE = re.compile(
    r"(?:AGPL|GPL|LGPL|SSPL|BUSL|Elastic|Commons[- ]Clause|PolyForm)",
    re.IGNORECASE,
)
_EXPRESSION_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*")
_CLASSIFIER_LICENSES = {
    "Apache Software License": "Apache-2.0",
    "BSD License": "BSD-3-Clause",
    "ISC License (ISCL)": "ISC",
    "MIT License": "MIT",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "Python Software Foundation License": "PSF-2.0",
    "The Unlicense (Unlicense)": "Unlicense",
}


def _expression_is_allowed(expression: str) -> bool:
    if _DENIED_LICENSE.search(expression):
        return False
    identifiers = {
        word
        for word in _EXPRESSION_WORD.findall(expression)
        if word.upper() not in {"AND", "OR", "WITH"}
    }
    return bool(identifiers) and identifiers <= _ALLOWED_SPDX


def classify_license(text: str) -> str | None:
    """Normalize common SPDX expressions, metadata labels, or license text."""

    rendered = str(text or "").strip()
    if not rendered or rendered.casefold() in {"unknown", "none", "n/a"}:
        return None
    if _DENIED_LICENSE.search(rendered):
        raise LicenseAuditError("prohibited dependency license")
    if _expression_is_allowed(rendered):
        return rendered
    folded = " ".join(rendered.casefold().split())
    labels = {
        "apache 2": "Apache-2.0",
        "apache-2.0": "Apache-2.0",
        "apache software license": "Apache-2.0",
        "bsd": "BSD-3-Clause",
        "bsd license": "BSD-3-Clause",
        "isc": "ISC",
        "mit": "MIT",
        "mit license": "MIT",
        "mpl-2.0": "MPL-2.0",
        "mozilla public license 2.0": "MPL-2.0",
        "psf": "PSF-2.0",
        "python software foundation license": "PSF-2.0",
        "the unlicense": "Unlicense",
        "unlicense": "Unlicense",
        "zlib": "Zlib",
    }
    if folded in labels:
        return labels[folded]
    if "permission is hereby granted, free of charge" in folded:
        return "MIT"
    if "apache license" in folded and "version 2.0" in folded:
        return "Apache-2.0"
    if "redistribution and use in source and binary forms" in folded:
        if "neither the name" in folded:
            return "BSD-3-Clause"
        return "BSD-2-Clause"
    if "permission to use, copy, modify, and/or distribute" in folded:
        return "ISC"
    if "mozilla public license" in folded and "2.0" in folded:
        return "MPL-2.0"
    if "python software foundation license" in folded:
        return "PSF-2.0"
    if (
        "this is free and unencumbered software released into the public domain"
        in folded
    ):
        return "Unlicense"
    if "creative commons zero" in folded or "cc0 1.0 universal" in folded:
        return "CC0-1.0"
    if "w3c software and document license" in folded:
        return "W3C-20150513"
    if "blue oak model license" in folded:
        return "BlueOak-1.0.0"
    if "unicode license agreement" in folded:
        return "Unicode-DFS-2016"
    if "zlib license" in folded:
        return "Zlib"
    return None


def _distribution_license(distribution: importlib.metadata.Distribution) -> str:
    metadata = distribution.metadata
    for header in ("License-Expression", "License"):
        value = metadata.get(header)
        if value:
            classified = classify_license(value)
            if classified:
                return classified
    for classifier in metadata.get_all("Classifier") or ():
        prefix = "License :: OSI Approved :: "
        if classifier.startswith(prefix):
            label = classifier.removeprefix(prefix)
            classified = _CLASSIFIER_LICENSES.get(label) or classify_license(label)
            if classified:
                return classified
    for relative in distribution.files or ():
        filename = Path(str(relative)).name.casefold()
        if not filename.startswith(("license", "licence", "copying")):
            continue
        try:
            path = distribution.locate_file(relative)
            if not path.is_file() or path.stat().st_size > 512 * 1024:
                continue
            classified = classify_license(
                path.read_text(encoding="utf-8", errors="replace")
            )
        except OSError:
            continue
        if classified:
            return classified
    name = str(metadata.get("Name") or "unnamed").strip()
    raise LicenseAuditError(f"dependency license is unknown: {name}")


def _dependency_closure(
    roots: tuple[str, ...],
) -> list[importlib.metadata.Distribution]:
    installed = {
        canonicalize_name(str(dist.metadata.get("Name") or "")): dist
        for dist in importlib.metadata.distributions()
        if str(dist.metadata.get("Name") or "").strip()
    }
    requested_extras: dict[str, set[str]] = {}
    processed_extras: dict[str, frozenset[str]] = {}
    queue: deque[str] = deque()
    for root in roots:
        name = canonicalize_name(root)
        requested_extras.setdefault(name, set())
        queue.append(name)
    selected: set[str] = set()
    while queue:
        name = queue.popleft()
        distribution = installed.get(name)
        if distribution is None:
            raise LicenseAuditError(f"required dependency is not installed: {name}")
        extras = frozenset(requested_extras[name])
        if processed_extras.get(name) == extras:
            continue
        processed_extras[name] = extras
        selected.add(name)
        for raw_requirement in distribution.requires or ():
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None:
                environments = [dict(default_environment(), extra="")]
                environments.extend(
                    dict(default_environment(), extra=extra) for extra in extras
                )
                if not any(requirement.marker.evaluate(env) for env in environments):
                    continue
            dependency = canonicalize_name(requirement.name)
            before = frozenset(requested_extras.get(dependency, set()))
            requested_extras.setdefault(dependency, set()).update(requirement.extras)
            after = frozenset(requested_extras[dependency])
            if dependency not in selected or before != after:
                queue.append(dependency)
    return [installed[name] for name in sorted(selected)]


def inventory(roots: tuple[str, ...] = ("agent-utilities",)) -> list[dict[str, Any]]:
    """Return components in the selected installed dependency closure."""

    components: dict[tuple[str, str], dict[str, Any]] = {}
    for distribution in _dependency_closure(roots):
        name = canonicalize_name(str(distribution.metadata.get("Name") or ""))
        version = str(distribution.version or "").strip()
        if not name or not version:
            raise LicenseAuditError("dependency identity metadata is incomplete")
        license_id = _distribution_license(distribution)
        key = (name, version)
        components[key] = {
            "type": "library",
            "bom-ref": f"pkg:pypi/{name}@{version}",
            "name": name,
            "version": version,
            "purl": f"pkg:pypi/{name}@{version}",
            "licenses": [{"expression": license_id}],
        }
    if not components:
        raise LicenseAuditError("dependency inventory is empty")
    return [components[key] for key in sorted(components)]


def build_sbom(roots: tuple[str, ...] = ("agent-utilities",)) -> dict[str, Any]:
    """Build a deterministic, path-free CycloneDX document."""

    components = inventory(roots)
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": "pkg:pypi/agent-utilities",
                "name": "agent-utilities",
            }
        },
        "components": components,
    }


def write_sbom(document: dict[str, Any], output: Path) -> None:
    """Write one bounded regular artifact without following a symlink."""

    if output.exists() and (output.is_symlink() or not output.is_file()):
        raise LicenseAuditError("SBOM output must be a regular file")
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if len(rendered.encode("utf-8")) > 16 * 1024 * 1024:
        raise LicenseAuditError("SBOM exceeds the artifact size bound")
    output.write_text(rendered, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-sbom-licenses")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Installed root distribution to inventory (default: agent-utilities)",
    )
    args = parser.parse_args(argv)
    try:
        roots = tuple(args.root) or ("agent-utilities",)
        document = build_sbom(roots)
        if args.output is not None:
            write_sbom(document, args.output)
    except Exception as exc:  # noqa: BLE001 - privacy-safe gate boundary
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {"ok": True, "components": len(document["components"])},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
