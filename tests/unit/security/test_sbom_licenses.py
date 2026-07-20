"""License policy and SBOM privacy contracts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _gate():
    path = ROOT / "scripts" / "security" / "check_sbom_licenses.py"
    spec = importlib.util.spec_from_file_location("sbom_license_gate_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("MIT", "MIT"),
        ("MIT-CMU", "MIT-CMU"),
        ("Apache-2.0 OR BSD-3-Clause", "Apache-2.0 OR BSD-3-Clause"),
        ("Mozilla Public License 2.0", "MPL-2.0"),
        ("Permission is hereby granted, free of charge", "MIT"),
    ],
)
def test_license_classifier_normalizes_allowlisted_licenses(value, expected):
    assert _gate().classify_license(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "GPL-3.0",
        "AGPL-3.0",
        "SSPL-1.0",
        "BUSL-1.1",
        "MIT-CMU OR GPL-3.0",
    ],
)
def test_license_classifier_rejects_prohibited_licenses(value):
    gate = _gate()
    with pytest.raises(gate.LicenseAuditError, match="prohibited"):
        gate.classify_license(value)


def test_sbom_writer_never_serializes_installation_context(tmp_path):
    gate = _gate()
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "components": [
            {
                "type": "library",
                "name": "example",
                "version": "1.0",
                "purl": "pkg:pypi/example@1.0",
                "licenses": [{"expression": "MIT"}],
            }
        ],
    }
    output = tmp_path / "sbom.json"
    gate.write_sbom(document, output)
    rendered = output.read_text(encoding="utf-8")
    assert json.loads(rendered) == document
    assert str(tmp_path) not in rendered
