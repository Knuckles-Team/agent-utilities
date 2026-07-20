from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    source = Path(__file__).parents[3] / "scripts" / "check_fleet_tls_policy.py"
    spec = importlib.util.spec_from_file_location("check_fleet_tls_policy", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_detects_production_and_documented_insecure_defaults(tmp_path):
    module = _module()
    package = tmp_path / "sample-agent"
    package.mkdir()
    (package / ".git").mkdir()
    (package / "client.py").write_text(
        "def connect(verify_ssl: bool = False):\n    return verify_ssl\n"
    )
    (package / ".env.example").write_text("SAMPLE_SSL_VERIFY=False\n")

    findings = module.scan_package(package)

    assert {finding.rule for finding in findings} >= {
        "python_false_default",
        "environment_false_default",
    }
    assert all(str(tmp_path) not in finding.path for finding in findings)


def test_secure_and_test_only_values_pass(tmp_path):
    module = _module()
    package = tmp_path / "secure-agent"
    package.mkdir()
    (package / ".git").mkdir()
    (package / "client.py").write_text(
        "def connect(verify_ssl: bool = True):\n    return verify_ssl\n"
    )
    tests = package / "tests"
    tests.mkdir()
    (tests / "test_client.py").write_text("connect(verify_ssl=False)\n")

    assert module.scan_package(package) == []
