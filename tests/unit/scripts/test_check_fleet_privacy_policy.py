from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    source = Path(__file__).parents[3] / "scripts" / "check_fleet_privacy_policy.py"
    spec = importlib.util.spec_from_file_location("check_fleet_privacy_policy", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reports_only_location_and_rule_for_sensitive_material(tmp_path):
    module = _module()
    package = tmp_path / "sample-agent"
    package.mkdir()
    (package / ".git").mkdir()
    private_endpoint = "https://private.service." + "kob." + "example/graphql"
    synthetic_token = "sk" + "-lf-" + "syntheticvalue"
    (package / "config.md").write_text(
        f"endpoint={private_endpoint}\ntoken={synthetic_token}\n"
    )

    findings = module.scan_package(
        package,
        all_files=True,
        denied_identifiers=(),
    )

    assert {finding.rule for finding in findings} == {
        "credential_token_material",
        "internal_endpoint",
    }
    assert all(finding.path == "config.md" for finding in findings)


def test_portable_placeholders_and_standard_ca_settings_pass(tmp_path):
    module = _module()
    package = tmp_path / "secure-agent"
    package.mkdir()
    (package / ".git").mkdir()
    (package / ".env.example").write_text(
        "SERVICE_URL=https://service.example.invalid\n"
        "SSL_CERT_FILE=/run/secrets/private-ca-bundle.pem\n"
        "WORKSPACE_ROOT=${AGENT_UTILITIES_WORKSPACE_ROOT}\n"
    )

    assert (
        module.scan_package(
            package,
            all_files=True,
            denied_identifiers=(),
        )
        == []
    )


def test_standard_container_host_alias_is_not_an_internal_endpoint(tmp_path):
    module = _module()
    package = tmp_path / "container-agent"
    package.mkdir()
    (package / ".git").mkdir()
    (package / "README.md").write_text(
        "Configure http://host.docker.internal:8000/v1 at deployment time.\n"
    )

    assert (
        module.scan_package(
            package,
            all_files=True,
            denied_identifiers=(),
        )
        == []
    )


def test_non_organizational_internal_test_endpoint_is_a_fixture(tmp_path):
    module = _module()
    package = tmp_path / "fixture-agent"
    tests = package / "tests"
    tests.mkdir(parents=True)
    (package / ".git").mkdir()
    endpoint = "https://service." + "internal/api"
    (tests / "test_transport.py").write_text(f'ENDPOINT = "{endpoint}"\n')

    assert module.scan_package(package, all_files=True, denied_identifiers=()) == []


def test_organizational_test_endpoint_remains_in_scope(tmp_path):
    module = _module()
    package = tmp_path / "fixture-agent"
    tests = package / "tests"
    tests.mkdir(parents=True)
    (package / ".git").mkdir()
    endpoint = "https://service." + "kob." + "example/api"
    (tests / "test_transport.py").write_text(f'ENDPOINT = "{endpoint}"\n')

    findings = module.scan_package(package, all_files=True, denied_identifiers=())

    assert [finding.rule for finding in findings] == ["internal_endpoint"]


def test_portable_path_identity_remains_valid_in_test_fixture(tmp_path):
    module = _module()
    package = tmp_path / "fixture-agent"
    tests = package / "tests"
    tests.mkdir(parents=True)
    (package / ".git").mkdir()
    portable_path = "C:\\Users\\agent-user\\Workspace\\project"
    (tests / "test_paths.py").write_text(f'ROOT = r"{portable_path}"\n')

    assert module.scan_package(package, all_files=True, denied_identifiers=()) == []


def test_terminal_portable_identity_and_route_id_are_not_local_paths(tmp_path):
    module = _module()
    package = tmp_path / "fixture-agent"
    tests = package / "tests"
    tests.mkdir(parents=True)
    (package / ".git").mkdir()
    (tests / "test_paths.py").write_text(
        'ROOT = r"C:\\Users\\agent-user"\nROUTE = "route:GET:/users/{id}"\n'
    )

    assert module.scan_package(package, all_files=True, denied_identifiers=()) == []


def test_nonportable_path_identity_remains_in_scope_in_tests(tmp_path):
    module = _module()
    package = tmp_path / "fixture-agent"
    tests = package / "tests"
    tests.mkdir(parents=True)
    (package / ".git").mkdir()
    windows_user_root = "C:\\Users\\"
    nonportable_path = windows_user_root + "environment-owner\\Workspace\\project"
    (tests / "test_paths.py").write_text(f'ROOT = r"{nonportable_path}"\n')

    findings = module.scan_package(package, all_files=True, denied_identifiers=())

    assert {finding.rule for finding in findings} == {
        "local_user_path",
        "workspace_absolute_path",
    }


def test_wsl_virtual_environment_is_outside_source_scope(tmp_path):
    module = _module()
    package = tmp_path / "secure-agent"
    environment = package / ".venv-wsl" / "lib"
    environment.mkdir(parents=True)
    (package / ".git").mkdir()
    (package / "runtime.py").write_text("SAFE = True\n")
    synthetic_token = "sk-" + "lf-" + "syntheticvalue"
    (environment / "generated.py").write_text('TOKEN = "' + synthetic_token + '"\n')

    assert (
        module.scan_package(
            package,
            all_files=True,
            denied_identifiers=(),
        )
        == []
    )


def test_scanner_credential_fixtures_do_not_self_report(tmp_path):
    module = _module()
    package = tmp_path / "scanner-agent"
    scanner = package / "scripts" / "security_sanitizer.py"
    scanner.parent.mkdir(parents=True)
    (package / ".git").mkdir()
    synthetic_token = "sk-" + "lf-" + "syntheticvalue"
    scanner.write_text(f'PLACEHOLDERS = {{"{synthetic_token}"}}\n')
    (package / "runtime.py").write_text(f'TOKEN = "{synthetic_token}"\n')

    findings = module.scan_package(
        package,
        all_files=True,
        denied_identifiers=(),
    )

    assert [(finding.path, finding.rule) for finding in findings] == [
        ("runtime.py", "credential_token_material")
    ]
