from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path
from types import ModuleType

_REQUIRED_RUNTIME_MEMBERS = {
    "agent_utilities/observability/langfuse_trust.py": b"",
    "agent_utilities/security/persistence_privacy.py": b"",
}


def _gate_module() -> ModuleType:
    source = Path(__file__).parents[2] / "scripts" / "check_wheel_privacy.py"
    spec = importlib.util.spec_from_file_location("check_wheel_privacy", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel(tmp_path: Path, members: dict[str, bytes]) -> Path:
    path = tmp_path / "synthetic.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def test_wheel_privacy_gate_accepts_runtime_only_content(tmp_path: Path) -> None:
    gate = _gate_module()
    wheel = _wheel(
        tmp_path,
        {
            "agent_utilities/__init__.py": b'__version__ = "9.8.7"',
            "agent_utilities-9.8.7.dist-info/METADATA": b"Name: agent-utilities",
            "scripts/__init__.py": b"",
            "scripts/release/promote_local_release.py": b"def main(): pass",
            "scripts/certification/campaign.py": b"def main(): pass",
            "scripts/scale/loadgen.py": b"def main(): pass",
            "deploy/__init__.py": b"",
            "deploy/release/exact-local-release-evidence.schema.json": b"{}",
            **_REQUIRED_RUNTIME_MEMBERS,
        },
    )

    assert (
        gate.wheel_privacy_findings(
            wheel, local_identifiers=frozenset({"sensitive-account"})
        )
        == ()
    )


def test_wheel_privacy_gate_rejects_unbounded_top_level_tooling(
    tmp_path: Path,
) -> None:
    gate = _gate_module()
    wheel = _wheel(
        tmp_path,
        {
            "scripts/developer_helper.py": b"pass",
            "deploy/local_override.json": b"{}",
            **_REQUIRED_RUNTIME_MEMBERS,
        },
    )

    assert gate.wheel_privacy_findings(
        wheel, local_identifiers=frozenset({"sensitive-account"})
    ) == ("non-runtime-content",)


def test_wheel_privacy_gate_rejects_missing_required_runtime_member(
    tmp_path: Path,
) -> None:
    gate = _gate_module()
    wheel = _wheel(
        tmp_path,
        {
            "agent_utilities/__init__.py": b"",
            "agent_utilities/observability/langfuse_trust.py": b"",
        },
    )

    assert gate.wheel_privacy_findings(wheel, local_identifiers=frozenset()) == (
        "missing-required-runtime-member",
    )


def test_wheel_privacy_gate_rejects_non_runtime_local_content(
    tmp_path: Path,
) -> None:
    gate = _gate_module()
    wheel = _wheel(
        tmp_path,
        {
            "tests/test_fixture.py": b"sensitive-account",
            "agent_utilities/__pycache__/module.pyc": b"/home/sensitive-account/project",
            "agent_utilities/.agent_data/images/generated.png": b"synthetic",
            **_REQUIRED_RUNTIME_MEMBERS,
        },
    )

    assert gate.wheel_privacy_findings(
        wheel, local_identifiers=frozenset({"sensitive-account"})
    ) == (
        "bytecode-content",
        "local-identity-content",
        "machine-path-content",
        "non-runtime-content",
        "runtime-state-content",
    )


def test_wheel_privacy_gate_rejects_a_planted_credential_uri(tmp_path: Path) -> None:
    """D-CIP-15/D-ORC-23: proof-of-catch for the gate's blind spot.

    Before this fix, ``check_wheel_privacy.py`` had no credential-URI,
    password, or private-key pattern at all -- a real ``user:pass@host`` DSN
    shipped in a runtime file passed silently (this is exactly the shape of
    the hardcoded ``postgresql://agent:agent@pggraph:5432/agent_kg`` default
    that shipped in the deploy wizard).
    """
    gate = _gate_module()
    wheel = _wheel(
        tmp_path,
        {
            "agent_utilities/deployment/scripts/deploy_wizard.py": (
                b'GRAPH_DB_URI = "postgresql://agent:hunter2@pggraph:5432/agent_kg"'
            ),
            **_REQUIRED_RUNTIME_MEMBERS,
        },
    )

    assert gate.wheel_privacy_findings(wheel, local_identifiers=frozenset()) == (
        "credential-content",
    )


def test_wheel_privacy_gate_rejects_a_planted_private_key(tmp_path: Path) -> None:
    gate = _gate_module()
    wheel = _wheel(
        tmp_path,
        {
            "agent_utilities/deployment/secret.pem": (
                b"-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJB\n"
                b"-----END RSA PRIVATE KEY-----\n"
            ),
            **_REQUIRED_RUNTIME_MEMBERS,
        },
    )

    assert gate.wheel_privacy_findings(wheel, local_identifiers=frozenset()) == (
        "credential-content",
    )


def test_wheel_privacy_gate_rejects_a_planted_secret_assignment(
    tmp_path: Path,
) -> None:
    gate = _gate_module()
    wheel = _wheel(
        tmp_path,
        {
            "agent_utilities/deployment/config.py": (
                b'API_KEY = "sk-live-abcdef1234567890"'
            ),
            **_REQUIRED_RUNTIME_MEMBERS,
        },
    )

    assert gate.wheel_privacy_findings(wheel, local_identifiers=frozenset()) == (
        "credential-content",
    )


def test_wheel_privacy_gate_does_not_flag_documented_placeholder_credentials(
    tmp_path: Path,
) -> None:
    """A DSN whose secret slot is an obvious, documented placeholder (the
    fixed form of the deploy-wizard default) must NOT trip the gate -- only
    a value that looks like it could actually authenticate somewhere."""
    gate = _gate_module()
    wheel = _wheel(
        tmp_path,
        {
            "agent_utilities/deployment/scripts/deploy_wizard.py": (
                b'GRAPH_DB_URI = "postgresql://agent:CHANGEME@pggraph:5432/agent_kg"'
            ),
            **_REQUIRED_RUNTIME_MEMBERS,
        },
    )

    assert gate.wheel_privacy_findings(wheel, local_identifiers=frozenset()) == ()
