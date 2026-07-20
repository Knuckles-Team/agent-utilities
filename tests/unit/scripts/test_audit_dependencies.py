"""Security-contract tests for the fail-closed dependency audit."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from scripts import audit_dependencies as dependency_audit


def _write_lock(path: Path) -> None:
    path.write_text(
        """\
version = 1

[[package]]
name = "Example_Name"
version = "1.0.0"
source = { registry = "https://packages.example.invalid/simple" }
sdist = { url = "https://packages.example.invalid/example-1.0.0.tar.gz", hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }

[[package]]
name = "example-name"
version = "2.0.0"
source = { registry = "https://packages.example.invalid/simple" }
wheels = [{ url = "https://packages.example.invalid/example-2.0.0.whl", hash = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" }]
""",
        encoding="utf-8",
    )


def test_lock_parser_audits_every_platform_selection(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    _write_lock(lock)

    assert dependency_audit.parse_lock(lock) == (
        ("example-name", "1.0.0"),
        ("example-name", "2.0.0"),
    )


def test_lock_parser_refuses_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.lock"
    link = tmp_path / "uv.lock"
    _write_lock(target)
    link.symlink_to(target)

    with pytest.raises(dependency_audit.AuditError, match="safe bound"):
        dependency_audit.parse_lock(link)


def test_lock_parser_refuses_registry_artifact_without_hash(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(
        """\
version = 1
[[package]]
name = "example"
version = "1.0.0"
source = { registry = "https://packages.example.invalid/simple" }
wheels = [{ url = "https://packages.example.invalid/example.whl" }]
""",
        encoding="utf-8",
    )

    with pytest.raises(dependency_audit.AuditError, match="unverified artifact"):
        dependency_audit.parse_lock(lock)


def test_acceptance_requires_exact_advisory_and_short_expiry(tmp_path: Path) -> None:
    expiry = dt.date.today() + dt.timedelta(days=10)
    (tmp_path / ".security-audit-allow.txt").write_text(
        f"OSV-EXAMPLE example-name expires={expiry.isoformat()} # Reviewed temporary exposure.\n",
        encoding="utf-8",
    )

    accepted = dependency_audit.load_acceptances(tmp_path)

    assert set(accepted) == {("osv-example", "example-name")}


@pytest.mark.parametrize(
    "declaration",
    [
        "example-name # Package-wide suppression is forbidden.",
        "OSV-EXAMPLE example-name # Missing expiry is forbidden.",
    ],
)
def test_acceptance_rejects_broad_or_incomplete_suppressions(
    tmp_path: Path,
    declaration: str,
) -> None:
    (tmp_path / ".security-audit-allow.txt").write_text(
        declaration + "\n",
        encoding="utf-8",
    )

    with pytest.raises(dependency_audit.AuditError):
        dependency_audit.load_acceptances(tmp_path)


def test_service_failure_is_closed_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = tmp_path / "uv.lock"
    _write_lock(lock)
    monkeypatch.delenv("SECURITY_AUDIT_OFFLINE_POLICY", raising=False)
    monkeypatch.setattr(
        dependency_audit,
        "audit",
        lambda _packages: (_ for _ in ()).throw(
            dependency_audit.AuditError("OSV service is unavailable")
        ),
    )

    assert dependency_audit.main([str(lock)]) == 2


def test_explicit_local_offline_policy_may_warn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = tmp_path / "uv.lock"
    _write_lock(lock)
    monkeypatch.setenv("SECURITY_AUDIT_OFFLINE_POLICY", "warn")
    monkeypatch.setattr(
        dependency_audit,
        "audit",
        lambda _packages: (_ for _ in ()).throw(
            dependency_audit.AuditError("OSV service is unavailable")
        ),
    )

    assert dependency_audit.main([str(lock)]) == 0
