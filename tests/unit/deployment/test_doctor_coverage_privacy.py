"""Doctor coverage checks expose aggregate health, never deployment identities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_utilities.deployment import doctor as D

_LOCAL_PATH = "/private/home/agent-user/workspace/workspace.yml"
_PERSON = "Ada Lovelace"
_ENDPOINT = "https://private.example.internal/graphql"
_REPOSITORY = "customer-secret-repository"
_STALE_REPOSITORY = "person-owned-repository"
_CONNECTOR = "tenant-servicenow-production"
_STALE_CONNECTOR = "named-graphql-production"
_PRIVATE_VALUES = {
    _LOCAL_PATH,
    _PERSON,
    _ENDPOINT,
    _REPOSITORY,
    _STALE_REPOSITORY,
    _CONNECTOR,
    _STALE_CONNECTOR,
}


def _assert_private_values_absent(result: dict[str, Any]) -> None:
    detail = json.dumps(result["detail"], sort_keys=True)
    data = json.dumps(result["data"], sort_keys=True)
    report = json.dumps({"checks": [result]}, sort_keys=True)
    for value in _PRIVATE_VALUES:
        assert value not in detail
        assert value not in data
        assert value not in report
    assert result["data"]["redacted"] is True


def test_workspace_config_doctor_redacts_manifest_and_validation_details(
    monkeypatch,
) -> None:
    from agent_utilities.core import workspace_config

    monkeypatch.setattr(
        workspace_config,
        "validate_workspace_yml",
        lambda: {
            "found": True,
            "path": _LOCAL_PATH,
            "parsed": True,
            "errors": [
                f"{_PERSON} configured {_ENDPOINT} for {_REPOSITORY} at {_LOCAL_PATH}"
            ],
            "warnings": [f"{_STALE_REPOSITORY} is owned by {_PERSON}"],
            "repo_count": 2,
        },
    )

    result = D._check_workspace_config()

    assert result["status"] == "fail"
    assert result["data"] == {
        "found": True,
        "parsed": True,
        "repo_count": 2,
        "error_count": 1,
        "warning_count": 1,
        "redacted": True,
    }
    assert "path" not in result["data"]
    _assert_private_values_absent(result)


def test_ingestion_coverage_doctor_reports_counts_only(monkeypatch) -> None:
    from agent_utilities.knowledge_graph import backends
    from agent_utilities.knowledge_graph.ingestion import coverage, manifest

    monkeypatch.setattr(coverage, "find_workspace_manifest", lambda: Path(_LOCAL_PATH))
    monkeypatch.setattr(
        coverage,
        "enumerate_agent_packages_repos",
        lambda _path: [_REPOSITORY, _STALE_REPOSITORY, _PERSON],
    )
    monkeypatch.setattr(
        coverage,
        "repo_symbol_counts",
        lambda _backend, _repos: {
            _REPOSITORY: 0,
            _STALE_REPOSITORY: 9,
            _PERSON: 0,
        },
    )
    monkeypatch.setattr(
        coverage,
        "assess_coverage",
        lambda _repos, _counts, _freshness: {
            "total": 3,
            "covered": 1,
            "missing": [_REPOSITORY, _PERSON],
            "stale": [{"repo": _STALE_REPOSITORY, "age_days": 30.0}],
            "coverage_pct": 33.3,
            "total_symbols": 9,
            "sla_days": 7,
        },
    )
    monkeypatch.setattr(backends, "get_active_backend", lambda: object())

    class PrivateManifest:
        def __init__(self, *, backend: object) -> None:
            self.backend = backend

        def freshness(self, _graph: str, _category: str) -> dict[str, str]:
            return {_ENDPOINT: "2026-01-01T00:00:00+00:00"}

    monkeypatch.setattr(manifest, "DeltaManifest", PrivateManifest)

    result = D._check_ingestion_coverage()

    assert result["status"] == "fail"
    assert result["data"] == {
        "total": 3,
        "covered": 1,
        "missing_count": 2,
        "stale_count": 1,
        "coverage_pct": 33.3,
        "total_symbols": 9,
        "sla_days": 7,
        "redacted": True,
    }
    assert "missing" not in result["data"]
    assert "stale" not in result["data"]
    _assert_private_values_absent(result)


def test_connector_coverage_doctor_reports_counts_only(monkeypatch) -> None:
    from agent_utilities.knowledge_graph import backends
    from agent_utilities.knowledge_graph.ingestion import connector_coverage, manifest

    monkeypatch.setattr(
        connector_coverage,
        "enumerate_expected_connectors",
        lambda: [_CONNECTOR, _STALE_CONNECTOR],
    )
    monkeypatch.setattr(
        connector_coverage,
        "assess_connector_coverage",
        lambda _expected, _freshness: {
            "total": 2,
            "covered": 1,
            "missing": [_CONNECTOR],
            "stale": [{"connector": _STALE_CONNECTOR, "age_days": 30.0}],
            "coverage_pct": 50.0,
            "sla_days": 7,
        },
    )
    monkeypatch.setattr(backends, "get_active_backend", lambda: object())

    class PrivateManifest:
        def __init__(self, *, backend: object) -> None:
            self.backend = backend

        def freshness(self, _graph: str, _category: str) -> dict[str, str]:
            return {_ENDPOINT: "2026-01-01T00:00:00+00:00"}

    monkeypatch.setattr(manifest, "DeltaManifest", PrivateManifest)

    result = D._check_connector_coverage()

    assert result["status"] == "warn"
    assert result["data"] == {
        "total": 2,
        "covered": 1,
        "missing_count": 1,
        "stale_count": 1,
        "coverage_pct": 50.0,
        "sla_days": 7,
        "redacted": True,
    }
    assert "missing" not in result["data"]
    assert "stale" not in result["data"]
    _assert_private_values_absent(result)
