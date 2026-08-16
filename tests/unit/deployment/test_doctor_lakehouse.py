"""Tests for the doctor's prescriptive lakehouse/data-plane checks and the
`interactive_apply` confirmation gate.

Covers CONCEPT:AU-OS.deployment.lakehouse-doctor: detection (absent / present-
but-unhealthy / healthy) for kafka, fuseki, seaweedfs_s3, lakekeeper,
lakekeeper_db, trino, spark_runner; the machine-readable `_prescription()`
structure attached to non-ok results; and, safety-critical, that
`interactive_apply` never applies anything without an explicit confirmation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.deployment import doctor as D


def _cfg(**overrides):
    base = dict(
        kafka_bootstrap_servers="",
        kg_fuseki_endpoint="",
        lakehouse_s3_endpoint="",
        lakekeeper_catalog_uri="",
        lakekeeper_oauth2_scope="catalog",
        lakekeeper_db_uri_ref=None,
        trino_endpoint="",
        spark_runner_endpoint="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ── absence is always non-fatal (skip), regardless of profile ───────────────


@pytest.mark.parametrize(
    ("check_name", "check_fn"),
    [
        ("kafka", D._check_kafka),
        ("fuseki", D._check_fuseki),
        ("seaweedfs_s3", D._check_seaweedfs_s3),
        ("lakekeeper", D._check_lakekeeper),
        ("lakekeeper_db", D._check_lakekeeper_db),
        ("trino", D._check_trino),
        ("spark_runner", D._check_spark_runner),
    ],
)
def test_absent_data_plane_service_is_skip_not_fatal(monkeypatch, check_name, check_fn):
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: _cfg())

    result = check_fn()

    assert result["name"] == check_name
    assert result["status"] == "skip"
    assert result["data"]["configured"] is False
    # A "laptop"/absent-service doctor run must never look unhealthy.
    assert result["status"] not in ("fail", "error")


# ── configured + non-live: declared ok, prescription withheld (nothing wrong) ──


def test_kafka_configured_non_live_is_ok_without_probing(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: _cfg(kafka_bootstrap_servers="kafka.example.invalid:9092"),
    )

    result = D._check_kafka(live=False)

    assert result["status"] == "ok"
    assert result["data"]["live_probed"] is False


# ── live probe: fail closed when unreachable, with a prescription attached ──


def test_kafka_live_probe_unreachable_fails_with_prescription(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: _cfg(kafka_bootstrap_servers="kafka.example.invalid:9092"),
    )
    monkeypatch.setattr(D, "_probe_tcp", lambda *_a, **_k: False)

    result = D._check_kafka(live=True)

    assert result["status"] == "fail"
    prescription = result["prescription"]
    assert prescription["manifest_path"] == "services/kafka/k8s/manifests.yaml"
    assert "KAFKA_BOOTSTRAP_SERVERS" in prescription["config_keys"]
    assert prescription["gotcha"]


def test_kafka_live_probe_reachable_is_ok(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: _cfg(kafka_bootstrap_servers="kafka.example.invalid:9092"),
    )
    monkeypatch.setattr(D, "_probe_tcp", lambda *_a, **_k: True)

    result = D._check_kafka(live=True)

    assert result["status"] == "ok"
    assert result["data"]["reachable"] is True


def test_fuseki_live_probe_uses_ping_endpoint(monkeypatch):
    seen_urls = []

    def fake_probe_http(url, timeout=2.0):
        seen_urls.append(url)
        return True, 200

    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: _cfg(kg_fuseki_endpoint="http://fuseki.example.invalid"),
    )
    monkeypatch.setattr(D, "_probe_http", fake_probe_http)

    result = D._check_fuseki(live=True)

    assert result["status"] == "ok"
    assert seen_urls == ["http://fuseki.example.invalid/$/ping"]


def test_seaweedfs_s3_live_probe_unreachable_fails(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: _cfg(
            lakehouse_s3_endpoint="http://s3gw.example.invalid:8333"
        ),
    )
    monkeypatch.setattr(D, "_probe_tcp", lambda *_a, **_k: False)

    result = D._check_seaweedfs_s3(live=True)

    assert result["status"] == "fail"
    assert (
        result["prescription"]["manifest_path"]
        == "services/lakehouse-seaweedfs/k8s/manifests.yaml"
    )


def test_trino_live_probe_hits_v1_info(monkeypatch):
    seen_urls = []

    def fake_probe_http(url, timeout=2.0):
        seen_urls.append(url)
        return False, None

    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: _cfg(trino_endpoint="http://trino.example.invalid:8080"),
    )
    monkeypatch.setattr(D, "_probe_http", fake_probe_http)

    result = D._check_trino(live=True)

    assert result["status"] == "fail"
    assert seen_urls == ["http://trino.example.invalid:8080/v1/info"]
    assert "476" in result["prescription"]["gotcha"]


def test_spark_runner_unreachable_live_is_warn_not_fail(monkeypatch):
    """The UI listener is disabled between jobs -- unreachable is expected, not fatal."""
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: _cfg(
            spark_runner_endpoint="http://spark.example.invalid:4040"
        ),
    )
    monkeypatch.setattr(D, "_probe_tcp", lambda *_a, **_k: False)

    result = D._check_spark_runner(live=True)

    assert result["status"] == "warn"
    assert "exec-driven" in result["prescription"]["gotcha"]


# ── lakekeeper's known gotcha: wrong catalog path / wrong oauth2 scope ───────


def test_lakekeeper_flags_missing_catalog_suffix(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: _cfg(
            lakekeeper_catalog_uri="http://catalog.example.invalid:8181/v1",
            lakekeeper_oauth2_scope="lakekeeper",
        ),
    )

    result = D._check_lakekeeper()

    assert result["status"] == "fail"
    assert "catalog" in result["detail"]
    assert result["data"]["gotcha_findings"]


def test_lakekeeper_flags_default_oauth2_scope(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: _cfg(
            lakekeeper_catalog_uri="http://catalog.example.invalid:8181/catalog",
            lakekeeper_oauth2_scope="catalog",  # left at the Iceberg client's own default
        ),
    )

    result = D._check_lakekeeper()

    assert result["status"] == "fail"
    assert "invalid_scope" in result["detail"]


def test_lakekeeper_correctly_configured_is_ok_non_live(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: _cfg(
            lakekeeper_catalog_uri="http://catalog.example.invalid:8181/catalog",
            lakekeeper_oauth2_scope="lakekeeper",
        ),
    )

    result = D._check_lakekeeper(live=False)

    assert result["status"] == "ok"
    assert result["data"]["gotcha_findings"] == []


# ── lakekeeper_db: never claims "reachable" when it cannot prove it (fail closed) ──


def test_lakekeeper_db_configured_live_is_warn_never_ok_claiming_reachable(
    monkeypatch,
):
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: _cfg(lakekeeper_db_uri_ref="vault://apps/lakekeeper-db#uri"),
    )

    result = D._check_lakekeeper_db(live=True)

    # Fail closed: this doctor cannot open a DB connection, so it must not
    # report "ok"/reachable=True just because live proof was requested.
    assert result["status"] == "warn"
    assert result["data"]["reachable"] is None


def test_lakekeeper_db_configured_non_live_is_ok(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: _cfg(lakekeeper_db_uri_ref="vault://apps/lakekeeper-db#uri"),
    )

    result = D._check_lakekeeper_db(live=False)

    assert result["status"] == "ok"


# ── wired into the real CHECKS registry + live dispatch set ─────────────────


@pytest.mark.parametrize(
    "name",
    [
        "kafka",
        "fuseki",
        "seaweedfs_s3",
        "lakekeeper",
        "lakekeeper_db",
        "trino",
        "spark_runner",
    ],
)
def test_new_checks_registered_and_live_dispatchable(name):
    assert name in D.CHECKS
    assert name in D._LIVE_CHECK_NAMES


def test_run_doctor_wires_a_new_check_end_to_end(monkeypatch):
    monkeypatch.setattr("agent_utilities.core.config.load_config", lambda: None)
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: _cfg())

    report = D.run_doctor(only=["kafka"])

    assert report["checks"][0]["name"] == "kafka"
    assert report["checks"][0]["status"] == "skip"
    assert report["status"] == "healthy"


# ── interactive_apply: the safety-critical confirmation gate ────────────────


def _failing_report_with_prescription():
    return {
        "status": "unhealthy",
        "checks": [
            D._result(
                "kafka",
                "fail",
                "Kafka bootstrap server is configured but unreachable",
                prescription=D._prescription(
                    manifest_path="services/kafka/k8s/manifests.yaml",
                    config_keys={"KAFKA_BOOTSTRAP_SERVERS": "kafka:9092"},
                    gotcha="single-broker KRaft",
                ),
            )
        ],
    }


def test_interactive_apply_default_never_applies_anything():
    """No confirm/executor supplied: this must be a strict no-op."""
    report = _failing_report_with_prescription()

    outcomes = D.interactive_apply(report, output=lambda _line: None)

    assert len(outcomes) == 1
    assert outcomes[0]["confirmed"] is False
    assert outcomes[0]["applied"] is False


def test_interactive_apply_confirm_false_never_applies_even_with_executor():
    """Explicit confirm=False must still refuse, even though an executor exists."""
    report = _failing_report_with_prescription()
    executor_called = []

    def executor(_check):
        executor_called.append(True)
        return {"applied": True}

    outcomes = D.interactive_apply(
        report, confirm=lambda _p: False, executor=executor, output=lambda _l: None
    )

    assert outcomes[0]["confirmed"] is False
    assert outcomes[0]["applied"] is False
    assert executor_called == []


def test_interactive_apply_confirmed_without_executor_stays_plan_only():
    """Confirmed, but no executor configured -- must NOT silently apply."""
    report = _failing_report_with_prescription()

    outcomes = D.interactive_apply(
        report, confirm=lambda _p: True, output=lambda _l: None
    )

    assert outcomes[0]["confirmed"] is True
    assert outcomes[0]["applied"] is False
    assert "PLAN-ONLY" in outcomes[0]["reason"]


def test_interactive_apply_confirmed_with_executor_applies_and_reruns_proof(monkeypatch):
    """Only when BOTH confirmed AND an executor is supplied does it apply --
    and the underlying check is re-run live to prove the result, not trusted
    blindly from the executor's own report."""
    report = _failing_report_with_prescription()
    rerun_calls = []

    def fake_kafka_check(*, live=False):
        rerun_calls.append(live)
        return D._result("kafka", "ok", "Kafka bootstrap server is reachable")

    def executor(_check):
        return {"applied": True, "detail": "handed off to operator, applied"}

    patched_checks = dict(D.CHECKS)
    patched_checks["kafka"] = fake_kafka_check
    monkeypatch.setattr(D, "CHECKS", patched_checks)

    outcomes = D.interactive_apply(
        report, confirm=lambda _p: True, executor=executor, output=lambda _l: None
    )

    assert outcomes[0]["confirmed"] is True
    assert outcomes[0]["applied"] is True
    assert rerun_calls == [True]  # kafka is in _LIVE_CHECK_NAMES -> re-run with live=True
    assert outcomes[0]["proof"]["status"] == "ok"


def test_interactive_apply_ignores_ok_and_skip_checks():
    report = {
        "checks": [
            D._result("kafka", "ok", "fine"),
            D._result("trino", "skip", "not configured"),
        ]
    }

    outcomes = D.interactive_apply(report, confirm=lambda _p: True, output=lambda _l: None)

    assert outcomes == []


def test_interactive_apply_skips_checks_without_a_prescription():
    """A warn/fail check with no prescription attached is never prompted for --
    there is nothing actionable to confirm."""
    report = {"checks": [D._result("python_env", "warn", "old python")]}

    outcomes = D.interactive_apply(report, confirm=lambda _p: True, output=lambda _l: None)

    assert outcomes == []


def test_main_interactive_flag_declines_on_non_interactive_terminal(monkeypatch, capsys):
    """`--interactive` on a non-TTY (CI) must decline every prompt, never hang,
    and never apply anything -- non-interactive/CI invocation must keep working
    exactly as before."""
    monkeypatch.setattr(
        D,
        "run_doctor",
        lambda *a, **k: _failing_report_with_prescription()
        | {"summary": "unhealthy: attend to ['kafka']."},
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    exit_code = D.main(["--interactive", "--only", "kafka"])

    captured = capsys.readouterr()
    assert "declining" in captured.out
    assert exit_code == 1  # unhealthy report; interactive flag does not change the verdict
