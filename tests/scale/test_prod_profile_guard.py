#!/usr/bin/python
"""Tests for the production profile guard (core/profile_guard.py)."""

import pytest

from agent_utilities.core.config import AgentConfig
from agent_utilities.core.profile_guard import (
    PROFILE_ENV_VAR,
    ProductionProfileError,
    assert_production_safe,
    collect_production_violations,
    is_production_profile,
)


def _make_config(**overrides):
    # Build a base config and override the relevant attributes after construction.
    # (pydantic-settings sources can shadow init kwargs; direct mutation is the
    # reliable way to pin the exact attribute values the guard inspects.)
    cfg = AgentConfig()
    # Hermetic baseline: the guard inspects these connection fields, and
    # pydantic-settings would otherwise source them from a leaked os.environ (test
    # order pollution — e.g. a DSN set by an earlier test). Reset to None unless a
    # test overrides them, so each scenario is pinned regardless of run order.
    for field in ("graph_db_uri", "pggraph_dsn", "graph_backend_l2"):
        if field not in overrides:
            setattr(cfg, field, None)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _toy_config(**overrides):
    base = dict(
        graph_persistence_type="file",
        graph_backend="memory",
        a2a_broker="in-memory",
        a2a_storage="in-memory",
    )
    base.update(overrides)
    return _make_config(**base)


def _prod_config(**overrides):
    base = dict(
        graph_persistence_type="postgresql",
        graph_backend="postgresql",
        a2a_broker="kafka",
        a2a_storage="postgresql",
        kafka_bootstrap_servers="redpanda-0:9092,redpanda-1:9092",
    )
    base.update(overrides)
    return _make_config(**base)


def test_unset_kafka_ledger_is_flagged_in_prod():
    # Plan 08 Synergy 2: prod requires a distributed reactive ledger.
    cfg = _prod_config(kafka_bootstrap_servers=None)
    violations = collect_production_violations(cfg)
    assert any("kafka_bootstrap_servers" in v for v in violations)


def test_is_production_profile_detection():
    assert is_production_profile("prod")
    assert is_production_profile("PROD")
    assert is_production_profile(" production ")
    assert not is_production_profile("dev")
    assert not is_production_profile("")
    assert not is_production_profile(None)


def test_prod_profile_with_file_persistence_fails():
    cfg = _toy_config()
    with pytest.raises(ProductionProfileError) as exc:
        assert_production_safe(cfg, profile="prod")
    # The error must list every offending setting (persistence, single-host graph
    # backend, broker, storage, and the unset reactive ledger — Plan 08 Synergy 2).
    assert len(exc.value.offending) == 5
    joined = "\n".join(exc.value.offending)
    assert "graph_persistence_type" in joined
    assert "graph_backend" in joined
    assert "a2a_broker" in joined
    assert "kafka_bootstrap_servers" in joined
    assert "a2a_storage" in joined


def test_sqlite_persistence_also_fails_under_prod():
    cfg = _prod_config(graph_persistence_type="sqlite")
    with pytest.raises(ProductionProfileError) as exc:
        assert_production_safe(cfg, profile="prod")
    assert any("graph_persistence_type" in o for o in exc.value.offending)
    assert len(exc.value.offending) == 1


def test_proper_prod_config_passes():
    cfg = _prod_config()
    # Should not raise.
    assert_production_safe(cfg, profile="prod")
    assert collect_production_violations(cfg) == []


def test_dev_profile_unaffected_even_with_toy_settings():
    cfg = _toy_config()
    # No raise under dev / default profiles, even though settings are toy.
    assert_production_safe(cfg, profile="dev")
    assert_production_safe(cfg, profile="")
    assert_production_safe(cfg, profile=None)


def test_default_profile_via_env_is_unaffected(monkeypatch):
    monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)
    cfg = _toy_config()
    assert_production_safe(cfg)  # reads APP_PROFILE -> unset -> no-op


def test_env_var_prod_triggers_guard(monkeypatch):
    monkeypatch.setenv(PROFILE_ENV_VAR, "prod")
    cfg = _toy_config()
    with pytest.raises(ProductionProfileError):
        assert_production_safe(cfg)


def test_config_method_delegates(monkeypatch):
    monkeypatch.setenv(PROFILE_ENV_VAR, "prod")
    with pytest.raises(ProductionProfileError):
        _toy_config().assert_production_safe()
    # proper prod config passes through the method too
    _prod_config().assert_production_safe()


def test_collect_violations_is_profile_independent():
    # collect_* always evaluates rules regardless of APP_PROFILE.
    assert len(collect_production_violations(_toy_config())) == 5
    assert collect_production_violations(_prod_config()) == []


def test_tiered_with_ladybug_l2_fails_under_prod():
    # The zero-infra default (tiered + embedded LadybugDB L2) is single-host and
    # must be rejected in production unless a durable L2 is configured.
    cfg = _prod_config(graph_backend="tiered", graph_backend_l2=None, graph_db_uri=None)
    with pytest.raises(ProductionProfileError) as exc:
        assert_production_safe(cfg, profile="prod")
    assert any("graph_backend=tiered" in o for o in exc.value.offending)


def test_tiered_with_postgres_dsn_passes_under_prod():
    # A DSN auto-promotes the tiered L2 to PostgreSQL -> production-safe.
    cfg = _prod_config(
        graph_backend="tiered",
        graph_backend_l2=None,
        graph_db_uri="postgresql://agent:agent@pg:5432/kg",
    )
    assert_production_safe(cfg, profile="prod")
    assert collect_production_violations(cfg) == []
