#!/usr/bin/python
"""Tests for the SCALE-P2-1 workload contract (docs/scaling/workload_contract.py).

Mirrors ``test_capacity_model.py``'s style: loads the module dynamically by path
(neither ``docs/scaling`` is a packaged/installed module), asserts the contract
validates, is internally consistent with the EXISTING ``capacity_model.py``
constants it is anchored to (so the two docs cannot silently drift apart), and
that scaling behaves as documented (SLOs/per-unit sizes never scale; population/
rate axes do).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT_MODULE_PATH = _REPO_ROOT / "docs" / "scaling" / "workload_contract.py"
_CAPACITY_MODEL_PATH = _REPO_ROOT / "docs" / "scaling" / "capacity_model.py"
_CONTRACT_YAML_PATH = _REPO_ROOT / "docs" / "scaling" / "workload_contract.yml"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wc = _load("agent_utilities_workload_contract", _CONTRACT_MODULE_PATH)
cm = _load("agent_utilities_capacity_model_for_wc", _CAPACITY_MODEL_PATH)


def test_yaml_file_exists():
    assert _CONTRACT_YAML_PATH.is_file()


def test_loads_and_validates():
    contract = wc.load_workload_contract(_CONTRACT_YAML_PATH)
    assert contract.registered_agents == 1_000_000
    assert contract.tenants.count > 0


def test_missing_file_raises_contract_error(tmp_path):
    with pytest.raises(wc.WorkloadContractError):
        wc.load_workload_contract(tmp_path / "does-not-exist.yml")


def test_malformed_yaml_raises_contract_error(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(wc.WorkloadContractError):
        wc.load_workload_contract(bad)


def test_missing_top_level_key_raises(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("name: x\nversion: 1\n")
    with pytest.raises(wc.WorkloadContractError):
        wc.load_workload_contract(bad)


def test_mix_fractions_must_sum_to_one(tmp_path):
    contract = wc.load_workload_contract(_CONTRACT_YAML_PATH)
    raw = dict(contract.raw)
    raw["mix"] = {"interactive_fraction": 0.5, "background_fraction": 0.6}
    import yaml

    bad = tmp_path / "bad.yml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(wc.WorkloadContractError):
        wc.load_workload_contract(bad)


def test_slo_percentiles_must_be_nondecreasing(tmp_path):
    contract = wc.load_workload_contract(_CONTRACT_YAML_PATH)
    raw = dict(contract.raw)
    raw = {**raw, "slo": {**raw["slo"]}}
    raw["slo"]["queue_latency_ms"] = {"p50": 100, "p95": 50, "p99": 200, "p99_9": 300}
    import yaml

    bad = tmp_path / "bad.yml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(wc.WorkloadContractError):
        wc.load_workload_contract(bad)


# --------------------------------------------------------------------------- #
# Anchored consistency with capacity_model.py — the two docs must not drift.
# --------------------------------------------------------------------------- #


def test_concurrency_anchored_to_capacity_model_active_agents():
    contract = wc.load_workload_contract(_CONTRACT_YAML_PATH)
    expected_active = cm.active_agents(
        contract.registered_agents, contract.reference_active_fraction
    )
    assert contract.concurrent_active_sessions == expected_active == 20_000


def test_graph_mutations_anchored_to_capacity_model_event_throughput():
    contract = wc.load_workload_contract(_CONTRACT_YAML_PATH)
    expected = cm.event_throughput_per_sec(
        contract.registered_agents, contract.reference_active_fraction
    )
    assert contract.graph_mutations_per_sec == expected == 40_000.0


def test_working_set_anchored_to_measured_footprint():
    contract = wc.load_workload_contract(_CONTRACT_YAML_PATH)
    # capacity_model.md's measured ~52 KiB working-set anchor.
    assert contract.working_set_bytes_avg == contract.resident_metadata_bytes_avg
    assert 50_000 <= contract.working_set_bytes_avg <= 55_000


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #


def test_scaled_workload_shrinks_population_and_rates():
    contract = wc.load_workload_contract(_CONTRACT_YAML_PATH)
    scaled = wc.ScaledWorkload.for_scale(contract, 0.001)
    assert scaled.registered_agents == round(contract.registered_agents * 0.001)
    assert scaled.turns_per_sec < contract.turns_per_sec
    assert scaled.tenant_count >= 2


def test_scale_one_reproduces_full_contract_population():
    contract = wc.load_workload_contract(_CONTRACT_YAML_PATH)
    scaled = wc.ScaledWorkload.for_scale(contract, 1.0)
    assert scaled.registered_agents == contract.registered_agents
    assert scaled.tenant_count == contract.tenants.count


@pytest.mark.parametrize("bad_scale", [0.0, -0.5, 1.5])
def test_invalid_scale_rejected(bad_scale):
    contract = wc.load_workload_contract(_CONTRACT_YAML_PATH)
    with pytest.raises(wc.WorkloadContractError):
        wc.ScaledWorkload.for_scale(contract, bad_scale)


def test_elephant_tenant_gets_disproportionate_weight():
    contract = wc.load_workload_contract(_CONTRACT_YAML_PATH)
    scaled = wc.ScaledWorkload.for_scale(contract, 0.01)
    elephant_idx = scaled.elephant_tenant_index()
    assert elephant_idx == 0
    # Weight function is defined for the ordinary long tail only (elephant is
    # handled via the explicit residents/active/messages fractions).
    assert scaled.tenant_weight(elephant_idx) == 0.0
    assert scaled.tenant_weight(1) > scaled.tenant_weight(2)  # zipf: earlier rank wins
