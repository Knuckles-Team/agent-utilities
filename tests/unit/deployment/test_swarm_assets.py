"""Static contracts for the hardened GraphOS Swarm profile."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _gate():
    path = ROOT / "scripts" / "deployment" / "check_swarm_assets.py"
    spec = importlib.util.spec_from_file_location("graphos_swarm_assets_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_swarm_source_satisfies_fail_closed_contract():
    report = _gate().validate(ROOT / "deploy" / "swarm" / "graphos.stack.yml")
    assert report == {
        "ok": True,
        "services": ["engine", "front"],
        "externalSecrets": 23,
    }


def test_swarm_gate_meta_check_rejects_security_regressions():
    gate = _gate()
    gate.self_check(ROOT / "deploy" / "swarm" / "graphos.stack.yml")


def test_swarm_gate_rejects_plaintext_endpoint(tmp_path):
    gate = _gate()
    source = (ROOT / "deploy" / "swarm" / "graphos.stack.yml").read_text(
        encoding="utf-8"
    )
    broken = source.replace(
        "MCP_TOOL_MODE: intent",
        "MCP_TOOL_MODE: intent\n      GRAPH_SERVICE_ENDPOINTS: tcp://engine.invalid:9100",
    )
    path = tmp_path / "broken.yml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(gate.SwarmAssetError, match="plaintext"):
        gate.validate(path)


def test_swarm_gate_rejects_privileged_container(tmp_path):
    gate = _gate()
    source = (ROOT / "deploy" / "swarm" / "graphos.stack.yml").read_text(
        encoding="utf-8"
    )
    broken = source.replace('cap_drop: ["ALL"]', "cap_drop: []")
    path = tmp_path / "broken.yml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(gate.SwarmAssetError, match="restrictions"):
        gate.validate(path)
