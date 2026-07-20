"""Meta-tests for the repository tenant-identity source contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _gate():
    path = ROOT / "scripts" / "security" / "check_tenant_identity_contract.py"
    spec = importlib.util.spec_from_file_location("tenant_identity_gate_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repository_tenant_identity_contract_is_fail_closed():
    assert _gate().check(ROOT) == {
        "ok": True,
        "boundaries": ["rest", "mcp", "tenant-session"],
    }


def test_tenant_identity_contract_meta_check_rejects_regressions():
    gate = _gate()
    gate.self_check(ROOT)


def test_tenant_identity_contract_rejects_bypassed_verified_projection():
    gate = _gate()
    sources = gate._read_sources(ROOT)
    sources["identity"] = sources["identity"].replace(
        "return _mint_graph_session(",
        "return unverified_graph_session(",
        1,
    )
    try:
        gate.check_sources(sources)
    except gate.TenantIdentityContractError:
        return
    raise AssertionError(
        "tenant identity gate accepted an unverified session projection"
    )
