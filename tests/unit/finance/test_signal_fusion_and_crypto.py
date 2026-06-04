"""Plan 02: real alpha-combination math + typed crypto provider errors.

No network: crypto HTTP is monkeypatched. Proves the no-stub remediation —
no NotImplementedError, no [Mock], no silent equal-weight.
"""

from __future__ import annotations

import math

import pytest

from agent_utilities.domains.finance import crypto_connector as cc
from agent_utilities.domains.finance.errors import ProviderNotConfigured
from agent_utilities.domains.finance.signal_fusion import AlphaCombinationEngine


# ---- signal_fusion -----------------------------------------------------------

def test_alpha_weights_not_uniform_for_distinct_signals():
    eng = AlphaCombinationEngine()
    # Signal 0 has a clear positive mean edge; signal 1 is noise around zero.
    returns = [
        [0.02, 0.03, 0.01, 0.025, 0.02],
        [0.001, -0.002, 0.0015, -0.001, 0.0005],
    ]
    w = eng.compute_weights(returns)
    assert len(w) == 2
    assert math.isclose(sum(w), 1.0, abs_tol=1e-9)
    # The edge-carrying signal must not get an equal share.
    assert abs(w[0] - 0.5) > 1e-6


def test_alpha_weights_single_and_empty():
    eng = AlphaCombinationEngine()
    assert eng.compute_weights([]) == []
    assert eng.compute_weights([[0.1, 0.2, 0.3]]) == [1.0]


def test_alpha_weights_reject_degenerate_input():
    eng = AlphaCombinationEngine()
    with pytest.raises(ValueError):
        eng.compute_weights([[0.1], [0.2]])  # only 1 observation each


# ---- crypto_connector --------------------------------------------------------

def test_get_tvl_parses_defillama(monkeypatch):
    monkeypatch.setattr(cc, "_http_get_json", lambda url, timeout=10.0: 1234567.5)
    assert cc.OnChainAnalytics().get_tvl("aave") == pytest.approx(1234567.5)


def test_get_dex_volume_parses_defillama(monkeypatch):
    monkeypatch.setattr(
        cc, "_http_get_json", lambda url, timeout=10.0: {"total24h": 9.99e8}
    )
    assert cc.OnChainAnalytics().get_dex_volume("uniswap") == pytest.approx(9.99e8)


def test_get_funding_rate_parses_binance(monkeypatch):
    monkeypatch.setattr(
        cc,
        "_http_get_json",
        lambda url, timeout=10.0: {"lastFundingRate": "0.0001", "time": 1_700_000_000_000},
    )
    fr = cc.CryptoDerivatives().get_funding_rate("BTC/USDT")
    assert fr.rate == pytest.approx(0.0001)
    assert fr.exchange == "binance"


def test_whale_transactions_typed_error_when_unconfigured(monkeypatch):
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    with pytest.raises(ProviderNotConfigured):
        cc.OnChainAnalytics().get_whale_transactions("ETH")


def test_liquidation_levels_typed_error_when_unconfigured(monkeypatch):
    monkeypatch.delenv("DERIVATIVES_API_KEY", raising=False)
    with pytest.raises(ProviderNotConfigured):
        cc.CryptoDerivatives().get_liquidation_levels("BTC/USDT")


def test_no_notimplemented_or_mock_in_finance_source():
    """The remediation must not reintroduce the very patterns it removed."""
    import pathlib

    finance_dir = pathlib.Path(cc.__file__).parent
    offenders = []
    for py in finance_dir.glob("*.py"):
        if py.name == "errors.py":
            continue  # the errors module legitimately names the patterns it replaces
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                continue  # skip comments / docstring lines
            if '"[Mock]' in line or "Fallback equal-weight" in line:
                offenders.append(f"{py.name}:{i}:mock")
            if "raise NotImplementedError" in line and "# ABSTRACT-OK" not in line:
                offenders.append(f"{py.name}:{i}:NotImplementedError")
    assert not offenders, f"stub patterns present in finance/: {offenders}"
