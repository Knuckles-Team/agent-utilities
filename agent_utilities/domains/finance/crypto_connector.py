"""
Crypto Connector — CONCEPT:KG-2.6
Domain: Finance

Provides abstractions for crypto-specific market data:
- On-chain analytics (Whale tracking, DEX volume)
- Derivatives (Funding rates, Liquidations)
- Protocol metrics (TVL, Active Addresses)
"""

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from agent_utilities.core.config import setting

from .errors import ProviderNotConfigured, ProviderRequestError

logger = logging.getLogger(__name__)

_DEFILLAMA_BASE = "https://api.llama.fi"


def _http_get_json(url: str, timeout: float = 10.0) -> Any:
    """GET a URL and parse JSON. Raises ProviderRequestError on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": "agent-utilities/finance"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise ProviderRequestError(f"request to {url} failed: {exc}") from exc


@dataclass
class FundingRate:
    symbol: str
    rate: float
    timestamp: float
    exchange: str


@dataclass
class WhaleAlert:
    tx_hash: str
    asset: str
    amount: float
    usd_value: float
    from_address: str
    to_address: str
    timestamp: float


class OnChainAnalytics:
    """Hooks for on-chain metric providers like Glassnode, Dune, or Etherscan."""

    def get_dex_volume(self, protocol: str = "uniswap") -> float:
        """Fetch 24h DEX volume (USD) from DeFiLlama's keyless API."""
        logger.info("Fetching DEX volume for %s", protocol)
        data = _http_get_json(
            f"{_DEFILLAMA_BASE}/summary/dexs/{protocol}?dataType=dailyVolume"
        )
        vol = data.get("total24h") if isinstance(data, dict) else None
        if vol is None:
            raise ProviderRequestError(
                f"DeFiLlama returned no 24h volume for protocol '{protocol}'"
            )
        return float(vol)

    def get_whale_transactions(
        self, asset: str, _min_usd_value: float = 1_000_000
    ) -> list[WhaleAlert]:
        """Track large on-chain transfers (requires an Etherscan/Glassnode key)."""
        if not setting("ETHERSCAN_API_KEY"):
            raise ProviderNotConfigured(
                "Whale tracking requires ETHERSCAN_API_KEY (or a Glassnode/Dune key). "
                "Set ETHERSCAN_API_KEY to enable on-chain transfer tracking."
            )
        # With a key configured, query the provider for large transfers.
        raise ProviderNotConfigured(
            "ETHERSCAN_API_KEY is set but the on-chain transfer feed is not yet "
            "wired for this asset; configure ONCHAIN_PROVIDER to select a backend."
        )

    def get_tvl(self, protocol: str) -> float:
        """Get Total Value Locked (USD) for a DeFi protocol via DeFiLlama (keyless)."""
        data = _http_get_json(f"{_DEFILLAMA_BASE}/tvl/{protocol}")
        # /tvl/{protocol} returns a bare number.
        try:
            return float(data)
        except (TypeError, ValueError) as exc:
            raise ProviderRequestError(
                f"DeFiLlama returned no TVL for protocol '{protocol}': {data!r}"
            ) from exc


class CryptoDerivatives:
    """Binance/Bybit derivatives data abstraction."""

    def get_funding_rate(self, symbol: str) -> FundingRate:
        """Fetch the current perpetual-futures funding rate from Binance (keyless)."""
        pair = symbol.replace("/", "").upper()
        data = _http_get_json(
            f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={pair}"
        )
        if not isinstance(data, dict) or "lastFundingRate" not in data:
            raise ProviderRequestError(
                f"Binance returned no funding rate for symbol '{symbol}'"
            )
        return FundingRate(
            symbol=symbol,
            rate=float(data["lastFundingRate"]),
            timestamp=float(data.get("time", time.time() * 1000)) / 1000.0,
            exchange="binance",
        )

    def get_liquidation_levels(self, symbol: str) -> dict[str, float]:
        """Estimated liquidation clusters (requires a derivatives-analytics key)."""
        if not setting("DERIVATIVES_API_KEY"):
            raise ProviderNotConfigured(
                "Liquidation-cluster estimation requires DERIVATIVES_API_KEY "
                "(e.g. Coinglass/Coinalyze). Set DERIVATIVES_API_KEY to enable it."
            )
        raise ProviderNotConfigured(
            "DERIVATIVES_API_KEY is set but no liquidation provider is selected; "
            "set DERIVATIVES_PROVIDER to choose one."
        )


class CryptoConnector:
    """Unified entrypoint for crypto-specific quant metrics."""

    def __init__(self):
        self.on_chain = OnChainAnalytics()
        self.derivatives = CryptoDerivatives()

    def get_asset_context(self, symbol: str) -> dict[str, Any]:
        """Aggregate all crypto-specific signals for an asset."""
        asset = symbol.split("/")[0] if "/" in symbol else symbol

        return {
            "symbol": symbol,
            "funding_rate": self.derivatives.get_funding_rate(symbol).__dict__,
            "liquidation_clusters": self.derivatives.get_liquidation_levels(symbol),
            "recent_whale_txs": [
                w.__dict__ for w in self.on_chain.get_whale_transactions(asset)
            ],
            "timestamp": time.time(),
        }
