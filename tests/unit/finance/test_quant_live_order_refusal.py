"""AU-CONTEXT-R006.1: no agent-utilities model output or UI payload can grant
live-order authority.

The registered ``quant`` MCP tool is the only AU entry point that accepts an
``execute``/``submit_order`` request from a caller (model, UI, or otherwise).
It must refuse every such request outright -- including a caller that passes
``mode="live"`` -- rather than reaching a broker. Governed live execution is
the agent connector SDK's write-back contract (AU-CONTEXT-R006.2), not an AU
model or UI payload.
"""

from __future__ import annotations

import pytest

from agent_utilities.domains.finance import quant_mcp_tools
from agent_utilities.domains.finance.errors import ProviderNotConfigured


class _CollectingMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def _deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _deco


def _register() -> object:
    return quant_mcp_tools.register_quant_tools(_CollectingMCP(), engine_default=None)


@pytest.mark.spec("AU-CONTEXT-R006.1")
@pytest.mark.parametrize("mode", ["paper", "live"])
@pytest.mark.parametrize("action", ["submit_order", "cancel_order", "status"])
def test_execute_domain_never_reaches_a_broker(mode: str, action: str) -> None:
    """Every execute action is refused regardless of the caller-supplied mode.

    ``mode`` is caller-controlled input (a UI payload or a model's tool call);
    it must never be the thing that decides whether an order reaches a broker.
    """
    quant = _register()

    result = quant(
        domain="execute",
        action=action,
        ticker="AAPL",
        side="buy",
        quantity=1.0,
        order_type="market",
        mode=mode,
    )

    assert "Execution broker not connected" in result
    assert "Mock fallback disabled" in result


@pytest.mark.spec("AU-CONTEXT-R006.1")
def test_execute_domain_raises_the_typed_refusal_not_a_silent_mock() -> None:
    """The underlying refusal is a typed error, not a value the caller could
    mistake for a placed order."""
    with pytest.raises(ProviderNotConfigured):
        quant_mcp_tools._quant_execute("submit_order")
