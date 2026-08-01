#!/usr/bin/python
from __future__ import annotations

"""Wire pydantic-ai's native ``Instrumentation`` capability onto our own OTel pipeline.

CONCEPT:AU-OS.observability.telemetry-observability — Track 3 of the pydantic-ai
native-adoption program (``reports/program/pydantic-ai-native-adoption.md``).

``pydantic_ai.capabilities.Instrumentation`` puts model-request, tool-call, and hook
spans on OpenTelemetry — but its ``InstrumentationSettings.tracer_provider`` defaults
to the AMBIENT GLOBAL provider ("typically configured via ``logfire.configure()``"),
which this codebase does not run. ``agent_utilities.observability.TelemetryEngine``
already builds a REAL ``TracerProvider`` wired with a ``BatchSpanProcessor`` +
``OTLPSpanExporter`` pointed at our live collector (``EPISTEMIC_GRAPH_OBS_ADDR`` /
``OTEL_EXPORTER_OTLP_ENDPOINT`` — the same Tempo OTLP receiver the engine's own
``graph.*`` spans already export to). This module is the ONE seam that hands THAT
provider to ``Instrumentation``, so a pydantic-ai agent's model/tool/hook spans land
on the SAME timeline as the engine's own spans instead of a second, unconfigured
tracer.

Adopting the native ``Instrumentation`` here means we did NOT hand-roll a
pydantic-ai-specific OTel bridge next to the engine's own — the ONLY new code is
this ~20-line factory that passes our existing provider into pydantic-ai's own
capability.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic_ai.capabilities import Instrumentation

    from agent_utilities.observability import TelemetryEngine

logger = logging.getLogger(__name__)

__all__ = ["build_fleet_instrumentation"]


def build_fleet_instrumentation(
    *,
    telemetry: "TelemetryEngine | None" = None,
    **settings_kwargs: Any,
) -> "Instrumentation | None":
    """Return an ``Instrumentation`` capability bound to our live OTel pipeline.

    Returns ``None`` — a clean no-op, never a placeholder capability — when no
    OTLP collector endpoint is configured (mirrors every other conditionally
    attached default capability in this package, see
    ``agent_utilities.capabilities.composition.default_runtime_capabilities``).

    Args:
        telemetry: The :class:`TelemetryEngine` whose tracer/meter provider to
            reuse. Defaults to the process-wide singleton
            (:func:`agent_utilities.observability.get_telemetry_engine`), so
            every caller shares the ONE configured pipeline unless a test
            explicitly injects its own engine.
        **settings_kwargs: Extra ``InstrumentationSettings`` fields (e.g.
            ``include_content=False``) layered on top of the reused providers.
    """
    from pydantic_ai.capabilities import Instrumentation
    from pydantic_ai.models.instrumented import InstrumentationSettings

    from agent_utilities.observability import get_telemetry_engine

    engine = telemetry if telemetry is not None else get_telemetry_engine()
    if not engine.is_otel_configured():
        logger.debug(
            "build_fleet_instrumentation: no OTLP collector configured "
            "(EPISTEMIC_GRAPH_OBS_ADDR / OTEL_EXPORTER_OTLP_ENDPOINT) — "
            "Instrumentation capability not attached."
        )
        return None

    settings_kwargs.setdefault("tracer_provider", engine.tracer_provider)
    meter_provider = engine.meter_provider
    if meter_provider is not None:
        settings_kwargs.setdefault("meter_provider", meter_provider)
    settings = InstrumentationSettings(**settings_kwargs)
    return Instrumentation(settings=settings)
