#!/usr/bin/python
"""Concrete event-stream adapters for the Company Brain (CONCEPT:KG-2.6).

Real Kafka/NATS implementations of ``BaseStreamAdapter`` (the in-module adapter
in ``core/company_brain.py`` is a no-dep simulation). Both are import-guarded so
the package is import-safe without the optional brokers installed:

    pip install agent-utilities[kafka]   # aiokafka
    pip install agent-utilities[nats]    # nats-py

``make_stream_adapter(config)`` picks the right adapter from the stream's
``source_type``.
"""

from __future__ import annotations

from typing import Any

from ..core.company_brain import BaseStreamAdapter


def make_stream_adapter(config: Any, **kwargs: Any) -> BaseStreamAdapter:
    """Build the concrete adapter for ``config.source_type`` (kafka/nats)."""
    source = str(getattr(config, "source_type", "")).lower()
    if "nats" in source:
        from .nats_adapter import NatsStreamAdapter

        return NatsStreamAdapter(config, **kwargs)
    # Default to Kafka for kafka/redpanda/redis_stream-style sources.
    from .kafka_adapter import KafkaStreamAdapter

    return KafkaStreamAdapter(config, **kwargs)


__all__ = ["make_stream_adapter"]
