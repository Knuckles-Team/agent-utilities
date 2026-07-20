#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-OS.observability.langfuse-exporter — Langfuse exporter for spans, token usage and traces.

The orchestration engine already opens an OpenTelemetry span around every graph
run (``orchestration/engine.py`` → ``tracer.start_as_current_span``) and the
4-bucket :class:`~agent_utilities.observability.token_tracker.TokenUsageRecord`
captures token usage. What was missing is a *sink*: a Langfuse exporter that
ships those spans/traces/token-usage to a Langfuse project for LLM observability.

This module provides that sink with **zero hard dependency**: the ``langfuse``
package is imported lazily, and the exporter only activates when
``TRACE_EXPORT_ENABLED=true`` and a Langfuse credential-reference pair is
configured. When the flag, credentials, or package are absent,
every method is a clean no-op so the live graph path is never affected.

Wiring: :func:`get_langfuse_exporter` returns a process-wide singleton; the
orchestration engine calls :meth:`LangfuseExporter.export_graph_run` from inside
its existing tracing block on every run when the exporter is enabled.
"""

import logging
import re
import time
from typing import Any

from agent_utilities.core.config import resolve_langfuse_host, setting
from agent_utilities.observability.langfuse_trust import (
    configure_langfuse_trust,
    langfuse_credentials_configured,
    resolve_langfuse_credentials,
)
from agent_utilities.security.persistence_privacy import (
    PersistencePrivacyGuard,
    persistence_reference,
)
from agent_utilities.usage.privacy import normalize_run_id, normalize_tenant_id

logger = logging.getLogger(__name__)

_TRACE_EVIDENCE_PATTERNS = {
    "model_ref": re.compile(r"pref_model_[a-f0-9]{64}"),
    "skill_ref": re.compile(r"pref_skill_[a-f0-9]{64}"),
    "skill_body_ref": re.compile(r"pref_skill_body_[a-f0-9]{64}"),
    "model_class": re.compile(r"(?:economy|standard)"),
}


def _validated_trace_evidence(evidence: dict[str, str] | None) -> dict[str, str]:
    """Accept only the closed opaque attribution contract."""

    if evidence is None:
        return {}
    if set(evidence) - set(_TRACE_EVIDENCE_PATTERNS):
        raise ValueError("langfuse_trace_evidence_invalid")
    validated: dict[str, str] = {}
    for key, value in evidence.items():
        if (
            not isinstance(value, str)
            or _TRACE_EVIDENCE_PATTERNS[key].fullmatch(value) is None
        ):
            raise ValueError("langfuse_trace_evidence_invalid")
        validated[key] = value
    return validated


class LangfuseExporter:
    """Optional Langfuse sink for graph-run traces + token usage. CONCEPT:AU-OS.observability.langfuse-exporter.

    Parameters
    ----------
    client:
        An injected Langfuse client (used by tests / advanced wiring). When
        ``None`` the exporter lazily constructs a real ``langfuse.Langfuse``
        from the ``LANGFUSE_*`` environment variables.
    public_key, secret_key, host:
        Ephemeral credentials injected programmatically; otherwise
        ``LANGFUSE_PUBLIC_KEY_REF`` / ``LANGFUSE_SECRET_KEY_REF`` are resolved
        in memory. The host defaults to ``LANGFUSE_HOST``.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ) -> None:
        self._public_key = public_key or ""
        self._secret_key = secret_key or ""
        self._host = host or resolve_langfuse_host("")
        self._client = client
        self._probed = client is not None
        self._privacy = PersistencePrivacyGuard()
        # Counters expose activity for health widgets + tests.
        self.exported_traces = 0
        self.exported_observations = 0

    # ── lifecycle ──────────────────────────────────────────────────────
    @property
    def configured(self) -> bool:
        """True when credentials are present (or a client was injected)."""
        return (
            bool(self._client)
            or bool(self._public_key and self._secret_key)
            or langfuse_credentials_configured()
        )

    def _ensure_client(self) -> Any:
        """Lazily build the real Langfuse client; cache None if unavailable."""
        if self._client is not None:
            return self._client
        if self._probed:
            return None
        self._probed = True
        try:
            if not (self._public_key and self._secret_key):
                self._public_key, self._secret_key = resolve_langfuse_credentials()
            trust = configure_langfuse_trust()
            if not trust.valid:
                logger.error("Langfuse exporter disabled: invalid CA bundle")
                return None
            from langfuse import Langfuse  # type: ignore[import-not-found]
            from opentelemetry.sdk.trace import TracerProvider

            kwargs: dict[str, Any] = {
                "public_key": self._public_key,
                "secret_key": self._secret_key,
                # Langfuse v4 is OpenTelemetry-backed.  Supplying a dedicated
                # SDK provider keeps its span processor independent from the
                # process-global provider installed by GraphOS/logfire.  In
                # particular, a global ProxyTracerProvider must not absorb (or
                # silently drop) Langfuse observations.  The Langfuse client
                # owns processor registration, flush, and provider shutdown.
                "tracer_provider": TracerProvider(),
            }
            if self._host:
                kwargs["host"] = self._host
            self._client = Langfuse(**kwargs)
            logger.info("Langfuse exporter active.")
        except Exception as exc:  # noqa: BLE001 — optional dep / offline
            logger.debug(
                "Langfuse client unavailable; exporter no-ops (%s).",
                type(exc).__name__,
            )
            self._client = None
        return self._client

    @property
    def enabled(self) -> bool:
        """True when an actual client can be obtained (keys + dep present)."""
        return self._ensure_client() is not None

    # ── export API ─────────────────────────────────────────────────────
    def export_graph_run(
        self,
        *,
        run_id: str,
        query: str = "",
        status: str = "success",
        duration_ms: float = 0.0,
        token_usage: dict[str, int] | None = None,
        model: str = "",
        metadata: dict[str, Any] | None = None,
        evidence: dict[str, str] | None = None,
    ) -> bool:
        """Export one graph run as a Langfuse trace. CONCEPT:AU-OS.observability.langfuse-exporter.

        Emits a ``trace`` (the graph run) with a nested ``generation``
        observation carrying token usage when present. Returns ``True`` when a
        trace was sent, ``False`` when the exporter is inactive (no keys/dep) —
        in which case it is a clean no-op.
        """
        client = self._ensure_client()
        if client is None:
            return False
        try:
            try:
                from agent_utilities.security.brain_context import current_actor

                tenant_id = current_actor().tenant_id
            except Exception:  # noqa: BLE001 — empty namespace is dev-safe
                tenant_id = ""
            clean_run_id = normalize_run_id(run_id, tenant_id=tenant_id)
            controlled_evidence = _validated_trace_evidence(evidence)
            capture_content = str(
                setting("LANGFUSE_CAPTURE_CONTENT", "false") or "false"
            ).strip().lower() in {"1", "true", "yes", "on"}
            clean_query, query_privacy = self._privacy.sanitize_text(query)
            if not capture_content:
                clean_query = ""
            clean_metadata, metadata_privacy = self._privacy.sanitize(metadata or {})
            clean_status, status_privacy = self._privacy.sanitize_text(status)
            # Metadata supplied by callers may contain prompt/tool/result content.
            # Metadata-only mode drops it rather than treating sanitization as a
            # retention grant. Production pins this mode off in its ConfigMaps.
            md = dict(clean_metadata) if capture_content else {}
            md.update(
                {
                    "status": clean_status,
                    "duration_ms": duration_ms,
                    "input_character_count": len(query),
                    "content_retention": "sanitized" if capture_content else "metadata",
                    "tenant_ref": normalize_tenant_id(tenant_id),
                    "run_ref": clean_run_id,
                    **controlled_evidence,
                }
            )
            redactions = (
                query_privacy.redactions
                + metadata_privacy.redactions
                + status_privacy.redactions
            )
            if redactions:
                md["privacy_redactions"] = redactions
                md["privacy_detected_types"] = sorted(
                    {
                        *query_privacy.detected_types,
                        *metadata_privacy.detected_types,
                        *status_privacy.detected_types,
                    }
                )
            # CONCEPT:AU-OS.observability.run-wide-correlation-id — stamp the run-wide correlation id so every agent's
            # trace in a multi-agent run is joinable ("which agents touched X?").
            try:
                from agent_utilities.observability.correlation import get_correlation_id

                cid = get_correlation_id()
                if cid and "correlation_id" not in md:
                    tenant_ref = normalize_tenant_id(tenant_id)
                    md["correlation_id"] = persistence_reference(
                        "correlation", cid, namespace=tenant_ref
                    )
            except Exception:  # noqa: BLE001 — never let correlation crash export
                pass
            trace = self._make_trace(
                client,
                run_id=str(clean_run_id),
                query=str(clean_query),
                metadata=md,
            )
            observation_emitted = False
            try:
                if token_usage:
                    self._make_generation(
                        trace=trace,
                        run_id=str(clean_run_id),
                        model=model,
                        token_usage=token_usage,
                    )
                    observation_emitted = True
            finally:
                self._finish_observation(trace)
            self.exported_traces += 1
            if observation_emitted:
                self.exported_observations += 1
            return True
        except Exception as exc:  # noqa: BLE001 — observability must never crash a run
            logger.debug(
                "Langfuse export skipped (%s).",
                type(exc).__name__,
            )
            return False

    def _make_trace(
        self, client: Any, *, run_id: str, query: str, metadata: dict[str, Any]
    ) -> Any:
        """Create one independent Langfuse v4 root-trace observation.

        OpenTelemetry span context is process-global even when a tracer uses a
        dedicated provider.  GraphOS invokes this exporter while its request
        span is current, so omitting ``trace_context`` makes Langfuse attach the
        graph run below that unrelated span.  Langfuse then stores the request
        span's name as the trace name and the graph-run doctor cannot find the
        emitted ``graph_run:*`` root.  A Langfuse-derived trace id seeded from
        the already tenant-qualified opaque run reference explicitly starts the
        governed export as its own trace while preserving logical-run
        idempotency; nested generations still attach through the returned
        observation.
        """
        return client.start_observation(
            trace_context={"trace_id": client.create_trace_id(seed=run_id)},
            name=f"graph_run:{run_id}",
            as_type="span",
            input=query[:2000],
            metadata=metadata,
        )

    def _make_generation(
        self,
        *,
        trace: Any,
        run_id: str,
        model: str,
        token_usage: dict[str, int],
    ) -> None:
        """Attach a token-usage generation observation to ``trace``."""
        clean_model, _ = self._privacy.sanitize_text(model)
        model_ref = persistence_reference(
            "model", clean_model, namespace="langfuse-generation"
        )
        usage = {
            "input": int(token_usage.get("prompt", 0)),
            "output": int(token_usage.get("response", 0)),
            "total": int(
                token_usage.get("total") or sum(int(v) for v in token_usage.values())
            ),
        }
        generation = trace.start_observation(
            name=f"llm:{run_id}",
            as_type="generation",
            model=model_ref,
            usage_details=usage,
        )
        self._finish_observation(generation)

    @staticmethod
    def _finish_observation(observation: Any) -> None:
        """End one Langfuse v4 observation."""
        observation.end()

    def flush(self) -> None:
        """Flush buffered events to Langfuse (no-op when inactive)."""
        client = self._client
        if client is not None:
            try:
                client.flush()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Langfuse flush skipped (%s).", type(exc).__name__)


# ── process-wide singleton (wired into the live graph path) ───────────────────
_EXPORTER: LangfuseExporter | None = None
_EXPORTER_BUILT = False


def get_langfuse_exporter() -> LangfuseExporter | None:
    """Return the process-wide exporter, or ``None`` when not configured.

    Production presets explicitly enable export. Credentials alone are not an
    authorization to emit traces; absent opt-in or keys returns ``None``.
    """
    global _EXPORTER, _EXPORTER_BUILT
    if _EXPORTER_BUILT:
        return _EXPORTER
    _EXPORTER_BUILT = True
    enabled = str(setting("TRACE_EXPORT_ENABLED", "false") or "false").strip()
    if enabled.casefold() not in {"1", "true", "yes", "on"}:
        _EXPORTER = None
        return None
    exporter = LangfuseExporter()
    _EXPORTER = exporter if exporter.configured else None
    return _EXPORTER


def set_langfuse_exporter(exporter: LangfuseExporter | None) -> None:
    """Install a specific exporter (used by tests to inject a fake client)."""
    global _EXPORTER, _EXPORTER_BUILT
    _EXPORTER = exporter
    _EXPORTER_BUILT = True


def reset_langfuse_exporter() -> None:
    """Reset the cached singleton so the next call re-probes the environment."""
    global _EXPORTER, _EXPORTER_BUILT
    _EXPORTER = None
    _EXPORTER_BUILT = False


def _now_ms() -> float:
    return time.time() * 1000.0
