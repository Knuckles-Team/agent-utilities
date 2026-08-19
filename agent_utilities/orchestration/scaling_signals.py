#!/usr/bin/python
from __future__ import annotations

"""Typed, bounded scaling signal providers.

CONCEPT:AU-OS.scaling.reactive-replica-autoscaling — Reactive replica autoscaling
(signal seam).

The autoscaler consumes immutable :class:`ScalingSignalSample` values rather
than untyped floats. A sample is bound to the service, symbolic signal,
aggregation mode, source, observation time, unit, and tenant/fleet scope.
Invalid, stale, replayed, cross-service, cross-unit, or cross-scope samples are
treated as no data by the autoscaler and therefore cannot trigger a scale-down.

One autoscaler tick calls the optional ``signal_values`` bulk extension once.
Prometheus batches reuse one client only when its transport is declared
thread-safe (custom transports use one client per worker), deduplicate
allowlisted queries, cap concurrency, and enforce an overall deadline. Timed-out work occupies a
process-wide bounded in-flight slot until it actually finishes; later ticks
return no data while all slots are occupied, so ignored transport cancellation
cannot create an unbounded thread pool. A single-read fallback exists only for
providers explicitly trusted to run in-process.

Prometheus signals are symbolic names resolved through the built-in definitions
or an explicitly deployment-injected allowlist.  A caller cannot submit raw
PromQL through the autoscaling signal interface.  Query length, timeout, and
series-cardinality limits are enforced at this seam.

Local queue/lag gauges pair each value family with an explicit observed-at gauge
family using identical labels. Missing, mismatched, duplicate, stale, or
high-cardinality pairs are no data; local scrape/read time is never substituted.
"""

import atexit
import logging
import math
import queue
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

logger = logging.getLogger(__name__)

SignalAggregation = Literal["fleet_total", "per_replica"]

# Well-known signal names → the shipped metric families they read.
SIGNAL_QUEUE_DEPTH = "queue_depth"
SIGNAL_CONSUMER_LAG = "consumer_lag"
SIGNAL_CPU = "cpu"

MAX_SIGNAL_AGE_S = 120.0
MAX_FUTURE_SKEW_S = 30.0
MAX_QUERY_LENGTH = 4096
MAX_QUERY_TIMEOUT_S = 10.0
MAX_RESULT_SERIES = 100
MAX_BULK_REQUESTS = 128
MAX_BULK_QUERIES = 64
MAX_QUERY_CONCURRENCY = 4
MAX_BULK_DEADLINE_S = 10.0
_MAX_SYMBOL_LENGTH = 128
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

UNIT_QUEUE_DEPTH = "items"
UNIT_CONSUMER_LAG = "messages"
UNIT_CPU = "percent"
SCOPE_FLEET = "fleet"
SCOPE_SERVICE = "service"
KG_INGEST_WORKER_SERVICE = "kg-ingest-worker"
KG_INGEST_QUEUE_DEPTH_OBSERVED_AT = "agent_utilities_kg_ingest_queue_depth_observed_at"
KG_INGEST_CONSUMER_LAG_OBSERVED_AT = (
    "agent_utilities_kg_ingest_consumer_lag_observed_at"
)

SignalRequest = tuple[str, str]
ScalingSignalBatch: TypeAlias = Mapping[SignalRequest, "ScalingSignalSample | None"]


class _BoundedDaemonExecutor:
    """Tiny fixed daemon pool for deadline-enforced signal reads.

    ``ThreadPoolExecutor`` registers an interpreter-exit hook that joins every
    worker before normal ``atexit`` handlers run.  A transport that ignores its
    timeout can therefore keep the whole process alive forever.  These workers
    are daemon threads, and the global in-flight authority below ensures that
    no more than ``max_workers`` calls can ever be running or retained.
    """

    def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
        self._queue: queue.Queue[
            tuple[Future[Any], Callable[..., Any], tuple[Any, ...]] | None
        ] = queue.Queue(maxsize=max_workers)
        self._lock = threading.Lock()
        self._shutdown = False
        self._threads = tuple(
            threading.Thread(
                target=self._worker,
                name=f"{thread_name_prefix}-{index}",
                daemon=True,
            )
            for index in range(max_workers)
        )
        for thread in self._threads:
            thread.start()

    def _worker(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                future, function, args = task
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    result = function(*args)
                except BaseException as exc:  # noqa: BLE001 - Future contract
                    future.set_exception(exc)
                else:
                    future.set_result(result)
            finally:
                self._queue.task_done()

    def submit(self, function: Callable[..., Any], *args: Any) -> Future[Any]:
        with self._lock:
            if self._shutdown:
                raise RuntimeError("signal query executor is shut down")
            future: Future[Any] = Future()
            try:
                self._queue.put_nowait((future, function, args))
            except queue.Full as exc:
                raise RuntimeError("signal query executor capacity exhausted") from exc
            return future

    def shutdown(self, *, wait_for_workers: bool, cancel_futures: bool) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        if cancel_futures:
            while True:
                try:
                    task = self._queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if task is not None:
                        task[0].cancel()
                finally:
                    self._queue.task_done()
        for _thread in self._threads:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                break
        if wait_for_workers:
            for thread in self._threads:
                thread.join()


_PROMETHEUS_EXECUTOR_LOCK = threading.Lock()
_PROMETHEUS_EXECUTOR: _BoundedDaemonExecutor | None = None
_PROMETHEUS_INFLIGHT: dict[Any, tuple[Any, Any, bool]] = {}


def _close_prometheus_client(client: Any) -> None:
    try:
        client.close()
    except Exception:  # noqa: BLE001 — cleanup cannot make a sample valid
        pass


def _reap_prometheus_inflight() -> None:
    """Release completed query slots and clients retained past a deadline."""

    closing: list[Any] = []
    with _PROMETHEUS_EXECUTOR_LOCK:
        completed = [future for future in _PROMETHEUS_INFLIGHT if future.done()]
        for future in completed:
            owner, client, close_when_idle = _PROMETHEUS_INFLIGHT.pop(future)
            if owner is not None:
                owner._prometheus_inflight = max(  # noqa: SLF001
                    0,
                    owner._prometheus_inflight - 1,  # noqa: SLF001
                )
            if close_when_idle and not any(
                client is active_record[1]
                for active_record in _PROMETHEUS_INFLIGHT.values()
            ):
                closing.append(client)
    for client in closing:
        _close_prometheus_client(client)


def _prometheus_submit(
    function: Any,
    owner: Any,
    client: Any,
    close_client_when_idle: bool,
    *args: Any,
) -> Any | None:
    """Submit within process-wide and provider-owned in-flight budgets."""

    global _PROMETHEUS_EXECUTOR
    _reap_prometheus_inflight()
    with _PROMETHEUS_EXECUTOR_LOCK:
        if len(_PROMETHEUS_INFLIGHT) >= MAX_QUERY_CONCURRENCY:
            return None
        if owner._prometheus_inflight >= owner.max_concurrency:  # noqa: SLF001
            return None
        if _PROMETHEUS_EXECUTOR is None:
            _PROMETHEUS_EXECUTOR = _BoundedDaemonExecutor(
                max_workers=MAX_QUERY_CONCURRENCY,
                thread_name_prefix="au-prometheus-signal",
            )
        future = _PROMETHEUS_EXECUTOR.submit(function, client, *args)
        owner._prometheus_inflight += 1  # noqa: SLF001
        _PROMETHEUS_INFLIGHT[future] = (
            owner,
            client,
            close_client_when_idle,
        )
        return future


def _close_prometheus_client_when_idle(client: Any) -> None:
    """Transfer a shared client's lifetime to its remaining future(s)."""

    retained = False
    with _PROMETHEUS_EXECUTOR_LOCK:
        for future, (owner, active_client, _close_when_idle) in tuple(
            _PROMETHEUS_INFLIGHT.items()
        ):
            if active_client is client:
                _PROMETHEUS_INFLIGHT[future] = (owner, active_client, True)
                retained = True
    if not retained:
        _close_prometheus_client(client)


def _shutdown_prometheus_executor() -> None:
    """Cancel queued work and close retained clients during process shutdown."""

    global _PROMETHEUS_EXECUTOR
    with _PROMETHEUS_EXECUTOR_LOCK:
        executor = _PROMETHEUS_EXECUTOR
        _PROMETHEUS_EXECUTOR = None
        futures = list(_PROMETHEUS_INFLIGHT)
        clients: list[Any] = []
        for _owner, client, _close_when_idle in _PROMETHEUS_INFLIGHT.values():
            if not any(existing is client for existing in clients):
                clients.append(client)
        _PROMETHEUS_INFLIGHT.clear()
    for future in futures:
        future.cancel()
    if executor is not None:
        executor.shutdown(wait_for_workers=False, cancel_futures=True)
    for client in clients:
        _close_prometheus_client(client)


atexit.register(_shutdown_prometheus_executor)


def _valid_symbol(value: str) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _MAX_SYMBOL_LENGTH
        and _SYMBOL_RE.fullmatch(value) is not None
    )


@dataclass(frozen=True, slots=True)
class SignalDefinition:
    """Trusted deployment definition for one symbolic scaling signal."""

    name: str
    aggregation: SignalAggregation
    query_template: str = ""
    local_metric_family: str | None = None
    service_label: str | None = None
    service_binding: str | None = None
    unit: str = ""
    scope: str = ""
    local_observed_at_metric_family: str | None = None

    def __post_init__(self) -> None:
        if not _valid_symbol(self.name):
            raise ValueError("signal definition name is invalid")
        if self.aggregation not in ("fleet_total", "per_replica"):
            raise ValueError("signal aggregation must be fleet_total or per_replica")
        if not isinstance(self.query_template, str):
            raise TypeError("signal query_template must be a string")
        if len(self.query_template.encode("utf-8")) > MAX_QUERY_LENGTH:
            raise ValueError("signal query exceeds the bounded query length")
        if self.local_metric_family is not None and not _valid_symbol(
            self.local_metric_family
        ):
            raise ValueError("local metric family is invalid")
        if self.local_observed_at_metric_family is not None and not _valid_symbol(
            self.local_observed_at_metric_family
        ):
            raise ValueError("local observed-at metric family is invalid")
        if (
            self.local_observed_at_metric_family is not None
            and self.local_metric_family is None
        ):
            raise ValueError("local observed-at metric requires a local metric family")
        if (
            self.local_observed_at_metric_family is not None
            and self.local_observed_at_metric_family == self.local_metric_family
        ):
            raise ValueError("local observed-at metric family must be distinct")
        if self.service_label is not None and not _valid_symbol(self.service_label):
            raise ValueError("service label is invalid")
        if self.service_binding is not None and not _valid_symbol(self.service_binding):
            raise ValueError("service binding is invalid")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("signal unit is required")
        if len(self.unit.encode("utf-8")) > _MAX_SYMBOL_LENGTH:
            raise ValueError("signal unit is too long")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("signal scope is required")
        if len(self.scope.encode("utf-8")) > _MAX_SYMBOL_LENGTH:
            raise ValueError("signal scope is too long")
        # A per-replica remote query must be bound to the requested service by
        # either the explicit service placeholder or a documented metric label.
        if (
            self.aggregation == "per_replica"
            and self.query_template
            and "{service}" not in self.query_template
            and self.service_label is None
            and self.service_binding is None
        ):
            raise ValueError("per-replica signal must bind the requested service")

    def binds_service(self, service: str) -> bool:
        """Whether this trusted definition proves which service it measures."""

        if not _valid_symbol(service):
            return False
        if self.service_binding is not None:
            return self.service_binding == service
        return self.service_label is not None or "{service}" in self.query_template


@dataclass(frozen=True, slots=True)
class ScalingSignalSample:
    """Immutable, identity-bound load observation consumed by the autoscaler."""

    value: float
    source: str
    service: str
    signal: str
    aggregation: SignalAggregation
    observed_at: float
    unit: str = ""
    scope: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.value, bool):
            raise ValueError("signal value must be numeric")
        try:
            value = float(self.value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("signal value must be numeric") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("signal value must be finite and non-negative")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("signal source is required")
        if not _valid_symbol(self.service):
            raise ValueError("signal service is invalid")
        if not _valid_symbol(self.signal):
            raise ValueError("signal name is invalid")
        if self.aggregation not in ("fleet_total", "per_replica"):
            raise ValueError("signal aggregation is invalid")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("signal unit is required")
        if len(self.unit.encode("utf-8")) > _MAX_SYMBOL_LENGTH:
            raise ValueError("signal unit is too long")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("signal scope is required")
        if len(self.scope.encode("utf-8")) > _MAX_SYMBOL_LENGTH:
            raise ValueError("signal scope is too long")
        try:
            observed_at = float(self.observed_at)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("signal observation time must be numeric") from exc
        if not math.isfinite(observed_at) or observed_at <= 0:
            raise ValueError("signal observation time is invalid")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "observed_at", observed_at)


_BUILTIN_SIGNAL_DEFINITIONS: Mapping[str, SignalDefinition] = MappingProxyType(
    {
        SIGNAL_QUEUE_DEPTH: SignalDefinition(
            name=SIGNAL_QUEUE_DEPTH,
            aggregation="fleet_total",
            query_template="sum(agent_utilities_kg_ingest_queue_depth)",
            local_metric_family="agent_utilities_kg_ingest_queue_depth",
            service_binding=KG_INGEST_WORKER_SERVICE,
            unit=UNIT_QUEUE_DEPTH,
            scope=SCOPE_FLEET,
            local_observed_at_metric_family=KG_INGEST_QUEUE_DEPTH_OBSERVED_AT,
        ),
        SIGNAL_CONSUMER_LAG: SignalDefinition(
            name=SIGNAL_CONSUMER_LAG,
            aggregation="fleet_total",
            query_template="sum(agent_utilities_kg_ingest_consumer_lag)",
            local_metric_family="agent_utilities_kg_ingest_consumer_lag",
            service_binding=KG_INGEST_WORKER_SERVICE,
            unit=UNIT_CONSUMER_LAG,
            scope=SCOPE_FLEET,
            local_observed_at_metric_family=KG_INGEST_CONSUMER_LAG_OBSERVED_AT,
        ),
        SIGNAL_CPU: SignalDefinition(
            name=SIGNAL_CPU,
            aggregation="per_replica",
            query_template=(
                "100 * avg(rate(container_cpu_usage_seconds_total"
                '{container_label_com_docker_swarm_service_name="{service}"}[5m]))'
            ),
            service_label="container_label_com_docker_swarm_service_name",
            unit=UNIT_CPU,
            scope=SCOPE_SERVICE,
        ),
    }
)


def _definitions_with_overrides(
    overrides: Mapping[str, SignalDefinition] | None,
) -> Mapping[str, SignalDefinition]:
    definitions = dict(_BUILTIN_SIGNAL_DEFINITIONS)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise TypeError("signal definitions must be a mapping")
        for name, definition in overrides.items():
            if not isinstance(name, str) or not isinstance(
                definition, SignalDefinition
            ):
                raise TypeError("signal definitions must map names to SignalDefinition")
            if name != definition.name:
                raise ValueError("signal definition key does not match its name")
            definitions[name] = definition
    return MappingProxyType(definitions)


def get_signal_definition(signal: str) -> SignalDefinition | None:
    """Return the built-in definition for a symbolic signal, if one exists."""

    return _BUILTIN_SIGNAL_DEFINITIONS.get(signal)


def validate_scaling_signal_sample(
    sample: Any,
    *,
    service: str,
    signal: str,
    aggregation: SignalAggregation | None,
    unit: str | None = None,
    scope: str | None = None,
    now: float | None = None,
    previous_observed_at: float | None = None,
) -> ScalingSignalSample | None:
    """Validate one provider result at the autoscaler consumption boundary."""

    if (
        not isinstance(sample, ScalingSignalSample)
        or aggregation is None
        or unit is None
        or scope is None
    ):
        return None
    if (
        sample.service != service
        or sample.signal != signal
        or sample.aggregation != aggregation
        or sample.unit != unit
        or sample.scope != scope
    ):
        return None
    try:
        current_time = time.time() if now is None else float(now)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(current_time):
        return None
    age = current_time - sample.observed_at
    if age > MAX_SIGNAL_AGE_S or age < -MAX_FUTURE_SKEW_S:
        return None
    if previous_observed_at is not None:
        try:
            if sample.observed_at <= float(previous_observed_at):
                return None
        except (TypeError, ValueError, OverflowError):
            return None
    return sample


def _bounded_timeout(timeout: float) -> float:
    if isinstance(timeout, bool):
        raise ValueError("signal query timeout must be numeric")
    try:
        value = float(timeout)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("signal query timeout must be numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("signal query timeout must be positive and finite")
    return min(value, MAX_QUERY_TIMEOUT_S)


def _bounded_series_limit(max_series: int) -> int:
    if isinstance(max_series, bool) or not isinstance(max_series, int):
        raise ValueError("signal series limit must be an integer")
    value = max_series
    if value <= 0:
        raise ValueError("signal series limit must be positive")
    return min(value, MAX_RESULT_SERIES)


def _bounded_concurrency(max_concurrency: int) -> int:
    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
        raise ValueError("signal query concurrency must be an integer")
    if max_concurrency <= 0:
        raise ValueError("signal query concurrency must be positive")
    return min(max_concurrency, MAX_QUERY_CONCURRENCY)


def _bounded_bulk_deadline(timeout: float) -> float:
    if isinstance(timeout, bool):
        raise ValueError("signal bulk deadline must be numeric")
    try:
        value = float(timeout)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("signal bulk deadline must be numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("signal bulk deadline must be positive and finite")
    return min(value, MAX_BULK_DEADLINE_S)


def _normalise_requests(
    requests: Sequence[SignalRequest],
) -> tuple[SignalRequest, ...] | None:
    try:
        raw_requests = list(requests)
    except (TypeError, ValueError):
        return None
    if len(raw_requests) > MAX_BULK_REQUESTS:
        return None
    normalised: list[SignalRequest] = []
    seen: set[SignalRequest] = set()
    for raw in raw_requests:
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            continue
        service, signal = raw
        if not _valid_symbol(service) or not _valid_symbol(signal):
            continue
        request = (service, signal)
        if request not in seen:
            seen.add(request)
            normalised.append(request)
    return tuple(normalised)


@runtime_checkable
class ScalingSignalProvider(Protocol):
    """Provider of typed, identity-bound observations for symbolic signals."""

    name: str

    def signal_definition(
        self, signal: str, service: str | None = None
    ) -> SignalDefinition | None:
        """Return the trusted allowlisted definition for ``signal``."""
        ...  # ABSTRACT-OK

    def signal_value(self, service: str, signal: str) -> ScalingSignalSample | None:
        """Return the current typed observation; ``None`` means no data."""
        ...  # ABSTRACT-OK


@runtime_checkable
class BulkScalingSignalProvider(Protocol):
    """Optional bounded batch extension for a scaling signal provider."""

    def signal_values(self, requests: Sequence[SignalRequest]) -> ScalingSignalBatch:
        """Return one no-data-safe result mapping for all ``requests``."""
        ...  # ABSTRACT-OK


def read_scaling_signal_samples(
    provider: ScalingSignalProvider,
    requests: Sequence[SignalRequest],
) -> ScalingSignalBatch:
    """Read a bounded batch without issuing unsafe remote single-read loops.

    Providers with ``signal_values`` own the batch and may be remote.  A
    provider that only has the historical single-read method is accepted only
    when it explicitly marks itself ``trusted_in_process``; this preserves
    local custom providers without allowing an unbounded remote timeout loop.
    """

    normalised = _normalise_requests(requests)
    if normalised is None:
        return {}
    empty: dict[SignalRequest, ScalingSignalSample | None] = {
        request: None for request in normalised
    }
    if isinstance(provider, BulkScalingSignalProvider):
        try:
            values = provider.signal_values(normalised)
        except Exception as exc:  # noqa: BLE001 — provider failures are no data
            logger.debug("scaling signal bulk read failed: %s", type(exc).__name__)
            return empty
        if not isinstance(values, Mapping):
            return empty
        results: dict[SignalRequest, ScalingSignalSample | None] = {}
        try:
            for request in normalised:
                sample = values.get(request)
                results[request] = (
                    sample if isinstance(sample, ScalingSignalSample) else None
                )
        except Exception as exc:  # noqa: BLE001 — malformed mapping is no data
            logger.debug("scaling signal bulk result failed: %s", type(exc).__name__)
            return empty
        return results
    if getattr(provider, "trusted_in_process", False) is not True:
        return empty
    fallback_results: dict[SignalRequest, ScalingSignalSample | None] = {}
    for service, signal in normalised:
        try:
            sample = provider.signal_value(service, signal)
        except Exception as exc:  # noqa: BLE001 — provider failures are no data
            logger.debug(
                "trusted in-process scaling read failed for %s/%s: %s",
                service,
                signal,
                type(exc).__name__,
            )
            sample = None
        fallback_results[(service, signal)] = (
            sample if isinstance(sample, ScalingSignalSample) else None
        )
    return fallback_results


MetricLabelKey = tuple[tuple[str, str], ...]


def _metric_label_key(labels: Any) -> MetricLabelKey | None:
    if not isinstance(labels, Mapping):
        return None
    try:
        items = tuple(labels.items())
    except Exception:  # noqa: BLE001 — malformed labels are no data
        return None
    if not all(
        isinstance(label, str) and isinstance(value, str) for label, value in items
    ):
        return None
    return tuple(sorted(items))


class LocalMetricsProvider:
    """Bounded provider over this process's Prometheus registry.

    Local observations are accepted only when the value family has a paired
    observed-at gauge with identical labels. The companion's Unix timestamp is
    durable freshness evidence; the provider never substitutes scrape/read
    time for it.
    """

    name = "local"
    trusted_in_process = True

    def __init__(
        self,
        signal_definitions: Mapping[str, SignalDefinition] | None = None,
        *,
        max_series: int = MAX_RESULT_SERIES,
    ) -> None:
        self._definitions = _definitions_with_overrides(signal_definitions)
        self._max_series = _bounded_series_limit(max_series)

    def signal_definition(
        self, signal: str, service: str | None = None
    ) -> SignalDefinition | None:
        definition = self._definitions.get(signal, get_signal_definition(signal))
        if service is not None and (
            definition is None or not definition.binds_service(service)
        ):
            return None
        return definition

    def signal_values(self, requests: Sequence[SignalRequest]) -> ScalingSignalBatch:
        normalised = _normalise_requests(requests)
        if normalised is None:
            return {}
        results: dict[SignalRequest, ScalingSignalSample | None] = {
            request: None for request in normalised
        }
        definitions: dict[SignalRequest, SignalDefinition] = {}
        families: set[str] = set()
        for service, signal in normalised:
            definition = self.signal_definition(signal, service)
            if (
                definition is None
                or definition.local_metric_family is None
                or definition.local_observed_at_metric_family is None
            ):
                continue
            request = (service, signal)
            definitions[request] = definition
            families.add(definition.local_metric_family)
            families.add(definition.local_observed_at_metric_family)
        if not families:
            return results
        try:
            from prometheus_client import REGISTRY
        except ImportError:
            return results
        records: dict[str, dict[MetricLabelKey, float]] = {
            family: {} for family in families
        }
        invalid_families: set[str] = set()
        duplicate_families: set[str] = set()
        high_cardinality_families: set[str] = set()
        try:
            for metric in REGISTRY.collect():
                family = getattr(metric, "name", "")
                if family not in families:
                    continue
                for metric_sample in getattr(metric, "samples", ()):
                    if getattr(metric_sample, "name", "") != family:
                        continue
                    labels = _metric_label_key(getattr(metric_sample, "labels", {}))
                    if labels is None:
                        invalid_families.add(family)
                        continue
                    try:
                        value = float(metric_sample.value)
                    except (AttributeError, TypeError, ValueError, OverflowError):
                        invalid_families.add(family)
                        continue
                    if not math.isfinite(value) or value < 0:
                        invalid_families.add(family)
                        continue
                    if labels in records[family]:
                        duplicate_families.add(family)
                        continue
                    if len(records[family]) >= self._max_series:
                        high_cardinality_families.add(family)
                        continue
                    records[family][labels] = value
        except Exception as exc:  # noqa: BLE001 — malformed metrics are no data
            logger.debug(
                "LocalMetricsProvider bulk read failed: %s", type(exc).__name__
            )
            return results
        now = time.time()
        for request, definition in definitions.items():
            value_family = definition.local_metric_family
            observed_family = definition.local_observed_at_metric_family
            if value_family is None or observed_family is None:
                continue
            if (
                value_family in invalid_families
                or observed_family in invalid_families
                or value_family in duplicate_families
                or observed_family in duplicate_families
                or value_family in high_cardinality_families
                or observed_family in high_cardinality_families
            ):
                continue
            service, signal = request
            value_records = records[value_family]
            observed_records = records[observed_family]
            value_keys = set(value_records)
            observed_keys = set(observed_records)
            if definition.service_label is not None:
                value_keys = {
                    labels
                    for labels in value_keys
                    if dict(labels).get(definition.service_label) == service
                }
                observed_keys = {
                    labels
                    for labels in observed_keys
                    if dict(labels).get(definition.service_label) == service
                }
            if (
                not value_keys
                or value_keys != observed_keys
                or len(value_keys) > self._max_series
            ):
                continue
            values = [value_records[labels] for labels in value_keys]
            observed = [observed_records[labels] for labels in value_keys]
            aggregate_value = (
                sum(values)
                if definition.aggregation == "fleet_total"
                else sum(values) / len(values)
            )
            try:
                sample = ScalingSignalSample(
                    value=aggregate_value,
                    source=self.name,
                    service=service,
                    signal=signal,
                    aggregation=definition.aggregation,
                    observed_at=min(observed),
                    unit=definition.unit,
                    scope=definition.scope,
                )
            except (TypeError, ValueError, OverflowError):
                continue
            results[request] = validate_scaling_signal_sample(
                sample,
                service=service,
                signal=signal,
                aggregation=definition.aggregation,
                unit=definition.unit,
                scope=definition.scope,
                now=now,
            )
        return results

    def signal_value(self, service: str, signal: str) -> ScalingSignalSample | None:
        return read_scaling_signal_samples(self, [(service, signal)]).get(
            (service, signal)
        )


class PrometheusHttpProvider:
    """Bounded instant-query provider for trusted symbolic signal definitions."""

    name = "prometheus"
    trusted_in_process = False

    def __init__(
        self,
        base_url: str,
        timeout: float = 5.0,
        transport: Any = None,
        signal_definitions: Mapping[str, SignalDefinition] | None = None,
        *,
        max_series: int = MAX_RESULT_SERIES,
        max_concurrency: int = MAX_QUERY_CONCURRENCY,
        overall_timeout: float = MAX_BULK_DEADLINE_S,
        transport_thread_safe: bool | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = _bounded_timeout(timeout)
        self.max_concurrency = _bounded_concurrency(max_concurrency)
        self.overall_timeout = _bounded_bulk_deadline(overall_timeout)
        self.transport = transport
        if transport_thread_safe is not None and not isinstance(
            transport_thread_safe, bool
        ):
            raise ValueError("transport_thread_safe must be a boolean when provided")
        # The standard httpx transport is designed for a shared Client. An
        # injected transport has no such contract unless it declares one, so
        # use one client per worker for custom transports by default.
        inferred_thread_safety = (
            transport is None or getattr(transport, "thread_safe", False) is True
        )
        self._transport_thread_safe = (
            inferred_thread_safety
            if transport_thread_safe is None
            else transport_thread_safe
        )
        self._definitions = _definitions_with_overrides(signal_definitions)
        self._max_series = _bounded_series_limit(max_series)
        self._prometheus_inflight = 0

    def signal_definition(
        self, signal: str, service: str | None = None
    ) -> SignalDefinition | None:
        definition = self._definitions.get(signal, get_signal_definition(signal))
        if service is not None and (
            definition is None or not definition.binds_service(service)
        ):
            return None
        return definition

    def _query_for(
        self, service: str, signal: str
    ) -> tuple[str, SignalDefinition] | None:
        definition = self.signal_definition(signal, service)
        if definition is None or not definition.query_template:
            return None
        if not _valid_symbol(service):
            return None
        query = definition.query_template.replace("{service}", service)
        if len(query.encode("utf-8")) > MAX_QUERY_LENGTH:
            return None
        return query, definition

    def _sample_from_payload(
        self,
        payload: Any,
        service: str,
        signal: str,
        definition: SignalDefinition,
        *,
        now: float,
    ) -> ScalingSignalSample | None:
        if not isinstance(payload, Mapping) or payload.get("status") != "success":
            return None
        data = payload.get("data")
        if not isinstance(data, Mapping) or data.get("resultType") != "vector":
            return None
        result = data.get("result")
        if not isinstance(result, list) or not result or len(result) > self._max_series:
            return None
        values: list[float] = []
        observed: list[float] = []
        for entry in result:
            if not isinstance(entry, Mapping):
                return None
            metric = entry.get("metric")
            if not isinstance(metric, Mapping):
                return None
            if definition.service_label is not None:
                label_value = metric.get(definition.service_label)
                if label_value != service:
                    return None
            raw_value = entry.get("value")
            if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
                return None
            try:
                timestamp = float(raw_value[0])
                value = float(raw_value[1])
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(timestamp) or not math.isfinite(value) or value < 0:
                return None
            values.append(value)
            observed.append(timestamp)
        try:
            aggregate_value = (
                sum(values)
                if definition.aggregation == "fleet_total"
                else sum(values) / len(values)
            )
            sample = ScalingSignalSample(
                value=aggregate_value,
                source=self.name,
                service=service,
                signal=signal,
                aggregation=definition.aggregation,
                observed_at=min(observed),
                unit=definition.unit,
                scope=definition.scope,
            )
        except (TypeError, ValueError, OverflowError):
            return None
        return validate_scaling_signal_sample(
            sample,
            service=service,
            signal=signal,
            aggregation=definition.aggregation,
            unit=definition.unit,
            scope=definition.scope,
            now=now,
        )

    def _fetch_payload(self, client: Any, query: str, timeout: float) -> Any:
        response = client.get(
            f"{self.base_url}/api/v1/query",
            params={"query": query},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    def signal_values(self, requests: Sequence[SignalRequest]) -> ScalingSignalBatch:
        normalised = _normalise_requests(requests)
        if normalised is None:
            return {}
        results: dict[SignalRequest, ScalingSignalSample | None] = {
            request: None for request in normalised
        }
        jobs: dict[str, list[tuple[SignalRequest, SignalDefinition]]] = {}
        for request in normalised:
            service, signal = request
            resolved = self._query_for(service, signal)
            if resolved is None:
                continue
            query, definition = resolved
            if query not in jobs and len(jobs) >= MAX_BULK_QUERIES:
                continue
            jobs.setdefault(query, []).append((request, definition))
        if not jobs:
            return results

        from agent_utilities.core.http_client import create_http_client

        deadline = time.monotonic() + self.overall_timeout
        shared_client = None
        pending: dict[Any, str] = {}
        payloads: dict[str, Any] = {}
        queries = list(jobs)
        next_query = 0
        try:
            if self._transport_thread_safe:
                shared_client = create_http_client(
                    timeout=self.timeout, transport=self.transport
                )

            def submit_available() -> None:
                nonlocal next_query
                while next_query < len(queries) and len(pending) < self.max_concurrency:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    _reap_prometheus_inflight()
                    with _PROMETHEUS_EXECUTOR_LOCK:
                        global_available = MAX_QUERY_CONCURRENCY - len(
                            _PROMETHEUS_INFLIGHT
                        )
                        provider_available = (
                            self.max_concurrency - self._prometheus_inflight
                        )
                    if global_available <= 0 or provider_available <= 0:
                        return
                    query = queries[next_query]
                    client = shared_client
                    client_owned_by_future = False
                    if client is None:
                        client = create_http_client(
                            timeout=self.timeout, transport=self.transport
                        )
                        client_owned_by_future = True
                    try:
                        future = _prometheus_submit(
                            self._fetch_payload,
                            self,
                            client,
                            client_owned_by_future,
                            query,
                            min(self.timeout, remaining),
                        )
                    except Exception:
                        if client_owned_by_future:
                            _close_prometheus_client(client)
                        raise
                    if future is None:
                        if client_owned_by_future:
                            _close_prometheus_client(client)
                        return
                    pending[future] = query
                    next_query += 1

            while pending or next_query < len(queries):
                submit_available()
                if not pending:
                    # Another provider may own every global slot, or this
                    # provider may still have a timed-out query. Do not spin or
                    # start unbounded work; remaining queries are no data.
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                done, _ = wait(
                    set(pending),
                    timeout=remaining,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    break
                for future in done:
                    query = pending.pop(future)
                    try:
                        payloads[query] = future.result()
                    except Exception as exc:  # noqa: BLE001 — one query is no data
                        logger.debug(
                            "PrometheusHttpProvider query failed: %s",
                            type(exc).__name__,
                        )
                        payloads[query] = None
                _reap_prometheus_inflight()
        except Exception as exc:  # noqa: BLE001 — provider failures are no data
            logger.debug(
                "PrometheusHttpProvider bulk setup failed: %s", type(exc).__name__
            )
        finally:
            for future in pending:
                future.cancel()
            if shared_client is not None:
                if pending:
                    _close_prometheus_client_when_idle(shared_client)
                else:
                    _close_prometheus_client(shared_client)
            _reap_prometheus_inflight()
        now = time.time()
        for query, query_requests in jobs.items():
            payload = payloads.get(query)
            if payload is None:
                continue
            for request, definition in query_requests:
                service, signal = request
                results[request] = self._sample_from_payload(
                    payload,
                    service,
                    signal,
                    definition,
                    now=now,
                )
        return results

    def signal_value(self, service: str, signal: str) -> ScalingSignalSample | None:
        return read_scaling_signal_samples(self, [(service, signal)]).get(
            (service, signal)
        )


# ── registry (deployment injection point) ───────────────────────────

_PROVIDER: ScalingSignalProvider | None = None


def set_scaling_signal_provider(provider: ScalingSignalProvider | None) -> None:
    """Register the process-wide typed provider; ``None`` resets defaults."""

    global _PROVIDER
    if provider is not None and not isinstance(provider, ScalingSignalProvider):
        raise TypeError("scaling provider must implement the typed signal contract")
    _PROVIDER = provider


def get_scaling_signal_provider() -> ScalingSignalProvider:
    """Resolve injected provider, configured Prometheus, or bounded local metrics."""

    if _PROVIDER is not None:
        return _PROVIDER
    url: str | None = None
    try:
        from agent_utilities.core.config import config as _cfg

        url = getattr(_cfg, "scaling_prometheus_url", None) or None
    except Exception:  # noqa: BLE001 — unavailable config means local no-data mode
        url = None
    if url:
        return PrometheusHttpProvider(url)
    return LocalMetricsProvider()
