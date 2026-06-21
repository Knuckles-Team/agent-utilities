"""Adaptive per-model concurrency controller (CONCEPT:KG-2.145).

The static per-model capacity (``Config.model_capacity`` =
``parallel_instances × max_parallel_calls``, see
:mod:`agent_utilities.core.model_concurrency`) is a *floor*, not a ceiling. It
declares the concurrency we know the backend can take, but it cannot know how
much headroom a beefier GPU / extra vLLM instance actually gives us, nor when the
serving tier is saturated. This module closes that gap: it watches each model's
vLLM Prometheus signals and **auto-tunes the concurrency target up toward the real
serving capacity and back off the moment the engine reports it is saturated** — so
fan-out scales with the hardware that is actually deployed, with no hardcoded
ceiling.

Design (an AIMD controller, one per model, cached):

* **Metrics URL.** Derived from the model's ``base_url`` — ``http://host[/v1]`` →
  ``http://host/metrics`` (drop a trailing ``/v1``, append ``/metrics``).
* **Signals** (per ``model_name`` label, from vLLM's ``/metrics``):
  - ``vllm:num_requests_running`` — in-flight now.
  - ``vllm:num_requests_waiting_by_reason{reason="capacity"}`` — requests queued
    because the engine is at scheduling capacity. ``> 0`` ⇒ SATURATED (primary
    saturation signal; GPU memory is unreliable on unified-memory parts).
* **AIMD tune** of the target, scraped lazily on a throttled interval:
  - SATURATED → multiplicative **decrease** toward the sustainable level
    (``max(floor, running)``, but at least ``target × 0.8``).
  - HEALTHY & near-full (``running >= 0.8 × target`` and no capacity-waiting) →
    additive **increase** (``+max(1, target × 0.25)``) to discover headroom.
  - idle / low utilisation → hold.
  - Bounds: ``floor`` = the static configured capacity (never below — no
    regression), ``ceiling`` = ``MODEL_MAX_CONCURRENCY`` (default 512) so it can
    ramp ``4 → … → 512`` as hardware allows. No hardcoded small cap.
* **Fail-safe.** If ``/metrics`` is unreachable or unparseable, the target stays
  at the static floor (ingestion never breaks). When ``KG_ADAPTIVE_CONCURRENCY``
  is off, behaviour is exactly the static value.

The semaphore in :mod:`model_concurrency` is keyed by ``(model, capacity)``, so a
changed target simply yields a fresh, larger/smaller gate on the next fan-out —
the gate "resizes" by being re-created at the new size (old gates are dropped by
:func:`reset_controllers`). Nothing here imports a heavy dep; the scrape uses
``urllib`` from the stdlib.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "AdaptiveCapacityController",
    "adaptive_capacity",
    "get_utilization",
    "reset_adaptive_controllers",
    "parse_vllm_gauge",
]

# A metrics fetcher: takes a URL, returns the raw Prometheus text. Injectable for
# tests so we never hit the network.
MetricsFetcher = Callable[[str], str]

_DEFAULT_CEILING = 512
_DEFAULT_FLOOR = 1
_MIN_POLL_INTERVAL_S = 12.0  # throttle: don't scrape on every call
_FETCH_TIMEOUT_S = 2.0
_NEAR_FULL = 0.8  # running/target ratio that counts as "near-full"
_DECREASE_FACTOR = 0.8
_INCREASE_FRACTION = 0.25


def _http_get(url: str) -> str:
    """Default metrics fetcher — a short-timeout stdlib GET (no heavy deps)."""
    with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_S) as resp:  # noqa: S310
        return resp.read().decode("utf-8", "replace")


def metrics_url_from_base(base_url: str) -> str:
    """Derive a vLLM ``/metrics`` URL from an OpenAI-style ``base_url``.

    ``http://host/v1`` → ``http://host/metrics``; ``http://host`` →
    ``http://host/metrics``. Trailing slashes and a trailing ``/v1`` segment are
    stripped before appending ``/metrics``. CONCEPT:KG-2.145.
    """
    u = (base_url or "").strip().rstrip("/")
    if u.endswith("/v1"):
        u = u[: -len("/v1")]
    u = u.rstrip("/")
    return f"{u}/metrics"


def parse_vllm_gauge(
    text: str, metric: str, *, model_name: str | None = None, reason: str | None = None
) -> float | None:
    """Sum a vLLM Prometheus gauge across matching label sets (CONCEPT:KG-2.145).

    Parses the exposition text directly (no ``prometheus_client`` dep). Lines look
    like ``vllm:num_requests_running{model_name="bge-m3"} 3.0``. When
    ``model_name``/``reason`` are given, only label sets carrying those values are
    summed. Returns ``None`` when the metric is entirely absent (so callers can
    distinguish "absent" from "0"), else the summed value.
    """
    found = False
    total = 0.0
    prefix = metric + "{"
    bare = metric + " "
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(prefix):
            close = line.find("}")
            if close == -1:
                continue
            labels = line[len(prefix) : close]
            value_part = line[close + 1 :].strip()
            if model_name is not None and f'model_name="{model_name}"' not in labels:
                continue
            if reason is not None and f'reason="{reason}"' not in labels:
                continue
        elif line.startswith(bare) and model_name is None and reason is None:
            value_part = line[len(bare) :].strip()
        else:
            continue
        try:
            total += float(value_part.split()[0])
            found = True
        except (ValueError, IndexError):
            continue
    return total if found else None


@dataclass
class _Sample:
    running: float = 0.0
    waiting_capacity: float = 0.0
    ok: bool = False  # True only when the scrape + parse succeeded


@dataclass
class AdaptiveCapacityController:
    """AIMD concurrency auto-tuner for ONE model (CONCEPT:KG-2.145).

    Starts at ``floor`` (the static configured capacity) and ramps toward
    ``ceiling`` as the model's vLLM signals show headroom, backing off on
    capacity-waiting. ``current_target`` is what :func:`adaptive_capacity` returns.
    """

    model_key: str
    model_name: str | None
    metrics_url: str | None
    floor: int
    ceiling: int
    fetcher: MetricsFetcher = _http_get
    min_poll_interval_s: float = _MIN_POLL_INTERVAL_S

    current_target: int = field(init=False)
    last_sample: _Sample = field(init=False, default_factory=_Sample)
    last_poll_ts: float = field(init=False, default=0.0)
    _lock: threading.Lock = field(
        init=False, default_factory=threading.Lock, repr=False
    )

    def __post_init__(self) -> None:
        self.floor = max(_DEFAULT_FLOOR, int(self.floor))
        self.ceiling = max(self.floor, int(self.ceiling))
        self.current_target = self.floor

    # -- internals ---------------------------------------------------------
    def _scrape(self) -> _Sample:
        """Fetch + parse the model's gauges. Any failure → ``ok=False`` (fail-safe)."""
        if not self.metrics_url:
            return _Sample(ok=False)
        try:
            text = self.fetcher(self.metrics_url)
        except Exception:  # noqa: BLE001 — unreachable metrics must not break anything
            return _Sample(ok=False)
        running = parse_vllm_gauge(
            text, "vllm:num_requests_running", model_name=self.model_name
        )
        waiting_cap = parse_vllm_gauge(
            text,
            "vllm:num_requests_waiting_by_reason",
            model_name=self.model_name,
            reason="capacity",
        )
        if running is None and waiting_cap is None:
            # Neither gauge present → we cannot read this model; stay safe.
            return _Sample(ok=False)
        return _Sample(
            running=running or 0.0,
            waiting_capacity=waiting_cap or 0.0,
            ok=True,
        )

    def _tune(self, sample: _Sample) -> None:
        """Apply one AIMD step from a fresh sample (caller holds ``_lock``)."""
        if not sample.ok:
            # Fail-safe: cannot read the backend → never tune above the floor.
            self.current_target = self.floor
            return
        target = self.current_target
        if sample.waiting_capacity > 0:
            # SATURATED → multiplicative decrease toward the sustainable level.
            sustainable = max(float(self.floor), sample.running)
            target = max(sustainable, target * _DECREASE_FACTOR)
        elif sample.running >= _NEAR_FULL * target:
            # HEALTHY & near-full → additive increase to discover headroom.
            target = target + max(1.0, target * _INCREASE_FRACTION)
        # else: idle/low → hold.
        self.current_target = int(
            max(self.floor, min(float(self.ceiling), round(target)))
        )

    def _maybe_poll(self, now: float) -> None:
        """Throttled scrape+tune; cheap no-op between intervals (caller holds lock)."""
        if now - self.last_poll_ts < self.min_poll_interval_s and self.last_poll_ts:
            return
        self.last_poll_ts = now
        sample = self._scrape()
        self.last_sample = sample
        self._tune(sample)

    # -- public API --------------------------------------------------------
    def resolve(self) -> int:
        """Return the current adaptive target, polling at most once per interval.

        Fail-safe: any scrape/parse failure pins the target at ``floor``.
        """
        with self._lock:
            self._maybe_poll(time.monotonic())
            return self.current_target

    def utilization(self) -> dict[str, object]:
        """Observability snapshot for this model (CONCEPT:KG-2.145).

        Triggers a throttled poll so the numbers are reasonably fresh, then
        returns the parsed gauges, the live target, and bounds.
        """
        with self._lock:
            self._maybe_poll(time.monotonic())
            s = self.last_sample
            return {
                "model": self.model_key,
                "model_name": self.model_name,
                "metrics_url": self.metrics_url,
                "running": s.running,
                "waiting_capacity": s.waiting_capacity,
                "metrics_ok": s.ok,
                "current_target": self.current_target,
                "floor": self.floor,
                "ceiling": self.ceiling,
                "last_poll": self.last_poll_ts,
                "saturated": bool(s.ok and s.waiting_capacity > 0),
            }


# --- Cached per-model controllers -------------------------------------------
_lock = threading.Lock()
_controllers: dict[str, AdaptiveCapacityController] = {}


def _key(model: str | None) -> str:
    return (model or "__default__").strip().lower() or "__default__"


def _enabled() -> bool:
    from agent_utilities.core._env import setting

    return bool(setting("KG_ADAPTIVE_CONCURRENCY", True))


def _ceiling() -> int:
    from agent_utilities.core._env import setting

    return max(1, int(setting("MODEL_MAX_CONCURRENCY", _DEFAULT_CEILING)))


def _get_controller(
    model: str | None, floor: int, *, fetcher: MetricsFetcher | None = None
) -> AdaptiveCapacityController | None:
    """Return (creating if needed) the cached controller for ``model``.

    Returns ``None`` when the model has no resolvable ``base_url`` (nothing to
    scrape) — callers then fall back to the static floor.
    """
    try:
        from agent_utilities.core.config import config

        model_name, base_url = config.model_endpoint(model)
    except Exception:  # noqa: BLE001 — no config → no adaptation, fall back to static
        return None
    if not base_url:
        return None
    k = _key(model)
    with _lock:
        ctrl = _controllers.get(k)
        if ctrl is None:
            ctrl = AdaptiveCapacityController(
                model_key=k,
                model_name=model_name,
                metrics_url=metrics_url_from_base(base_url),
                floor=max(_DEFAULT_FLOOR, int(floor)),
                ceiling=_ceiling(),
                fetcher=fetcher or _http_get,
            )
            _controllers[k] = ctrl
        else:
            # Keep the floor in lockstep with config (a reload may raise capacity);
            # never let the target drop below the new floor.
            new_floor = max(_DEFAULT_FLOOR, int(floor))
            if new_floor != ctrl.floor:
                ctrl.floor = new_floor
                if ctrl.current_target < new_floor:
                    ctrl.current_target = new_floor
            if fetcher is not None:
                ctrl.fetcher = fetcher
        return ctrl


def adaptive_capacity(
    model: str | None, floor: int, *, fetcher: MetricsFetcher | None = None
) -> int:
    """Resolve the adaptive concurrency target for ``model`` (CONCEPT:KG-2.145).

    ``floor`` is the static configured capacity (``Config.model_capacity``). When
    ``KG_ADAPTIVE_CONCURRENCY`` is off, or the model has no scrapeable endpoint, or
    any scrape/parse fails, this returns ``floor`` unchanged (fail-safe — never
    breaks ingestion). Otherwise it returns the AIMD-tuned target, bounded to
    ``[floor, MODEL_MAX_CONCURRENCY]``.

    ``fetcher`` injects a metrics getter (tests pass a fake; production omits it).
    """
    floor = max(_DEFAULT_FLOOR, int(floor))
    if not _enabled():
        return floor
    ctrl = _get_controller(model, floor, fetcher=fetcher)
    if ctrl is None:
        return floor
    return max(floor, ctrl.resolve())


def get_utilization(model: str | None = None) -> dict[str, object]:
    """Profiling/observability view of a model's adaptive concurrency state.

    CONCEPT:KG-2.145. Returns ``{running, waiting_capacity, current_target,
    last_poll, floor, ceiling, saturated, metrics_ok, ...}``. When adaptation is
    disabled or the model has no endpoint, returns a static snapshot reporting the
    floor as the target (so an operator always gets a coherent answer).
    """
    floor = 1
    try:
        from agent_utilities.core.config import config

        floor = max(1, int(config.model_capacity(model)))
    except Exception:  # noqa: BLE001
        pass
    ctrl = _get_controller(model, floor)
    if ctrl is None or not _enabled():
        return {
            "model": _key(model),
            "adaptive": _enabled() and ctrl is not None,
            "running": 0.0,
            "waiting_capacity": 0.0,
            "current_target": floor,
            "floor": floor,
            "ceiling": _ceiling(),
            "metrics_ok": False,
            "last_poll": 0.0,
            "saturated": False,
        }
    snap = ctrl.utilization()
    snap["adaptive"] = True
    return snap


def reset_adaptive_controllers() -> None:
    """Drop all cached controllers (test isolation / config reload). CONCEPT:KG-2.145."""
    with _lock:
        _controllers.clear()
