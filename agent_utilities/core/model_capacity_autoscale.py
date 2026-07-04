"""Adaptive per-model concurrency controller (CONCEPT:AU-KG.compute.surfaces-universal-latency-signal).

The static per-model capacity (``Config.model_capacity`` =
``parallel_instances × max_parallel_calls``, see
:mod:`agent_utilities.core.model_concurrency`) is a *floor*, not a ceiling. It
declares the concurrency we know the backend can take, but it cannot know how
much headroom a beefier GPU / extra serving instance actually gives us, nor when
the serving tier is saturated. This module closes that gap: it auto-tunes the
concurrency target up toward the real serving capacity and backs off the moment
the endpoint shows it is congested — so fan-out scales with the hardware that is
actually deployed, with no hardcoded ceiling.

**The primary signal is client-observed and therefore universal.** It works
against ANY OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp server,
OpenAI, …) because it needs nothing from the server — only the latency and status
of each call, both of which we already orchestrate. This is the proven Netflix
adaptive-concurrency-limits / TCP-Vegas approach: watch how response latency
inflates as you push more concurrency, and treat that inflation (a positive
latency *gradient* away from the low-load baseline) as the congestion signal.

Design (an AIMD controller, one per model, cached):

* **Universal latency-gradient AIMD (default signal):**
  - Each fan-out call reports ``record_sample(latency_s, ok, status)``.
  - A low-load **baseline** RTT is tracked per model: the EWMA of the *smallest*
    observed latencies (it only moves down toward fast samples, so transient
    spikes never poison it).
  - ``gradient = baseline / max(avg_recent_latency, baseline)`` ∈ (0, 1]. Near
    ``1`` ⇒ latency hasn't inflated ⇒ not queueing ⇒ **additive increase** the
    target. Well below the configured target (default ``0.9``) ⇒ latency is
    inflated ⇒ queueing ⇒ **multiplicative decrease**.
  - Any ``429``/``503``/overload sample is a congestion event ⇒ **immediate
    multiplicative decrease** (back off), regardless of the gradient.
  - Idle / too-few-samples ⇒ hold. The target is re-evaluated on a throttle
    (every ``MODEL_AUTOSCALE_UPDATE_SAMPLES`` samples or
    ``MODEL_AUTOSCALE_UPDATE_INTERVAL_S`` seconds, whichever first) over a rolling
    window of recent samples.

* **vLLM ``/metrics`` (optional precision booster):**
  - The metrics URL is derived from the model's ``base_url`` — ``http://host[/v1]``
    → ``http://host/metrics``. It is **auto-detected**: if the endpoint serves
    vLLM-style gauges (``vllm:num_requests_running`` /
    ``vllm:num_requests_waiting_by_reason{reason="capacity"}``), then
    ``waiting{capacity} > 0`` is layered on top as a hard saturation signal that
    forces a back-off. If ``/metrics`` is absent or non-vLLM, the controller
    relies purely on the universal latency-gradient. **/metrics is never
    required.**

* **Bounds:** ``floor`` = the static configured capacity (never below — no
  regression), ``ceiling`` = ``MODEL_MAX_CONCURRENCY`` (default 512), so it can
  ramp ``4 → … → 512`` as hardware allows. No hardcoded small cap.

* **Fail-safe.** When ``KG_ADAPTIVE_CONCURRENCY`` is off, behaviour is exactly the
  static value. With no samples and no readable metrics the target stays at the
  static floor (ingestion never breaks).

The semaphore in :mod:`model_concurrency` is keyed by ``(model, capacity)``, so a
changed target simply yields a fresh, larger/smaller gate on the next fan-out —
the gate "resizes" by being re-created at the new size (old gates are dropped by
:func:`reset_controllers`). Nothing here imports a heavy dep; the optional scrape
uses ``urllib`` from the stdlib.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "AdaptiveCapacityController",
    "adaptive_capacity",
    "get_utilization",
    "record_sample",
    "reset_adaptive_controllers",
    "parse_vllm_gauge",
    "metrics_url_from_base",
]

# A metrics fetcher: takes a URL, returns the raw Prometheus text. Injectable for
# tests so we never hit the network.
MetricsFetcher = Callable[[str], str]

_DEFAULT_CEILING = 512
_DEFAULT_FLOOR = 1
_MIN_POLL_INTERVAL_S = 12.0  # throttle: don't scrape /metrics on every call
_FETCH_TIMEOUT_S = 2.0
_NEAR_FULL = 0.8  # running/target ratio that counts as "near-full" (vLLM path)
_DECREASE_FACTOR = 0.8
_INCREASE_FRACTION = 0.25

# --- Universal latency-gradient AIMD defaults -------------------------------
_GRADIENT_TARGET = 0.9  # gradient >= this ⇒ increase; well below ⇒ decrease
_WINDOW = 40  # rolling window of recent samples for avg latency
_UPDATE_SAMPLES = 8  # re-tune the target at most once per N samples …
_UPDATE_INTERVAL_S = 1.0  # … or once per T seconds, whichever comes first
_MIN_SAMPLES = 4  # below this many samples in the window → hold
_BASELINE_ALPHA = 0.2  # EWMA weight when a new minimum-ish latency arrives
# Statuses that mean "the endpoint is overloaded / shedding load".
_OVERLOAD_STATUSES = frozenset({429, 502, 503, 504, 529})


def _http_get(url: str) -> str:
    """Default metrics fetcher — a short-timeout stdlib GET (no heavy deps)."""
    # internal metrics endpoint (http://…/metrics), fixed scheme — not user input
    with urllib.request.urlopen(  # noqa: S310  # nosec B310
        url, timeout=_FETCH_TIMEOUT_S
    ) as resp:
        return resp.read().decode("utf-8", "replace")


def metrics_url_from_base(base_url: str) -> str:
    """Derive a vLLM ``/metrics`` URL from an OpenAI-style ``base_url``.

    ``http://host/v1`` → ``http://host/metrics``; ``http://host`` →
    ``http://host/metrics``. Trailing slashes and a trailing ``/v1`` segment are
    stripped before appending ``/metrics``. CONCEPT:AU-KG.compute.surfaces-universal-latency-signal.
    """
    u = (base_url or "").strip().rstrip("/")
    if u.endswith("/v1"):
        u = u[: -len("/v1")]
    u = u.rstrip("/")
    return f"{u}/metrics"


def parse_vllm_gauge(
    text: str, metric: str, *, model_name: str | None = None, reason: str | None = None
) -> float | None:
    """Sum a vLLM Prometheus gauge across matching label sets (CONCEPT:AU-KG.compute.surfaces-universal-latency-signal).

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
class _MetricsSample:
    """A parsed vLLM /metrics scrape (optional precision booster)."""

    running: float = 0.0
    waiting_capacity: float = 0.0
    ok: bool = False  # True only when the scrape + parse succeeded
    available: bool = False  # True if this endpoint serves vLLM-style gauges


@dataclass
class AdaptiveCapacityController:
    """AIMD concurrency auto-tuner for ONE model (CONCEPT:AU-KG.compute.surfaces-universal-latency-signal).

    Starts at ``floor`` (the static configured capacity) and ramps toward
    ``ceiling``. The **primary, universal signal** is client-observed latency +
    status, fed via :meth:`record_sample`: a near-baseline latency gradient ramps
    the target up (additive), an inflated gradient or an overload status backs it
    off (multiplicative). When the endpoint happens to expose vLLM ``/metrics``,
    capacity-waiting is layered on as an additional hard back-off signal — but it
    is never required. ``current_target`` is what :func:`adaptive_capacity`
    returns.
    """

    model_key: str
    model_name: str | None
    metrics_url: str | None
    floor: int
    ceiling: int
    gpu_group: str | None = None
    fetcher: MetricsFetcher = _http_get
    min_poll_interval_s: float = _MIN_POLL_INTERVAL_S
    gradient_target: float = _GRADIENT_TARGET
    window: int = _WINDOW
    update_samples: int = _UPDATE_SAMPLES
    update_interval_s: float = _UPDATE_INTERVAL_S
    metrics_enabled: bool = True

    current_target: int = field(init=False)
    last_metrics: _MetricsSample = field(init=False, default_factory=_MetricsSample)
    last_poll_ts: float = field(init=False, default=0.0)
    # latency-gradient state
    baseline_latency: float | None = field(init=False, default=None)
    _latencies: deque[float] = field(init=False, repr=False, default_factory=deque)
    _errors: deque[bool] = field(init=False, repr=False, default_factory=deque)
    _samples_since_tune: int = field(init=False, default=0)
    _last_tune_ts: float = field(init=False, default=0.0)
    _last_gradient: float = field(init=False, default=1.0)
    _last_signal: str = field(init=False, default="latency")
    _lock: threading.Lock = field(
        init=False, default_factory=threading.Lock, repr=False
    )

    def __post_init__(self) -> None:
        self.floor = max(_DEFAULT_FLOOR, int(self.floor))
        self.ceiling = max(self.floor, int(self.ceiling))
        self.current_target = self.floor
        self._latencies = deque(maxlen=max(1, int(self.window)))
        self._errors = deque(maxlen=max(1, int(self.window)))

    # -- universal latency-gradient signal ---------------------------------
    def record_sample(
        self, *, latency_s: float, ok: bool, status: int | None = None
    ) -> None:
        """Record one observed call (the universal congestion signal).

        ``latency_s`` is the wall-clock duration of the call; ``ok`` is whether it
        succeeded; ``status`` is the HTTP status when known. Overload statuses
        (429/503/…) — or ``ok=False`` whose status looks like an overload — trigger
        an immediate multiplicative back-off (a congestion event), like a TCP loss.
        Otherwise the sample updates the rolling baseline/avg, and the target is
        re-tuned on a throttle. CONCEPT:AU-KG.compute.surfaces-universal-latency-signal.
        """
        try:
            lat = float(latency_s)
        except (TypeError, ValueError):
            return
        if lat < 0:
            lat = 0.0
        is_overload = (status in _OVERLOAD_STATUSES) or (
            not ok
            and status is None  # an opaque failure under load → treat as congestion-ish
        )
        with self._lock:
            self._latencies.append(lat)
            self._errors.append(not ok)
            self._update_baseline(lat, ok)
            self._samples_since_tune += 1
            if is_overload:
                # Congestion event → immediate multiplicative decrease, no throttle.
                self._decrease(signal="latency", reason="overload")
                self._samples_since_tune = 0
                self._last_tune_ts = time.monotonic()
                return
            self._maybe_tune_latency(time.monotonic())

    def _update_baseline(self, lat: float, ok: bool) -> None:
        """EWMA-track the low-load baseline RTT (only moves toward fast samples)."""
        if not ok or lat <= 0:
            return
        if self.baseline_latency is None:
            self.baseline_latency = lat
        elif lat <= self.baseline_latency:
            # A new fast sample: snap most of the way toward it.
            self.baseline_latency = (
                _BASELINE_ALPHA * lat + (1 - _BASELINE_ALPHA) * self.baseline_latency
            )
        else:
            # Slower sample: let the baseline drift up very gently so it can track
            # a genuinely changed floor, but mostly hold (don't chase inflation).
            self.baseline_latency = 0.02 * lat + 0.98 * self.baseline_latency

    def _avg_recent_latency(self) -> float | None:
        if not self._latencies:
            return None
        return sum(self._latencies) / len(self._latencies)

    def _error_rate(self) -> float:
        if not self._errors:
            return 0.0
        return sum(1 for e in self._errors if e) / len(self._errors)

    def _gradient(self) -> float:
        avg = self._avg_recent_latency()
        base = self.baseline_latency
        if avg is None or base is None or avg <= 0 or base <= 0:
            return 1.0
        return min(1.0, base / max(avg, base))

    def _maybe_tune_latency(self, now: float) -> None:
        """Throttled latency-gradient AIMD step (caller holds ``_lock``)."""
        due = (
            self._samples_since_tune >= self.update_samples
            or (now - self._last_tune_ts) >= self.update_interval_s
        )
        if not due:
            return
        self._last_tune_ts = now
        self._samples_since_tune = 0
        if len(self._latencies) < _MIN_SAMPLES:
            return  # too few samples → hold
        grad = self._gradient()
        self._last_gradient = grad
        if grad >= self.gradient_target:
            # Not queueing → additive increase to discover headroom.
            self._increase(signal="latency")
        elif grad < self.gradient_target:
            # Latency inflated relative to baseline → queueing → back off.
            self._decrease(signal="latency", reason="gradient")

    # -- AIMD primitives (caller holds ``_lock``) --------------------------
    def _increase(self, *, signal: str) -> None:
        self._last_signal = signal
        target = self.current_target + max(
            1.0, self.current_target * _INCREASE_FRACTION
        )
        self._commit(target)

    def _decrease(self, *, signal: str, reason: str = "") -> None:
        self._last_signal = signal
        target = max(float(self.floor), self.current_target * _DECREASE_FACTOR)
        self._commit(target)

    def _commit(self, target: float) -> None:
        self.current_target = int(
            max(self.floor, min(float(self.ceiling), round(target)))
        )

    # -- optional vLLM /metrics booster ------------------------------------
    def _scrape(self) -> _MetricsSample:
        """Fetch + parse the model's gauges. Absent/garbage ⇒ unavailable (fail-safe)."""
        if not self.metrics_url or not self.metrics_enabled:
            return _MetricsSample(ok=False, available=False)
        try:
            text = self.fetcher(self.metrics_url)
        except Exception:  # noqa: BLE001 — unreachable metrics must not break anything
            return _MetricsSample(ok=False, available=False)
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
            # Not a vLLM-style endpoint (LM Studio / llama.cpp / OpenAI / …):
            # auto-detect failure → rely purely on the universal latency signal.
            return _MetricsSample(ok=False, available=False)
        return _MetricsSample(
            running=running or 0.0,
            waiting_capacity=waiting_cap or 0.0,
            ok=True,
            available=True,
        )

    def _maybe_poll_metrics(self, now: float) -> None:
        """Throttled optional scrape; a precision booster layered on the latency tuner.

        When a vLLM endpoint is detected this applies the engine's own scheduler
        signals on top of the universal latency signal:

        * ``waiting{capacity} > 0`` ⇒ **hard saturation** → forced multiplicative
          back-off (the headline guarantee — never queue requests at the engine).
        * else ``running >= 0.8 × target`` ⇒ near-full with headroom → additive
          increase (a faster, more precise ramp than waiting for latency to move).
        * else idle/low ⇒ hold (let the latency signal decide).

        When ``/metrics`` is absent/non-vLLM the scrape reports ``available=False``
        and this is a no-op — the controller relies purely on the latency signal.
        """
        if not self.metrics_enabled or not self.metrics_url:
            return
        if now - self.last_poll_ts < self.min_poll_interval_s and self.last_poll_ts:
            return
        self.last_poll_ts = now
        sample = self._scrape()
        self.last_metrics = sample
        if not (sample.available and sample.ok):
            return
        if sample.waiting_capacity > 0:
            # Hard saturation signal from the engine → force a back-off, layered
            # on top of whatever the latency tuner decided. Don't go below a
            # sustainable level (the work actually running right now).
            sustainable = max(float(self.floor), sample.running)
            target = max(sustainable, self.current_target * _DECREASE_FACTOR)
            self._last_signal = "vllm_metrics"
            self._commit(target)
        elif sample.running >= _NEAR_FULL * self.current_target:
            # HEALTHY & near-full → additive increase to discover headroom.
            self._last_signal = "vllm_metrics"
            self._increase(signal="vllm_metrics")

    # -- public API --------------------------------------------------------
    def resolve(self) -> int:
        """Return the current adaptive target.

        The target is driven primarily by :meth:`record_sample` (universal latency
        signal). This call additionally performs a throttled optional vLLM
        ``/metrics`` poll, which only ever *forces a back-off* when the engine
        reports capacity-waiting. Fail-safe: with no samples and no readable
        metrics the target stays at ``floor``.
        """
        with self._lock:
            self._maybe_poll_metrics(time.monotonic())
            return self.current_target

    def _active_signal(self) -> str:
        have_latency = len(self._latencies) > 0
        have_metrics = self.last_metrics.available
        if have_latency and have_metrics:
            return "hybrid"
        if have_metrics:
            return "vllm_metrics"
        return "latency"

    def utilization(self) -> dict[str, object]:
        """Observability snapshot for this model (CONCEPT:AU-KG.compute.surfaces-universal-latency-signal).

        Triggers a throttled optional /metrics poll so the numbers are reasonably
        fresh, then returns the latency-gradient state, the optional vLLM gauges,
        the live target, bounds, and which signal is active.
        """
        with self._lock:
            self._maybe_poll_metrics(time.monotonic())
            m = self.last_metrics
            avg = self._avg_recent_latency()
            grad = self._gradient()
            snap: dict[str, object] = {
                "model": self.model_key,
                "model_name": self.model_name,
                "metrics_url": self.metrics_url,
                # vLLM /metrics (optional)
                "running": m.running,
                "waiting_capacity": m.waiting_capacity,
                "metrics_ok": m.ok,
                "metrics_available": m.available,
                # universal latency signal
                "baseline_latency": self.baseline_latency,
                "recent_avg_latency": avg,
                "gradient": grad,
                "error_rate": self._error_rate(),
                "samples": len(self._latencies),
                "signal": self._active_signal(),
                # target + bounds
                "current_target": self.current_target,
                "floor": self.floor,
                "ceiling": self.ceiling,
                "last_poll": self.last_poll_ts,
                "saturated": bool(m.ok and m.waiting_capacity > 0),
            }
        snap.update(_gpu_group_fields(self.gpu_group, self.model_key))
        return snap


def _gpu_group_fields(gpu_group: str | None, model_key: str) -> dict[str, object]:
    """Shared-GPU budget fields for a utilization snapshot (CONCEPT:AU-KG.compute.pure-config-enumeration-fail).

    Always returns the four keys (``gpu_group``/``group_budget``/``group_used``/
    ``group_allowed_for_this_model``) so the surface is stable; budget/used/allowed
    are ``None`` when no budget is configured for the group.
    """
    fields: dict[str, object] = {
        "gpu_group": gpu_group,
        "group_budget": None,
        "group_used": None,
        "group_allowed_for_this_model": None,
    }
    if not gpu_group:
        return fields
    try:
        from agent_utilities.core.gpu_group_budget import group_snapshot

        snap = group_snapshot(gpu_group, model_key)
        if snap:
            fields["group_budget"] = snap.get("group_budget")
            fields["group_used"] = snap.get("group_used")
            fields["group_allowed_for_this_model"] = snap.get(
                "group_allowed_for_this_model"
            )
    except Exception:  # noqa: BLE001 — observability must never raise
        return fields
    return fields


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


def _tunables() -> dict[str, float]:
    from agent_utilities.core._env import setting

    return {
        "gradient_target": float(
            setting("MODEL_LATENCY_GRADIENT_TARGET", _GRADIENT_TARGET)
        ),
        "window": int(setting("MODEL_AUTOSCALE_WINDOW", _WINDOW)),
        "update_samples": int(
            setting("MODEL_AUTOSCALE_UPDATE_SAMPLES", _UPDATE_SAMPLES)
        ),
        "update_interval_s": float(
            setting("MODEL_AUTOSCALE_UPDATE_INTERVAL_S", _UPDATE_INTERVAL_S)
        ),
        "metrics_enabled": bool(setting("MODEL_AUTOSCALE_VLLM_METRICS", True)),
    }


def _resolve_gpu_group(model: str | None) -> str | None:
    """Resolve a model's shared-GPU group key (CONCEPT:AU-KG.compute.pure-config-enumeration-fail); never raises.

    ``None`` whenever config lacks a ``gpu_group`` resolver or it errors — the
    budget layer then applies no cap (per-model behaviour, no regression).
    """
    try:
        from agent_utilities.core.config import config

        resolver = getattr(config, "gpu_group", None)
        if resolver is None:
            return None
        return resolver(model)
    except Exception:  # noqa: BLE001 — grouping is best-effort
        return None


def _role_hint(model: str | None) -> str | None:
    """Best-effort role label for GPU-budget priority classification (CONCEPT:AU-KG.compute.pure-config-enumeration-fail).

    A model id that matches a configured chat/embedding model is classified by which
    registry it lives in; a bare role string (``"chat"``/``"embedding"``/…) is
    returned as-is. ``None`` ⇒ the group coordinator falls back to the model key.
    """
    key = (model or "").strip().lower()
    # The failover embedder key (CONCEPT:AU-KG.enrichment.each-call-resolves-active) is a best-effort embedding role,
    # so it yields the shared GPU's headroom to the latency-sensitive generator.
    if key in ("embedding:fallback", "embed:fallback", "embedding-fallback"):
        return "embedding"
    if key in ("", "chat", "default", "lite", "super", "embedding", "embed"):
        return key or "default"
    try:
        from agent_utilities.core.config import config

        for em in config.embedding_models:
            if em.id == model:
                return "embedding"
    except Exception:  # noqa: BLE001 — classification is best-effort
        return None
    return None


def _register_gpu_member(
    gpu_group: str | None, model_key: str, *, floor: int, model: str | None
) -> None:
    """Register a model into its shared-GPU budget (CONCEPT:AU-KG.compute.pure-config-enumeration-fail); never raises."""
    if not gpu_group:
        return
    try:
        from agent_utilities.core.gpu_group_budget import register_member

        register_member(gpu_group, model_key, floor=floor, role_hint=_role_hint(model))
    except Exception:  # noqa: BLE001 — the budget layer must never break adaptation
        return


def _register_gpu_group_peers(gpu_group: str | None) -> None:
    """Proactively register EVERY configured model sharing ``gpu_group`` (CONCEPT:AU-KG.compute.pure-config-enumeration-fail).

    The shared-GPU budget reserves a *priority* peer's floor off the top of every
    member's allowance — but only for peers that are actually registered. A model is
    registered when its per-model controller is first created (on its first call), so
    an **idle** priority model (e.g. chat that hasn't been called yet) would NOT be a
    member, and its floor would not be reserved while it is idle. That lets a
    best-effort peer (embedding) transiently exceed ``budget − chat_floor``.

    To make a priority peer's floor reserved at ALL times, this enumerates the
    configured models — ``config.chat_models`` + ``config.embedding_models`` — selects
    those whose :meth:`Config.gpu_group` matches ``gpu_group``, and registers each with
    its static floor (:meth:`Config.model_capacity`) and role classification. It is
    pure config enumeration: NO hardcoded model names or GPU types, so it works for any
    ``gpu_group`` value and any GPU (GB10/GB200/H100/clusters).

    Idempotent (``upsert`` only refreshes floor/role) and fail-safe: any enumeration
    error falls back to the current active-only behaviour and never raises. A group
    with no configured budget is a no-op (``register_member`` short-circuits).
    """
    if not gpu_group:
        return
    try:
        from agent_utilities.core.config import config

        models = [*config.chat_models, *config.embedding_models]
    except Exception:  # noqa: BLE001 — enumeration is best-effort; fall back to active-only
        return
    for cfg_model in models:
        try:
            model_id = getattr(cfg_model, "id", None)
            if not model_id:
                continue
            if _resolve_gpu_group(model_id) != gpu_group:
                continue
            floor = max(_DEFAULT_FLOOR, int(config.model_capacity(model_id)))
            _register_gpu_member(gpu_group, _key(model_id), floor=floor, model=model_id)
        except Exception:  # noqa: BLE001 — one bad peer must not break the rest
            continue


def _apply_gpu_cap(
    gpu_group: str | None, model_key: str, target: int, floor: int
) -> int:
    """Cap ``target`` at the model's allowed share of its GPU budget (CONCEPT:AU-KG.compute.pure-config-enumeration-fail).

    Reports the model's current target into the group, then returns
    ``min(target, group_allowed)`` floored at ``floor``. When no budget is
    configured (``group_allowed is None``) the target passes through unchanged —
    zero regression. Conservative: under contention a best-effort model is driven
    toward its floor so priority (chat) keeps its reserved slice.
    """
    if not gpu_group:
        return target
    try:
        from agent_utilities.core.gpu_group_budget import group_allowed, report_target

        report_target(gpu_group, model_key, target)
        allowed = group_allowed(gpu_group, model_key)
        if allowed is None:
            return target
        return max(floor, min(int(target), int(allowed)))
    except Exception:  # noqa: BLE001 — the budget layer must never break adaptation
        return target


def _get_controller(
    model: str | None, floor: int, *, fetcher: MetricsFetcher | None = None
) -> AdaptiveCapacityController | None:
    """Return (creating if needed) the cached controller for ``model``.

    Returns ``None`` only when no config is resolvable at all. A model with no
    ``base_url`` still gets a controller (the universal latency signal needs no
    endpoint); its optional /metrics booster is simply disabled.
    """
    try:
        from agent_utilities.core.config import config

        model_name, base_url = config.model_endpoint(model)
    except Exception:  # noqa: BLE001 — no config → no adaptation, fall back to static
        return None
    gpu_group = _resolve_gpu_group(model)
    k = _key(model)
    tun = _tunables()
    # Register this model as a member of its shared-GPU budget (CONCEPT:AU-KG.compute.pure-config-enumeration-fail);
    # a no-op when no budget is configured for the group → pure per-model behaviour.
    _register_gpu_member(
        gpu_group, k, floor=max(_DEFAULT_FLOOR, int(floor)), model=model
    )
    # Proactively register ALL configured peers sharing this GPU group, so an idle
    # priority peer (e.g. chat with no calls yet) still reserves its floor off every
    # other member's allowance (CONCEPT:AU-KG.compute.pure-config-enumeration-fail). Pure config enumeration; fail-safe.
    _register_gpu_group_peers(gpu_group)
    with _lock:
        ctrl = _controllers.get(k)
        if ctrl is None:
            ctrl = AdaptiveCapacityController(
                model_key=k,
                model_name=model_name,
                metrics_url=metrics_url_from_base(base_url) if base_url else None,
                floor=max(_DEFAULT_FLOOR, int(floor)),
                ceiling=_ceiling(),
                gpu_group=gpu_group,
                fetcher=fetcher or _http_get,
                gradient_target=tun["gradient_target"],
                window=int(tun["window"]),
                update_samples=int(tun["update_samples"]),
                update_interval_s=tun["update_interval_s"],
                metrics_enabled=bool(tun["metrics_enabled"]),
            )
            _controllers[k] = ctrl
        else:
            ctrl.gpu_group = gpu_group
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
    """Resolve the adaptive concurrency target for ``model`` (CONCEPT:AU-KG.compute.surfaces-universal-latency-signal).

    ``floor`` is the static configured capacity (``Config.model_capacity``). When
    ``KG_ADAPTIVE_CONCURRENCY`` is off, or no config is resolvable, this returns
    ``floor`` unchanged (fail-safe — never breaks ingestion). Otherwise it returns
    the AIMD-tuned target, bounded to ``[floor, MODEL_MAX_CONCURRENCY]``. The
    target is driven by client-observed samples (universal) plus, when present, a
    vLLM ``/metrics`` saturation back-off.

    ``fetcher`` injects a metrics getter (tests pass a fake; production omits it).
    """
    floor = max(_DEFAULT_FLOOR, int(floor))
    if not _enabled():
        return floor
    ctrl = _get_controller(model, floor, fetcher=fetcher)
    if ctrl is None:
        return floor
    target = max(floor, ctrl.resolve())
    # Cap at this model's allowed share of any shared-GPU budget (CONCEPT:AU-KG.compute.pure-config-enumeration-fail).
    # No budget configured → returns the target unchanged (no regression).
    return _apply_gpu_cap(ctrl.gpu_group, ctrl.model_key, target, floor)


def record_sample(
    model: str | None,
    *,
    latency_s: float,
    ok: bool,
    status: int | None = None,
) -> None:
    """Feed one observed call into a model's adaptive controller (CONCEPT:AU-KG.compute.surfaces-universal-latency-signal).

    The universal congestion signal. Called by the fan-out helpers in
    :mod:`model_concurrency` after each ``fn`` invocation. A no-op (never raises)
    when adaptation is disabled or no controller exists — observation is a pure
    side-channel and must never affect the fan-out contract.
    """
    try:
        if not _enabled():
            return
        # Don't create a controller solely to record (avoids touching config on a
        # hot path when adaptation has never been engaged for this model); but if a
        # controller already exists OR config resolves, route to it.
        k = _key(model)
        with _lock:
            ctrl = _controllers.get(k)
        if ctrl is None:
            ctrl = _get_controller(model, _DEFAULT_FLOOR)
        if ctrl is None:
            return
        ctrl.record_sample(latency_s=latency_s, ok=ok, status=status)
    except Exception:  # noqa: BLE001 — observation must never break fan-out
        return


def get_utilization(model: str | None = None) -> dict[str, object]:
    """Profiling/observability view of a model's adaptive concurrency state.

    CONCEPT:AU-KG.compute.surfaces-universal-latency-signal. Surfaces the universal latency signal (``baseline_latency``,
    ``recent_avg_latency``, ``gradient``, ``error_rate``), the active ``signal``
    (``"latency"``|``"vllm_metrics"``|``"hybrid"``), the optional vLLM gauges, the
    live ``current_target``, and bounds. When adaptation is disabled or no config
    resolves, returns a static snapshot reporting the floor as the target.
    """
    floor = 1
    try:
        from agent_utilities.core.config import config

        floor = max(1, int(config.model_capacity(model)))
    except Exception:  # noqa: BLE001
        pass
    ctrl = _get_controller(model, floor)
    if ctrl is None or not _enabled():
        gpu_group = _resolve_gpu_group(model)
        static = {
            "model": _key(model),
            "adaptive": _enabled() and ctrl is not None,
            "running": 0.0,
            "waiting_capacity": 0.0,
            "metrics_ok": False,
            "metrics_available": False,
            "baseline_latency": None,
            "recent_avg_latency": None,
            "gradient": 1.0,
            "error_rate": 0.0,
            "samples": 0,
            "signal": "latency",
            "current_target": floor,
            "floor": floor,
            "ceiling": _ceiling(),
            "last_poll": 0.0,
            "saturated": False,
        }
        static.update(_gpu_group_fields(gpu_group, _key(model)))
        return static
    snap = ctrl.utilization()
    snap["adaptive"] = True
    return snap


def reset_adaptive_controllers() -> None:
    """Drop all cached controllers (test isolation / config reload). CONCEPT:AU-KG.compute.surfaces-universal-latency-signal.

    Also drops the shared-GPU budget registry (CONCEPT:AU-KG.compute.pure-config-enumeration-fail) so a reload
    re-derives both the per-model state and the per-GPU member set from fresh config.
    """
    with _lock:
        _controllers.clear()
    try:
        from agent_utilities.core.gpu_group_budget import reset_gpu_group_budgets

        reset_gpu_group_budgets()
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass
