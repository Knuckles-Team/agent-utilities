"""Shared-GPU concurrency budget coordinator (CONCEPT:KG-2.146).

The per-model adaptive controller (CONCEPT:KG-2.145,
:mod:`model_capacity_autoscale`) tunes each model's concurrency target on its own,
toward that model's real serving capacity. But several models can be served from
**one physical GPU** — on our homelab the embedder ``bge-m3`` (``vllm-embed.arpa``)
and the chat model ``qwen3.5-9b`` (``vllm.arpa``) are *different endpoints sharing
the same GB10* (unified memory). Each per-model controller, tuning in isolation,
would happily ramp both at once and **jointly oversubscribe** the device — bulk
embedding would starve interactive chat of GPU time.

This module adds the missing layer: a **per-GPU budget** that caps the *sum* of
the concurrency targets of all models on one GPU, with a **reserved slice for
latency-sensitive roles** (chat/generator) so interactive latency is protected
even while best-effort work (embedding/batch) is saturating the leftover headroom.

Design (a thin, conservative cap layered on top of the per-model controller):

* **Grouping** is resolved upstream by :meth:`Config.gpu_group` — explicit
  ``gpu_group`` tag wins, else the ``base_url`` host. This module is told the group
  key, the group's budget, and each member's (floor, role, current target).
* **Members are seeded PROACTIVELY from config**, not only when a model first runs.
  The autoscale layer (:mod:`model_capacity_autoscale`,
  ``_register_gpu_group_peers``) enumerates every configured model sharing a group
  on first touch and registers each with its static floor + role — so an **idle**
  priority peer (chat that has never been called) still reserves its floor off every
  other member's allowance. This is the hard guarantee: a best-effort peer can never
  transiently exceed ``budget − Σ priority floors`` while a priority peer sits idle.
* **Roles split into two classes.** *Priority* roles (``GPU_RESERVED_ROLES`` —
  default ``chat``/``generator``/``default``) always keep their static floor
  reserved off the top of the budget. *Best-effort* roles (embedding, batch) may be
  squeezed down toward their own floor when the priority members are active.
* **The cap for one model** is the budget minus what its peers are entitled to /
  currently using:

      allowed(m) = budget
                   − Σ floor(p)            for priority peers p ≠ m   (reserved)
                   − Σ target(q)           for best-effort peers q ≠ m (in use)

  floored at ``floor(m)`` for a **priority** model (a priority model is *never*
  squeezed below its own floor — that floor is its guaranteed reserved slice), and
  floored at ``floor(m)`` for a best-effort model too, but only after the priority
  reservations are subtracted (so when chat is busy, embedding is driven down to —
  but never below — its own floor; when chat is idle, embedding reclaims up to the
  whole budget).

  The headline guarantee: ``Σ allowed`` across the group never exceeds the budget
  given each member's actual demand, and a priority model can always obtain at
  least its floor (its reservation is subtracted from *every* peer's allowance, so
  no peer can consume it).

* **No budget configured ⇒ no cap.** When the group is unknown or has no budget,
  :func:`group_allowed` returns ``None`` and the per-model target passes through
  unchanged — zero regression.

This module holds **no network or heavy deps**; it is pure arithmetic over a small
registry the per-model controllers feed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

__all__ = [
    "GpuGroupBudget",
    "register_member",
    "report_target",
    "group_allowed",
    "group_snapshot",
    "reset_gpu_group_budgets",
    "DEFAULT_RESERVED_ROLES",
]

# Latency-sensitive roles that get their floor reserved FIRST off the GPU budget.
DEFAULT_RESERVED_ROLES = frozenset({"chat", "generator", "default", "lite", "super"})


@dataclass
class _Member:
    """One model's slice of a shared-GPU group (CONCEPT:KG-2.146)."""

    model_key: str
    floor: int = 1
    priority: bool = False
    # The model's current resolved per-model target (its demand), reported by the
    # per-model controller each time it re-tunes. Starts at the floor.
    current_target: int = 1


@dataclass
class GpuGroupBudget:
    """Tracks one GPU's budget + its member models' demand (CONCEPT:KG-2.146).

    ``budget`` is the total concurrent in-flight calls allowed across *all* models
    sharing this GPU. Members register their floor + role; each re-tune reports the
    member's current per-model target. :meth:`allowed_for` returns the cap to apply
    to a given model so the group never oversubscribes and priority roles keep their
    reserved floors.
    """

    group: str
    budget: int
    members: dict[str, _Member] = field(default_factory=dict)
    _lock: threading.Lock = field(
        init=False, default_factory=threading.Lock, repr=False
    )

    def upsert(self, model_key: str, *, floor: int, priority: bool) -> None:
        with self._lock:
            m = self.members.get(model_key)
            if m is None:
                self.members[model_key] = _Member(
                    model_key=model_key,
                    floor=max(1, int(floor)),
                    priority=bool(priority),
                    current_target=max(1, int(floor)),
                )
            else:
                m.floor = max(1, int(floor))
                m.priority = bool(priority)
                if m.current_target < m.floor:
                    m.current_target = m.floor

    def report(self, model_key: str, target: int) -> None:
        with self._lock:
            m = self.members.get(model_key)
            if m is not None:
                m.current_target = max(1, int(target))

    def allowed_for(self, model_key: str) -> int | None:
        """Cap for ``model_key`` under this GPU's budget (CONCEPT:KG-2.146).

        ``None`` when the model isn't a registered member (caller applies no cap).
        Otherwise: budget − reserved floors of *priority* peers − current targets
        of *best-effort* peers, floored at this model's own floor.
        """
        with self._lock:
            me = self.members.get(model_key)
            if me is None:
                return None
            reserved_priority_peers = 0
            best_effort_peers_in_use = 0
            for k, other in self.members.items():
                if k == model_key:
                    continue
                if other.priority:
                    reserved_priority_peers += other.floor
                else:
                    best_effort_peers_in_use += other.current_target
            allowed = self.budget - reserved_priority_peers - best_effort_peers_in_use
            # A model can always have at least its own floor: a priority model's
            # floor is its guaranteed reserved slice; a best-effort model is never
            # driven below its floor (it just stops growing / yields its surplus).
            return max(me.floor, int(allowed))

    def snapshot(self, model_key: str | None = None) -> dict[str, object]:
        with self._lock:
            used = sum(m.current_target for m in self.members.values())
            out: dict[str, object] = {
                "gpu_group": self.group,
                "group_budget": self.budget,
                "group_used": used,
                "members": {
                    k: {
                        "floor": m.floor,
                        "priority": m.priority,
                        "current_target": m.current_target,
                    }
                    for k, m in self.members.items()
                },
            }
            if model_key is not None and model_key in self.members:
                # Inline the per-model allowed so callers get a consistent view.
                me = self.members[model_key]
                reserved = sum(
                    o.floor
                    for k, o in self.members.items()
                    if k != model_key and o.priority
                )
                in_use = sum(
                    o.current_target
                    for k, o in self.members.items()
                    if k != model_key and not o.priority
                )
                out["group_allowed_for_this_model"] = max(
                    me.floor, int(self.budget - reserved - in_use)
                )
            return out


# --- Module-level registry (one budget per GPU group) -----------------------
_lock = threading.Lock()
_budgets: dict[str, GpuGroupBudget] = {}


def _budget_for_group(group: str) -> int | None:
    """Configured concurrent budget for ``group`` (CONCEPT:KG-2.146).

    Read from ``GPU_CONCURRENCY_BUDGETS`` — a JSON/dict mapping ``group -> int``.
    An unmapped group (or unparsable config) returns ``None`` ⇒ no budget ⇒ no cap.
    """
    from agent_utilities.core._env import setting

    raw = setting("GPU_CONCURRENCY_BUDGETS", {})
    mapping: dict[str, object]
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            import json

            parsed = json.loads(raw)
            mapping = parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return None
    elif isinstance(raw, dict):
        mapping = raw
    else:
        return None
    val = mapping.get(group)
    if not isinstance(val, int | float | str):
        return None
    try:
        b = int(val)
    except (ValueError, TypeError):
        return None
    return b if b > 0 else None


def _reserved_roles() -> frozenset[str]:
    """The latency-sensitive roles whose floor is reserved first (CONCEPT:KG-2.146)."""
    from agent_utilities.core._env import setting

    raw = setting("GPU_RESERVED_ROLES", "")
    if not raw:
        return DEFAULT_RESERVED_ROLES
    if isinstance(raw, list | tuple | set | frozenset):
        items = [str(x) for x in raw]
    else:
        items = str(raw).split(",")
    roles = {x.strip().lower() for x in items if x and x.strip()}
    return frozenset(roles) if roles else DEFAULT_RESERVED_ROLES


def _is_priority_role(model_key: str, role_hint: str | None) -> bool:
    """Classify a model as priority (latency-sensitive) vs best-effort.

    ``role_hint`` (e.g. ``"chat"``/``"embedding"``) is checked against the reserved
    roles; the model key itself is also checked so an embedding model keyed
    ``"embedding"`` is best-effort while ``"chat"`` is priority.
    """
    reserved = _reserved_roles()
    candidates = {(role_hint or "").strip().lower(), (model_key or "").strip().lower()}
    return bool(candidates & reserved)


def _get_budget(group: str) -> GpuGroupBudget | None:
    budget = _budget_for_group(group)
    if budget is None:
        return None
    with _lock:
        gb = _budgets.get(group)
        if gb is None:
            gb = GpuGroupBudget(group=group, budget=budget)
            _budgets[group] = gb
        elif gb.budget != budget:
            gb.budget = budget  # live config reload may raise/lower the budget
        return gb


def register_member(
    group: str | None,
    model_key: str,
    *,
    floor: int,
    role_hint: str | None = None,
) -> None:
    """Register a model as a member of its GPU group (CONCEPT:KG-2.146).

    No-op when ``group`` is falsy or the group has no configured budget — so an
    un-budgeted deployment behaves exactly as per-model (no regression).
    """
    if not group:
        return
    gb = _get_budget(group)
    if gb is None:
        return
    gb.upsert(model_key, floor=floor, priority=_is_priority_role(model_key, role_hint))


def report_target(group: str | None, model_key: str, target: int) -> None:
    """Report a member's current per-model target into its group (CONCEPT:KG-2.146)."""
    if not group:
        return
    gb = _get_budget(group)
    if gb is None:
        return
    gb.report(model_key, target)


def group_allowed(group: str | None, model_key: str) -> int | None:
    """Group cap for ``model_key`` (CONCEPT:KG-2.146); ``None`` ⇒ no cap.

    ``None`` whenever there is no group, no configured budget, or the model isn't a
    registered member — the per-model target then passes through unchanged.
    """
    if not group:
        return None
    gb = _get_budget(group)
    if gb is None:
        return None
    return gb.allowed_for(model_key)


def group_snapshot(
    group: str | None, model_key: str | None = None
) -> dict[str, object]:
    """Observability view of a GPU group (CONCEPT:KG-2.146).

    Returns the budget, total used, and per-member targets; when ``model_key`` is a
    member it also includes ``group_allowed_for_this_model``. An empty dict when no
    budget is configured for the group.
    """
    if not group:
        return {}
    gb = _get_budget(group)
    if gb is None:
        return {}
    return gb.snapshot(model_key)


def reset_gpu_group_budgets() -> None:
    """Drop all cached group budgets (test isolation / config reload). CONCEPT:KG-2.146."""
    with _lock:
        _budgets.clear()
