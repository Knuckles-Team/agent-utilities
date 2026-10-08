"""AU-CONTEXT-R004: context sizing as a certified multi-resolution 0-1 knapsack.

The context compiler used to fit its ranked evidence greedily into a flat
token budget, counting tokens as ``chars / 3.5``. Here the choice is an
optimisation EG solves and certifies (``Method::Solve`` -- a bounded
0-1 programme with a verifiable certificate; EG runs no model):

* items are evidence units at several RESOLUTIONS (the unit's summary, or its
  full text), at most one resolution per unit;
* an item's weight is its EXACT token count under the target model's
  tokenizer -- AU's, because the tokenizer belongs to the model;
* capacity is ``min(window - reserved output, budget / price per token,
  latency budget x tokens per second, the caller's token budget)``;
* value is calibrated relevance x evidence weight, minus a marginal floor per
  token, so an item whose value does not pay for its tokens is never taken --
  easy tasks naturally get less context;
* among equal-value selections the one with fewer tokens wins (second level).

When EG is unreachable or does not certify, the deterministic greedy fit runs
instead and the result says so (``certified=False``). The sizing identity
(tokenizer, capacity, floor) is part of the bundle cache key, so a different
sizing never reuses another's bundle.

The sizer is per MODEL: ``compile_model_context`` -- the one entrypoint every
model invocation's evidence goes through -- resolves the invoked model's
:class:`ContextSizer` (:func:`sizer_for_model`: its registry definition's
window/output/price, its exact tokenizer, EG's ``Solve`` over the installed
decision transport) and compiles inside :func:`sizing_scope`. A model with no
registry definition or no exact tokenizer keeps the greedy fit.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import logging
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Integer scale of item values in the solver's objective.
VALUE_SCALE = 1_000_000
#: Solve statuses that certify optimality.
_CERTIFIED = frozenset({"optimal", "optimal_by_deterministic_search"})


class TokenCounter(Protocol):
    """Exact token counts under one model's tokenizer."""

    @property
    def identity(self) -> str: ...

    def count(self, texts: Sequence[str]) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class TiktokenCounter:
    """A ``tiktoken`` encoding (the OpenAI-family tokenizers)."""

    encoding: Any
    identity: str

    def count(self, texts: Sequence[str]) -> list[int]:
        return [
            len(self.encoding.encode(text, disallowed_special=())) for text in texts
        ]


@functools.lru_cache(maxsize=64)
def tiktoken_counter(model_name: str) -> TiktokenCounter | None:
    """The exact counter for ``model_name`` when tiktoken knows its encoding."""
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model_name)
    except (ImportError, KeyError) as exc:
        logger.warning(
            "no exact tokenizer for %s (the greedy estimate sizes it): %s",
            model_name,
            exc,
        )
        return None
    return TiktokenCounter(encoding, f"tiktoken:{encoding.name}")


def _whole(tokens: float) -> int:
    """Whole tokens within a float limit (a rounding error never costs one)."""
    return math.floor(tokens + 1e-9)


@dataclass(frozen=True, slots=True)
class Capacity:
    """The token capacity one context may use, from the model's facts."""

    window: int
    reserved_output: int = 0
    budget_usd: float | None = None
    usd_per_token: float | None = None
    latency_s: float | None = None
    tokens_per_second: float | None = None

    def tokens(self, caller_budget: int | None = None) -> int:
        limits = [self.window - self.reserved_output]
        if self.budget_usd is not None and self.usd_per_token:
            limits.append(_whole(self.budget_usd / self.usd_per_token))
        if self.latency_s is not None and self.tokens_per_second:
            limits.append(_whole(self.latency_s * self.tokens_per_second))
        if caller_budget is not None:
            limits.append(int(caller_budget))
        return max(0, min(limits))


def capacity_of(definition: Any, **limits: Any) -> Capacity:
    """A :class:`Capacity` from a ``ModelDefinition`` (window, output, price)."""
    cost = getattr(definition, "cost", None)
    per_million = getattr(cost, "input", None) if cost is not None else None
    return Capacity(
        window=int(getattr(definition, "context_window", None) or 0),
        reserved_output=int(getattr(definition, "max_output_tokens", None) or 0),
        usd_per_token=None if per_million is None else float(per_million) / 1_000_000,
        **limits,
    )


@dataclass(frozen=True, slots=True)
class Slice:
    """One evidence unit at one resolution."""

    group: str
    resolution: str
    text: str
    value: float


@dataclass(frozen=True, slots=True)
class SliceSelection:
    """The chosen slices, their exact token total and how they were chosen."""

    chosen: tuple[Slice, ...]
    tokens: int
    capacity: int
    certified: bool
    reason: str
    certificate_digest: str | None = None


#: Solve one EG ``SolveRequest``; returns EG's ``SolveResult`` (body or model).
Solver = Callable[[Mapping[str, Any]], Any]


def _net_value(item: Slice, weight: int, floor_per_token: float) -> int:
    return round((item.value - floor_per_token * weight) * VALUE_SCALE)


def knapsack_model(
    items: Sequence[Slice],
    weights: Sequence[int],
    capacity: int,
    floor_per_token: float,
) -> dict[str, Any]:
    """The 0-1 programme: at most one resolution per unit, total tokens within
    capacity; minimise the negated net value, then the tokens."""
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        groups.setdefault(item.group, []).append(index)
    constraints: list[dict[str, Any]] = [
        {
            "label": f"one-resolution:{group}",
            "body": {"at_most": {"vars": members, "k": 1}},
        }
        for group, members in groups.items()
    ]
    constraints.append(
        {
            "label": "capacity",
            "body": {
                "linear": {
                    "terms": [
                        {"var": i, "coefficient": w} for i, w in enumerate(weights)
                    ],
                    "relation": "less_equal",
                    "rhs": int(capacity),
                }
            },
        }
    )
    net = [
        _net_value(item, w, floor_per_token)
        for item, w in zip(items, weights, strict=True)
    ]
    return {
        "variables": [f"{item.group}@{item.resolution}" for item in items],
        "constraints": constraints,
        "objective": [
            {"label": "net-value", "terms": _terms([-v for v in net])},
            {"label": "tokens", "terms": _terms(list(weights))},
        ],
    }


def _terms(coefficients: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {"var": i, "coefficient": {"known": int(c)}} for i, c in enumerate(coefficients)
    ]


def _status_kind(status: Any) -> str:
    """EG's externally tagged ``SolveStatus``: a unit variant is its name, a
    struct variant a one-key mapping."""
    if isinstance(status, Mapping):
        return next(iter(status), "")
    return str(status)


def _certified_choice(answer: Any) -> tuple[list[bool], str] | None:
    body = answer.model_dump(mode="json") if hasattr(answer, "model_dump") else answer
    certificate = (body or {}).get("certificate") or {}
    selected = (certificate.get("incumbent") or {}).get("selected")
    if _status_kind(certificate.get("status")) not in _CERTIFIED:
        return None
    if not isinstance(selected, list):
        return None
    return [bool(s) for s in selected], str(body.get("certificate_digest") or "")


def greedy_choice(
    items: Sequence[Slice],
    weights: Sequence[int],
    capacity: int,
    floor_per_token: float,
) -> list[bool]:
    """The deterministic fallback: best net value per token first, one
    resolution per unit, within capacity."""
    order = sorted(
        range(len(items)),
        key=lambda i: (
            -_net_value(items[i], weights[i], floor_per_token) / max(1, weights[i]),
            i,
        ),
    )
    chosen = [False] * len(items)
    used, taken = 0, set()
    for i in order:
        fits = used + weights[i] <= capacity
        pays = _net_value(items[i], weights[i], floor_per_token) > 0
        if fits and pays and items[i].group not in taken:
            chosen[i], used = True, used + weights[i]
            taken.add(items[i].group)
    return chosen


def _solved(
    solve: Solver | None, model: Mapping[str, Any]
) -> tuple[tuple[list[bool], str] | None, str]:
    """EG's certified choice for ``model`` and why there is none."""
    if solve is None:
        return None, "no_solver"
    try:
        certified = _certified_choice(solve({"model": model, "config": None}))
    except Exception as exc:
        logger.warning("context knapsack solve failed; greedy fit: %s", exc)
        return None, "solve_failed"
    return certified, "certified" if certified else "not_certified"


def select_slices(
    items: Sequence[Slice],
    counter: TokenCounter,
    capacity: int,
    *,
    solve: Solver | None = None,
    floor_per_token: float = 0.0,
) -> SliceSelection:
    """Choose the slices: EG's certified solve, or the greedy fallback."""
    weights = counter.count([item.text for item in items])
    model = knapsack_model(items, weights, capacity, floor_per_token)
    certified, reason = _solved(solve if items else None, model)
    mask = (
        certified[0]
        if certified
        else greedy_choice(items, weights, capacity, floor_per_token)
    )
    kept = [
        (item, w) for item, w, keep in zip(items, weights, mask, strict=True) if keep
    ]
    return SliceSelection(
        chosen=tuple(item for item, _ in kept),
        tokens=sum(w for _, w in kept),
        capacity=capacity,
        certified=certified is not None,
        reason=reason,
        certificate_digest=certified[1] if certified else None,
    )


@dataclass(frozen=True, slots=True)
class ContextSizer:
    """A model's sizing policy: its tokenizer, capacity and marginal floor."""

    counter: TokenCounter
    capacity: Capacity
    solve: Solver | None = None
    floor_per_token: float = 0.0
    #: What a unit's summary is worth relative to its full text.
    summary_value_share: float = 0.6

    def identity(self) -> str:
        """What a bundle compiled under this sizing is keyed by."""
        body = json.dumps(
            {
                "counter": self.counter.identity,
                "capacity": [
                    self.capacity.window,
                    self.capacity.reserved_output,
                    self.capacity.budget_usd,
                    self.capacity.usd_per_token,
                    self.capacity.latency_s,
                    self.capacity.tokens_per_second,
                ],
                "floor": self.floor_per_token,
                "solver": self.solve is not None,
                "summary_share": self.summary_value_share,
            },
            sort_keys=True,
        )
        return "sizer:" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


_SIZER: contextvars.ContextVar[ContextSizer | None] = contextvars.ContextVar(
    "context_sizer", default=None
)


@contextmanager
def sizing_scope(sizer: ContextSizer | None) -> Iterator[None]:
    """Compile under ``sizer`` (``None``: the greedy fit) for this call only."""
    token = _SIZER.set(sizer)
    try:
        yield
    finally:
        _SIZER.reset(token)


def current_sizer() -> ContextSizer | None:
    return _SIZER.get()


def _definition_of(model_name: str) -> Any | None:
    """The registry's definition of ``model_name`` (by id, model id or name)."""
    from agent_utilities.models.model_registry import load_active_registry

    for definition in load_active_registry().models:
        if model_name in (definition.id, definition.model_id, definition.name):
            return definition
    return None


def engine_solver() -> Solver | None:
    """EG ``Solve`` over the installed decision runner's transport, or ``None``
    when no runner (or a transport without ``solve``) is installed."""
    from agent_utilities import decide

    runner = decide.current_runner()
    transport = None if runner is None else runner.transport
    if transport is None or not hasattr(transport, "solve"):
        return None
    return lambda request: transport.run(transport.solve(request))


def sizer_for_model(model_name: str) -> ContextSizer | None:
    """The sizing policy of the invoked model: its registry definition's
    capacity, its exact tokenizer, EG's certified solve. ``None`` (the greedy
    fit) for an unregistered model, one with no window, or no exact tokenizer."""
    definition = _definition_of(model_name) if model_name else None
    if definition is None:
        return None
    capacity = capacity_of(definition)
    counter = tiktoken_counter(str(definition.model_id))
    if counter is None or capacity.tokens() <= 0:
        return None
    return ContextSizer(counter, capacity, solve=engine_solver())


def sizing_key(model_version: str) -> str:
    """``model_version`` extended with the installed sizing identity."""
    sizer = current_sizer()
    return model_version if sizer is None else f"{model_version}|{sizer.identity()}"


#: A record's body fields; its summary resolution is the record without them.
BODY_FIELDS = ("content", "text")


def summary_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """``record`` at summary resolution: its node without the body fields."""
    node = {k: v for k, v in dict(record["node"]).items() if k not in BODY_FIELDS}
    return {**record, "node": node, "resolution": "summary"}


def _slices(
    records: Sequence[Mapping[str, Any]],
    text_of: Callable[[Any], str],
    share: float,
) -> list[Slice]:
    items = []
    for record in records:
        group, value = str(record["nid"]), float(record.get("composite") or 0.0)
        full = text_of(record)
        items.append(Slice(group, "full", full, value))
        summary = text_of(summary_view(record))
        if summary and summary != full:
            items.append(Slice(group, "summary", summary, value * share))
    return items


def fit_to_budget(
    records: Sequence[Mapping[str, Any]],
    token_budget: int,
    *,
    text_of: Callable[[Any], str],
) -> Any:
    """Fit ranked compiler records to the budget: the installed sizer's
    certified knapsack, else the greedy :class:`RetrievalBudgetManager` fit."""
    from .budget import BudgetResult, RetrievalBudgetManager

    sizer = current_sizer()
    if sizer is None:
        return RetrievalBudgetManager(token_budget).fit(list(records), text_of=text_of)
    capacity = sizer.capacity.tokens(token_budget)
    selection = select_slices(
        _slices(records, text_of, sizer.summary_value_share),
        sizer.counter,
        capacity,
        solve=sizer.solve,
        floor_per_token=sizer.floor_per_token,
    )
    picked = {s.group: s.resolution for s in selection.chosen}
    kept = [
        summary_view(r) if picked[str(r["nid"])] == "summary" else r
        for r in records
        if str(r["nid"]) in picked
    ]
    return BudgetResult(
        kept=kept,
        dropped=len(records) - len(kept),
        tokens_used=selection.tokens,
        token_budget=capacity,
    )


__all__ = [
    "BODY_FIELDS",
    "VALUE_SCALE",
    "Capacity",
    "ContextSizer",
    "Slice",
    "SliceSelection",
    "Solver",
    "TiktokenCounter",
    "TokenCounter",
    "capacity_of",
    "current_sizer",
    "fit_to_budget",
    "engine_solver",
    "greedy_choice",
    "knapsack_model",
    "select_slices",
    "sizer_for_model",
    "sizing_key",
    "sizing_scope",
    "summary_view",
    "tiktoken_counter",
]
