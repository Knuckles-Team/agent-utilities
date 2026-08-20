"""Fixtures for CONCEPT:AU-OS.scaling.single-replica-controller-authority.

The tests use a small durable-CAS double rather than an actuator emulator.  A
native autoscaler may create an accepted intent, but only the reconciler is
allowed to apply it.  HPA/KEDA fixtures prove the other valid mode: observe and
report, never write replicas.  Execution fixtures additionally prove the
proposed → intent_accepted → simulated/executed → observed → verified state
machine and that only a real executed ledger row consumes cooldown.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from agent_utilities.orchestration.action_policy import ActionPolicy, ActionRequest
from agent_utilities.orchestration.fleet_actuation import DryRunActuator
from agent_utilities.orchestration.fleet_autoscaler import FleetAutoscaler
from agent_utilities.orchestration.fleet_health import FleetHealthSnapshot
from agent_utilities.orchestration.fleet_reconciler import (
    FleetReconciler,
    load_desired_state,
    parse_scaling_spec,
)

from .fleet_autonomy_fakes import (
    FakeEngine,
    FakeObserver,
    FakeSignalProvider,
    healthy_fleet_evidence,
    obs,
    write_policy,
)


def _healthy_snapshot() -> FleetHealthSnapshot:
    return FleetHealthSnapshot(
        evidence=healthy_fleet_evidence(),
        sessions=None,
        goals=None,
        domains=None,
        dispatch_workers=None,
    )


PERMISSIVE = (
    "defaults: {tier: auto, rate_limit: {max: 100, window_s: 60},"
    " blast_radius: {max_targets: 100, window_s: 60}}\n"
    "rules:\n  - {kind: '*', target: '*', tier: auto}\n"
)


class DurableIntentCAS:
    """Minimal atomic revision authority used only by these source fixtures."""

    def __init__(self) -> None:
        self.intents: dict[str, dict[str, Any]] = {}
        self.cas_calls: list[dict[str, Any]] = []

    def latest(self, service: str) -> tuple[bool, dict[str, Any] | None]:
        value = self.intents.get(service)
        return True, deepcopy(value) if value is not None else None

    def cas(self, request: dict[str, Any]) -> dict[str, Any]:
        self.cas_calls.append(deepcopy(request))
        service = str(request.get("service") or "")
        current = self.intents.get(service)
        current_revision = int(current.get("revision", 0)) if current else 0
        expected = int(request.get("expected_revision", -1))
        if (
            request.get("operation") == "put"
            and current is not None
            and current.get("intent_id") == request.get("intent_id")
        ):
            # A true replay resubmits the EXACT same request a caller already
            # won with (every field, not just the target-state subset) — two
            # independent leaders that blindly read the same stale revision
            # and coincidentally compute the identical desired intent still
            # differ on caller-scoped fields (decision_id, created_unix), so
            # comparing the full payload — not just the immutable subset —
            # is what actually distinguishes "same caller retrying" from "a
            # second leader racing on a stale revision" (the latter must be
            # a CAS rejection, per docs/architecture/fleet-scale-authority.md).
            resubmitted = {k: v for k, v in request.items() if k != "operation"}
            if current == resubmitted:
                return {
                    "accepted": True,
                    "replayed": True,
                    "revision": current_revision,
                }
            return {"accepted": False, "reason": "intent identity conflict"}
        if expected != current_revision:
            return {"accepted": False, "reason": "revision conflict"}
        if request.get("operation") == "put":
            value = dict(request)
            value.pop("operation", None)
            value["revision"] = current_revision + 1
            self.intents[service] = value
            return {"accepted": True, "revision": value["revision"]}
        if request.get("operation") == "transition":
            if current is None or current.get("intent_id") != request.get("intent_id"):
                return {"accepted": False, "reason": "intent identity conflict"}
            if expected != current_revision:
                return {"accepted": False, "reason": "revision conflict"}
            for key, value in request.items():
                if key not in {
                    "operation",
                    "service",
                    "intent_id",
                    "expected_revision",
                }:
                    current[key] = value
            return {"accepted": True, "status": current["status"]}
        return {"accepted": False, "reason": "unknown operation"}


class RecordingActuator:
    name = "recording"

    def __init__(self) -> None:
        self.applied: list[Any] = []

    def apply(self, request: Any) -> dict[str, Any]:
        self.applied.append(request)
        return {"ok": True, "dry_run": False, "detail": "recorded"}


class FailingActuator:
    name = "failing"

    def apply(self, request: Any) -> dict[str, Any]:
        return {"ok": False, "dry_run": False, "detail": "failed"}


class ScaleLedgerEngine(FakeEngine):
    """Fake engine that preserves the execution fields cooldown requires."""

    def __init__(self) -> None:
        super().__init__()
        self.execution_queries: list[str] = []

    def query_cypher(self, query: str, params: dict | None = None):
        if "ActionExecution" in query:
            self.execution_queries.append(query)
            params = params or {}
            rows = [
                {
                    "ok": node.get("ok"),
                    "dry_run": node.get("dry_run"),
                    "state": node.get("state"),
                    "ts": node.get("executed_unix"),
                }
                for node in self.by_type("ActionExecution")
                if node.get("kind") == params.get("kind")
                and node.get("target") == params.get("target")
            ]
            if "ORDER BY x.executed_unix DESC" in query:
                rows.sort(key=lambda row: row["ts"] or 0, reverse=True)
            return rows[:200]
        return super().query_cypher(query, params)


def _registry(*, controller_mode: str = "native", replicas: int = 1) -> str:
    return f"""
version: 1
services:
  - name: vector-mcp
    replicas: {replicas}
    scaling:
      min: 1
      max: 5
      signal: queue_depth
      target: 100
      scale_up_step: 2
      scale_down_step: 1
      cooldown_s: 300
      controller_mode: {controller_mode}
"""


def _setup(
    tmp_path,
    monkeypatch,
    store,
    *,
    controller_mode: str = "native",
    replicas: int = 1,
    engine: FakeEngine | None = None,
):
    registry = tmp_path / "registry.yml"
    registry.write_text(
        _registry(controller_mode=controller_mode, replicas=replicas),
        encoding="utf-8",
    )
    policy = ActionPolicy(
        engine=engine or FakeEngine(),
        policy_path=write_policy(tmp_path, PERMISSIVE),
    )
    engine = policy.engine
    from .test_fleet_action_outbox import DurableActionOutbox

    engine.action_outbox_store = DurableActionOutbox(engine)
    observer = FakeObserver({"vector-mcp": obs("vector-mcp", "up", replicas=1)})
    signals = FakeSignalProvider(default=450.0)
    import agent_utilities.orchestration.fleet_autoscaler as autoscaler_module
    import agent_utilities.orchestration.fleet_reconciler as reconciler_module

    monkeypatch.setattr(
        autoscaler_module,
        "load_desired_state",
        lambda *args, **kwargs: load_desired_state(registry_path=str(registry)),
    )
    monkeypatch.setattr(
        reconciler_module,
        "load_desired_state",
        lambda *args, **kwargs: load_desired_state(registry_path=str(registry)),
    )
    monkeypatch.setattr(autoscaler_module, "collect_fleet_health", _healthy_snapshot)
    monkeypatch.setattr(reconciler_module, "collect_fleet_health", _healthy_snapshot)
    return engine, observer, signals, policy


def test_native_autoscaler_persists_intent_and_never_actuates(tmp_path, monkeypatch):
    store = DurableIntentCAS()
    engine, observer, signals, policy = _setup(tmp_path, monkeypatch, store)
    autoscaler_actuator = DryRunActuator()
    scaler = FleetAutoscaler(
        engine,
        observer=observer,
        actuator=autoscaler_actuator,
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    )

    report = scaler.evaluate()

    assert report["intents_accepted"] == 1
    assert autoscaler_actuator.applied == []
    assert store.intents["vector-mcp"]["status"] == "intent_accepted"
    assert store.intents["vector-mcp"]["revision"] == 1
    replay = store.cas(store.cas_calls[0])
    assert replay["accepted"] is True
    assert replay["replayed"] is True


def test_reconciler_is_only_native_actuator_and_state_progression_is_not_replayed(
    tmp_path, monkeypatch
):
    store = DurableIntentCAS()
    engine, observer, signals, policy = _setup(tmp_path, monkeypatch, store)
    scaler = FleetAutoscaler(
        engine,
        observer=observer,
        actuator=DryRunActuator(),
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    )
    scaler.evaluate()
    reconciler_actuator = RecordingActuator()
    reconciler = FleetReconciler(
        engine,
        observer=observer,
        actuator=reconciler_actuator,
        policy=policy,
        intent_store=store,
    )

    first = reconciler.reconcile()
    second = reconciler.reconcile()
    observer.observations["vector-mcp"] = obs("vector-mcp", "up", replicas=3)
    third = reconciler.reconcile()

    assert first["actions"][0]["decision"] == "accepted_intent"
    assert len(reconciler_actuator.applied) == 1
    assert first["actions"][0]["state"] == "executed"
    assert second["actions"] == []
    # The third reconcile is the FIRST stable observation at the desired
    # replica count: executed -> observed.
    assert store.intents["vector-mcp"]["status"] == "observed"
    assert third["actions"] == []

    fourth = reconciler.reconcile()

    # The fourth reconcile is the SECOND stable observation: observed ->
    # verified. Reading state must happen before this call runs, since
    # ``reconcile()`` advances the durable intent as a side effect.
    assert fourth["actions"] == []
    assert store.intents["vector-mcp"]["status"] == "verified"
    assert len(reconciler_actuator.applied) == 1


def test_reconciler_first_then_autoscaler_still_has_one_authority(
    tmp_path, monkeypatch
):
    store = DurableIntentCAS()
    engine, observer, signals, policy = _setup(tmp_path, monkeypatch, store)
    reconciler_actuator = RecordingActuator()
    reconciler = FleetReconciler(
        engine,
        observer=observer,
        actuator=reconciler_actuator,
        policy=policy,
        intent_store=store,
    )

    assert reconciler.reconcile()["actions"] == []
    scaler = FleetAutoscaler(
        engine,
        observer=observer,
        actuator=DryRunActuator(),
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    )
    scaler.evaluate()
    reconciler.reconcile()

    assert len(reconciler_actuator.applied) == 1
    assert store.intents["vector-mcp"]["status"] == "executed"
    observer.observations["vector-mcp"] = obs("vector-mcp", "up", replicas=3)
    restarted = FleetReconciler(
        engine,
        observer=observer,
        actuator=reconciler_actuator,
        policy=policy,
        intent_store=store,
    )
    restarted.reconcile()
    assert len(reconciler_actuator.applied) == 1
    assert store.intents["vector-mcp"]["status"] == "observed"


class BlindReadIntentCAS(DurableIntentCAS):
    """Two leaders read revision zero while the shared CAS arbitrates writes."""

    def latest(self, service: str) -> tuple[bool, dict[str, Any] | None]:
        return True, None


def test_concurrent_leaders_cannot_both_commit_revision_one(tmp_path, monkeypatch):
    store = BlindReadIntentCAS()
    engine, observer, signals, policy = _setup(tmp_path, monkeypatch, store)
    first = FleetAutoscaler(
        engine,
        observer=observer,
        actuator=DryRunActuator(),
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    )
    second = FleetAutoscaler(
        engine,
        observer=observer,
        actuator=DryRunActuator(),
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    )

    first_report = first.evaluate()
    second_report = second.evaluate()

    assert first_report["intents_accepted"] == 1
    assert second_report["intents_accepted"] == 0
    assert "CAS rejected" in second_report["evaluations"][0]["reason"]
    assert store.intents["vector-mcp"]["revision"] == 1


def test_restart_reuses_pending_intent_and_cooldown_prevents_duplicate(
    tmp_path, monkeypatch
):
    store = DurableIntentCAS()
    engine, observer, signals, policy = _setup(tmp_path, monkeypatch, store)
    first = FleetAutoscaler(
        engine,
        observer=observer,
        actuator=DryRunActuator(),
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    )
    first.evaluate()
    calls = len(store.cas_calls)

    restarted = FleetAutoscaler(
        engine,
        observer=observer,
        actuator=DryRunActuator(),
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    )
    report = restarted.evaluate()

    assert report["actions"] == 0
    assert "awaiting execution or observation" in report["evaluations"][0]["reason"]
    assert len(store.cas_calls) == calls


def test_terminal_intent_without_real_execution_does_not_consume_cooldown(
    tmp_path, monkeypatch
):
    store = DurableIntentCAS()
    engine, observer, signals, policy = _setup(tmp_path, monkeypatch, store)
    first = FleetAutoscaler(
        engine,
        observer=observer,
        actuator=DryRunActuator(),
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    )
    first.evaluate()
    # Simulate a durable terminal outcome from another controller. A policy
    # decision alone is not a real execution and must not consume cooldown.
    store.intents["vector-mcp"]["status"] = "rejected"

    restarted = FleetAutoscaler(
        engine,
        observer=observer,
        actuator=DryRunActuator(),
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    )
    report = restarted.evaluate()

    assert report["actions"] == 1
    assert report["evaluations"][0]["outcome"] == "intent_accepted"


def test_reconciler_rejects_stale_accepted_intent_without_actuation(
    tmp_path, monkeypatch
):
    store = DurableIntentCAS()
    engine, observer, signals, policy = _setup(tmp_path, monkeypatch, store)
    FleetAutoscaler(
        engine,
        observer=observer,
        actuator=DryRunActuator(),
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    ).evaluate()
    actuator = RecordingActuator()
    reconciler = FleetReconciler(
        engine,
        observer=observer,
        actuator=actuator,
        policy=policy,
        intent_store=store,
    )
    proposal = reconciler.diff()[0]
    replacement = deepcopy(store.intents["vector-mcp"])
    replacement["intent_id"] = "scale-intent:replacement"
    replacement["revision"] = 2
    store.intents["vector-mcp"] = replacement

    entry = reconciler._converge_one(proposal)

    assert entry["decision"] == "stale_intent"
    assert "stale" in entry["reason"]
    assert actuator.applied == []


def test_cooldown_selects_latest_real_execution_after_two_hundred_rows(
    tmp_path, monkeypatch
):
    store = DurableIntentCAS()
    engine = ScaleLedgerEngine()
    engine, observer, signals, policy = _setup(
        tmp_path, monkeypatch, store, engine=engine
    )
    now = 10_000.0
    for index in range(205):
        engine.add_node(
            f"execution:{index}",
            "ActionExecution",
            properties={
                "kind": "scale_service",
                "target": "vector-mcp",
                "ok": True,
                "dry_run": False,
                "state": "executed",
                "executed_unix": now - 1_000 + index,
            },
        )
    # A simulation is newer but cannot become cooldown evidence.
    engine.add_node(
        "execution:simulation",
        "ActionExecution",
        properties={
            "kind": "scale_service",
            "target": "vector-mcp",
            "ok": True,
            "dry_run": True,
            "state": "simulated",
            "executed_unix": now + 1,
        },
    )
    scaler = FleetAutoscaler(
        engine,
        observer=observer,
        policy=policy,
        signal_provider=signals,
        intent_store=store,
        clock=lambda: now,
    )

    latest = scaler._last_scale_unix("vector-mcp")

    assert latest == now - 1_000 + 204
    assert "ORDER BY x.executed_unix DESC LIMIT 200" in engine.execution_queries[-1]


def test_clock_skew_clamps_small_future_and_fails_closed_on_large_future(
    tmp_path, monkeypatch
):
    store = DurableIntentCAS()
    engine = ScaleLedgerEngine()
    engine, observer, signals, policy = _setup(
        tmp_path, monkeypatch, store, engine=engine
    )
    engine.add_node(
        "execution:future-small",
        "ActionExecution",
        properties={
            "kind": "scale_service",
            "target": "vector-mcp",
            "ok": True,
            "dry_run": False,
            "state": "executed",
            "executed_unix": 1_005.0,
        },
    )
    scaler = FleetAutoscaler(
        engine,
        observer=observer,
        policy=policy,
        signal_provider=signals,
        intent_store=store,
        clock=lambda: 1_000.0,
    )
    assert scaler._last_scale_unix("vector-mcp") == 1_000.0

    engine.add_node(
        "execution:future-large",
        "ActionExecution",
        properties={
            "kind": "scale_service",
            "target": "vector-mcp",
            "ok": True,
            "dry_run": False,
            "state": "executed",
            "executed_unix": 2_000.0,
        },
    )
    assert scaler._last_scale_unix("vector-mcp") is None


def test_dry_run_is_simulated_without_watch_or_real_cooldown(tmp_path, monkeypatch):
    store = DurableIntentCAS()
    engine = ScaleLedgerEngine()
    engine, observer, signals, policy = _setup(
        tmp_path, monkeypatch, store, engine=engine
    )
    FleetAutoscaler(
        engine,
        observer=observer,
        actuator=DryRunActuator(),
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    ).evaluate()
    reconciler = FleetReconciler(
        engine,
        observer=observer,
        actuator=DryRunActuator(),
        policy=policy,
        intent_store=store,
    )

    report = reconciler.reconcile()
    action = report["actions"][0]
    probe = FleetAutoscaler(
        engine,
        observer=observer,
        policy=policy,
        signal_provider=signals,
        intent_store=store,
        clock=lambda: time.time(),
    )

    assert action["state"] == "simulated"
    assert action["execution"]["state"] == "simulated"
    assert "watch_job" not in action
    assert store.intents["vector-mcp"]["status"] == "simulated"
    assert probe._last_scale_unix("vector-mcp") == 0.0


def test_failed_execution_is_explicit_and_does_not_consume_cooldown(
    tmp_path, monkeypatch
):
    store = DurableIntentCAS()
    engine = ScaleLedgerEngine()
    engine, observer, signals, policy = _setup(
        tmp_path, monkeypatch, store, engine=engine
    )
    FleetAutoscaler(
        engine,
        observer=observer,
        actuator=DryRunActuator(),
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    ).evaluate()
    report = FleetReconciler(
        engine,
        observer=observer,
        actuator=FailingActuator(),
        policy=policy,
        intent_store=store,
    ).reconcile()
    probe = FleetAutoscaler(
        engine,
        observer=observer,
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    )

    assert report["actions"][0]["state"] == "failed"
    assert report["actions"][0]["execution"]["state"] == "failed"
    assert store.intents["vector-mcp"]["status"] == "failed"
    assert probe._last_scale_unix("vector-mcp") == 0.0


def test_observer_reconciles_lost_outcome_without_replaying_actuator(
    tmp_path, monkeypatch
):
    store = DurableIntentCAS()
    engine, observer, signals, policy = _setup(tmp_path, monkeypatch, store)
    outbox = engine.action_outbox_store
    outbox.fail_complete = True
    actuator = RecordingActuator()
    scaler = FleetAutoscaler(
        engine,
        observer=observer,
        actuator=DryRunActuator(),
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    )
    scaler.evaluate()
    reconciler = FleetReconciler(
        engine,
        observer=observer,
        actuator=actuator,
        policy=policy,
        intent_store=store,
    )

    first = reconciler.reconcile()
    assert first["actions"][0]["state"] == "recovery_pending"
    assert store.intents["vector-mcp"]["status"] == "recovery_pending"
    outbox.fail_complete = False
    observer.observations["vector-mcp"] = obs("vector-mcp", "up", replicas=3)

    recovered = reconciler.reconcile()
    verified = reconciler.reconcile()

    assert recovered["actions"] == []
    assert verified["actions"] == []
    assert store.intents["vector-mcp"]["status"] == "verified"
    assert len(actuator.applied) == 1


def test_direct_scale_bypass_is_rejected_for_native_and_external_authorities(
    tmp_path, monkeypatch
):
    for mode in ("native", "hpa"):
        mode_path = tmp_path / f"direct-{mode}"
        mode_path.mkdir()
        store = DurableIntentCAS()
        engine, observer, signals, policy = _setup(
            mode_path, monkeypatch, store, controller_mode=mode
        )
        actuator = RecordingActuator()
        reconciler = FleetReconciler(
            engine,
            observer=observer,
            actuator=actuator,
            policy=policy,
            intent_store=store,
        )

        result = reconciler._converge_one(
            ActionRequest(
                kind="scale_service",
                target="vector-mcp",
                params={"replicas": 3},
                source="fixture",
                reason="bypass attempt",
            )
        )

        assert result["decision"] == "rejected"
        assert result["state"] == "rejected"
        assert actuator.applied == []


def test_external_hpa_and_keda_modes_report_without_au_replica_writes(
    tmp_path, monkeypatch
):
    for mode in ("hpa", "keda"):
        mode_path = tmp_path / mode
        mode_path.mkdir()
        store = DurableIntentCAS()
        engine, observer, signals, policy = _setup(
            mode_path, monkeypatch, store, controller_mode=mode, replicas=2
        )
        autoscaler_actuator = DryRunActuator()
        scaler = FleetAutoscaler(
            engine,
            observer=observer,
            actuator=autoscaler_actuator,
            policy=policy,
            signal_provider=signals,
            intent_store=store,
        )
        report = scaler.evaluate()
        reconciler_actuator = RecordingActuator()
        reconciler = FleetReconciler(
            engine,
            observer=observer,
            actuator=reconciler_actuator,
            policy=policy,
            intent_store=store,
        )

        assert report["evaluations"][0]["outcome"] == "reported"
        assert report["actions"] == 0
        assert store.cas_calls == []
        assert reconciler.diff() == []
        assert reconciler_actuator.applied == []


def test_conflicting_controller_aliases_are_rejected():
    assert (
        parse_scaling_spec(
            {
                "min": 1,
                "max": 5,
                "signal": "queue_depth",
                "target": 100,
                "controller_mode": "native",
                "controller": "hpa",
            },
            "vector-mcp",
        )
        is None
    )


def test_operator_override_wins_over_pending_native_intent(tmp_path, monkeypatch):
    store = DurableIntentCAS()
    engine, observer, signals, policy = _setup(tmp_path, monkeypatch, store)
    scaler = FleetAutoscaler(
        engine,
        observer=observer,
        actuator=DryRunActuator(),
        policy=policy,
        signal_provider=signals,
        intent_store=store,
    )
    scaler.evaluate()
    override = tmp_path / "override.yml"
    override.write_text(
        "services:\n  - name: vector-mcp\n    replicas: 2\n", encoding="utf-8"
    )
    import agent_utilities.orchestration.fleet_autoscaler as autoscaler_module
    import agent_utilities.orchestration.fleet_reconciler as reconciler_module

    original = load_desired_state
    monkeypatch.setattr(
        autoscaler_module,
        "load_desired_state",
        lambda *args, **kwargs: original(
            registry_path=str(tmp_path / "registry.yml"), override_path=str(override)
        ),
    )
    monkeypatch.setattr(
        reconciler_module,
        "load_desired_state",
        lambda *args, **kwargs: original(
            registry_path=str(tmp_path / "registry.yml"), override_path=str(override)
        ),
    )
    report = scaler.evaluate()
    proposals = FleetReconciler(
        engine,
        observer=observer,
        actuator=RecordingActuator(),
        policy=policy,
        intent_store=store,
    ).diff()

    assert "operator replica override" in report["evaluations"][0]["reason"]
    assert len(proposals) == 1
    assert "scale_intent_id" not in proposals[0].params
    assert proposals[0].params["replicas"] == 2
