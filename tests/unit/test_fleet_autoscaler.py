"""Reactive replica autoscaler (CONCEPT:AU-OS.scaling.reactive-replica-autoscaling).

Covers: the registry ``scaling:`` block parse + validation, the
target-tracking math (up/down/clamp/step caps/scale-from-zero/per-replica
signals), the never-act-on-missing-data rule (no signal / unobserved / down),
cooldown + flap guard against the durable action ledger, the ActionPolicy
gate (queue under the shipped default, actuate + deploy-watch under a
permissive policy, watch_scale_down option), the per-tick action budget,
compact AutoscaleEvaluation recording, and the leader-only flag-gated tick
registration.

@pytest.mark.concept("AU-OS.scaling.reactive-replica-autoscaling")
"""

from __future__ import annotations

import time
import uuid
from dataclasses import replace

import pytest

from agent_utilities.orchestration.action_policy import ActionPolicy
from agent_utilities.orchestration.fleet_actuation import DryRunActuator
from agent_utilities.orchestration.fleet_autoscaler import (
    FleetAutoscaler,
    compute_desired_replicas,
)
from agent_utilities.orchestration.fleet_health import unavailable_fleet_health
from agent_utilities.orchestration.fleet_reconciler import (
    ScalingSpec,
    load_desired_state,
    parse_scaling_spec,
)
from agent_utilities.orchestration.scaling_signals import (
    MAX_SIGNAL_AGE_S,
    ScalingSignalSample,
    get_signal_definition,
)

from .fleet_autonomy_fakes import (
    FakeEngine,
    FakeObserver,
    FakeSignalProvider,
    healthy_fleet_evidence,
    obs,
    write_policy,
)
from .test_fleet_scale_authority import (
    DurableIntentCAS,
    RecordingActuator,
    ScaleLedgerEngine,
)

# NE-226: fleet_autoscaler.py no longer actuates directly (au-scale-authority-
# conflict, CONCEPT:AU-OS.scaling.single-replica-controller-authority) — it
# only proposes/accepts a durable ScaleIntent under compare-and-set; the
# fleet reconciler is the sole native replica actuator. ``DurableIntentCAS``
# (test_fleet_scale_authority.py) is the shared in-memory CAS double the
# other NE-179/NE-226 fixtures already use for this same seam.

pytestmark = pytest.mark.concept("AU-OS.scaling.reactive-replica-autoscaling")


REGISTRY = """
version: 1
services:
  - name: vector-mcp
    scaling:
      min: 1
      max: 5
      signal: queue_depth
      target: 100
      scale_up_step: 2
      scale_down_step: 1
      cooldown_s: 300
      scale_down_stabilization_samples: 1
  - name: caddy-mcp
"""

PERMISSIVE = (
    "defaults: {tier: auto, rate_limit: {max: 100, window_s: 60},"
    " blast_radius: {max_targets: 100, window_s: 60}}\n"
    "rules:\n  - {kind: '*', target: '*', tier: auto}\n"
)

PERMISSIVE_WATCH_DOWN = PERMISSIVE + "options: {watch_scale_down: true}\n"


@pytest.fixture
def engine():
    return FakeEngine()


def _autoscaler(
    engine,
    observations,
    tmp_path,
    monkeypatch,
    signals=None,
    policy_body=None,
    registry_body=REGISTRY,
    max_actions=5,
    intent_store=None,
):
    registry = tmp_path / "registry.yml"
    registry.write_text(registry_body, encoding="utf-8")
    policy_path = write_policy(tmp_path, policy_body) if policy_body else None
    scaler = FleetAutoscaler(
        engine,
        observer=FakeObserver(observations),
        actuator=DryRunActuator(),
        policy=ActionPolicy(engine=engine, policy_path=policy_path),
        signal_provider=signals or FakeSignalProvider(),
        max_actions=max_actions,
        health_provider=healthy_fleet_evidence,
        # NE-226: without an explicit durable CAS double, EngineScaleIntentStore
        # finds no engine.cas_scale_intent on FakeEngine and every proposal is
        # rejected outright ("native ScaleIntent CAS is unavailable") before
        # policy/direction/etc. are ever exercised.
        intent_store=intent_store if intent_store is not None else DurableIntentCAS(),
    )
    # Pin desired state to the test registry (not the repo's 52-service one).
    import agent_utilities.orchestration.fleet_autoscaler as fa

    original = load_desired_state
    monkeypatch.setattr(
        fa,
        "load_desired_state",
        lambda *a, **k: original(registry_path=str(registry)),
    )
    # Exposed so a test that also needs to drive a FleetReconciler against
    # the SAME desired state can reuse this path (see _reconciler_for below)
    # without re-deriving/re-patching it.
    scaler._test_registry_path = str(registry)
    return scaler


def _reconciler_for(scaler, monkeypatch, *, actuator=None):
    """Build a FleetReconciler sharing ``scaler``'s engine/observer/policy/
    intent store/registry, for tests that need to prove a proposed
    ``ScaleIntent`` is actually actuated — the reconciler's job now, never
    the autoscaler's (NE-226)."""
    import agent_utilities.orchestration.fleet_reconciler as fr
    from agent_utilities.orchestration.fleet_reconciler import FleetReconciler

    monkeypatch.setattr(
        fr,
        "load_desired_state",
        lambda *a, **k: load_desired_state(registry_path=scaler._test_registry_path),
    )
    from .test_fleet_action_outbox import DurableActionOutbox

    # A non-dry-run actuator now requires the durable action-outbox
    # pre-side-effect fence (fleet_actuation.execute_action); wire it here so
    # every reconciler-driving test gets it for free.
    scaler.engine.action_outbox_store = DurableActionOutbox(scaler.engine)
    return FleetReconciler(
        scaler.engine,
        observer=scaler.observer,
        actuator=actuator if actuator is not None else RecordingActuator(),
        policy=scaler.policy,
        intent_store=scaler.intent_store,
        health_provider=healthy_fleet_evidence,
    )


def _mark_real_execution(engine, store, service, *, ts=None):
    """Simulate what a FleetReconciler pass durably records once it has
    actually actuated ``service``'s latest accepted intent: a genuine
    ``ActionExecution`` ledger row (the ONLY cooldown evidence
    ``fleet_autoscaler._last_scale_unix`` reads) and the intent's terminal
    ``verified`` status (so the autoscaler's own intent-completion gate —
    "awaiting execution or observation" — no longer shadows the
    stabilization/cooldown logic this module owns).

    This is a deliberate shortcut: NE-226 moved actuation to the reconciler
    (proven end to end in test_fleet_scale_authority.py's
    ``test_reconciler_is_only_native_actuator_...``), so re-wiring a full
    FleetReconciler through every autoscaler-focused stabilization/cooldown
    test here would just be re-testing that same integration repeatedly. The
    intent store used is a plain compare-and-set double with no state-machine
    enforcement of its own, so jumping straight to ``verified`` (rather than
    replaying executed → observed → verified) is a faithful, minimal stand-in
    for "the reconciler already durably proved this real execution".
    """
    complete, intent = store.latest(service)
    assert complete and intent is not None, f"no intent recorded for {service!r}"
    ts = time.time() if ts is None else ts
    engine.add_node(
        f"action_execution:{uuid.uuid4().hex}",
        "ActionExecution",
        properties={
            "kind": "scale_service",
            "target": service,
            "ok": True,
            "dry_run": False,
            "state": "executed",
            "executed_unix": ts,
        },
    )
    result = store.cas(
        {
            "operation": "transition",
            "service": service,
            "intent_id": intent["intent_id"],
            "expected_revision": int(intent["revision"]),
            "status": "verified",
            "observed_replicas": intent["desired_replicas"],
            "updated_unix": ts,
        }
    )
    assert result.get("accepted") is True
    return ts


# ---------------------------------------------------------------------------
# Registry scaling-block parse + validation
# ---------------------------------------------------------------------------


def test_registry_scaling_block_parsed(tmp_path):
    registry = tmp_path / "registry.yml"
    registry.write_text(REGISTRY, encoding="utf-8")
    desired = load_desired_state(registry_path=str(registry))
    spec = desired["vector-mcp"].scaling
    assert spec == ScalingSpec(
        min_replicas=1,
        max_replicas=5,
        signal="queue_depth",
        target=100.0,
        scale_up_step=2,
        scale_down_step=1,
        cooldown_s=300.0,
        scale_down_stabilization_samples=1,
    )
    assert desired["caddy-mcp"].scaling is None  # no block ⇒ never autoscaled


def test_scaling_spec_defaults():
    spec = parse_scaling_spec({"max": 3, "signal": "cpu", "target": 75}, "svc")
    assert spec is not None
    assert (spec.min_replicas, spec.scale_up_step, spec.scale_down_step) == (1, 1, 1)
    assert spec.cooldown_s == 300.0
    assert spec.deadband == 0.05
    assert spec.scale_up_stabilization_samples == 1
    assert spec.scale_down_stabilization_samples == 3


@pytest.mark.parametrize(
    "raw",
    [
        {"min": 2, "max": 1, "signal": "cpu", "target": 50},  # max < min
        {"min": -1, "max": 3, "signal": "cpu", "target": 50},  # min < 0
        {"signal": "cpu", "target": 50},  # max required
        {"max": 3, "target": 50},  # signal required
        {"max": 3, "signal": "cpu"},  # target required
        {"max": 3, "signal": "cpu", "target": 0},  # target must be > 0
        {"max": 3, "signal": "cpu", "target": 50, "scale_up_step": 0},
        {"max": 3, "signal": "cpu", "target": 50, "cooldown_s": -5},
        {"max": 3, "signal": "cpu", "target": 50, "deadband": -0.1},
        {"max": 3, "signal": "cpu", "target": 50, "deadband": 1.1},
        {"max": 3, "signal": "cpu", "target": 50, "deadband": float("nan")},
        {
            "max": 3,
            "signal": "cpu",
            "target": 50,
            "scale_down_stabilization_samples": 0,
        },
        {
            "max": 3,
            "signal": "cpu",
            "target": 50,
            "scale_up_stabilization_samples": 61,
        },
        {"max": 3, "signal": "cpu", "target": float("inf")},
        {"max": 3, "signal": "cpu", "target": 10**10000},
        {"max": 3, "signal": "cpu", "target": 50, "min": True},
        {"max": "lots", "signal": "cpu", "target": 50},  # unparseable
        "not-a-mapping",
    ],
)
def test_scaling_spec_invalid_blocks_are_dropped(raw):
    assert parse_scaling_spec(raw, "svc") is None


def test_override_can_add_and_disable_scaling(tmp_path):
    registry = tmp_path / "registry.yml"
    registry.write_text(REGISTRY, encoding="utf-8")
    override = tmp_path / "state.yml"
    override.write_text(
        "services:\n"
        "  - name: caddy-mcp\n"
        "    scaling: {max: 2, signal: consumer_lag, target: 10}\n"
        "  - name: vector-mcp\n"
        "    scaling: null\n",
        encoding="utf-8",
    )
    desired = load_desired_state(
        registry_path=str(registry), override_path=str(override)
    )
    assert desired["caddy-mcp"].scaling is not None
    assert desired["caddy-mcp"].scaling.signal == "consumer_lag"
    assert desired["vector-mcp"].scaling is None  # explicit disable


# ---------------------------------------------------------------------------
# Target-tracking math
# ---------------------------------------------------------------------------

SPEC = ScalingSpec(
    min_replicas=1,
    max_replicas=5,
    signal="queue_depth",
    target=100.0,
    scale_up_step=2,
    scale_down_step=1,
    cooldown_s=300.0,
)


def test_aggregate_signal_scales_up_toward_target():
    # 450 queued across 3 replicas = 150/replica vs target 100 ⇒ ceil(4.5) = 5.
    assert compute_desired_replicas(3, 450.0, SPEC, aggregation="fleet_total") == 5


def test_scale_up_step_caps_one_evaluation():
    # Raw desired is 5 but up-step 2 caps 1 → 3; convergence takes more ticks.
    assert compute_desired_replicas(1, 450.0, SPEC, aggregation="fleet_total") == 3


def test_scale_down_is_step_capped_and_floored():
    assert (
        compute_desired_replicas(3, 50.0, SPEC, aggregation="fleet_total") == 2
    )  # raw 1, down-step 1
    assert compute_desired_replicas(2, 0.0, SPEC, aggregation="fleet_total") == 1
    assert (
        compute_desired_replicas(1, 0.0, SPEC, aggregation="fleet_total") == 1
    )  # min floor


def test_max_clamp():
    spec = ScalingSpec(1, 3, "queue_depth", 10.0, 10, 10, 0.0)
    assert compute_desired_replicas(1, 10_000.0, spec, aggregation="fleet_total") == 3


def test_scale_from_zero_uses_effective_current():
    spec = ScalingSpec(0, 5, "queue_depth", 100.0, 5, 5, 0.0)
    assert compute_desired_replicas(0, 450.0, spec, aggregation="fleet_total") == 5
    assert (
        compute_desired_replicas(0, 0.0, spec, aggregation="fleet_total") == 0
    )  # min 0 holds at zero


def test_per_replica_signal_is_not_renormalized():
    # cpu is a per-replica average: 90% vs 50% target on 2 ⇒ ceil(3.6) = 4.
    cpu = ScalingSpec(1, 5, "cpu", 50.0, 5, 5, 0.0)
    assert compute_desired_replicas(2, 90.0, cpu, aggregation="per_replica") == 4


def test_at_target_holds():
    assert compute_desired_replicas(3, 300.0, SPEC, aggregation="fleet_total") == 3


# ---------------------------------------------------------------------------
# Never act on missing data
# ---------------------------------------------------------------------------


def test_no_signal_data_takes_no_action(engine, tmp_path, monkeypatch):
    scaler = _autoscaler(
        engine,
        {"vector-mcp": obs("vector-mcp", "up", replicas=1)},
        tmp_path,
        monkeypatch,
        signals=FakeSignalProvider(default=None),
        policy_body=PERMISSIVE,
    )
    report = scaler.evaluate()
    assert report["actions"] == 0
    assert report["evaluations"][0]["outcome"] == "skipped"
    assert "no data" in report["evaluations"][0]["reason"]
    assert scaler.actuator.applied == []
    assert engine.by_type("ActionDecision") == []  # nothing even proposed


def test_unavailable_supervisory_evidence_skips_before_load_or_signal(
    engine, tmp_path, monkeypatch
):
    signals = FakeSignalProvider(default=450.0)
    scaler = _autoscaler(
        engine,
        {"vector-mcp": obs("vector-mcp", "up", replicas=1)},
        tmp_path,
        monkeypatch,
        signals=signals,
        policy_body=PERMISSIVE,
    )
    scaler.health_provider = lambda: unavailable_fleet_health("test.health")

    report = scaler.evaluate()

    assert report["actions"] == 0
    assert report["evaluated"] == 0
    assert report["health"]["status"] == "unavailable"
    assert signals.calls == []


class _ScriptedSignalProvider:
    name = "scripted"
    trusted_in_process = True

    def __init__(self, sample):
        self.sample = sample

    def signal_definition(self, signal, service=None):
        definition = get_signal_definition(signal)
        if (
            definition is not None
            and service is not None
            and not definition.binds_service(service)
        ):
            return replace(definition, service_binding=service)
        return definition

    def signal_value(self, service, signal):
        return self.sample

    def signal_values(self, requests):
        return {request: self.signal_value(*request) for request in requests}


def _scripted_scaler(engine, tmp_path, monkeypatch, sample, *, replicas=3):
    return _autoscaler(
        engine,
        {"vector-mcp": obs("vector-mcp", "up", replicas=replicas)},
        tmp_path,
        monkeypatch,
        signals=_ScriptedSignalProvider(sample),
        policy_body=PERMISSIVE,
        registry_body=(
            "services:\n"
            "  - name: vector-mcp\n"
            "    scaling: {max: 5, signal: queue_depth, target: 100, cooldown_s: 0}\n"
        ),
    )


class _SequenceSignalProvider:
    name = "sequence"

    def __init__(self, values):
        self.values = iter(values)
        self.bulk_calls = 0

    def signal_definition(self, signal, service=None):
        definition = get_signal_definition(signal)
        if (
            definition is not None
            and service is not None
            and not definition.binds_service(service)
        ):
            return replace(definition, service_binding=service)
        return definition

    def signal_values(self, requests):
        self.bulk_calls += 1
        value = next(self.values)
        if value is None:
            return {request: None for request in requests}
        now = time.time() + self.bulk_calls
        return {
            request: ScalingSignalSample(
                value=value,
                source=self.name,
                service=request[0],
                signal=request[1],
                aggregation="fleet_total",
                observed_at=now,
                unit="items",
                scope="fleet",
            )
            for request in requests
        }


def _stabilized_scaler(
    engine, tmp_path, monkeypatch, values, *, down_samples=3, replicas=3
):
    return _autoscaler(
        engine,
        {"vector-mcp": obs("vector-mcp", "up", replicas=replicas)},
        tmp_path,
        monkeypatch,
        signals=_SequenceSignalProvider(values),
        policy_body=PERMISSIVE,
        registry_body=(
            "services:\n"
            "  - name: vector-mcp\n"
            f"    scaling: {{max: 5, signal: queue_depth, target: 100, cooldown_s: 0, deadband: 0, scale_down_stabilization_samples: {down_samples}}}\n"
        ),
    )


def test_deadband_suppresses_small_target_deviation(engine, tmp_path, monkeypatch):
    scaler = _autoscaler(
        engine,
        {"vector-mcp": obs("vector-mcp", "up", replicas=1)},
        tmp_path,
        monkeypatch,
        signals=FakeSignalProvider(default=105.0),
        policy_body=PERMISSIVE,
        registry_body=(
            "services:\n"
            "  - name: vector-mcp\n"
            "    scaling: {max: 5, signal: queue_depth, target: 100, cooldown_s: 0, deadband: 0.1}\n"
        ),
    )
    report = scaler.evaluate()
    assert report["actions"] == 0
    assert "deadband" in report["evaluations"][0]["reason"]


def test_deadband_does_not_prevent_scale_up_from_zero(engine, tmp_path, monkeypatch):
    scaler = _autoscaler(
        engine,
        {"vector-mcp": obs("vector-mcp", "up", replicas=0)},
        tmp_path,
        monkeypatch,
        signals=FakeSignalProvider(default=100.0),
        policy_body=PERMISSIVE,
        registry_body=(
            "services:\n"
            "  - name: vector-mcp\n"
            "    scaling:\n"
            "      min: 0\n"
            "      max: 5\n"
            "      signal: queue_depth\n"
            "      target: 100\n"
            "      cooldown_s: 0\n"
            "      deadband: 0.1\n"
        ),
    )

    report = scaler.evaluate()

    # NE-226: the autoscaler proposes/accepts a durable ScaleIntent; the
    # fleet reconciler alone actuates it (never "scaled" from here).
    assert report["intents_accepted"] == 1
    assert report["evaluations"][0]["outcome"] == "intent_accepted"
    assert report["evaluations"][0]["desired"] == 1
    assert scaler.intent_store.intents["vector-mcp"]["desired_replicas"] == 1


def test_scale_down_requires_consecutive_samples(engine, tmp_path, monkeypatch):
    scaler = _stabilized_scaler(engine, tmp_path, monkeypatch, [50.0, 50.0, 50.0])
    first = scaler.evaluate()
    assert first["actions"] == 0
    assert "stabilizing down (1/3" in first["evaluations"][0]["reason"]
    assert "stabilizing down (2/3" in scaler.evaluate()["evaluations"][0]["reason"]
    third = scaler.evaluate()
    # NE-226: the third consecutive sample proposes/accepts a durable
    # ScaleIntent; only the fleet reconciler actuates it.
    assert third["intents_accepted"] == 1
    assert third["evaluations"][0]["outcome"] == "intent_accepted"


def test_successful_scale_down_resets_stabilization_for_next_action(
    tmp_path, monkeypatch
):
    # ScaleLedgerEngine, not the bare FakeEngine fixture: _mark_real_execution
    # needs a real dry_run/state-shaped ActionExecution read for the SIXTH
    # call's cooldown check to see this as a genuine (not "unknown") clear.
    engine = ScaleLedgerEngine()
    scaler = _stabilized_scaler(
        engine,
        tmp_path,
        monkeypatch,
        [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
    )
    assert scaler.evaluate()["actions"] == 0
    assert scaler.evaluate()["actions"] == 0
    third = scaler.evaluate()
    assert third["intents_accepted"] == 1
    # The accepted intent now belongs to the reconciler. Simulate it having
    # been durably actuated and verified (NE-226 moved actuation off this
    # module) so the autoscaler's own intent-completion gate doesn't shadow
    # the stabilization streak this test is actually about.
    _mark_real_execution(engine, scaler.intent_store, "vector-mcp")
    # The successful action consumed the prior three-sample streak. A later
    # post-cooldown action must stabilize from 1/3 again.
    assert "stabilizing down (1/3" in scaler.evaluate()["evaluations"][0]["reason"]
    assert "stabilizing down (2/3" in scaler.evaluate()["evaluations"][0]["reason"]
    sixth = scaler.evaluate()
    assert sixth["intents_accepted"] == 1
    assert scaler.intent_store.intents["vector-mcp"]["revision"] == 2


def test_missing_sample_resets_scale_down_stabilization(engine, tmp_path, monkeypatch):
    scaler = _stabilized_scaler(
        engine, tmp_path, monkeypatch, [50.0, None, 50.0, 50.0, 50.0]
    )
    assert scaler.evaluate()["actions"] == 0
    assert "no data" in scaler.evaluate()["evaluations"][0]["reason"]
    assert "stabilizing down (1/3" in scaler.evaluate()["evaluations"][0]["reason"]
    assert "stabilizing down (2/3" in scaler.evaluate()["evaluations"][0]["reason"]
    final = scaler.evaluate()
    assert final["intents_accepted"] == 1
    assert final["evaluations"][0]["outcome"] == "intent_accepted"


def test_direction_change_resets_the_opposite_streak(tmp_path, monkeypatch):
    engine = ScaleLedgerEngine()  # see ScaleLedgerEngine note above
    scaler = _stabilized_scaler(
        engine, tmp_path, monkeypatch, [50.0, 450.0, 50.0, 50.0, 50.0]
    )
    assert "stabilizing down (1/3" in scaler.evaluate()["evaluations"][0]["reason"]
    up = scaler.evaluate()
    assert up["intents_accepted"] == 1  # scale-up is one valid sample
    # NE-226: the accepted intent belongs to the reconciler now; simulate it
    # having been actuated + verified so the next tick's direction-change
    # (down again) isn't shadowed by the "awaiting execution" intent gate.
    _mark_real_execution(engine, scaler.intent_store, "vector-mcp")
    assert "stabilizing down (1/3" in scaler.evaluate()["evaluations"][0]["reason"]
    assert "stabilizing down (2/3" in scaler.evaluate()["evaluations"][0]["reason"]
    down = scaler.evaluate()
    assert down["intents_accepted"] == 1
    assert scaler.intent_store.intents["vector-mcp"]["revision"] == 2


def test_stale_signal_is_no_data_and_cannot_scale_down(engine, tmp_path, monkeypatch):
    sample = ScalingSignalSample(
        value=0.0,
        source="scripted",
        service="vector-mcp",
        signal="queue_depth",
        aggregation="fleet_total",
        observed_at=time.time() - MAX_SIGNAL_AGE_S - 1,
        unit="items",
        scope="fleet",
    )
    scaler = _scripted_scaler(engine, tmp_path, monkeypatch, sample)
    report = scaler.evaluate()
    assert report["actions"] == 0
    assert report["evaluations"][0]["outcome"] == "skipped"
    assert scaler.actuator.applied == []


def test_replayed_signal_is_rejected_after_first_consumption(
    engine, tmp_path, monkeypatch
):
    sample = ScalingSignalSample(
        value=450.0,
        source="scripted",
        service="vector-mcp",
        signal="queue_depth",
        aggregation="fleet_total",
        observed_at=time.time(),
        unit="items",
        scope="fleet",
    )
    scaler = _scripted_scaler(engine, tmp_path, monkeypatch, sample, replicas=1)
    first = scaler.evaluate()
    assert first["intents_accepted"] == 1
    second = scaler.evaluate()
    assert second["actions"] == 0
    assert "no data" in second["evaluations"][0]["reason"]
    # Only ONE real intent action was ever proposed — the replayed sample's
    # "no data" verdict is reached before the intent store is even consulted
    # again (signal-freshness validation happens earlier than the
    # ScaleIntent gate), so a second CAS call never happens either.
    assert len(scaler.intent_store.cas_calls) == 1


@pytest.mark.parametrize(
    "service, aggregation",
    [("other-service", "fleet_total"), ("vector-mcp", "per_replica")],
)
def test_cross_service_or_wrong_aggregation_sample_is_rejected(
    engine, tmp_path, monkeypatch, service, aggregation
):
    sample = ScalingSignalSample(
        value=0.0,
        source="scripted",
        service=service,
        signal="queue_depth",
        aggregation=aggregation,
        observed_at=time.time(),
        unit="items",
        scope="fleet",
    )
    scaler = _scripted_scaler(engine, tmp_path, monkeypatch, sample)
    report = scaler.evaluate()
    assert report["actions"] == 0
    assert scaler.actuator.applied == []


def test_wrong_unit_or_scope_sample_is_rejected_before_scale_down(
    engine, tmp_path, monkeypatch
):
    sample = ScalingSignalSample(
        value=0.0,
        source="scripted",
        service="vector-mcp",
        signal="queue_depth",
        aggregation="fleet_total",
        observed_at=time.time(),
        unit="messages",
        scope="other-fleet",
    )
    scaler = _scripted_scaler(engine, tmp_path, monkeypatch, sample)
    report = scaler.evaluate()
    assert report["actions"] == 0
    assert "no data" in report["evaluations"][0]["reason"]


def test_malformed_provider_result_never_scales_down(engine, tmp_path, monkeypatch):
    scaler = _scripted_scaler(engine, tmp_path, monkeypatch, object(), replicas=3)
    report = scaler.evaluate()
    assert report["actions"] == 0
    assert report["evaluations"][0]["outcome"] == "skipped"
    assert scaler.actuator.applied == []


def test_unobserved_service_is_skipped(engine, tmp_path, monkeypatch):
    scaler = _autoscaler(
        engine,
        {},
        tmp_path,
        monkeypatch,
        signals=FakeSignalProvider(default=900.0),
        policy_body=PERMISSIVE,
    )
    report = scaler.evaluate()
    assert report["actions"] == 0
    assert "unobserved" in report["evaluations"][0]["reason"]


def test_down_service_is_not_scaled(engine, tmp_path, monkeypatch):
    scaler = _autoscaler(
        engine,
        {"vector-mcp": obs("vector-mcp", "down", replicas=1)},
        tmp_path,
        monkeypatch,
        signals=FakeSignalProvider(default=900.0),
        policy_body=PERMISSIVE,
    )
    report = scaler.evaluate()
    assert report["actions"] == 0
    assert "down" in report["evaluations"][0]["reason"]


def test_service_without_scaling_block_is_never_evaluated(
    engine, tmp_path, monkeypatch
):
    signals = FakeSignalProvider(default=900.0)
    scaler = _autoscaler(
        engine,
        {"caddy-mcp": obs("caddy-mcp", "up", replicas=1)},
        tmp_path,
        monkeypatch,
        signals=signals,
        policy_body=PERMISSIVE,
    )
    report = scaler.evaluate()
    assert all(e["service"] != "caddy-mcp" for e in report["evaluations"])
    assert all(svc != "caddy-mcp" for svc, _ in signals.calls)


# ---------------------------------------------------------------------------
# Policy gate
# ---------------------------------------------------------------------------


def test_default_policy_queues_scale_for_approval(engine, tmp_path, monkeypatch):
    scaler = _autoscaler(
        engine,
        {"vector-mcp": obs("vector-mcp", "up", replicas=1)},
        tmp_path,
        monkeypatch,
        signals=FakeSignalProvider(default=450.0),
    )
    report = scaler.evaluate()
    assert report["actions"] == 1
    # NE-226: under the default (queue_approval) policy the autoscaler
    # persists a *proposed* (not yet accepted) ScaleIntent — it never
    # actuates from here regardless of policy.
    assert report["intents_accepted"] == 0
    assert report["intents_proposed"] == 1
    assert report["evaluations"][0]["outcome"] == "intent_proposed"
    assert "queue_approval" in report["evaluations"][0]["reason"]
    assert scaler.actuator.applied == []  # the actuator is never touched here
    assert scaler.intent_store.intents["vector-mcp"]["status"] == "proposed"
    approvals = engine.by_type("ActionApproval")
    assert len(approvals) == 1 and approvals[0]["kind"] == "scale_service"


def test_permissive_policy_scales_up_and_schedules_watch(engine, tmp_path, monkeypatch):
    scaler = _autoscaler(
        engine,
        {"vector-mcp": obs("vector-mcp", "up", replicas=1)},
        tmp_path,
        monkeypatch,
        signals=FakeSignalProvider(default=450.0),
        policy_body=PERMISSIVE,
    )
    report = scaler.evaluate()
    # NE-226: the autoscaler only proposes/accepts the intent now.
    assert report["intents_accepted"] == 1
    assert report["evaluations"][0]["outcome"] == "intent_accepted"
    intent = scaler.intent_store.intents["vector-mcp"]
    assert intent["desired_replicas"] == 3  # up-step capped from raw 5
    assert scaler.actuator.applied == []  # the autoscaler never touches it

    # Real actuation is the reconciler's job alone — drive it to prove the
    # accepted intent actually converges.
    reconciler_actuator = RecordingActuator()
    reconciler = _reconciler_for(scaler, monkeypatch, actuator=reconciler_actuator)
    converged = reconciler.reconcile()
    action = converged["actions"][0]
    assert action["decision"] == "accepted_intent"
    assert action["state"] == "executed"
    applied = reconciler_actuator.applied
    assert [(r.kind, r.target, r.params["replicas"]) for r in applied] == [
        ("scale_service", "vector-mcp", 3)
    ]
    assert len(engine.by_type("ActionExecution")) == 1
    # NE-226 REGRESSION (see report — not edited around): the OS-5.27
    # scale-up health-watch scheduling that fleet_autoscaler.py used to do
    # directly (pre-redesign: `if execution.get("ok") and (direction == "up"
    # or watch_scale_down): watch_deploy(...)`) was never carried over to
    # the reconciler. fleet_reconciler._WATCHED_KINDS (fleet_reconciler.py)
    # omits "scale_service" entirely, so no real scale actuation — up or
    # down — schedules a watch job anymore, through either actuation path.
    assert [t["task_type"] for t in engine.submitted] == ["deploy_watch"]


def test_scale_down_skips_watch_unless_policy_opts_in(engine, tmp_path, monkeypatch):
    scaler = _autoscaler(
        engine,
        {"vector-mcp": obs("vector-mcp", "up", replicas=3)},
        tmp_path,
        monkeypatch,
        signals=FakeSignalProvider(default=50.0),
        policy_body=PERMISSIVE,
    )
    report = scaler.evaluate()
    assert report["intents_accepted"] == 1
    intent = scaler.intent_store.intents["vector-mcp"]
    assert intent["desired_replicas"] < 3  # scale-down

    reconciler_actuator = RecordingActuator()
    reconciler = _reconciler_for(scaler, monkeypatch, actuator=reconciler_actuator)
    reconciler.reconcile()
    # NE-226: "direction" was an autoscaler-request-only field; it never
    # survives into the persisted ScaleIntent or the reconciler's converge
    # request, so the direction is proven via the replica count instead.
    assert reconciler_actuator.applied[0].params["replicas"] < 3
    # No watch on scale-down by default — true today, but (see NE-226
    # REGRESSION note above) now for a broader reason than intended: NO
    # real scale actuation schedules a watch anymore, regardless of
    # direction or the watch_scale_down option.
    assert engine.submitted == []


def test_watch_scale_down_policy_option_schedules_watch(engine, tmp_path, monkeypatch):
    scaler = _autoscaler(
        engine,
        {"vector-mcp": obs("vector-mcp", "up", replicas=3)},
        tmp_path,
        monkeypatch,
        signals=FakeSignalProvider(default=50.0),
        policy_body=PERMISSIVE_WATCH_DOWN,
    )
    report = scaler.evaluate()
    assert report["intents_accepted"] == 1

    reconciler_actuator = RecordingActuator()
    reconciler = _reconciler_for(scaler, monkeypatch, actuator=reconciler_actuator)
    reconciler.reconcile()
    assert len(reconciler_actuator.applied) == 1
    # NE-226 REGRESSION: the `watch_scale_down` policy option
    # (action_policy.py's ActionPolicy.option docstring still documents it)
    # is now DEAD — nothing in fleet_autoscaler.py or fleet_reconciler.py
    # reads it anymore (grep confirms zero references outside that
    # docstring), so this opt-in can no longer do anything.
    assert [t["task_type"] for t in engine.submitted] == ["deploy_watch"]


# ---------------------------------------------------------------------------
# Cooldown + flap guard (durable ledger)
# ---------------------------------------------------------------------------


def test_cooldown_blocks_repeat_scale(tmp_path, monkeypatch):
    # NE-226: cooldown is read only from a REAL ActionExecution ledger row
    # (fleet_autoscaler._last_scale_unix), which the reconciler alone now
    # writes. ScaleLedgerEngine (test_fleet_scale_authority.py) is the
    # shared fake that actually returns the ok/dry_run/state/ts shape that
    # read requires — the bare FakeEngine fixture used elsewhere in this
    # file does not, and would misreport every cooldown check as "unknown".
    engine = ScaleLedgerEngine()
    scaler = _autoscaler(
        engine,
        {"vector-mcp": obs("vector-mcp", "up", replicas=1)},
        tmp_path,
        monkeypatch,
        signals=FakeSignalProvider(default=450.0),
        policy_body=PERMISSIVE,
    )
    first = scaler.evaluate()
    assert first["intents_accepted"] == 1
    # Simulate the reconciler having actually (and verifiably) executed this
    # intent — real cooldown evidence — while the observer STILL reports 1
    # replica (actuation not yet visible): without the cooldown this would
    # immediately re-propose the same scale-up.
    _mark_real_execution(engine, scaler.intent_store, "vector-mcp")
    second = scaler.evaluate()
    assert second["actions"] == 0
    assert "cooldown" in second["evaluations"][0]["reason"]
    assert scaler.intent_store.intents["vector-mcp"]["revision"] == 1  # no 2nd intent


def test_flap_guard_blocks_opposite_direction_within_cooldown(tmp_path, monkeypatch):
    engine = ScaleLedgerEngine()
    signals = FakeSignalProvider(default=450.0)
    scaler = _autoscaler(
        engine,
        {"vector-mcp": obs("vector-mcp", "up", replicas=1)},
        tmp_path,
        monkeypatch,
        signals=signals,
        policy_body=PERMISSIVE,
    )
    first = scaler.evaluate()
    assert first["intents_accepted"] == 1  # scale up
    _mark_real_execution(engine, scaler.intent_store, "vector-mcp")
    # Load evaporates and the observer now sees 3 replicas: the raw verdict
    # is scale-DOWN, but it lands inside the cooldown window.
    signals.default = 0.0
    scaler.observer.observations["vector-mcp"] = obs("vector-mcp", "up", replicas=3)
    second = scaler.evaluate()
    assert second["actions"] == 0
    assert "cooldown" in second["evaluations"][0]["reason"]


def test_expired_cooldown_allows_scaling_again(tmp_path, monkeypatch):
    engine = ScaleLedgerEngine()
    scaler = _autoscaler(
        engine,
        {"vector-mcp": obs("vector-mcp", "up", replicas=1)},
        tmp_path,
        monkeypatch,
        signals=FakeSignalProvider(default=450.0),
        policy_body=PERMISSIVE,
    )
    first = scaler.evaluate()
    assert first["intents_accepted"] == 1
    _mark_real_execution(engine, scaler.intent_store, "vector-mcp")
    # Age the ledger entries beyond the 300s cooldown.
    stale = time.time() - 1000
    for node in engine.nodes.values():
        if node["type"] == "ActionDecision":
            node["decided_unix"] = stale
        if node["type"] == "ActionExecution":
            node["executed_unix"] = stale
    second = scaler.evaluate()
    assert second["intents_accepted"] == 1
    assert scaler.intent_store.intents["vector-mcp"]["revision"] == 2  # scaled again


# ---------------------------------------------------------------------------
# Per-tick budget + compact recording
# ---------------------------------------------------------------------------

TWO_SCALED_REGISTRY = """
services:
  - name: a-svc
    scaling: {max: 5, signal: queue_depth, target: 100, cooldown_s: 0}
  - name: b-svc
    scaling: {max: 5, signal: queue_depth, target: 100, cooldown_s: 0}
"""


def test_action_budget_defers_excess_services(engine, tmp_path, monkeypatch):
    scaler = _autoscaler(
        engine,
        {
            "a-svc": obs("a-svc", "up", replicas=1),
            "b-svc": obs("b-svc", "up", replicas=1),
        },
        tmp_path,
        monkeypatch,
        signals=FakeSignalProvider(default=450.0),
        policy_body=PERMISSIVE,
        registry_body=TWO_SCALED_REGISTRY,
        max_actions=1,
    )
    report = scaler.evaluate()
    assert report["actions"] == 1
    budgeted = [e for e in report["evaluations"] if "budget" in e.get("reason", "")]
    assert [e["service"] for e in budgeted] == ["b-svc"]


def test_one_bulk_signal_read_per_tick_for_all_services(engine, tmp_path, monkeypatch):
    signals = FakeSignalProvider(default=450.0)
    scaler = _autoscaler(
        engine,
        {
            "a-svc": obs("a-svc", "up", replicas=1),
            "b-svc": obs("b-svc", "up", replicas=1),
        },
        tmp_path,
        monkeypatch,
        signals=signals,
        policy_body=PERMISSIVE,
        registry_body=TWO_SCALED_REGISTRY,
    )
    scaler.evaluate()
    assert signals.bulk_calls == 1
    assert signals.calls == [("a-svc", "queue_depth"), ("b-svc", "queue_depth")]


def test_one_compact_evaluation_node_per_tick(engine, tmp_path, monkeypatch):
    scaler = _autoscaler(
        engine,
        {
            "a-svc": obs("a-svc", "up", replicas=1),
            "b-svc": obs("b-svc", "up", replicas=1),
        },
        tmp_path,
        monkeypatch,
        signals=FakeSignalProvider(default=450.0),
        policy_body=PERMISSIVE,
        registry_body=TWO_SCALED_REGISTRY,
    )
    scaler.evaluate()
    records = engine.by_type("AutoscaleEvaluation")
    assert len(records) == 1  # one node per tick, never per service
    assert records[0]["evaluated"] == 2
    scaler.evaluate()
    assert len(engine.by_type("AutoscaleEvaluation")) == 2


def test_quiet_tick_writes_nothing(engine, tmp_path, monkeypatch):
    scaler = _autoscaler(
        engine,
        {},
        tmp_path,
        monkeypatch,
        signals=FakeSignalProvider(default=450.0),
        policy_body=PERMISSIVE,
        registry_body="services:\n  - name: plain-svc\n",
    )
    scaler.evaluate()
    assert engine.by_type("AutoscaleEvaluation") == []


# ---------------------------------------------------------------------------
# Daemon wiring (leader-only maintenance tick, opt-in flag)
# ---------------------------------------------------------------------------


def _enabled_maintenance_names() -> set[str]:
    """Names of ENABLED maintenance :Schedule nodes for the current config
    (the unified-scheduler analog of a tick being registered; CONCEPT:AU-OS.state.unified-scheduling-one-intelligent)."""
    from agent_utilities.core import schedule_engine as _se
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    inst = TaskManagerMixin.__new__(TaskManagerMixin)  # type: ignore[type-abstract]
    # This helper is called twice per test (flag True, then False); each call
    # constructs its own GraphComputeEngine directly, which trips the
    # "process graph transport already exists" guard on the second call
    # unless the singleton left by the first call's internal machinery is
    # cleared first.
    GraphComputeEngine._PROCESS_ENGINE = None
    # A bare EpistemicGraphBackend() resolves its own routing graph via
    # resolve_routing_graph(None) BEFORE GraphComputeEngine is ever asked for
    # one, bypassing the isolate_graph_compute_engine fixture's redirect (same
    # family as test_kg_native_orchestration.py). Bind directly to an
    # already-isolated GraphComputeEngine instead.
    compute = GraphComputeEngine(backend_type="rust")
    backend = object.__new__(EpistemicGraphBackend)
    backend._graph = compute
    backend.graph_name = compute.graph_name
    backend.create_schema()
    inst.backend = backend
    # _control_backend() (schedule_engine.py) reads engine.control_backend,
    # not engine.backend -- a real IntelligenceGraphEngine sets both
    # (_build_control_backend() returns self.backend for the single-client
    # profile); this hand-built TaskManagerMixin instance needs the same.
    inst.control_backend = backend
    inst._register_maintenance_schedules()
    return {s.name for s in _se._load_all(inst) if s.enabled}


def test_autoscaler_tick_registration_is_flag_gated(monkeypatch):
    from agent_utilities.core.config import config

    monkeypatch.setattr(config, "fleet_autoscaler", True)
    assert "fleet_autoscaler" in _enabled_maintenance_names()

    monkeypatch.setattr(config, "fleet_autoscaler", False)
    assert "fleet_autoscaler" not in _enabled_maintenance_names()
