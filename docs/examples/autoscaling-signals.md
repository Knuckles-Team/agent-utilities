# Worked Example: Autoscaling Signals and Target Tracking

## What this demonstrates

How the reactive replica autoscaler (CONCEPT:OS-5.29,
`agent_utilities/orchestration/fleet_autoscaler.py`) turns a declared
`scaling:` block plus a live load signal into a policy-gated
`scale_service` proposal — with the exact target-tracking math, the
cooldown/flap guard, and what lands in the KG.

Deep dive: [fleet_autonomy.md](../architecture/fleet_autonomy.md) and
[gateway_scaling.md](../architecture/gateway_scaling.md).

## Prerequisites (ladder rung)

The "gateway + host daemon" rung or above — see
[deployment configurations](../guides/deployment-configurations.md). The
autoscaler is a **leader-only maintenance tick** registered by the engine's
consolidated scheduler (`knowledge_graph/core/engine_tasks.py`):

| Flag | Default | Meaning |
|---|---|---|
| `FLEET_AUTOSCALER` | `False` (off) | Opt-in: register the leader-only autoscale tick. With the default dry-run actuator (`FLEET_ACTUATOR=dryrun`) it records intent without mutating anything. |
| `FLEET_AUTOSCALER_INTERVAL` | `60.0` | Seconds between ticks. |
| `SCALING_PROMETHEUS_URL` | unset | Set → signals come from instant HTTP queries against that Prometheus (`PrometheusHttpProvider`); unset → the zero-infra `LocalMetricsProvider` reads this process's own gauges. |
| `FLEET_RECONCILER_MAX_ACTIONS` | `5` | Shared per-tick action budget (also used by the autoscaler). |

```bash
export FLEET_AUTOSCALER=1
```

## 1. Declaring scaling bounds (`scaling:` block)

A service is **never autoscaled without a declared block** — `max`, `signal`
and `target` are required (no implicit ceiling, no implicit metric), and any
invalid block is dropped with a warning rather than guessed at
(`parse_scaling_spec` in `orchestration/fleet_reconciler.py`).

Schema (verified against `ScalingSpec` / `parse_scaling_spec`):

```yaml
scaling:
  min: 1            # replica floor (>= 0; default 1)
  max: 3            # replica ceiling (REQUIRED; >= min)
  signal: queue_depth  # REQUIRED: queue_depth | consumer_lag | cpu | custom
  target: 200       # REQUIRED (> 0): per-replica target value for the signal
  scale_up_step: 1  # max replicas added per evaluation (default 1)
  scale_down_step: 1  # max replicas removed per evaluation (default 1)
  cooldown_s: 300   # min seconds between scale actions (default 300)
```

The block rides on a `services:` entry in the fleet registry
(`deploy/mcp-fleet.registry.yml`) **or** — because that registry is
machine-generated — in the `FLEET_DESIRED_STATE_PATH` override YAML, which
layers per-service `replicas` / `desired` / `version` / `scaling` on top of
the registry.

### Block A — queue-depth signal (ingest workers)

```yaml
# desired-state override file (FLEET_DESIRED_STATE_PATH=…/fleet-overrides.yml)
services:
  - name: kg-ingest-worker
    scaling:
      min: 1
      max: 6
      signal: queue_depth      # fleet-total pending KG ingest tasks
      target: 200              # aim for <= 200 queued tasks PER replica
      scale_up_step: 2         # backlog spikes deserve a bigger step up
      scale_down_step: 1       # drain slowly to avoid flapping
      cooldown_s: 300
```

### Block B — consumer-lag signal (Kafka-backed dispatch)

```yaml
services:
  - name: agent-dispatch-worker
    scaling:
      min: 2
      max: 8
      signal: consumer_lag     # fleet-total unconsumed messages
      target: 500              # aim for <= 500 lag PER replica
      scale_up_step: 1
      scale_down_step: 1
      cooldown_s: 600          # lag is bursty; damp harder
```

## 2. Where the signal value comes from

Provider resolution (`orchestration/scaling_signals.py`,
`get_scaling_signal_provider()`): an injected provider
(`set_scaling_signal_provider`) → `SCALING_PROMETHEUS_URL` → the local
default. **`None` from the provider means "no data" — and no data means no
scaling action**, mirroring the reconciler's unobserved⇒skip rule.

| Signal name | Semantics | LocalMetricsProvider reads | PrometheusHttpProvider query |
|---|---|---|---|
| `queue_depth` | FLEET-TOTAL | `agent_utilities_kg_ingest_queue_depth` | `sum(agent_utilities_kg_ingest_queue_depth)` |
| `consumer_lag` | FLEET-TOTAL | `agent_utilities_kg_ingest_consumer_lag` | `sum(agent_utilities_kg_ingest_consumer_lag)` |
| `cpu` | per-replica avg | (metric family named `cpu`, normally absent) | `100 * avg(rate(container_cpu_usage_seconds_total{container_label_com_docker_swarm_service_name="<service>"}[5m]))` |
| anything else | per-replica avg | metric family of that name in the local registry | the signal string itself, verbatim PromQL (`{service}` placeholder substituted) |

Convention that matters for the math: `queue_depth` and `consumer_lag` are
fleet totals (divided by current replicas first); every other signal is
treated as a per-replica average, so custom PromQL should `avg(...)`, not
`sum(...)`.

## 3. The target-tracking math (the actual formula)

From `compute_desired_replicas()` in `fleet_autoscaler.py`:

```text
eff          = max(current, 1)                  # a 0-replica service can recover
per_replica  = value / eff      if signal is fleet-total (queue_depth, consumer_lag)
             = value            otherwise (cpu, custom)
desired      = ceil(eff * per_replica / target)
desired      = clamp(desired, min, max)
desired      = step-cap: at most +scale_up_step / -scale_down_step vs current
```

Note the fleet-total case algebraically collapses to
`desired = ceil(value / target)` — independent of current replicas — which is
exactly what you want for a shared backlog.

Worked numbers (verified by executing the real function — Block A spec:
min=1, max=5 for this table, target=200, up_step=2, down_step=1):

| Signal value | Current | per_replica | Raw `ceil` | After clamp [1,5] | After step cap | Outcome |
|---|---|---|---|---|---|---|
| `queue_depth` = 900 | 2 | 450 | ceil(900/200) = 5 | 5 | min(5, 2+2) = **4** | scale up 2→4 |
| `queue_depth` = 900 | 4 | 225 | 5 | 5 | **5** | scale up 4→5 |
| `queue_depth` = 150 | 3 | 50 | ceil(150/200) = 1 | 1 | max(1, 3-1) = **2** | scale down 3→2 |
| `queue_depth` = 0 | 3 | 0 | 0 | clamp → 1 | max(1, 3-1) = **2** | scale down 3→2 |
| `cpu` = 85 (target 60, max 4) | 2 | 85 | ceil(2·85/60) = 3 | 3 | **3** | scale up 2→3 |
| `cpu` = 20 (target 60, max 4) | 3 | 20 | ceil(3·20/60) = 1 | 1 | max(1, 3-1) = **2** | scale down 3→2 |

## 4. Guards before any action

In `_evaluate_service`, in order:

1. **Unobserved ⇒ skip.** No replica evidence from the FleetObserver = no
   action.
2. **Down ⇒ skip.** A down service is the reconciler's restart problem;
   scaling a dead service masks the failure.
3. **No signal data ⇒ skip.**
4. **At target ⇒ skip** (`desired == current`).
5. **Cooldown/flap guard ⇒ skip.** No scale action in either direction
   within `cooldown_s` of the service's last allowed/executed
   `scale_service` entry — read from the **durable** `ActionDecision` and
   `ActionExecution` ledgers, so the guard holds across processes and
   restarts, and opposite-direction flapping inside the window is impossible.
6. **Per-tick action budget** (`FLEET_RECONCILER_MAX_ACTIONS`, default 5)
   exhausted ⇒ remaining services are deferred to the next tick.

## 5. Gate → actuate → watch

A surviving proposal becomes:

```python
ActionRequest(kind="scale_service", target="kg-ingest-worker",
              params={"replicas": 4, "from_replicas": 2, "direction": "up",
                      "signal": "queue_depth", "value": 900.0, "target": 200},
              source="autoscaler",
              reason="target tracking: queue_depth=900 vs target 200/replica → 2→4 (bounds 1-6)")
```

and goes through the ActionPolicy gate (CONCEPT:OS-5.24). Under the shipped
default policy `scale_service` is `approval_required` — the autoscaler then
*files an approval* instead of scaling; the
[scoped-autonomous posture](action-policy-postures.md) shows the
`auto_notify` rule (with rate/blast caps) that lets it act. Allowed
proposals execute through the FleetActuator seam (`FLEET_ACTUATOR=dryrun`
default records intent only; `docker` uses the docker CLI), and successful
scale-**ups** schedule an OS-5.27 deploy watch — scale-downs too when the
policy file sets `options: {watch_scale_down: true}`. The watch probes
service health for `DEPLOY_WATCH_WINDOW` seconds and escalates
(policy-gated rollback + notification) on failure.

## 6. What the autoscaler records

- **Per action**: the usual `ActionDecision` audit node (from the policy
  gate) and, when actuated, an `ActionExecution` node (`kind`, `target`,
  `ok`, `dry_run`, `executed_unix`, ...) — these double as the cooldown
  ledger.
- **Per tick**: at most ONE compact `AutoscaleEvaluation` node (no
  per-service nodes; ticks that evaluated nothing write nothing):
  `evaluated`, `actions`, `scaled`, `details_json` (per-service outcome:
  `scaled` | `proposed` | `skipped` + reason/current/desired/value),
  `actuator`, `signal_provider`, `created_at`, `created_unix`.
- **Log line** on the daemon when anything happened:

```text
[OS-5.29] fleet autoscale: evaluated=3 actions=1 scaled=1 actuator=dryrun signals=local
```

## Verification

```bash
python3 -m pytest tests/unit/test_fleet_autoscaler.py tests/unit/test_scaling_signals.py -q
```

---

*Smoke-run against this tree (2026-06-11): the table in section 3 was produced
by executing `compute_desired_replicas()` directly with the specs shown, and
`python3 -m pytest tests/unit/test_fleet_autoscaler.py
tests/unit/test_scaling_signals.py -q` passed as part of a 99-test green run.
The registry/override YAML blocks were validated against `parse_scaling_spec`
schema in code (reviewed, not deployed).*
