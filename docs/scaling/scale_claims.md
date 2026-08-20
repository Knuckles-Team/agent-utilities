# Agent-utilities scale claim register

This is the AU-side, source-anchored register for scaling, W3 deployment assets,
and capacity statements. It deliberately does not copy epistemic-graph claims:
the source anchors below are files in this repository, and a claim is only as
strong as the AU source or evidence named here.

## Status vocabulary

The status is an evidence level, not a promise about an operator's cluster:

- `DESIGNED` — an AU plan or workload model exists; no implementation or run is
  implied.
- `IMPLEMENTED` — the AU source path or committed reference asset exists. This
  does not mean it has been deployed.
- `UNIT-PROVEN` — a focused automated fixture exercises the AU path. The fixture
  still needs to pass in the current release before it is a release claim.
- `LAB-PROVEN` — a bounded external run passed against a named release and
  environment; no such run is implied by a unit fixture.
- `LIVE` — an operator observed the exact deployed artifact, configuration,
  and health evidence at a recorded time.
- `1M-CERTIFIED` — the signed production certification campaign passed the
  exact one-million-resident workload contract and SLOs.

`reference` and `staged` describe committed deployment inputs, not additional
status levels. A reference/staged manifest is not deployable until its release
compatibility gate, exact digest rendering, and asset checks pass. A deployable
render is not `LIVE` until an operator records deployment evidence. No AU claim
below is currently `LAB-PROVEN`, `LIVE`, or `1M-CERTIFIED`.

## Machine-checked claims

The checker in [`scripts/check_scale_claims.py`](../../scripts/check_scale_claims.py)
requires each source anchor and source fragment to remain present. It also
rejects duplicate IDs, path escape, unsupported statuses, and ungrounded
`LIVE`/`1M-CERTIFIED` rows. Run it from the repository root before changing a
scale claim.

| ID | Status | AU source anchor | Required source fragment | Evidence scope |
|---|---|---|---|---|
| `AU-SCALE-001` | `IMPLEMENTED` | `agent_utilities/orchestration/fleet_actuation.py#class KubernetesActuator` | `class KubernetesActuator:` | optional `kubectl` reference actuator; source capability, not live cluster proof |
| `AU-SCALE-002` | `IMPLEMENTED` | `agent_utilities/orchestration/fleet_actuation.py#selection in ("k8s", "kubernetes")` | `selection in ("k8s", "kubernetes")` | config selection falls back to dry-run when `kubectl` is unavailable |
| `AU-SCALE-003` | `IMPLEMENTED` | `agent_utilities/core/config.py#fleet_actuator: str` | `fleet_actuator: str = Field(default="dryrun"` | default is inert; mutation remains policy-gated |
| `AU-SCALE-004` | `IMPLEMENTED` | `agent_utilities/orchestration/fleet_reconciler.py#def parse_scaling_spec` | `Required: ``max`` (ceiling), ``signal`` and ``target`` (>0).` | explicit floor/ceiling/target contract; invalid blocks are ignored |
| `AU-SCALE-005` | `UNIT-PROVEN` | `agent_utilities/orchestration/fleet_autoscaler.py#def compute_desired_replicas` | `desired = math.ceil(eff * per_replica / spec.target)` | focused math path; root must run the AU unit suite |
| `AU-SCALE-006` | `IMPLEMENTED` | `agent_utilities/orchestration/scaling_signals.py#_PROMQL_TEMPLATES` | `container_label_com_docker_swarm_service_name` | built-in CPU query is Swarm-shaped; K8s needs a custom/injected metric mapping |
| `AU-SCALE-007` | `IMPLEMENTED` | `docs/scaling/capacity_model.py#RESIDENTS_PER_ENGINE_SHARD` | `RESIDENTS_PER_ENGINE_SHARD = 50_000` | AU planning constant, not a measured multi-host capacity |
| `AU-SCALE-008` | `DESIGNED` | `docs/scaling/capacity_model.md#1,000,000 residents` | `MODELED — the documented reference case` | one-million case is a workload/capacity model until certification evidence exists |
| `AU-SCALE-009` | `UNIT-PROVEN` | `tests/scale/test_capacity_model.py#def test_one_million_matches_documented_numbers` | `p.kafka_partitions == 8` | arithmetic/doc consistency only; not a 1M runtime certification |
| `AU-SCALE-010` | `IMPLEMENTED` | `deploy/k8s/production-cell/autoscaling.yaml#kind: HorizontalPodAutoscaler` | `kind: HorizontalPodAutoscaler` | committed HPA reference/staged asset; not evidence of a live HPA |
| `AU-SCALE-011` | `IMPLEMENTED` | `deploy/README.md#check-graphos-compatibility` | `kubectl apply -k RENDERED_DIRECTORY` | exact release gate + rendered directory are required before deployment |
| `AU-SCALE-012` | `UNIT-PROVEN` | `tests/unit/test_fleet_actuation.py#def test_scale_service_issues_real_kubectl_scale_call` | `deployment/graph-os-dispatch` | mocked argv contract only; no cluster was contacted by the fixture |
