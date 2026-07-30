# Engine topology and scaling

Epistemic Graph is authoritative durable state, not a disposable cache. Select a
topology from measured availability, latency, throughput, tenancy, and recovery
requirements.

## Unified in process

One graph-os workload owns one embedded/local engine lifecycle and persistent store.
This minimizes moving parts and IPC latency and is the default for development and a
single durable instance.

- replicas: one writer;
- storage: durable single-writer volume;
- scaling: increase resources or move to shared topology;
- upgrade: checkpoint, stop, migrate if required, replace, verify;
- failure domain: the workload and its storage.

Do not claim an in-process PyO3 path is live unless the deployed build and trace prove
it. A packaged, auto-started side process may provide the same operational
single-unit experience but remains a different execution boundary.

## Out-of-process shared

Stateless graph-os clients connect to an independently operated engine service or
cluster. Scale graph-os and engine partitions according to different signals.

- graph-os replicas may scale on concurrency/latency after session and cache safety
  are proven;
- engine replicas/partitions follow the engine’s supported consensus and sharding
  contract;
- client endpoints, authentication, policy version, and tenant context must be
  explicit;
- failure and retry budgets must prevent retry storms;
- cross-shard or replicated writes require the engine’s documented transactional
  semantics, not filesystem sharing.

## Native throughput rules

- send batches across the Python/Rust or network boundary;
- avoid one functional call per entity when a vectorized/graph operation exists;
- separate prefill/model concurrency from disk, connector, and reasoning queues;
- measure p50/p95/p99 latency, queue time, batch size, disk bytes/IOPS, memory,
  retries, and cache pressure per lane;
- checkpoint background ingestion/evolution and yield resources to foreground user
  runs;
- test degraded operation when a model, connector, mirror, or engine member is
  unavailable.

## Scaling gate

Scale only after a representative load test proves improved throughput without
violating correctness, isolation, tail latency, recovery objectives, or cost
budgets. Record the tested artifact, topology, data size, workload, and bottleneck.
