# Engine performance probes (lane-perf-0801)

Run each INSIDE the live graph-os pod — they mint a real process identity and use
the production backend, so they measure what delegation actually pays:

    POD=$(kubectl -n platform get pods -l app=graph-os -o jsonpath='{.items[0].metadata.name}')
    kubectl -n platform exec -i $POD -c graph-os -- python3 - < scripts/perf/<probe>.py

Detach anything that runs longer than ~90s; a foreground harness call is killed at 2 min.

| probe | answers |
|---|---|
| `kg_index_population_probe.py` | How many nodes carry embeddings/text? (retrieval *correctness*) |
| `engine_stall_probe.py` | Is engine latency per-op work or a contention STALL? 100 identical point-reads + the concurrent op mix. |
| `engine_write_contention_probe.py` | Correlates read latency against the concurrent WRITE rate over 90 buckets. |

## Why these exist

`epistemic_graph_request_duration_seconds` (engine `/metrics`, port 9101) reports a
*cumulative* mean that mixes quiet and contended periods, so it reads as "every op is
slow". The histogram buckets show the truth is bimodal — most calls are ~100us, a
minority stall for seconds. These probes separate the two.
