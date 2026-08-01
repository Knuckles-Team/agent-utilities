# Design Document: A large upload batch is enqueued as a background job, not ingested inline

CONCEPT:AU-KG.ingest.enqueue-large-batch

> `agent_utilities/ingestion/collector.py:190-215`.

## Decision — a large batch of received bundles returns `status=enqueued` + a `job_id` from a background `session_upload` job; only a small batch ingests inline

`_flush()` (`collector.py:204-213`) is explicit in its own comment: **"the
server now ENQUEUES a large batch as a background `session_upload` job
(returns `status=enqueued` + `job_id`) and only ingests a tiny batch inline
(`status=ingested`); count both honestly so the summary reflects
async-drain (not a silent `ingested=0`)."**

**The rejected alternative is synchronous inline ingestion of the whole
batch.** It is the naturally simpler implementation — the caller sends
bundles, the server ingests them, done — and it loses on the property this
decision is fixing: a large batch ingested synchronously blocks the
caller's request for however long ingestion takes, and on a genuinely large
batch that means a request timeout with no partial-success signal. The fix
splits the response into two honestly-counted outcomes (`received`,
`ingested`, `enqueued`) instead of forcing every batch through one blocking
path.

**The corollary decision named in the same comment**: the summary must
never silently report `ingested=0` when work actually happened — it has to
show `enqueued=N` so a caller polling `job_id` knows the batch is in flight,
not lost. This is a correctness requirement on the response shape, not just
a performance optimization: a caller that only checks `ingested` would
otherwise conclude a large batch failed silently.

## Risk Assessment

- **Blast Radius**: `agent_utilities/ingestion/collector.py`'s `_flush()`
  and its `session_upload` job counterpart.
- **Backward Compatible**: Yes for callers that already handle
  `status=enqueued`/`job_id`; a caller that only checks `status=ingested`
  synchronously would need updating to poll the job.
- **Breaking Changes**: None currently — this is the existing behavior being
  documented, not a proposed change.
- **Known weak point**: the threshold between "tiny batch, ingest inline"
  and "large batch, enqueue" is a size cutoff in this module; a caller
  sending batches right at that boundary sees inconsistent latency/response
  shape (sometimes synchronous, sometimes a job handle) depending on batch
  size alone.
