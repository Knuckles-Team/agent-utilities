# AU-RETRIEVAL-001 requirements

| ID | Requirement | Verification |
|---|---|---|
| `AU-RETRIEVAL-R001` | **Benchmark structure-aware retrieval against served hybrid search.** AU benchmarks constrained section-leaf selection against hierarchical-document retrieval and the production hybrid vector-plus-graph retrieval path, using a versioned synthetic fixture corpus and optionally held-out ecosystem documents, under identical visible candidate sets and token budgets, measuring citation validity, visibility correctness, p50/p95 latency and index/update cost. | A reproducible benchmark report compares all methods on the fixture corpus with Recall@1/3, nDCG, citation-validity and latency metrics, and adoption requires a statistically defensible gain on a held-out set with no visibility or citation regression. |
