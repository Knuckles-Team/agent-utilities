# AU native numeric call-site gap report

This is the NE-153 production audit for the `xp` imports in `agent_utilities/`.
It covers 36 production files and 75 unique `xp` attribute surfaces. The names
below are the historical audit inventory; every bounded production surface now
has an explicit disposition: a native-kernel call, a caller-owned builtin-list
loop, a deterministic random adapter, or a versioned artifact seam. The AU
adapter does not grow a replacement array runtime.

## Exact audited surfaces

| Production file | Surfaces observed |
|---|---|
| `domains/finance/alpha_factors.py` | `isnan`, `log`, `spearmanr` |
| `domains/finance/composite_backtest.py` | `abs`, `array`, `asarray`, `cumprod`, `full`, `maximum`, `maximum.accumulate`, `mean`, `min`, `ones`, `prod`, `sqrt`, `std`, `zeros` |
| `domains/finance/cross_market_arb.py` | `exp`, `linalg`, `linalg.lstsq`, `log`, `ones`, `sqrt`, `std`, `vstack` |
| `domains/finance/execution.py` | `ks_2samp` |
| `domains/finance/kronos_forecaster.py` | `abs`, `any`, `array`, `maximum`, `minimum`, `percentile`, `where` |
| `domains/finance/market_data.py` | `cumsum`, `exp`, `random`, `random.default_rng` |
| `domains/finance/microstructure.py` | `mean` |
| `domains/finance/profit_attribution.py` | `corrcoef`, `cumprod`, `max`, `maximum`, `maximum.accumulate`, `mean`, `min`, `prod`, `sqrt`, `std`, `sum` |
| `domains/finance/research_autopilot.py` | `array`, `cumsum`, `maximum`, `maximum.accumulate`, `mean`, `min`, `random`, `random.default_rng`, `sqrt`, `std`, `sum` |
| `domains/finance/risk_manager.py` | `mean`, `norm_pdf`, `norm_ppf`, `percentile`, `random`, `random.default_rng`, `sort`, `std` |
| `domains/finance/signal_fusion.py` | `abs`, `asarray`, `atleast_2d`, `cov`, `eye`, `linalg`, `linalg.LinAlgError`, `linalg.pinv`, `linalg.solve` |
| `domains/finance/visual_ta.py` | `arange`, `max`, `mean`, `min`, `sum` |
| `graph/test_time_diversity.py` | `asarray`, `dot`, `float32`, `linalg`, `linalg.norm` |
| `harness/assimilation_benchmark.py` | `array`, `linalg`, `linalg.norm`, `log2`, `random`, `random.default_rng`, `vstack` |
| `harness/latent_efficiency_benchmark.py` | `eye`, `float32`, `linalg`, `linalg.norm` |
| `harness/superhuman_gate.py` | `asarray`, `float64`, `quantile`, `random`, `random.default_rng` |
| `knowledge_graph/assimilation/concept_matcher.py` | `argsort`, `asarray`, `float32`, `linalg`, `linalg.norm` |
| `knowledge_graph/core/analogy_engine.py` | `array`, `dot`, `linalg`, `linalg.norm` |
| `knowledge_graph/core/engine_tasks.py` | `dot`, `linalg`, `linalg.norm`, `mean` |
| `knowledge_graph/core/formal_reasoning_core.py` | `int64`, `linalg`, `linalg.matrix_power`, `linalg.norm`, `ones`, `random`, `random.default_rng`, `zeros` |
| `knowledge_graph/core/hypergraph.py` | `concatenate`, `dot`, `maximum`, `random`, `random.default_rng`, `sqrt`, `zeros` |
| `knowledge_graph/core/markov_regime.py` | `argsort`, `array`, `concatenate`, `cumsum`, `full`, `inf`, `prod`, `random`, `random.randint`, `zeros` |
| `knowledge_graph/core/optimal_execution.py` | `array`, `float64`, `mean`, `std`, `sum`, `tanh`, `zeros` |
| `knowledge_graph/core/semantic_subsumption.py` | `array`, `dot`, `linalg`, `linalg.norm` |
| `knowledge_graph/core/spectral_navigator.py` | `argmax`, `argmin`, `array`, `clip`, `diag`, `diff`, `eigsh`, `empty`, `eye`, `fill_diagonal`, `float64`, `linalg`, `linalg.eigh`, `linalg.norm`, `random`, `random.default_rng`, `sort`, `sqrt`, `where` |
| `knowledge_graph/core/world_model.py` | `asarray`, `concatenate`, `eye`, `float64`, `linalg`, `linalg.norm`, `linalg.solve`, `stack`, `zeros` |
| `knowledge_graph/distillation/deduplicator.py` | `array`, `float32`, `linalg`, `linalg.norm` |
| `knowledge_graph/distillation/lsh_index.py` | `array`, `dot`, `float32`, `linalg`, `linalg.norm`, `random`, `random.RandomState` |
| `knowledge_graph/extraction/fact_extractor.py` | `asarray` |
| `knowledge_graph/memory/optimization_engine.py` | `abs`, `arange`, `argmin`, `argsort`, `array`, `clip`, `dot`, `exp`, `float64`, `linalg`, `linalg.LinAlgError`, `linalg.norm`, `linalg.svd`, `log2`, `max`, `mean`, `newaxis`, `random`, `random.default_rng`, `sqrt`, `std`, `sum`, `triu_indices`, `var`, `where`, `zeros` |
| `knowledge_graph/orchestration/engine_finance.py` | `array`, `zeros` |
| `knowledge_graph/retrieval/capability_index.py` | `argsort`, `array`, `asarray`, `dot`, `float32`, `linalg`, `linalg.norm`, `load`, `save`, `stack`, `zeros` |
| `knowledge_graph/retrieval/generative_recommender.py` | `argsort`, `array`, `empty`, `float64`, `linalg`, `linalg.norm`, `vstack`, `zeros` |
| `knowledge_graph/retrieval/semantic_retrieval_engine.py` | `array`, `dot`, `linalg`, `linalg.norm`, `mean` |
| `knowledge_graph/retrieval/temporal_semantic_id.py` | `argmin`, `asarray`, `float64`, `linalg`, `linalg.norm`, `nan_to_num`, `random`, `random.default_rng`, `sum`, `vstack` |
| `mcp/tools/analysis_tools.py` | `cumprod`, `random`, `random.normal`, `roll` |

## Final semantic dispositions

| Gap family | Representative callers | Required owner |
|---|---|---|
| constructors/dtypes/constants | all constructor-heavy callers, including capability index and spectral clustering | Removed from AU; bounded builtin-list conversion and explicit loops |
| stateful/random module | market-data, research, harness, LSH, temporal IDs, analysis tools | `_DeterministicRandom` over seeded one-shot EG exports, including native batch choice/permutation; fixed seeds and list outputs |
| array arithmetic/indexing/broadcasting | capability index, memory optimization, recommender, spectral/world models | Bulk native matmul/kmeans where applicable, with caller-owned bounded mapping loops; no array runtime |
| covariance/persistence/object dtype | finance signal paths and capability index save/load | Bulk native covariance/matrix operations; `save_numeric_artifact`/`load_numeric_artifact` JSON seam with exact schema, bounds, and verified digest |
| missing scalar/vector helpers | roll/cumulative/quantile/log2-style callers | Explicit list loops or existing native reduction/elementwise calls |
| linalg compatibility | regression, fusion, spectral, world-model callers | Native `solve`/`pinv`/`lstsq`/`svd`/`eigh`; unsupported names remain typed failures |

The focused facade tests cover both sides of this report: direct native calls
(`sum`, `sqrt`, `norm`, `where`), deterministic random/list behavior including
batch choice/permutation, bounded artifact round-trips with digest verification,
and typed rejection of representative unsupported gaps
(`array`, `asarray`, `zeros`, `stack`, `save`, `load`).
