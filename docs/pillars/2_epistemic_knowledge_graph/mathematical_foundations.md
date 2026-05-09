# Mathematical Foundations & Financial Engineering Reference

> **CONCEPT Registry**: KG-2.41 through KG-2.46
>
> This document provides full explanations of all mathematical, probabilistic,
> and financial engineering terms introduced by the comparative analysis
> integration.  It is the canonical glossary for the 6 new modules.

---

## Table of Contents

1. [Graph Theory Foundations (KG-2.41)](#graph-theory-foundations-kg-241)
2. [Embedding Alignment Diagnostics (KG-2.42)](#embedding-alignment-diagnostics-kg-242)
3. [Structural Causal Reasoning (KG-2.43)](#structural-causal-reasoning-kg-243)
4. [Latent Space Regularization (KG-2.44)](#latent-space-regularization-kg-244)
5. [Probabilistic Graph Reasoning (KG-2.45)](#probabilistic-graph-reasoning-kg-245)
6. [Optimal Execution & Market Making (KG-2.46)](#optimal-execution--market-making-kg-246)
7. [Source Materials](#source-materials)

---

## Graph Theory Foundations (KG-2.41)

**Source**: *Mathematics for Computer Science* (Lehman, Leighton, Meyer — MIT 6.042J)

### DAG Critical Path Analysis

A **DAG** (Directed Acyclic Graph) is a directed graph with no cycles.  The
**critical path** is the longest weighted path from any source node to any
sink node.  Its length equals the **makespan** — the minimum possible
completion time regardless of how many parallel workers you use.

- **Topological Sort**: A linear ordering of DAG vertices such that for
  every directed edge (u, v), u appears before v in the ordering.
  Prerequisite for critical path computation.
- **Earliest Start Time**: The earliest time a node (task) can begin,
  considering all predecessor dependencies.
- **Slack**: The amount of time a task can be delayed without increasing
  the makespan.  Critical path tasks have zero slack.

**Module**: `graph_theory_primitives.dag_critical_path()`

### Graph Connectivity

- **Vertex Connectivity κ(G)**: The minimum number of vertices whose
  removal disconnects the graph.  Measures structural resilience.
- **Edge Connectivity λ(G)**: The minimum number of edges whose removal
  disconnects the graph.
- **Whitney's Theorem**: κ(G) ≤ λ(G) ≤ δ(G) where δ is the minimum vertex degree.
- **Minimum Vertex Cut**: The actual set of vertices forming the minimum cut.
  These are the "chokepoint" nodes in the Knowledge Graph.

**Module**: `graph_theory_primitives.vertex_connectivity()`, `edge_connectivity()`, `minimum_vertex_cut()`

### Euler Tour

An **Euler tour** (or circuit) is a closed walk that traverses every edge of
a graph exactly once and returns to the starting vertex.

**Euler's Theorem**: A connected graph has an Euler tour if and only if every
vertex has even degree.

Used for O(E) serialization of the entire Knowledge Graph — visiting every
relationship exactly once for efficient checkpointing.

**Module**: `graph_theory_primitives.euler_tour()`

### Graph Coloring & Chromatic Scheduling

A **proper k-coloring** assigns one of k colors to each vertex such that no
two adjacent vertices share a color.  The **chromatic number χ(G)** is the
minimum k for which a proper k-coloring exists.

**Brooks' Theorem**: χ(G) ≤ Δ(G) + 1 where Δ is the maximum vertex degree
(with exceptions for complete graphs and odd cycles).

For agent scheduling, edges represent conflicts (two agents that cannot run
concurrently), and colors represent execution slots.  k colors = k parallel
execution slots.

**Module**: `graph_theory_primitives.chromatic_schedule()`, `chromatic_number_upper_bound()`

### Personalized PageRank

**PageRank** models a random walker on the graph.  At each step, the walker
either follows a random outgoing edge (with probability `damping`, typically
0.85) or "teleports" to a random node (with probability `1 - damping`).

The **stationary distribution** of this random walk gives the PageRank score
for each node — its long-run importance.

**Personalized PageRank** biases the teleport distribution toward specific
"seed" nodes, making the importance scores context-dependent.

**Module**: `graph_theory_primitives.personalized_pagerank()`

### Adjacency Matrix Power Theorem

For a graph with adjacency matrix A, the entry **(A^k)[i][j]** equals the
number of distinct walks of length k from vertex i to vertex j.

This enables answering queries like "How many reasoning paths of length 3
connect concept A to concept B?"

**Module**: `graph_theory_primitives.count_paths_of_length()`

---

## Embedding Alignment Diagnostics (KG-2.42)

**Source**: *MINER — Retrieval-Optimized VLMs* (arXiv:2605.06460v1)

### Centered Kernel Alignment (CKA)

CKA measures **structural similarity** between two embedding spaces at the
dataset level.  Unlike cosine similarity (which compares individual vectors),
CKA compares the *geometry* of entire representation spaces.

**Formula**: CKA(X, Y) = ||X^T Y||²_F / (||X^T X||_F · ||Y^T Y||_F)

Properties:
- Invariant to orthogonal transformations (rotations)
- Invariant to isotropic scaling
- Range: [0, 1] where 1 = structurally identical

**Module**: `EmbeddingDiagnostics.compute_cka()`

### Alignment Ratio (AR)

AR combines CKA (dataset-level structural similarity) with cosine similarity
(sample-level alignment):

**AR = mean_cosine_similarity / CKA_score**

- **High AR** (> 0.5): Embeddings are directly usable without transformation.
- **Low AR** (< 0.5): Structure exists but requires transformation to align.

**Module**: Part of `CKAResult` returned by `compute_cka()`

### Adaptive Sparse Fusion

When fusing embeddings from multiple sources (e.g., title embeddings + content
embeddings + graph-structural embeddings), **adaptive sparse fusion** applies:

1. **Neuron-level masking**: For each embedding layer, low-variance dimensions
   are masked out (set to zero).  Only high-information dimensions are retained.
2. **Cross-layer weighting**: Layers are weighted by their quality scores.
3. **Normalization**: Final fused embeddings are L2-normalized.

**Sparsity target**: Fraction of dimensions to mask.  Higher sparsity =
more compression but potential information loss.

**Module**: `EmbeddingDiagnostics.adaptive_sparse_fusion()`

### Embedding Health Monitor

Continuous monitoring of embedding quality through:
- **Effective dimensionality**: Via SVD, counts singular values that contribute
  meaningfully (> 1% of the maximum).
- **CKA drift**: Compares current embeddings against a baseline snapshot.
- **Severity levels**: "none", "mild", "severe" drift.

**Module**: `EmbeddingDiagnostics.embedding_health_check()`

---

## Structural Causal Reasoning (KG-2.43)

**Source**: *MedCausalX — Causal Reasoning* (arXiv:2603.23085v1)

### Structural Causal Model (SCM)

An SCM is a mathematical framework for representing causal relationships:

**M = ⟨V, U, F, P(U)⟩** where:
- **V**: Endogenous (observed) variables
- **U**: Exogenous (latent/unobserved) variables
- **F**: Structural equations (the causal mechanisms)
- **P(U)**: Probability distribution over exogenous variables

In agent-utilities, SCMs are implemented as directed acyclic graphs (DAGs)
where nodes are `CausalFactor` objects and edges are `CausalEdge` objects
with mechanism descriptions and strength weights.

**Module**: `StructuralCausalModel`

### do-Calculus & Interventions

Pearl's **do-operator** `do(X = x)` represents an intervention — physically
setting variable X to value x, as opposed to merely observing X = x.

The intervention is implemented by **graph mutilation**: removing all incoming
edges to the intervened variable while keeping outgoing edges.  This
distinguishes causal effects from correlations.

**Module**: `StructuralCausalModel.do_intervention()`

### d-Separation

Two variables X and Y are **d-separated** by a set Z in a DAG if every path
between X and Y is "blocked" by Z.  A path is blocked if it contains:

1. **Chain** (A → B → C) or **fork** (A ← B → C) with B ∈ Z
2. **Collider** (A → B ← C) with B ∉ Z and no descendant of B in Z

d-separation implies **conditional independence**: X ⊥ Y | Z

**Module**: `StructuralCausalModel.is_d_separated()`

### Causal Verification Protocol

Inspired by MedCausalX's `<causal>` and `<verify>` tokens.  Checks whether
a reasoning chain's intermediate steps maintain causal consistency with the
underlying SCM by verifying:

1. Causal direction matches the DAG
2. No reversed causality
3. Intermediate variables are not skipped

**Module**: `CausalVerifier.verify_chain()`

### Counterfactual Reasoning

"What if X had been different?"  Given a causal model and observed evidence,
generates counterfactual queries for each causal ancestor of a target variable,
sorted by causal proximity.

**Module**: `CounterfactualGenerator.generate_counterfactuals()`

### Spuriousness Detection

Identifies edges that represent **spurious correlations** rather than true
causal relationships, using the d-separation criterion.  An edge X → Y is
flagged as spurious if X and Y are d-separated when conditioning on the
parents of Y (potential confounders).

**Module**: `SpuriousnessDetector.detect_spurious_edges()`

---

## Latent Space Regularization (KG-2.44)

**Source**: *LeWorldModel* (arXiv:2603.19312v2)

### SIGReg Normality Test

**SIGReg** (Sigmoid Regularization) prevents embedding collapse by enforcing
Gaussian-distributed latent embeddings.

**Method**: Project all embeddings onto multiple random unit directions.
Each 1D projection should approximate a Gaussian distribution (by the Central
Limit Theorem for healthy embeddings).  A normality test (kurtosis + skewness)
is applied to each projection.  Failed tests indicate collapse.

**Module**: `LatentSpaceRegularizer.detect_collapse()`

### Embedding Collapse

**Collapse** occurs when a representation space loses its diversity — all
embeddings converge to a low-dimensional manifold or a single point.

Detection methods:
- **SVD effective dimensionality**: Count singular values > 1% of the maximum.
  If effective_dim / total_dim < threshold, collapse is detected.
- **Participation ratio**: (Σλᵢ)² / Σλᵢ² where λᵢ are eigenvalues.
  Low participation ratio = collapsed.

### Diversity-Preserving Consolidation

Extends **EWC (Elastic Weight Consolidation)** with a diversity constraint.
When consolidating an embedding:

1. Apply standard EWC: `new = old + delta × (1 - λ × Fisher)`
2. Check if the result reduces the overall participation ratio
3. If so, apply additional dampening to preserve diversity

**Module**: `LatentSpaceRegularizer.diversity_preserving_consolidation()`

### Predictive Consistency Score

Measures whether the KG's embedding-based predictions match observed outcomes.
Computed as the average cosine similarity between predicted and observed
embedding states across a sequence of actions.

**Module**: `LatentSpaceRegularizer.predictive_consistency_score()`

---

## Probabilistic Graph Reasoning (KG-2.45)

**Source**: *MCS* Chapters 17–21

### Bayes' Theorem

**P(H|E) = P(E|H) × P(H) / P(E)**

Updates a **prior** belief P(H) to a **posterior** belief P(H|E) given
observed evidence E.

In odds form: **O(H|E) = LR × O(H)** where LR = P(E|H) / P(E|¬H) is the
**likelihood ratio**.  LR > 1 supports the hypothesis; LR < 1 weakens it.

**Module**: `BayesianBeliefPropagator.observe_evidence()`

### Belief Propagation on Graphs

When a node's belief changes, the update propagates to neighbors through
graph edges.  The update strength decays exponentially with graph distance:

**dampened_LR = 1 + (LR - 1) × decay^depth**

**Module**: `BayesianBeliefPropagator.propagate()`

### Law of Total Probability

**P(B) = Σᵢ P(B|Aᵢ) × P(Aᵢ)** where {Aᵢ} partitions the sample space.

Applied to multi-source retrieval: combines relevance scores from multiple
KG sources weighted by each source's reliability.  Avoids **Simpson's Paradox**
(where aggregate trends reverse when data is partitioned by a confounding
variable).

**Module**: `total_probability_aggregation()`

### Birthday Paradox

In a set of n randomly chosen items from d possibilities, the probability
of at least one **collision** (duplicate) is:

**P(collision) ≈ 1 - e^(-n²/2d)**

The collision probability exceeds 50% when **n ≈ 1.2√d**.

Applied to KG: estimates the probability of hash collisions in node IDs,
enabling probabilistic deduplication that's faster than pairwise comparison.

**Module**: `birthday_collision_probability()`

### Random Walk Exploration

Stochastic KG exploration using random walks with restart.  At each step:
- With probability `1 - restart_prob`: follow a random outgoing edge
- With probability `restart_prob`: teleport back to the start node

**Surprise score** for each discovered node = visit_frequency × graph_distance.
High surprise = frequently visited but far away = novel connection.

**Module**: `RandomWalkExplorer.explore()`, `discover_unexpected_connections()`

---

## Optimal Execution & Market Making (KG-2.46)

**Source**: *High-Frequency Trading* lecture notes (Drissi, Oxford-Man Institute, 2024)

### Implementation Shortfall

The difference between the **decision price** (price when you decide to trade)
and the **actual average execution price**.  This is the total cost of trading,
composed of:

- **Permanent market impact**: Price change that persists (information leakage)
- **Temporary market impact**: Price change that reverses (liquidity displacement)
- **Timing risk**: Variance due to price volatility during execution

### Limit Order Book (LOB)

An electronic book of resting buy/sell orders organized by price and time
priority.  Key concepts:

- **Bid**: Highest price a buyer is willing to pay
- **Ask**: Lowest price a seller is willing to accept
- **Spread**: Ask - Bid (the cost of immediacy)
- **Mid-price**: (Bid + Ask) / 2
- **Market order**: Executes immediately at best available price (takes liquidity)
- **Limit order**: Rests in the book until matched or cancelled (provides liquidity)

### Almgren-Chriss Model

The foundational optimal execution model.  Minimizes:

**min E[Cost] + λ × Var[Cost]**

where λ is the **risk aversion** parameter.

**Discrete solution** (Ch 3): Trade schedule uses hyperbolic functions:
`nⱼ = X × sinh(κ(N-j)) / sinh(κN)`

where κ = arccosh(1 + τ²σ²λ/(2η)) is the risk parameter, X is total shares,
N is number of time steps.

**Continuous solution** (Ch 4): Remaining inventory trajectory:
`x(t) = X × sinh(κ(T-t)) / sinh(κT)`

where κ = √(λσ²/η).  Solved via the **Hamilton-Jacobi-Bellman (HJB)** equation.

**Module**: `AlmgrenChrissDiscrete`, `AlmgrenChrissContinuous`

### Hamilton-Jacobi-Bellman (HJB) Equation

The fundamental PDE for **optimal control** in continuous time:

**0 = ∂V/∂t + min_u [f(x,u,t) + ∂V/∂x × g(x,u,t) + ½σ²∂²V/∂x²]**

where V is the value function, u is the control (trading rate), x is the state
(inventory), and σ is volatility.

### Cartea-Jaimungal Framework

Extends Almgren-Chriss with a **running inventory penalty** φq²(t) that
penalizes holding inventory over time.  This produces more aggressive early
liquidation.  The optimal trading rate solves a **Riccati ODE**:

**h'(t) = -(φ + σ²h²(t)/(2η))**

with terminal condition h(T) = α (terminal penalty).

**Module**: `CarteaJaimungalExecutor`

### Riccati ODE

A nonlinear ordinary differential equation of the form:

**y' = q₀(t) + q₁(t)y + q₂(t)y²**

Arises naturally in optimal control problems.  In the Cartea-Jaimungal framework,
the Riccati ODE determines the time-varying optimal trading aggressiveness.

### Avellaneda-Stoikov Market Making

Computes optimal bid and ask quotes for a market maker:

- **Reservation price**: `r = s - q × γ × σ² × (T-t)` where s is mid-price,
  q is inventory, γ is risk aversion.  Long inventory → lower reservation
  (incentivize selling).
- **Optimal spread**: `δ = γσ²(T-t) + (2/γ)ln(1 + γ/k)` where k is the
  order arrival intensity.

**Module**: `AvellanedaStoikovMarketMaker`

### Cointegration & Pairs Trading

Two price series are **cointegrated** if a linear combination of them is
stationary (mean-reverting).

The spread is modeled as an **Ornstein-Uhlenbeck (OU) process**:

**dXₜ = θ(μ - Xₜ)dt + σdWₜ**

- **θ**: Mean-reversion speed (how fast the spread returns to the mean)
- **μ**: Long-term mean level
- **σ**: Volatility of the spread
- **Half-life**: Time to revert halfway = ln(2)/θ

**Trading signals** are generated using **z-scores**:
`z = (current_spread - μ) / σ`

- |z| > entry_threshold → enter mean-reversion trade
- |z| < exit_threshold → exit position

**Module**: `CointegrationPairsTrader`

### Signal-Adaptive Execution

Incorporates predictive signals (e.g., order flow imbalance, MACD) into the
execution schedule.  The Almgren-Chriss schedule is adjusted by:

`adjusted_qty = base_qty × (1 - signal_weight × tanh(signal))`

- Positive signal (favorable) → delay trading (reduce quantity)
- Negative signal (unfavorable) → accelerate trading (increase quantity)

The total shares are preserved by rescaling after adjustment.

**Module**: `SignalAdaptiveExecutor`

---

## Source Materials

| ID | Title | Authors | Type |
|---|---|---|---|
| MCS | Mathematics for Computer Science | Lehman, Leighton, Meyer (MIT) | Textbook, 1000+ pages |
| MINER | Multi-Layer Internal Representation Mining for Efficient Retrieval | arXiv:2605.06460v1 | Research Paper |
| MedCausalX | Causally-Grounded Medical AI with Adaptive Reflection | arXiv:2603.23085v1 | Research Paper |
| LeWM | Stable End-to-End JEPA from Pixels | arXiv:2603.19312v2 | Research Paper |
| Oxford HFT | High-Frequency Trading Lecture Notes | Drissi, Oxford-Man Institute, 2024 | Lecture Notes |
