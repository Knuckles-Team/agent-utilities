# Model Display Optimization (CONCEPT:KG-2.17)

## Overview
Display-predict decoupling engine: optimizes model `__str__()` for LLM consumption independently of `predict()` logic. 5 strategies (linear_collapse, piecewise_table, symbolic_equation, coefficient_summary, adaptive/SmartAdditive). Bounded complexity budgets. Based on arXiv:2605.03808.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/model_display.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
