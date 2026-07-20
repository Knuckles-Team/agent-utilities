# Comparative Analysis Report\n
**Date**: 2026-05-24 04:38 UTC\n**Projects Analyzed**: 2\n**Projects**: agent-utilities, activegraph-monid\n
## Executive Summary
\n**Overall Leader**: **agent-utilities** (GPA: 69.4)\n- agent-utilities: **69.4**/100 weighted GPA\n- activegraph-monid: **46.5**/100 weighted GPA\n
## Comparison Matrix
\n| Domain |**agent-utilities** | **activegraph-monid** | Winner |\n|--------|--- | --- | ------ |\n| Governance | 45 (F) | 5 (F) | agent-utilities |\n| Ecosystem Health | 65 (D) | 10 (F) | agent-utilities |\n| Architecture | 100 (A) | 53 (F) | agent-utilities |\n| Code Quality | 45 (F) | 75 (C+) | activegraph-monid |\n| Security | 65 (D) | 100 (A+) | activegraph-monid |\n| Testing | 90 (A) | 24 (F) | agent-utilities |\n| Documentation | 83 (B) | 33 (F) | agent-utilities |\n| Performance | 40 (F) | 25 (F) | agent-utilities |\n| Architecture Differential | — (—) | — (—) | agent-utilities |\n| Innovation Extraction | — (—) | — (—) | agent-utilities |\n| **Weighted GPA** |**69.4** | **46.5** | **agent-utilities** |\n
## Visual Comparison
\n```mermaid
%%{init: {'theme': 'dark'}}%%
radar-beta
  title Comparative Analysis Radar
  axis Gov, Health, Arch, Quality, Security, Test, Docs, Perf, Archit, Innova
  "agent-utilities" : [45, 65, 100, 45, 65, 90, 83, 40, 0, 0]
  "activegraph-monid" : [5, 10, 53, 75, 100, 24, 33, 25, 0, 0]
```\n
## Domain Deep Dives
\n### CA-001: Governance
\n**Winner**: agent-utilities (45/100, +40 delta)\n
#### agent-utilities\n- **Score**: 45/100 (F)\n  - Green license (MIT): +30\n  - LICENSE file present: +5\n  - 2+ contributors (4): +5\n  - Bus factor 1 (1): +5\n
#### activegraph-monid\n- **Score**: 5/100 (F)\n  - No license or unrecognized: +0\n  - Bus factor 1 (1): +5\n\n### CA-002: Ecosystem Health
\n**Winner**: agent-utilities (65/100, +55 delta)\n
#### agent-utilities\n- **Score**: 65/100 (D)\n  - High velocity (130 commits/90d): +15\n  - Multiple releases (75 tags): +10\n  - Strong SemVer compliance (100.0%): +15\n  - CI configured (github_actions): +15\n  - Multiple workflows (3): +5\n  - Mostly unpinned (0.0%): +5\n
#### activegraph-monid\n- **Score**: 10/100 (F)\n  - Low velocity (5 commits/90d): +5\n  - Mostly unpinned (0.0%): +5\n\n### CA-003: Architecture
\n**Winner**: agent-utilities (100/100, +47 delta)\n
#### agent-utilities\n- **Score**: 100/100 (A)\n  - Rich protocol support (7 protocols): +30\n  - Good type coverage (58.0%): +15\n  - Well-structured (55 packages): +15\n  - Appropriate nesting depth (4): +10\n  - 12-Factor signals (5/6): +20\n  - Architecture depth bonus: +10\n
#### activegraph-monid\n- **Score**: 53/100 (F)\n  - Good type coverage (75.0%): +15\n  - Basic structure (3 packages): +10\n  - Appropriate nesting depth (3): +10\n  - 12-Factor signals (2/6): +8\n  - Architecture depth bonus: +10\n\n### CA-004: Code Quality
\n**Winner**: activegraph-monid (75/100, +30 delta)\n
#### agent-utilities\n- **Score**: 45/100 (F)\n  - Good complexity (4.08): no penalty\n  - Many long functions (465): -15\n  - Many stubs (194): -20\n  - High duplication (21.9%): -20\n
#### activegraph-monid\n- **Score**: 75/100 (C+)\n  - Good complexity (4.09): no penalty\n  - Many long functions (11): -15\n  - Moderate duplication (10.3%): -10\n\n### CA-005: Security
\n**Winner**: activegraph-monid (100/100, +35 delta)\n
#### agent-utilities\n- **Score**: 65/100 (D)\n  - Hardcoded secrets detected (2): -40\n  - Some findings (10): -10\n  - Input validation (Pydantic): +5\n  - Auth framework detected: +5\n  - Security linter in pipeline: +5\n
#### activegraph-monid\n- **Score**: 100/100 (A+)\n  - Input validation (Pydantic): +5\n\n### CA-006: Testing
\n**Winner**: agent-utilities (90/100, +66 delta)\n
#### agent-utilities\n- **Score**: 90/100 (A)\n  - Comprehensive test suite (284 files): +20\n  - Many tests (3708): +20\n  - Low ratio (0.42:1): +10\n  - Multi-layer testing (unit:235, integ:47, e2e:0): +15\n  - Proper testing pyramid shape: +5\n  - Quality indicators (20/20): +20\n
#### activegraph-monid\n- **Score**: 24/100 (F)\n  - Tests present (2 files): +10\n  - Unit tests present: +10\n  - Quality indicators (4/20): +4\n\n### CA-007: Documentation
\n**Winner**: agent-utilities (83/100, +50 delta)\n
#### agent-utilities\n- **Score**: 83/100 (B)\n  - README quality (105/100 → 37pts)\n  - Good docstring coverage (61.4%): +18\n  - docs_directory: +10\n  - changelog: +8\n  - examples_directory: +8\n  - agents_md: +2\n
#### activegraph-monid\n- **Score**: 33/100 (F)\n  - README quality (79/100 → 28pts)\n  - Minimal docstrings (26.4%): +5\n\n### CA-008: Performance
\n**Winner**: agent-utilities (40/100, +15 delta)\n
#### agent-utilities\n- **Score**: 40/100 (F)\n  - Benchmark suite present: +25\n  - Very heavy deps (261): +5\n  - Some async (19.6%): +10\n
#### activegraph-monid\n- **Score**: 25/100 (F)\n  - Moderate deps (19): +20\n  - No async patterns: +5\n\n### CA-003b: Architecture Differential
\n**Winner**: agent-utilities (0/100, +0 delta)\n
#### agent-utilities\n- **Score**: N/A/100 (N/A)\n
#### activegraph-monid\n- **Score**: N/A/100 (N/A)\n\n### CA-010: Innovation Extraction
\n**Winner**: agent-utilities (0/100, +0 delta)\n
#### agent-utilities\n- **Score**: N/A/100 (N/A)\n
#### activegraph-monid\n- **Score**: N/A/100 (N/A)\n\n## Winner Summary
\n| Domain | Winner | Score | Delta |\n|--------|--------|-------|-------|\n| Governance | agent-utilities | 45 | +40 |\n| Ecosystem Health | agent-utilities | 65 | +55 |\n| Architecture | agent-utilities | 100 | +47 |\n| Code Quality | activegraph-monid | 75 | +30 |\n| Security | activegraph-monid | 100 | +35 |\n| Testing | agent-utilities | 90 | +66 |\n| Documentation | agent-utilities | 83 | +50 |\n| Performance | agent-utilities | 40 | +15 |\n| Architecture Differential | agent-utilities | 0 | +0 |\n| Innovation Extraction | agent-utilities | 0 | +0 |\n\n## Innovations & Recommendations\n\n### Cross-Domain Synergies (agent-utilities)\n\n**From activegraph-monid** (Concept: ):\n- **variant_pool**: parametric exploration (Value: low)\n\n### agent-utilities — Areas for Improvement\n- Governance (45/100)\n- Ecosystem Health (65/100)\n- Code Quality (45/100)\n- Security (65/100)\n- Performance (40/100)\n\n### activegraph-monid — Areas for Improvement\n- Governance (5/100)\n- Ecosystem Health (10/100)\n- Architecture (53/100)\n- Testing (24/100)\n- Documentation (33/100)\n- Performance (25/100)\n
