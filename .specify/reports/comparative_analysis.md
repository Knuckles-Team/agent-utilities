# Comparative Analysis Report

**Date**: 2026-05-08 17:42 UTC
**Projects Analyzed**: 3
**Projects**: agent-utilities, agent-runtimes, unknown

## Executive Summary

**Overall Leader**: **agent-utilities** (GPA: 73.4)
- agent-utilities: **73.4**/100 weighted GPA
- agent-runtimes: **68.3**/100 weighted GPA
- unknown: **0.0**/100 weighted GPA

## Comparison Matrix

| Domain |**agent-utilities** | **agent-runtimes** | **unknown** | Winner |
|--------|--- | --- | --- | ------ |
| Governance | 45 (F) | 50 (F) | — (—) | agent-runtimes |
| Ecosystem Health | 60 (D) | 35 (F) | — (—) | agent-utilities |
| Architecture | 90 (A) | 96 (A+) | — (—) | agent-runtimes |
| Code Quality | 45 (F) | 45 (F) | — (—) | agent-utilities |
| Security | 100 (A+) | 95 (A+) | — (—) | agent-utilities |
| Testing | 95 (A+) | 72 (C) | — (—) | agent-utilities |
| Documentation | 83 (B) | 78 (C+) | — (—) | agent-utilities |
| Performance | 45 (F) | 45 (F) | — (—) | agent-utilities |
| **Weighted GPA** |**73.4** | **68.3** | **0.0** | **agent-utilities** |

## Visual Comparison

```mermaid
%%{init: {'theme': 'dark'}}%%
radar-beta
  title Comparative Analysis Radar
  axis Gov, Health, Arch, Quality, Security, Test, Docs, Perf
  "agent-utilities" : [45, 60, 90, 45, 100, 95, 83, 45]
  "agent-runtimes" : [50, 35, 96, 45, 95, 72, 78, 45]
  "unknown" : [0, 0, 0, 0, 0, 0, 0, 0]
```

## Domain Deep Dives

### CA-001: Governance

**Winner**: agent-runtimes (50/100, +5 delta)

#### agent-utilities
- **Score**: 45/100 (F)
  - Green license (MIT): +30
  - LICENSE file present: +5
  - 2+ contributors (4): +5
  - Bus factor 1 (1): +5

#### agent-runtimes
- **Score**: 50/100 (F)
  - Green license (MIT): +30
  - LICENSE file present: +5
  - 5+ contributors (5): +10
  - Bus factor 1 (1): +5

#### unknown
- **Score**: N/A/100 (N/A)

### CA-002: Ecosystem Health

**Winner**: agent-utilities (60/100, +25 delta)

#### agent-utilities
- **Score**: 60/100 (D)
  - High velocity (105 commits/90d): +15
  - Multiple releases (63 tags): +10
  - Strong SemVer compliance (100.0%): +15
  - CI configured (github_actions): +15
  - Mostly unpinned (0.0%): +5

#### agent-runtimes
- **Score**: 35/100 (F)
  - Moderate velocity (43 commits/90d): +10
  - CI configured (github_actions): +15
  - Multiple workflows (10): +5
  - Mostly unpinned (1.6%): +5

#### unknown
- **Score**: N/A/100 (N/A)

### CA-003: Architecture

**Winner**: agent-runtimes (96/100, +6 delta)

#### agent-utilities
- **Score**: 90/100 (A)
  - Rich protocol support (6 protocols): +30
  - Good type coverage (56.9%): +15
  - Well-structured (41 packages): +15
  - Appropriate nesting depth (4): +10
  - 12-Factor signals (5/6): +20

#### agent-runtimes
- **Score**: 96/100 (A+)
  - Rich protocol support (5 protocols): +30
  - Excellent type coverage (98.6%): +25
  - Well-structured (44 packages): +15
  - Appropriate nesting depth (3): +10
  - 12-Factor signals (4/6): +16

#### unknown
- **Score**: N/A/100 (N/A)

### CA-004: Code Quality

**Winner**: agent-utilities (45/100, +0 delta)

#### agent-utilities
- **Score**: 45/100 (F)
  - Good complexity (4.21): no penalty
  - Many long functions (263): -15
  - Many stubs (112): -20
  - High duplication (22.3%): -20

#### agent-runtimes
- **Score**: 45/100 (F)
  - Good complexity (4.65): no penalty
  - Many long functions (249): -15
  - Many stubs (111): -20
  - High duplication (25.2%): -20

#### unknown
- **Score**: N/A/100 (N/A)

### CA-005: Security

**Winner**: agent-utilities (100/100, +5 delta)

#### agent-utilities
- **Score**: 100/100 (A+)
  - Input validation (Pydantic): +5
  - Auth framework detected: +5
  - Security linter in pipeline: +5

#### agent-runtimes
- **Score**: 95/100 (A+)
  - Hardcoded secrets detected (1): -20
  - Input validation (Pydantic): +5
  - Auth framework detected: +5
  - Security linter in pipeline: +5

#### unknown
- **Score**: N/A/100 (N/A)

### CA-006: Testing

**Winner**: agent-utilities (95/100, +23 delta)

#### agent-utilities
- **Score**: 95/100 (A+)
  - Comprehensive test suite (203 files): +20
  - Many tests (2933): +20
  - Good ratio (0.52:1): +15
  - Multi-layer testing (unit:158, integ:42, e2e:0): +15
  - Proper testing pyramid shape: +5
  - Quality indicators (20/20): +20

#### agent-runtimes
- **Score**: 72/100 (C)
  - Comprehensive test suite (24 files): +20
  - Many tests (159): +20
  - Multi-layer testing (unit:14, integ:2, e2e:0): +15
  - Proper testing pyramid shape: +5
  - Quality indicators (12/20): +12

#### unknown
- **Score**: N/A/100 (N/A)

### CA-007: Documentation

**Winner**: agent-utilities (83/100, +5 delta)

#### agent-utilities
- **Score**: 83/100 (B)
  - README quality (105/100 → 37pts)
  - Good docstring coverage (62.6%): +18
  - docs_directory: +10
  - changelog: +8
  - examples_directory: +8
  - agents_md: +2

#### agent-runtimes
- **Score**: 78/100 (C+)
  - README quality (77/100 → 27pts)
  - Excellent docstring coverage (78.0%): +25
  - docs_directory: +10
  - changelog: +8
  - examples_directory: +8

#### unknown
- **Score**: N/A/100 (N/A)

### CA-008: Performance

**Winner**: agent-utilities (45/100, +0 delta)

#### agent-utilities
- **Score**: 45/100 (F)
  - Benchmark suite present: +25
  - Very heavy deps (181): +5
  - Partial async (22.2%): +15

#### agent-runtimes
- **Score**: 45/100 (F)
  - Benchmark suite present: +25
  - Very heavy deps (124): +5
  - Partial async (30.4%): +15

#### unknown
- **Score**: N/A/100 (N/A)

## Winner Summary

| Domain | Winner | Score | Delta |
|--------|--------|-------|-------|
| Governance | agent-runtimes | 50 | +5 |
| Ecosystem Health | agent-utilities | 60 | +25 |
| Architecture | agent-runtimes | 96 | +6 |
| Code Quality | agent-utilities | 45 | +0 |
| Security | agent-utilities | 100 | +5 |
| Testing | agent-utilities | 95 | +23 |
| Documentation | agent-utilities | 83 | +5 |
| Performance | agent-utilities | 45 | +0 |

## Recommendations

Based on the analysis, the following integration opportunities exist:

### agent-utilities — Areas for Improvement
- Governance (45/100)
- Ecosystem Health (60/100)
- Code Quality (45/100)
- Performance (45/100)

### agent-runtimes — Areas for Improvement
- Governance (50/100)
- Ecosystem Health (35/100)
- Code Quality (45/100)
- Performance (45/100)
