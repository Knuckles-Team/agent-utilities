# Comparative Analysis Report

**Date**: 2026-05-06 05:30 UTC
**Projects Analyzed**: 2
**Projects**: agent-utilities, unknown

## Executive Summary

**Overall Leader**: **agent-utilities** (GPA: 73.4)
- agent-utilities: **73.4**/100 weighted GPA
- unknown: **0.0**/100 weighted GPA

## Comparison Matrix

| Domain |**agent-utilities** | **unknown** | Winner |
|--------|--- | --- | ------ |
| Governance | 45 (F) | — (—) | agent-utilities |
| Ecosystem Health | 60 (D) | — (—) | agent-utilities |
| Architecture | 90 (A) | — (—) | agent-utilities |
| Code Quality | 45 (F) | — (—) | agent-utilities |
| Security | 100 (A+) | — (—) | agent-utilities |
| Testing | 95 (A+) | — (—) | agent-utilities |
| Documentation | 83 (B) | — (—) | agent-utilities |
| Performance | 45 (F) | — (—) | agent-utilities |
| **Weighted GPA** |**73.4** | **0.0** | **agent-utilities** |

## Visual Comparison

```mermaid
%%{init: {'theme': 'dark'}}%%
radar-beta
  title Comparative Analysis Radar
  axis Gov, Health, Arch, Quality, Security, Test, Docs, Perf
  "agent-utilities" : [45, 60, 90, 45, 100, 95, 83, 45]
  "unknown" : [0, 0, 0, 0, 0, 0, 0, 0]
```

## Domain Deep Dives

### CA-001: Governance

**Winner**: agent-utilities (45/100, +0 delta)

#### agent-utilities
- **Score**: 45/100 (F)
  - Green license (MIT): +30
  - LICENSE file present: +5
  - 2+ contributors (4): +5
  - Bus factor 1 (1): +5

#### unknown
- **Score**: N/A/100 (N/A)

### CA-002: Ecosystem Health

**Winner**: agent-utilities (60/100, +0 delta)

#### agent-utilities
- **Score**: 60/100 (D)
  - High velocity (98 commits/90d): +15
  - Multiple releases (57 tags): +10
  - Strong SemVer compliance (100.0%): +15
  - CI configured (github_actions): +15
  - Mostly unpinned (0.0%): +5

#### unknown
- **Score**: N/A/100 (N/A)

### CA-003: Architecture

**Winner**: agent-utilities (90/100, +0 delta)

#### agent-utilities
- **Score**: 90/100 (A)
  - Rich protocol support (6 protocols): +30
  - Good type coverage (60.7%): +15
  - Well-structured (35 packages): +15
  - Appropriate nesting depth (4): +10
  - 12-Factor signals (5/6): +20

#### unknown
- **Score**: N/A/100 (N/A)

### CA-004: Code Quality

**Winner**: agent-utilities (45/100, +0 delta)

#### agent-utilities
- **Score**: 45/100 (F)
  - Good complexity (4.35): no penalty
  - Many long functions (195): -15
  - Many stubs (103): -20
  - High duplication (24.2%): -20

#### unknown
- **Score**: N/A/100 (N/A)

### CA-005: Security

**Winner**: agent-utilities (100/100, +0 delta)

#### agent-utilities
- **Score**: 100/100 (A+)
  - Input validation (Pydantic): +5
  - Auth framework detected: +5
  - Security linter in pipeline: +5

#### unknown
- **Score**: N/A/100 (N/A)

### CA-006: Testing

**Winner**: agent-utilities (95/100, +0 delta)

#### agent-utilities
- **Score**: 95/100 (A+)
  - Comprehensive test suite (165 files): +20
  - Many tests (2251): +20
  - Good ratio (0.61:1): +15
  - Multi-layer testing (unit:120, integ:42, e2e:0): +15
  - Proper testing pyramid shape: +5
  - Quality indicators (20/20): +20

#### unknown
- **Score**: N/A/100 (N/A)

### CA-007: Documentation

**Winner**: agent-utilities (83/100, +0 delta)

#### agent-utilities
- **Score**: 83/100 (B)
  - README quality (105/100 → 37pts)
  - Good docstring coverage (65.6%): +18
  - docs_directory: +10
  - changelog: +8
  - examples_directory: +8
  - agents_md: +2

#### unknown
- **Score**: N/A/100 (N/A)

### CA-008: Performance

**Winner**: agent-utilities (45/100, +0 delta)

#### agent-utilities
- **Score**: 45/100 (F)
  - Benchmark suite present: +25
  - Very heavy deps (174): +5
  - Partial async (27.7%): +15

#### unknown
- **Score**: N/A/100 (N/A)

## Winner Summary

| Domain | Winner | Score | Delta |
|--------|--------|-------|-------|
| Governance | agent-utilities | 45 | +0 |
| Ecosystem Health | agent-utilities | 60 | +0 |
| Architecture | agent-utilities | 90 | +0 |
| Code Quality | agent-utilities | 45 | +0 |
| Security | agent-utilities | 100 | +0 |
| Testing | agent-utilities | 95 | +0 |
| Documentation | agent-utilities | 83 | +0 |
| Performance | agent-utilities | 45 | +0 |

## Recommendations

Based on the analysis, the following integration opportunities exist:

### agent-utilities — Areas for Improvement
- Governance (45/100)
- Ecosystem Health (60/100)
- Code Quality (45/100)
- Performance (45/100)
