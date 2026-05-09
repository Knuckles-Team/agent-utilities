# Jailbreak Robustness Hardening (CONCEPT:OS-5.12)

## Overview
Extends Prompt Injection Scanner (OS-5.4) with 4-category jailbreak attack taxonomy from SoK research: template-based (DAN, AIM, UCAR, Grandma), optimization-based (GCG suffix, token smuggling), LLM-based (context confusion, multi-turn escalation), manual (role-play, authority override). 12 new threat patterns. Derived from SoK: Robustness against Jailbreak (arXiv:2605.05058v1).

## Implementation Details
- **Source Code**: ``agent_utilities/security/prompt_scanner.py``
- **Pillar**: OS

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
