# Test contract: agent control plane

| Scenario | Layer and fixture | Expected result |
|---|---|---|
| Typed task with published components | generated-client contract + served in-memory EG | committed graph/decision; digest and why-not recorded |
| Unknown or free-text-only task | unit and served negative | labelled claim or typed abstention; no execution |
| Missing required tool, skill, trace fidelity, budget meter, or account mode | each adapter conformance | refusal before launch; no side effect |
| Planted permitted and forbidden tools plus planted skill | all five adapter fixture launches | permitted tool/skill observed, forbidden tool absent |
| Sandbox and remote mode claims | port contract | no local lease for provider-managed remote; untrusted code refused on weak mode |
| Capacity denial during multi-cell admission | graph-os/AU integration | partial leases released, at most one new decision, no heuristic execution |
| Harness without enforceable child cap | adapter negative | children disabled or launch refused by policy |
| Continuation asks for greater width/depth/rounds | topology unit | refuse; new decision and lease needed |
| Fence loss, timeout after dispatch, missing event, unavailable usage | fault injection | stop/reconcile; `outcome-uncertain`/`trace-incomplete`; no blind retry or zero cost |
| Self-reported success or unpromoted model | routing negative | no selection authority or promotion |
| Agent saved through the library | `tests/unit/orchestration/test_agent_library.py` | listed and read back with every field; runnable-skill contract kept |
| Library agent through the intent router | `tests/unit/orchestration/test_agent_library_intent.py` | `find` ranks `agent_library`; `ask` lists it; `ask` cannot save; `manage` saves after a previewed plan |
| Packaged prompt with and without a role | `tests/unit/orchestration/test_agent_library.py` | role blueprint becomes a role agent with package tools; no role, no agent |
| Solved, uncommitted and failed-publish assemblies | `tests/unit/decide/test_assembly_consumer.py` | committed graph is published and saved; uncommitted graph is saved only; publish failure keeps the agent |

Fresh checkout: `python3 scripts/uv_workspace.py doctor`; install locked extras using the repository helper, then run `python3 scripts/uv_workspace.py run --all-extras pytest tests/unit tests/orchestration -q` and the focused new conformance tests. Tests must provide fake EG/gateway/provider adapters and temporary directories; ordinary PR checks must need no pre-existing deployment, vendor account or GPU. Live certification is a separately labelled optional environment test with captured receipts.

Quality: run `python3 scripts/safe_precommit_all_files.py` through the repo lease wrapper when the shared runner is available, plus the configured CCCC, jscpd and Dupehound gates. `pyproject.toml` pins Dupehound 0.1.2 (similarity 0.85, 40-token minimum) and jscpd 5.0.16 (50-token/5-line minimum, mild mode); use configured differential exclusions and no new duplicate findings. CCCC uses the repository's current hook configuration; do not invent a numeric threshold. Apply KISS: reuse EG decision/capacity, existing graph entry points and one adapter conformance suite; reject a second optimizer, event store or scheduler. A local missing tool is reported as not run, never pass.
