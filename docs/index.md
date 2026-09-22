<section class="site-hero" aria-labelledby="agent-utilities-title">
  <p class="site-hero__eyebrow">Knuckles Team agent platform</p>
  <h1 id="agent-utilities-title" class="site-hero__title">Coordinate AI agents without blurring system boundaries</h1>
  <p class="site-hero__summary">
    Agent Utilities is the Python control plane for model selection, skills,
    teams, workflows, governed execution, evaluation, and outcome-driven
    improvement.
  </p>
  <div class="site-hero__actions">
    <a class="md-button md-button--primary" href="guides/quick-start/">Run the quick start</a>
    <a class="md-button" href="architecture/">Explore the architecture</a>
  </div>
</section>

<div class="site-card-grid">
  <article class="site-card">
    <h2 class="site-card__title">Build an agent</h2>
    <p class="site-card__body">Create a provider-aware agent, attach typed tools and reusable skills, and run it through one governed harness.</p>
    <a href="guides/creating-an-agent/">Create your first agent →</a>
  </article>
  <article class="site-card">
    <h2 class="site-card__title">Coordinate work</h2>
    <p class="site-card__body">Route goals through planners, teams, durable loops, budgets, checkpoints, approvals, and observable execution.</p>
    <a href="guides/kg_native_orchestration/">Understand orchestration →</a>
  </article>
  <article class="site-card">
    <h2 class="site-card__title">Operate the control plane</h2>
    <p class="site-card__body">Configure identity, policies, providers, health checks, metrics, and deployment profiles without embedding environment secrets.</p>
    <a href="guides/deployment-configurations/">Choose a deployment →</a>
  </article>
</div>

## Runtime map

![Agent platform runtime architecture](assets/runtime-architecture.svg)

Agent Utilities occupies the agent-control boundary in the shared runtime. It
coordinates model and worker activity, reads and writes committed state through
Epistemic Graph contracts, and is exposed to clients through GraphOS. Connector
adapters reach external systems through the Agent Connector SDK, while Agent
WebUI presents the browser experience.

<div class="site-ownership">
  <div class="site-ownership__grid">
    <div class="site-ownership__item">
      <div class="site-ownership__label">Agent Utilities</div>
      <div class="site-ownership__value">Agents, routing, skills, workflows, execution policy, evaluation, and control-plane telemetry.</div>
    </div>
    <div class="site-ownership__item">
      <div class="site-ownership__label">Epistemic Graph</div>
      <div class="site-ownership__value">Durable graph, GraphSchema, ontology and SHACL authority, query, reasoning, provenance, and work state.</div>
    </div>
    <div class="site-ownership__item">
      <div class="site-ownership__label">GraphOS</div>
      <div class="site-ownership__value">Public MCP, REST, and A2A runtime; identity, deployment, and service composition.</div>
    </div>
    <div class="site-ownership__item">
      <div class="site-ownership__label">Connector SDK and WebUI</div>
      <div class="site-ownership__value">Governed source integration and the browser-facing application experience.</div>
    </div>
  </div>
</div>

!!! info "One contract at each boundary"
    The control plane does not become a second graph engine, public gateway,
    connector framework, or frontend. Typed contracts keep each authority
    independently testable and replaceable.

## How work moves

<ol class="site-flow">
  <li class="site-flow__step">
    <strong class="site-flow__title">Admit a goal.</strong>
    <span class="site-flow__body">GraphOS authenticates the caller and passes a scoped request into the control plane.</span>
  </li>
  <li class="site-flow__step">
    <strong class="site-flow__title">Compile context.</strong>
    <span class="site-flow__body">Agent Utilities selects models, skills, tools, and bounded graph evidence for the task.</span>
  </li>
  <li class="site-flow__step">
    <strong class="site-flow__title">Execute under policy.</strong>
    <span class="site-flow__body">Planners, agents, teams, and workflows operate within budgets, approvals, and safety constraints.</span>
  </li>
  <li class="site-flow__step">
    <strong class="site-flow__title">Commit and learn.</strong>
    <span class="site-flow__body">Outcomes, traces, and approved state changes return through authoritative contracts for replay and evaluation.</span>
  </li>
</ol>

## Explore the platform

| I want to… | Start here |
|---|---|
| Install and run a local evaluation | [Quick start](guides/quick-start.md) |
| Build an agent in Python | [Creating an agent](guides/creating-an-agent.md) |
| Choose library, service, or remote consumption | [Consumption models](guides/consumption-models.md) |
| Understand control-plane internals | [Architecture](architecture/index.md) |
| Configure a deployment | [Deployment configurations](guides/deployment-configurations.md) |
| Operate and observe the runtime | [Observability and usage](guides/observability-usage-tracking.md) |
| Browse generated Python APIs | [API reference](reference/api.md) |
| Check what ships in this release | [Status](status.md) |

!!! tip "Using the whole ecosystem?"
    Start with [GraphOS](https://knuckles-team.github.io/graph-os/) for the
    composed public runtime. Use these Agent Utilities docs when you are
    building or extending agent behavior inside that runtime.

## Control-plane capabilities

The implementation is organized around five practical areas rather than one
monolithic agent loop:

- **Agent runtime** — model providers, structured output, toolsets, skills, and
  guardrails.
- **Orchestration** — routing, planning, teams, workflows, loops, checkpoints,
  and scheduling.
- **Context coordination** — bounded retrieval and typed interaction with
  committed graph state.
- **Evaluation and improvement** — replay, scoring, failure analysis, and
  review-gated proposals.
- **Operations and governance** — identity, policy, tenant isolation, metrics,
  health checks, and runtime configuration.

Continue to the [architecture reference](architecture/index.md) for the module
map, or browse the [documentation catalog](reference/documentation-catalog.md)
when you need a specialized operational guide.
