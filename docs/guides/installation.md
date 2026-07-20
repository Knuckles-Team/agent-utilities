# Installation Guide


## Installation

```bash
# Core Python library plus the mandatory epistemic-graph[full] CPU engine
pip install agent-utilities

# ---------------------------------------------------------
# 1. Agent & Orchestration Environments
# ---------------------------------------------------------
# Supported self-contained GraphOS serving runtime (recommended for graph-os)
pip install "agent-utilities[serving]"

# Interactive model orchestration, including terminal and AG-UI frontends
pip install "agent-utilities[agent-runtime]"

# Protocol adapters & UI
pip install agent-utilities[acp]        # Standardized ACP protocol
pip install agent-utilities[ag-ui]      # Agent WebUI streaming
pip install agent-utilities[terminal]   # Terminal UI

# Browser & Web Automation
pip install agent-utilities[browser]    # Playwright browser integration

# ---------------------------------------------------------
# 2. Model Providers (Slim dependencies)
# ---------------------------------------------------------
pip install agent-utilities[agent-anthropic]
pip install agent-utilities[agent-google]
pip install agent-utilities[agent-groq]
pip install agent-utilities[agent-mistral]
pip install agent-utilities[agent-huggingface]

# ---------------------------------------------------------
# 3. Alternative Knowledge Graph Backends
# ---------------------------------------------------------

pip install agent-utilities[neo4j]
pip install agent-utilities[falkordb]

# ---------------------------------------------------------
# 4. RAG & Embeddings
# ---------------------------------------------------------
# Base embedding support
pip install agent-utilities[embeddings]

# Provider-specific embeddings
pip install agent-utilities[embeddings-openai]
pip install agent-utilities[embeddings-huggingface]
pip install agent-utilities[embeddings-ollama]

# ---------------------------------------------------------
# 5. OWL Reasoning & Ontologies
# ---------------------------------------------------------
# Core OWL reasoning (Owlready2 + HermiT)
# Note: Requires Java Runtime Environment (sudo apt install default-jre)
pip install agent-utilities[owl]

# Stardog OWL backend
pip install agent-utilities[stardog]

# ---------------------------------------------------------
# 6. Tools & Infrastructure
# ---------------------------------------------------------
pip install agent-utilities[mcp]        # MCP Server hosting capabilities
pip install agent-utilities[logfire]    # Observability & Tracing
pip install agent-utilities[vault]      # HashiCorp Vault & OpenBao secrets
pip install agent-utilities[auth]       # JWT/OIDC support

# ---------------------------------------------------------
# 7. Everything
# ---------------------------------------------------------
# Install every optional integration only when the host needs all of them
pip install "agent-utilities[all]"
```

## `uvx`, Windows, WSL, and private TLS

Use the supported self-contained serving profile for GraphOS. It includes the full
engine runtime, MCP surface, headless agent, native program optimizer, Langfuse integration,
ingestion providers, auth, and metrics. Use ``[mcp]`` only when intentionally building
a minimal custom runtime that does not need those capabilities:

```powershell
uvx --refresh --from "agent-utilities[serving]>=1.27.1,<2.0.0" graph-os
```

Current releases use the native epistemic-graph optimizer and do not install a second
Python optimization stack. Agent Utilities also requires
`epistemic-graph[full]>=2.23.1,<3.0.0` in its base dependencies, so every extra,
including `serving`, receives the folded numeric kernel and full CPU engine runtime.
Use only this single base `epistemic-graph[full]` requirement; do not add a
second engine requirement.

The characteristic `litellm==1.92.0` / `maturin.build_wheel` / "Rust not found"
failure from `agent-utilities==1.26.4` has two causes, not a GraphOS runtime defect:

1. that older dependency graph selected `dspy-ai`, which selected LiteLLM, and uv
   found no compatible LiteLLM wheel for the target interpreter, so it entered a
   source build; and
2. the temporary build backend then tried to download Rust, but its HTTPX TLS
   client could not validate the enterprise interception chain.

Release 1.27 removes the duplicate DSPy/LiteLLM optimizer path. Refresh to the
approved current wheel set with the command above. If an internal package mirror
offers only 1.26.4, the remaining issue is release/mirror availability; installing
Rust treats the obsolete source-build symptom rather than restoring the current
dependency contract.

If uv cannot reach the package index on a TLS-inspecting enterprise network, allow uv
to use the operating-system certificate store:

```powershell
$env:UV_NATIVE_TLS = "true"
uvx --refresh --from "agent-utilities[serving]>=1.27.1,<2.0.0" graph-os
```

Package bootstrap happens before AgentConfig exists. If a Python build backend must
also trust a private CA, the launcher may point `SSL_CERT_FILE` at an
administrator-approved PEM trust store. A `REQUESTS_CA_BUNDLE` setting configures
Requests only and does not, by itself, configure HTTPX.

The PEM store must include every required intermediate;
a binary certificate file or root-only bundle is insufficient when the target server
or proxy does not present a complete chain. Once GraphOS starts, use
`TLS_PROFILES_REF` plus a purpose-specific selector such as
`ENGINE_TLS_PROFILE_REF`, `MODEL_TLS_PROFILE_REF`, or
`LANGFUSE_TLS_PROFILE_REF`. GraphOS resolves and projects the profile for Requests,
HTTPX, SSL, database drivers, and child MCP processes. Keep bundle material and its
location in the runtime secret system, never in package source or a committed MCP
config. Verify resolution with the corresponding `agent-utilities-doctor --only`
check.

For Langfuse, persist only `LANGFUSE_PUBLIC_KEY_REF` and
`LANGFUSE_SECRET_KEY_REF`. Select trust with `LANGFUSE_TLS_PROFILE` plus an
external `TLS_PROFILES_REF` catalog. GraphOS then materializes the Langfuse MCP
child environment in memory. Metadata-only OTLP tracing automatically reuses that
credential pair and TLS profile only when the collector has the same HTTPS origin.
A different collector uses `OTEL_EXPORTER_OTLP_HEADERS_REF` or the OTLP key-reference
pair together with `OTEL_TLS_PROFILE`; raw keys and authorization headers are not
durable AgentConfig fields.

Do not disable certificate verification or add an insecure host exception. If a
dependency genuinely has to build from source on Windows, install the approved Rust
and MSVC build toolchains in advance and configure their network trust separately.
