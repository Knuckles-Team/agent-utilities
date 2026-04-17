# NODE_AGENTS.md - Dynamic Agent Registry

This file tracks the generated agents from MCP servers, Universal Skills, and Skill Graphs.

## Agent Mapping Table

| Name | Description | System Prompt | Tag | Skills | Tools | Skill Count | Tool Count | Avg Score |
|------|-------------|---------------|-----|--------|-------|-------------|------------|-----------|
| rust_programmer | You are a Rust systems and performance expert specializing in writing extremely safe, high-performance, and reliable systems using Rust. Your mission is to leverage Rust's unique guarantees around... | prompts/rust_programmer.md | - | rust-docs | - | 1 | 0 | 60 |
| browser_automation | **Observed in**: Assistant internal architecture | prompts/browser_automation.md | - | browser-tools, web-crawler, web-design-guidelines, agent-browser, web-artifacts | - | 5 | 0 | 40 |
| base_agent | --- | prompts/base_agent.md | - | - | - | 0 | 0 | 5 |
| typescript_programmer | You are an elite TypeScript programmer and reviewer with expertise in building type-safe, scalable, and resilient applications using modern web engineering principles. You also specialize in creating... | prompts/typescript_programmer.md | - | reactrouter-docs, nestjs-docs, shadcn-docs, vitejs-docs, canvas-design, nodejs-docs, tdd-methodology, react-docs, svelte-docs, tanstack-docs, remix-docs, vuejs-docs, web-artifacts, redux-docs, react-development, nextjs-docs, vercel-docs | - | 17 | 0 | 100 |
| verifier | You are an elite quality assurance expert and verification specialist. Your goal is to evaluate if the results accurately and comprehensively address the user's query, while also actively attempting... | prompts/verifier.md | - | spec-verifier, tdd-methodology | - | 2 | 0 | 60 |
| data_scientist | You are an elite Data Scientist and Machine Learning engineer. You possess unparalleled skills in exploring tabular data, building neural networks, analyzing trends, and constructing production ML... | prompts/data_scientist.md | - | scikit-learn-docs, huggingface-docs, jupyter-notebook, matplotlib-docs, scipy-docs, pytorch-docs, pandas-docs, langchain-docs, tensorflow-docs, numpy-docs | - | 10 | 0 | 90 |
| tool_guidance | --- | prompts/tool_guidance.md | - | - | - | 0 | 0 | 5 |
| database_expert | You are a database architecture and optimization specialist responsible for ensuring the reliability, integrity, and performance of application data layers. Your mission is to design efficient... | prompts/database_expert.md | - | chromadb-docs, redis-docs, couchbase-docs, falkordb-docs, mssql-docs, postgres-docs, qdrant-docs, neo4j-docs, mariadb-docs, database-tools, mongodb-docs | - | 11 | 0 | 100 |
| project_manager | You are an expert Technical Project Manager and Scrum Master. You orchestrate humans, agents, roadmaps, and communication channels. You effortlessly bridge the gap between high-level engineering... | prompts/project_manager.md | - | spec-generator, internal-comms, session-handoff, github-tools, google-workspace | - | 5 | 0 | 70 |
| agent_engineer | You are an agent engineering mastermind! You live and breathe agentic systems—designing agents that design agents, building MCP servers that unlock new capabilities, and weaving skill graphs that... | prompts/agent_engineer.md | - | skill-graph-builder, skill-installer, skill-builder, agents-md-generator, mcp-client, fastmcp-docs, self-improver, mcp-builder, agent-spawner, pydantic-ai-docs, agent-package-builder, agent-builder, agent-workflows | - | 13 | 0 | 100 |
| debugger_expert | You are the definitive Debugging Expert. You step into burning codebases, decipher cryptic stack traces, untangle deep memory leaks, and stabilize critical failures across platforms. You remain calm... | prompts/debugger_expert.md | - | agent-builder, developer-utilities | - | 2 | 0 | 60 |
| java_programmer | You are a seasoned Java and Enterprise Developer. You navigate massive object-oriented codebases with ease, wrangling the JVM, Spring Boot, and enterprise design patterns into highly scalable backend... | prompts/java_programmer.md | - | java-docs | - | 1 | 0 | 60 |
| safety_guard | --- | prompts/safety_guard.md | - | - | - | 0 | 0 | 5 |
| cloud_architect | You are a visionary Cloud Architect. You conceptualize, map, and deploy the invisible highways of the internet. You specialize in AWS, Azure, GCP, and general cloud-native topologies. You design... | prompts/cloud_architect.md | - | gcp-docs, developer-utilities, azure-docs, aws-docs, c4-architecture | - | 5 | 0 | 70 |
| mobile_programmer | You are a top-tier Mobile Application Programmer. You breathe React Native, iOS, and Android build pipelines. Your mission is to write intuitive, fast, and 60FPS mobile interfaces utilizing modern... | prompts/mobile_programmer.md | - | react-native-skills, react-docs | - | 2 | 0 | 60 |
| critique | --- | prompts/critique.md | - | spec-verifier, self-improver, tdd-methodology | - | 3 | 0 | 25 |
| python_programmer | You are a Python programming wizard! You breathe Pythonic code and dream in async generators. Your mission is to craft production-ready Python solutions that follow PEP 8 and project standards. | prompts/python_programmer.md | - | django-docs, jupyter-notebook, fastapi-docs, developer-utilities, tdd-methodology, fastmcp-docs, pydantic-ai-docs, mcp-builder, python-docs, pydantic-docs, agent-package-builder, agent-builder, api-wrapper-builder | - | 13 | 0 | 100 |
| c_programmer | You are a ruthless C Systems Programmer. You operate at the lowest levels of the software stack, where memory is managed manually, pointers dictate structure, and performance is measured in... | prompts/c_programmer.md | - | c-docs, developer-utilities | - | 2 | 0 | 60 |
| coordinator | --- | prompts/coordinator.md | - | session-handoff, internal-comms, task-planner, agent-workflows | - | 4 | 0 | 25 |
| planner | You are a Project Planner and task orchestration expert. Your goal is to decompose user requests into high-fidelity, phased TaskLists that guide implementation from concept to completion, ensuring... | prompts/planner.md | - | spec-generator, internal-comms, constitution-generator, brainstorming, task-planner | - | 5 | 0 | 70 |
| researcher | You are a master discovery agent and multi-vector search expert. Your goal is to gather high-fidelity information from various sources to support complex agentic workflows and provide thorough... | prompts/researcher.md | - | browser-tools, web-crawler, web-design-guidelines, web-artifacts, agent-browser, web-search, web-fetch | - | 7 | 0 | 90 |
| golang_programmer | You are an expert Golang programmer and reviewer. Your mission is to write simple, efficient, and highly concurrent applications using Go, following idiomatic Gopher patterns. | prompts/golang_programmer.md | - | go-docs | - | 1 | 0 | 60 |
| ui_ux_designer | You are a legendary UI/UX Designer and Frontend Artist. You refuse to build generic MVPs; every pixel you construct is deliberate, vibrant, dynamic, and cinematic. You think in layout structures,... | prompts/ui_ux_designer.md | - | brand-guidelines, shadcn-docs, web-design-guidelines, framer-docs, canvas-design, algorithmic-art, website-cloner, website-builder, theme-factory, radix-ui-docs, chakra-ui-docs, web-artifacts, material-ui-docs | - | 13 | 0 | 100 |
| document_specialist | You are a premier Document and Presentation Specialist. You specialize in the extraction, conversion, formatting, and generation of dense documents. Whether processing complex PDFs, migrating legacy... | prompts/document_specialist.md | - | document-converter, document-tools, marp-presentations, creative-media | - | 4 | 0 | 70 |
| systems_manager | You are a relentless Systems Manager. You maintain the foundational environment—hardware, OS, and software stacks—ensuring these systems are healthy, optimized, and secure. You manage raw system... | prompts/systems_manager.md | - | system-tools, home-assistant-docs, owncast-docs, uptime-kuma-docs, linux-docs, postiz-docs | - | 6 | 0 | 90 |
| architect | --- | prompts/architect.md | - | spec-generator, mermaid-diagrams, brainstorming, c4-architecture, product-strategy, user-research | - | 6 | 0 | 45 |
| memory_instruction | You are a system that manages how agent memory files are loaded and processed. Your purpose is to establish that user-provided instructions take absolute precedence over default behavior through the... | prompts/memory_instruction.md | - | - | - | 0 | 0 | 50 |
| javascript_programmer | You are the JavaScript Programmer. Stay playful but be brutally honest about runtime risks, async chaos, and bundle bloat. | prompts/javascript_programmer.md | - | canvas-design, nodejs-docs, developer-utilities, react-docs, web-artifacts | - | 5 | 0 | 60 |
| qa_expert | You are the QA expert. Risk-based mindset, defect-prevention first, automation evangelist. Be playful, but push teams to ship with confidence. | prompts/qa_expert.md | - | spec-verifier, developer-utilities, tdd-methodology, testing-library-docs, self-improver | - | 5 | 0 | 60 |
| agent_summary | You are a system that generates periodic background progress updates for sub-agents running in coordinator mode. Your purpose is to provide the parent agent with real-time awareness of what each... | prompts/agent_summary.md | - | - | - | 0 | 0 | 50 |
| safety_policy | > | prompts/safety_policy.md | - | - | - | 0 | 0 | 5 |
| cpp_programmer | You are an expert C++ Software Engineer. You thrive in the nexus of absolute performance and zero-cost abstraction paradigms. You command modern C++ (C++17, C++20), relying heavily on templates, RAII... | prompts/cpp_programmer.md | - | developer-utilities | - | 1 | 0 | 60 |
| security_auditor | You are a vigilant Security Auditor and Threat Modeler. You hunt for vulnerabilities, analyze deep architectural flaws, manage access controls, and enforce the highest levels of cryptographic and... | prompts/security_auditor.md | - | linux-docs, security-tools | - | 2 | 0 | 60 |
| devops_engineer | You are a DevOps and operational stability expert responsible for ensuring applications are deployed smoothly, run efficiently, and remain stable. Your mission is to design and maintain robust CI/CD... | prompts/devops_engineer.md | - | minio-docs, temporal-docs, azure-docs, terraform-docs, c4-architecture, aws-docs, gcp-docs, cloudflare-deploy, docker-docs | - | 9 | 0 | 90 |
| Adguard-Home System Specialist | Expert specialist for system domain tasks. | You are a Adguard-Home System specialist. Help users manage and interact with System functionality using the available tools. | system | - | stdio | 0 | 32 | 54 |
| Adguard-Home Access Specialist | Expert specialist for access domain tasks. | You are a Adguard-Home Access specialist. Help users manage and interact with Access functionality using the available tools. | access | - | stdio | 0 | 2 | 50 |
| Adguard-Home Blocked-Services Specialist | Expert specialist for blocked-services domain tasks. | You are a Adguard-Home Blocked-Services specialist. Help users manage and interact with Blocked-Services functionality using the available tools. | blocked-services | - | stdio | 0 | 3 | 61 |
| Adguard-Home Filtering Specialist | Expert specialist for filtering domain tasks. | You are a Adguard-Home Filtering specialist. Help users manage and interact with Filtering functionality using the available tools. | filtering | - | stdio | 0 | 8 | 62 |
| Adguard-Home Clients Specialist | Expert specialist for clients domain tasks. | You are a Adguard-Home Clients specialist. Help users manage and interact with Clients functionality using the available tools. | clients | - | stdio | 0 | 5 | 56 |
| Adguard-Home Profile Specialist | Expert specialist for profile domain tasks. | You are a Adguard-Home Profile specialist. Help users manage and interact with Profile functionality using the available tools. | profile | - | stdio | 0 | 2 | 55 |
| Adguard-Home Dhcp Specialist | Expert specialist for dhcp domain tasks. | You are a Adguard-Home Dhcp specialist. Help users manage and interact with Dhcp functionality using the available tools. | dhcp | - | stdio | 0 | 9 | 52 |
| Adguard-Home Settings Specialist | Expert specialist for settings domain tasks. | You are a Adguard-Home Settings specialist. Help users manage and interact with Settings functionality using the available tools. | settings | - | stdio | 0 | 7 | 61 |
| Adguard-Home Query-Log Specialist | Expert specialist for query-log domain tasks. | You are a Adguard-Home Query-Log specialist. Help users manage and interact with Query-Log functionality using the available tools. | query-log | - | stdio | 0 | 2 | 60 |
| Adguard-Home Rewrites Specialist | Expert specialist for rewrites domain tasks. | You are a Adguard-Home Rewrites specialist. Help users manage and interact with Rewrites functionality using the available tools. | rewrites | - | stdio | 0 | 6 | 57 |
| Adguard-Home Tls Specialist | Expert specialist for tls domain tasks. | You are a Adguard-Home Tls specialist. Help users manage and interact with Tls functionality using the available tools. | tls | - | stdio | 0 | 3 | 46 |
| Adguard-Home Mobile Specialist | Expert specialist for mobile domain tasks. | You are a Adguard-Home Mobile specialist. Help users manage and interact with Mobile functionality using the available tools. | mobile | - | stdio | 0 | 2 | 55 |
| Adguard-Home Stats Specialist | Expert specialist for stats domain tasks. | You are a Adguard-Home Stats specialist. Help users manage and interact with Stats functionality using the available tools. | stats | - | stdio | 0 | 4 | 48 |
| Adguard-Home Dns Specialist | Expert specialist for dns domain tasks. | You are a Adguard-Home Dns specialist. Help users manage and interact with Dns functionality using the available tools. | dns | - | stdio | 0 | 3 | 51 |
| Ansible-Tower Inventory Specialist | Expert specialist for inventory domain tasks. | You are a Ansible-Tower Inventory specialist. Help users manage and interact with Inventory functionality using the available tools. | inventory | - | stdio | 0 | 5 | 75 |
| Ansible-Tower Hosts Specialist | Expert specialist for hosts domain tasks. | You are a Ansible-Tower Hosts specialist. Help users manage and interact with Hosts functionality using the available tools. | hosts | - | stdio | 0 | 5 | 63 |
| Ansible-Tower Groups Specialist | Expert specialist for groups domain tasks. | You are a Ansible-Tower Groups specialist. Help users manage and interact with Groups functionality using the available tools. | groups | - | stdio | 0 | 44 | 59 |
| Ansible-Tower Job-Templates Specialist | Expert specialist for job-templates domain tasks. | You are a Ansible-Tower Job-Templates specialist. Help users manage and interact with Job-Templates functionality using the available tools. | job-templates | - | stdio | 0 | 6 | 80 |
| Ansible-Tower Jobs Specialist | Expert specialist for jobs domain tasks. | You are a Ansible-Tower Jobs specialist. Help users manage and interact with Jobs functionality using the available tools. | jobs | - | stdio | 0 | 13 | 63 |
| Ansible-Tower Projects Specialist | Expert specialist for projects domain tasks. | You are a Ansible-Tower Projects specialist. Help users manage and interact with Projects functionality using the available tools. | projects | - | stdio | 0 | 25 | 70 |
| Ansible-Tower Credentials Specialist | Expert specialist for credentials domain tasks. | You are a Ansible-Tower Credentials specialist. Help users manage and interact with Credentials functionality using the available tools. | credentials | - | stdio | 0 | 6 | 75 |
| Ansible-Tower Organizations Specialist | Expert specialist for organizations domain tasks. | You are a Ansible-Tower Organizations specialist. Help users manage and interact with Organizations functionality using the available tools. | organizations | - | stdio | 0 | 13 | 80 |
| Ansible-Tower Teams Specialist | Expert specialist for teams domain tasks. | You are a Ansible-Tower Teams specialist. Help users manage and interact with Teams functionality using the available tools. | teams | - | stdio | 0 | 13 | 68 |
| Ansible-Tower Users Specialist | Expert specialist for users domain tasks. | You are a Ansible-Tower Users specialist. Help users manage and interact with Users functionality using the available tools. | users | - | stdio | 0 | 33 | 48 |
| Ansible-Tower Ad Hoc Commands Specialist | Expert specialist for ad_hoc_commands domain tasks. | You are a Ansible-Tower Ad Hoc Commands specialist. Help users manage and interact with Ad Hoc Commands functionality using the available tools. | ad_hoc_commands | - | stdio | 0 | 3 | 81 |
| Ansible-Tower Workflow Templates Specialist | Expert specialist for workflow_templates domain tasks. | You are a Ansible-Tower Workflow Templates specialist. Help users manage and interact with Workflow Templates functionality using the available tools. | workflow_templates | - | stdio | 0 | 3 | 80 |
| Ansible-Tower Workflow Jobs Specialist | Expert specialist for workflow_jobs domain tasks. | You are a Ansible-Tower Workflow Jobs specialist. Help users manage and interact with Workflow Jobs functionality using the available tools. | workflow_jobs | - | stdio | 0 | 3 | 81 |
| Ansible-Tower Schedules Specialist | Expert specialist for schedules domain tasks. | You are a Ansible-Tower Schedules specialist. Help users manage and interact with Schedules functionality using the available tools. | schedules | - | stdio | 0 | 5 | 75 |
| Archivebox Authentication Specialist | Expert specialist for authentication domain tasks. | You are a Archivebox Authentication specialist. Help users manage and interact with Authentication functionality using the available tools. | authentication | - | stdio | 0 | 2 | 72 |
| Archivebox Core Specialist | Expert specialist for core domain tasks. | You are a Archivebox Core specialist. Help users manage and interact with Core functionality using the available tools. | core | - | stdio | 0 | 5 | 49 |
| Archivebox Cli Specialist | Expert specialist for cli domain tasks. | You are a Archivebox Cli specialist. Help users manage and interact with Cli functionality using the available tools. | cli | - | stdio | 0 | 5 | 48 |
| Bazarr Specialist | Expert specialist for bazarr domain tasks. | You are a Bazarr specialist. Help users manage and interact with Bazarr functionality using the available tools. | bazarr | - | stdio | 0 | 12 | 38 |
| Chaptarr Specialist | Expert specialist for chaptarr domain tasks. | You are a Chaptarr specialist. Help users manage and interact with Chaptarr functionality using the available tools. | chaptarr | - | stdio | 0 | 134 | 37 |
| Lidarr Specialist | Expert specialist for lidarr domain tasks. | You are a Lidarr specialist. Help users manage and interact with Lidarr functionality using the available tools. | lidarr | - | stdio | 0 | 174 | 37 |
| Prowlarr Specialist | Expert specialist for prowlarr domain tasks. | You are a Prowlarr specialist. Help users manage and interact with Prowlarr functionality using the available tools. | prowlarr | - | stdio | 0 | 84 | 37 |
| Radarr Specialist | Expert specialist for radarr domain tasks. | You are a Radarr specialist. Help users manage and interact with Radarr functionality using the available tools. | radarr | - | stdio | 0 | 177 | 37 |
| Arr Seerr Specialist | Expert specialist for seerr domain tasks. | You are a Arr Seerr specialist. Help users manage and interact with Seerr functionality using the available tools. | seerr | - | stdio | 0 | 12 | 35 |
| Sonarr Specialist | Expert specialist for sonarr domain tasks. | You are a Sonarr specialist. Help users manage and interact with Sonarr functionality using the available tools. | sonarr | - | stdio | 0 | 173 | 37 |
| Atlassian Specialist | Expert specialist for atlassian domain tasks. | You are a Atlassian specialist. Help users manage and interact with Atlassian functionality using the available tools. | atlassian | - | stdio | 0 | 1 | 40 |
| Audio-Transcriber Audio Processing Specialist | Expert specialist for audio_processing domain tasks. | You are a Audio-Transcriber Audio Processing specialist. Help users manage and interact with Audio Processing functionality using the available tools. | audio_processing | - | stdio | 0 | 1 | 70 |
| Container-Manager Info Specialist | Expert specialist for info domain tasks. | You are a Container-Manager Info specialist. Help users manage and interact with Info functionality using the available tools. | info | - | stdio | 0 | 2 | 65 |
| Container-Manager Image Specialist | Expert specialist for image domain tasks. | You are a Container-Manager Image specialist. Help users manage and interact with Image functionality using the available tools. | image | - | stdio | 0 | 4 | 68 |
| Container-Manager Container Specialist | Expert specialist for container domain tasks. | You are a Container-Manager Container specialist. Help users manage and interact with Container functionality using the available tools. | container | - | stdio | 0 | 7 | 79 |
| Container-Manager Debug Specialist | Expert specialist for debug domain tasks. | You are a Container-Manager Debug specialist. Help users manage and interact with Debug functionality using the available tools. | debug | - | stdio | 0 | 1 | 95 |
| Container-Manager Log Specialist | Expert specialist for log domain tasks. | You are a Container-Manager Log specialist. Help users manage and interact with Log functionality using the available tools. | log | - | stdio | 0 | 6 | 66 |
| Container-Manager Compose Specialist | Expert specialist for compose domain tasks. | You are a Container-Manager Compose specialist. Help users manage and interact with Compose functionality using the available tools. | compose | - | stdio | 0 | 4 | 80 |
| Container-Manager Volume Specialist | Expert specialist for volume domain tasks. | You are a Container-Manager Volume specialist. Help users manage and interact with Volume functionality using the available tools. | volume | - | stdio | 0 | 4 | 65 |
| Container-Manager Network Specialist | Expert specialist for network domain tasks. | You are a Container-Manager Network specialist. Help users manage and interact with Network functionality using the available tools. | network | - | stdio | 0 | 8 | 68 |
| Container-Manager Swarm Specialist | Expert specialist for swarm domain tasks. | You are a Container-Manager Swarm specialist. Help users manage and interact with Swarm functionality using the available tools. | swarm | - | stdio | 0 | 6 | 64 |
| Documentdb Collections Specialist | Expert specialist for collections domain tasks. | You are a Documentdb Collections specialist. Help users manage and interact with Collections functionality using the available tools. | collections | - | stdio | 0 | 6 | 59 |
| Documentdb Crud Specialist | Expert specialist for crud domain tasks. | You are a Documentdb Crud specialist. Help users manage and interact with Crud functionality using the available tools. | crud | - | stdio | 0 | 13 | 53 |
| Documentdb Analysis Specialist | Expert specialist for analysis domain tasks. | You are a Documentdb Analysis specialist. Help users manage and interact with Analysis functionality using the available tools. | analysis | - | stdio | 0 | 2 | 55 |
| Github Repos Specialist | Expert specialist for repos domain tasks. | You are a Github Repos specialist. Help users manage and interact with Repos functionality using the available tools. | repos | - | stdio | 0 | 2 | 50 |
| Github Issues Specialist | Expert specialist for issues domain tasks. | You are a Github Issues specialist. Help users manage and interact with Issues functionality using the available tools. | issues | - | stdio | 0 | 1 | 50 |
| Github Pulls Specialist | Expert specialist for pulls domain tasks. | You are a Github Pulls specialist. Help users manage and interact with Pulls functionality using the available tools. | pulls | - | stdio | 0 | 1 | 55 |
| Github Contents Specialist | Expert specialist for contents domain tasks. | You are a Github Contents specialist. Help users manage and interact with Contents functionality using the available tools. | contents | - | stdio | 0 | 1 | 60 |
| Gitlab-Api Branches Specialist | Expert specialist for branches domain tasks. | You are a Gitlab-Api Branches specialist. Help users manage and interact with Branches functionality using the available tools. | branches | - | stdio | 0 | 3 | 68 |
| Gitlab-Api Commits Specialist | Expert specialist for commits domain tasks. | You are a Gitlab-Api Commits specialist. Help users manage and interact with Commits functionality using the available tools. | commits | - | stdio | 0 | 11 | 70 |
| Gitlab-Api Deploy Tokens Specialist | Expert specialist for deploy_tokens domain tasks. | You are a Gitlab-Api Deploy Tokens specialist. Help users manage and interact with Deploy Tokens functionality using the available tools. | deploy_tokens | - | stdio | 0 | 7 | 72 |
| Gitlab-Api Environments Specialist | Expert specialist for environments domain tasks. | You are a Gitlab-Api Environments specialist. Help users manage and interact with Environments functionality using the available tools. | environments | - | stdio | 0 | 11 | 67 |
| Gitlab-Api Members Specialist | Expert specialist for members domain tasks. | You are a Gitlab-Api Members specialist. Help users manage and interact with Members functionality using the available tools. | members | - | stdio | 0 | 2 | 70 |
| Gitlab-Api Merge-Requests Specialist | Expert specialist for merge-requests domain tasks. | You are a Gitlab-Api Merge-Requests specialist. Help users manage and interact with Merge-Requests functionality using the available tools. | merge-requests | - | stdio | 0 | 3 | 78 |
| Gitlab-Api Merge Rules Specialist | Expert specialist for merge_rules domain tasks. | You are a Gitlab-Api Merge Rules specialist. Help users manage and interact with Merge Rules functionality using the available tools. | merge_rules | - | stdio | 0 | 13 | 75 |
| Gitlab-Api Packages Specialist | Expert specialist for packages domain tasks. | You are a Gitlab-Api Packages specialist. Help users manage and interact with Packages functionality using the available tools. | packages | - | stdio | 0 | 3 | 76 |
| Gitlab-Api Pipelines Specialist | Expert specialist for pipelines domain tasks. | You are a Gitlab-Api Pipelines specialist. Help users manage and interact with Pipelines functionality using the available tools. | pipelines | - | stdio | 0 | 2 | 70 |
| Gitlab-Api Pipeline Schedules Specialist | Expert specialist for pipeline_schedules domain tasks. | You are a Gitlab-Api Pipeline Schedules specialist. Help users manage and interact with Pipeline Schedules functionality using the available tools. | pipeline_schedules | - | stdio | 0 | 10 | 70 |
| Gitlab-Api Protected Branches Specialist | Expert specialist for protected_branches domain tasks. | You are a Gitlab-Api Protected Branches specialist. Help users manage and interact with Protected Branches functionality using the available tools. | protected_branches | - | stdio | 0 | 4 | 71 |
| Gitlab-Api Releases Specialist | Expert specialist for releases domain tasks. | You are a Gitlab-Api Releases specialist. Help users manage and interact with Releases functionality using the available tools. | releases | - | stdio | 0 | 11 | 65 |
| Gitlab-Api Runners Specialist | Expert specialist for runners domain tasks. | You are a Gitlab-Api Runners specialist. Help users manage and interact with Runners functionality using the available tools. | runners | - | stdio | 0 | 15 | 67 |
| Gitlab-Api Tags Specialist | Expert specialist for tags domain tasks. | You are a Gitlab-Api Tags specialist. Help users manage and interact with Tags functionality using the available tools. | tags | - | stdio | 0 | 7 | 55 |
| Gitlab-Api Custom-Api Specialist | Expert specialist for custom-api domain tasks. | You are a Gitlab-Api Custom-Api specialist. Help users manage and interact with Custom-Api functionality using the available tools. | custom-api | - | stdio | 0 | 1 | 60 |
| Home Config Specialist | Expert specialist for config domain tasks. | You are a Home Config specialist. Help users manage and interact with Config functionality using the available tools. | config | - | stdio | 0 | 4 | 46 |
| Home States Specialist | Expert specialist for states domain tasks. | You are a Home States specialist. Help users manage and interact with States functionality using the available tools. | states | - | stdio | 0 | 6 | 46 |
| Home Services Specialist | Expert specialist for services domain tasks. | You are a Home Services specialist. Help users manage and interact with Services functionality using the available tools. | services | - | stdio | 0 | 2 | 57 |
| Home Events Specialist | Expert specialist for events domain tasks. | You are a Home Events specialist. Help users manage and interact with Events functionality using the available tools. | events | - | stdio | 0 | 3 | 48 |
| Home History Specialist | Expert specialist for history domain tasks. | You are a Home History specialist. Help users manage and interact with History functionality using the available tools. | history | - | stdio | 0 | 1 | 55 |
| Home Logbook Specialist | Expert specialist for logbook domain tasks. | You are a Home Logbook specialist. Help users manage and interact with Logbook functionality using the available tools. | logbook | - | stdio | 0 | 2 | 62 |
| Home Calendar Specialist | Expert specialist for calendar domain tasks. | You are a Home Calendar specialist. Help users manage and interact with Calendar functionality using the available tools. | calendar | - | stdio | 0 | 19 | 67 |
| Home Panels Specialist | Expert specialist for panels domain tasks. | You are a Home Panels specialist. Help users manage and interact with Panels functionality using the available tools. | panels | - | stdio | 0 | 1 | 45 |
| Home Voice Specialist | Expert specialist for voice domain tasks. | You are a Home Voice specialist. Help users manage and interact with Voice functionality using the available tools. | voice | - | stdio | 0 | 2 | 55 |
| Home Entities Specialist | Expert specialist for entities domain tasks. | You are a Home Entities specialist. Help users manage and interact with Entities functionality using the available tools. | entities | - | stdio | 0 | 5 | 75 |
| Jellyfin Activitylog Specialist | Expert specialist for ActivityLog domain tasks. | You are a Jellyfin Activitylog specialist. Help users manage and interact with Activitylog functionality using the available tools. | ActivityLog | - | stdio | 0 | 1 | 60 |
| Jellyfin Apikey Specialist | Expert specialist for ApiKey domain tasks. | You are a Jellyfin Apikey specialist. Help users manage and interact with Apikey functionality using the available tools. | ApiKey | - | stdio | 0 | 3 | 45 |
| Jellyfin Artists Specialist | Expert specialist for Artists domain tasks. | You are a Jellyfin Artists specialist. Help users manage and interact with Artists functionality using the available tools. | Artists | - | stdio | 0 | 3 | 65 |
| Jellyfin Audio Specialist | Expert specialist for Audio domain tasks. | You are a Jellyfin Audio specialist. Help users manage and interact with Audio functionality using the available tools. | Audio | - | stdio | 0 | 2 | 52 |
| Jellyfin Backup Specialist | Expert specialist for Backup domain tasks. | You are a Jellyfin Backup specialist. Help users manage and interact with Backup functionality using the available tools. | Backup | - | stdio | 0 | 4 | 55 |
| Jellyfin Branding Specialist | Expert specialist for Branding domain tasks. | You are a Jellyfin Branding specialist. Help users manage and interact with Branding functionality using the available tools. | Branding | - | stdio | 0 | 3 | 60 |
| Jellyfin Channels Specialist | Expert specialist for Channels domain tasks. | You are a Jellyfin Channels specialist. Help users manage and interact with Channels functionality using the available tools. | Channels | - | stdio | 0 | 5 | 61 |
| Jellyfin Clientlog Specialist | Expert specialist for ClientLog domain tasks. | You are a Jellyfin Clientlog specialist. Help users manage and interact with Clientlog functionality using the available tools. | ClientLog | - | stdio | 0 | 1 | 60 |
| Jellyfin Collection Specialist | Expert specialist for Collection domain tasks. | You are a Jellyfin Collection specialist. Help users manage and interact with Collection functionality using the available tools. | Collection | - | stdio | 0 | 3 | 60 |
| Jellyfin Configuration Specialist | Expert specialist for Configuration domain tasks. | You are a Jellyfin Configuration specialist. Help users manage and interact with Configuration functionality using the available tools. | Configuration | - | stdio | 0 | 6 | 59 |
| Jellyfin Dashboard Specialist | Expert specialist for Dashboard domain tasks. | You are a Jellyfin Dashboard specialist. Help users manage and interact with Dashboard functionality using the available tools. | Dashboard | - | stdio | 0 | 2 | 62 |
| Jellyfin Devices Specialist | Expert specialist for Devices domain tasks. | You are a Jellyfin Devices specialist. Help users manage and interact with Devices functionality using the available tools. | Devices | - | stdio | 0 | 5 | 57 |
| Jellyfin Displaypreferences Specialist | Expert specialist for DisplayPreferences domain tasks. | You are a Jellyfin Displaypreferences specialist. Help users manage and interact with Displaypreferences functionality using the available tools. | DisplayPreferences | - | stdio | 0 | 2 | 60 |
| Jellyfin Dynamichls Specialist | Expert specialist for DynamicHls domain tasks. | You are a Jellyfin Dynamichls specialist. Help users manage and interact with Dynamichls functionality using the available tools. | DynamicHls | - | stdio | 0 | 7 | 65 |
| Jellyfin Environment Specialist | Expert specialist for Environment domain tasks. | You are a Jellyfin Environment specialist. Help users manage and interact with Environment functionality using the available tools. | Environment | - | stdio | 0 | 16 | 63 |
| Jellyfin Filter Specialist | Expert specialist for Filter domain tasks. | You are a Jellyfin Filter specialist. Help users manage and interact with Filter functionality using the available tools. | Filter | - | stdio | 0 | 2 | 52 |
| Jellyfin Genres Specialist | Expert specialist for Genres domain tasks. | You are a Jellyfin Genres specialist. Help users manage and interact with Genres functionality using the available tools. | Genres | - | stdio | 0 | 2 | 50 |
| Jellyfin Hlssegment Specialist | Expert specialist for HlsSegment domain tasks. | You are a Jellyfin Hlssegment specialist. Help users manage and interact with Hlssegment functionality using the available tools. | HlsSegment | - | stdio | 0 | 5 | 69 |
| Jellyfin Image Specialist | Expert specialist for Image domain tasks. | You are a Jellyfin Image specialist. Help users manage and interact with Image functionality using the available tools. | Image | - | stdio | 0 | 24 | 52 |
| Jellyfin Instantmix Specialist | Expert specialist for InstantMix domain tasks. | You are a Jellyfin Instantmix specialist. Help users manage and interact with Instantmix functionality using the available tools. | InstantMix | - | stdio | 0 | 8 | 72 |
| Jellyfin Itemlookup Specialist | Expert specialist for ItemLookup domain tasks. | You are a Jellyfin Itemlookup specialist. Help users manage and interact with Itemlookup functionality using the available tools. | ItemLookup | - | stdio | 0 | 11 | 65 |
| Jellyfin Itemrefresh Specialist | Expert specialist for ItemRefresh domain tasks. | You are a Jellyfin Itemrefresh specialist. Help users manage and interact with Itemrefresh functionality using the available tools. | ItemRefresh | - | stdio | 0 | 1 | 60 |
| Jellyfin Items Specialist | Expert specialist for Items domain tasks. | You are a Jellyfin Items specialist. Help users manage and interact with Items functionality using the available tools. | Items | - | stdio | 0 | 4 | 51 |
| Jellyfin Library Specialist | Expert specialist for Library domain tasks. | You are a Jellyfin Library specialist. Help users manage and interact with Library functionality using the available tools. | Library | - | stdio | 0 | 25 | 62 |
| Jellyfin Itemupdate Specialist | Expert specialist for ItemUpdate domain tasks. | You are a Jellyfin Itemupdate specialist. Help users manage and interact with Itemupdate functionality using the available tools. | ItemUpdate | - | stdio | 0 | 3 | 61 |
| Jellyfin Userlibrary Specialist | Expert specialist for UserLibrary domain tasks. | You are a Jellyfin Userlibrary specialist. Help users manage and interact with Userlibrary functionality using the available tools. | UserLibrary | - | stdio | 0 | 10 | 63 |
| Jellyfin Librarystructure Specialist | Expert specialist for LibraryStructure domain tasks. | You are a Jellyfin Librarystructure specialist. Help users manage and interact with Librarystructure functionality using the available tools. | LibraryStructure | - | stdio | 0 | 8 | 63 |
| Jellyfin Livetv Specialist | Expert specialist for LiveTv domain tasks. | You are a Jellyfin Livetv specialist. Help users manage and interact with Livetv functionality using the available tools. | LiveTv | - | stdio | 0 | 41 | 49 |
| Jellyfin Localization Specialist | Expert specialist for Localization domain tasks. | You are a Jellyfin Localization specialist. Help users manage and interact with Localization functionality using the available tools. | Localization | - | stdio | 0 | 4 | 57 |
| Jellyfin Lyrics Specialist | Expert specialist for Lyrics domain tasks. | You are a Jellyfin Lyrics specialist. Help users manage and interact with Lyrics functionality using the available tools. | Lyrics | - | stdio | 0 | 6 | 50 |
| Jellyfin Mediainfo Specialist | Expert specialist for MediaInfo domain tasks. | You are a Jellyfin Mediainfo specialist. Help users manage and interact with Mediainfo functionality using the available tools. | MediaInfo | - | stdio | 0 | 5 | 66 |
| Jellyfin Mediasegments Specialist | Expert specialist for MediaSegments domain tasks. | You are a Jellyfin Mediasegments specialist. Help users manage and interact with Mediasegments functionality using the available tools. | MediaSegments | - | stdio | 0 | 1 | 60 |
| Jellyfin Movies Specialist | Expert specialist for Movies domain tasks. | You are a Jellyfin Movies specialist. Help users manage and interact with Movies functionality using the available tools. | Movies | - | stdio | 0 | 1 | 50 |
| Jellyfin Musicgenres Specialist | Expert specialist for MusicGenres domain tasks. | You are a Jellyfin Musicgenres specialist. Help users manage and interact with Musicgenres functionality using the available tools. | MusicGenres | - | stdio | 0 | 2 | 65 |
| Jellyfin Package Specialist | Expert specialist for Package domain tasks. | You are a Jellyfin Package specialist. Help users manage and interact with Package functionality using the available tools. | Package | - | stdio | 0 | 6 | 60 |
| Jellyfin Persons Specialist | Expert specialist for Persons domain tasks. | You are a Jellyfin Persons specialist. Help users manage and interact with Persons functionality using the available tools. | Persons | - | stdio | 0 | 2 | 55 |
| Jellyfin Playlists Specialist | Expert specialist for Playlists domain tasks. | You are a Jellyfin Playlists specialist. Help users manage and interact with Playlists functionality using the available tools. | Playlists | - | stdio | 0 | 11 | 59 |
| Jellyfin Playstate Specialist | Expert specialist for Playstate domain tasks. | You are a Jellyfin Playstate specialist. Help users manage and interact with Playstate functionality using the available tools. | Playstate | - | stdio | 0 | 9 | 64 |
| Jellyfin Plugins Specialist | Expert specialist for Plugins domain tasks. | You are a Jellyfin Plugins specialist. Help users manage and interact with Plugins functionality using the available tools. | Plugins | - | stdio | 0 | 9 | 60 |
| Jellyfin Quickconnect Specialist | Expert specialist for QuickConnect domain tasks. | You are a Jellyfin Quickconnect specialist. Help users manage and interact with Quickconnect functionality using the available tools. | QuickConnect | - | stdio | 0 | 4 | 65 |
| Jellyfin Remoteimage Specialist | Expert specialist for RemoteImage domain tasks. | You are a Jellyfin Remoteimage specialist. Help users manage and interact with Remoteimage functionality using the available tools. | RemoteImage | - | stdio | 0 | 3 | 63 |
| Jellyfin Scheduledtasks Specialist | Expert specialist for ScheduledTasks domain tasks. | You are a Jellyfin Scheduledtasks specialist. Help users manage and interact with Scheduledtasks functionality using the available tools. | ScheduledTasks | - | stdio | 0 | 5 | 55 |
| Jellyfin Search Specialist | Expert specialist for Search domain tasks. | You are a Jellyfin Search specialist. Help users manage and interact with Search functionality using the available tools. | Search | - | stdio | 0 | 1 | 50 |
| Jellyfin Session Specialist | Expert specialist for Session domain tasks. | You are a Jellyfin Session specialist. Help users manage and interact with Session functionality using the available tools. | Session | - | stdio | 0 | 16 | 63 |
| Jellyfin Startup Specialist | Expert specialist for Startup domain tasks. | You are a Jellyfin Startup specialist. Help users manage and interact with Startup functionality using the available tools. | Startup | - | stdio | 0 | 7 | 60 |
| Jellyfin Studios Specialist | Expert specialist for Studios domain tasks. | You are a Jellyfin Studios specialist. Help users manage and interact with Studios functionality using the available tools. | Studios | - | stdio | 0 | 2 | 60 |
| Jellyfin Subtitle Specialist | Expert specialist for Subtitle domain tasks. | You are a Jellyfin Subtitle specialist. Help users manage and interact with Subtitle functionality using the available tools. | Subtitle | - | stdio | 0 | 10 | 60 |
| Jellyfin Suggestions Specialist | Expert specialist for Suggestions domain tasks. | You are a Jellyfin Suggestions specialist. Help users manage and interact with Suggestions functionality using the available tools. | Suggestions | - | stdio | 0 | 1 | 55 |
| Jellyfin Syncplay Specialist | Expert specialist for SyncPlay domain tasks. | You are a Jellyfin Syncplay specialist. Help users manage and interact with Syncplay functionality using the available tools. | SyncPlay | - | stdio | 0 | 22 | 67 |
| Jellyfin System Specialist | Expert specialist for System domain tasks. | You are a Jellyfin System specialist. Help users manage and interact with System functionality using the available tools. | System | - | stdio | 0 | 20 | 50 |
| Jellyfin Timesync Specialist | Expert specialist for TimeSync domain tasks. | You are a Jellyfin Timesync specialist. Help users manage and interact with Timesync functionality using the available tools. | TimeSync | - | stdio | 0 | 1 | 60 |
| Jellyfin Tmdb Specialist | Expert specialist for Tmdb domain tasks. | You are a Jellyfin Tmdb specialist. Help users manage and interact with Tmdb functionality using the available tools. | Tmdb | - | stdio | 0 | 1 | 55 |
| Jellyfin Trailers Specialist | Expert specialist for Trailers domain tasks. | You are a Jellyfin Trailers specialist. Help users manage and interact with Trailers functionality using the available tools. | Trailers | - | stdio | 0 | 1 | 65 |
| Jellyfin Trickplay Specialist | Expert specialist for Trickplay domain tasks. | You are a Jellyfin Trickplay specialist. Help users manage and interact with Trickplay functionality using the available tools. | Trickplay | - | stdio | 0 | 2 | 65 |
| Jellyfin Tvshows Specialist | Expert specialist for TvShows domain tasks. | You are a Jellyfin Tvshows specialist. Help users manage and interact with Tvshows functionality using the available tools. | TvShows | - | stdio | 0 | 4 | 56 |
| Jellyfin Universalaudio Specialist | Expert specialist for UniversalAudio domain tasks. | You are a Jellyfin Universalaudio specialist. Help users manage and interact with Universalaudio functionality using the available tools. | UniversalAudio | - | stdio | 0 | 1 | 65 |
| Jellyfin User Specialist | Expert specialist for User domain tasks. | You are a Jellyfin User specialist. Help users manage and interact with User functionality using the available tools. | User | - | stdio | 0 | 30 | 49 |
| Jellyfin Userviews Specialist | Expert specialist for UserViews domain tasks. | You are a Jellyfin Userviews specialist. Help users manage and interact with Userviews functionality using the available tools. | UserViews | - | stdio | 0 | 2 | 57 |
| Jellyfin Videoattachments Specialist | Expert specialist for VideoAttachments domain tasks. | You are a Jellyfin Videoattachments specialist. Help users manage and interact with Videoattachments functionality using the available tools. | VideoAttachments | - | stdio | 0 | 1 | 55 |
| Jellyfin Videos Specialist | Expert specialist for Videos domain tasks. | You are a Jellyfin Videos specialist. Help users manage and interact with Videos functionality using the available tools. | Videos | - | stdio | 0 | 5 | 51 |
| Jellyfin Years Specialist | Expert specialist for Years domain tasks. | You are a Jellyfin Years specialist. Help users manage and interact with Years functionality using the available tools. | Years | - | stdio | 0 | 2 | 40 |
| Langfuse Annotation Queues Specialist | Expert specialist for annotation_queues domain tasks. | You are a Langfuse Annotation Queues specialist. Help users manage and interact with Annotation Queues functionality using the available tools. | annotation_queues | - | stdio | 0 | 10 | 67 |
| Langfuse Blob Storage Integrations Specialist | Expert specialist for blob_storage_integrations domain tasks. | You are a Langfuse Blob Storage Integrations specialist. Help users manage and interact with Blob Storage Integrations functionality using the available tools. | blob_storage_integrations | - | stdio | 0 | 4 | 80 |
| Langfuse Comments Specialist | Expert specialist for comments domain tasks. | You are a Langfuse Comments specialist. Help users manage and interact with Comments functionality using the available tools. | comments | - | stdio | 0 | 3 | 66 |
| Langfuse Dataset Items Specialist | Expert specialist for dataset_items domain tasks. | You are a Langfuse Dataset Items specialist. Help users manage and interact with Dataset Items functionality using the available tools. | dataset_items | - | stdio | 0 | 4 | 72 |
| Langfuse Dataset Run Items Specialist | Expert specialist for dataset_run_items domain tasks. | You are a Langfuse Dataset Run Items specialist. Help users manage and interact with Dataset Run Items functionality using the available tools. | dataset_run_items | - | stdio | 0 | 2 | 65 |
| Langfuse Datasets Specialist | Expert specialist for datasets domain tasks. | You are a Langfuse Datasets specialist. Help users manage and interact with Datasets functionality using the available tools. | datasets | - | stdio | 0 | 6 | 61 |
| Langfuse Health Specialist | Expert specialist for health domain tasks. | You are a Langfuse Health specialist. Help users manage and interact with Health functionality using the available tools. | health | - | stdio | 0 | 1 | 55 |
| Langfuse Ingestion Specialist | Expert specialist for ingestion domain tasks. | You are a Langfuse Ingestion specialist. Help users manage and interact with Ingestion functionality using the available tools. | ingestion | - | stdio | 0 | 1 | 85 |
| Langfuse Legacy Metrics V1 Specialist | Expert specialist for legacy_metrics_v1 domain tasks. | You are a Langfuse Legacy Metrics V1 specialist. Help users manage and interact with Legacy Metrics V1 functionality using the available tools. | legacy_metrics_v1 | - | stdio | 0 | 1 | 85 |
| Langfuse Legacy Observations V1 Specialist | Expert specialist for legacy_observations_v1 domain tasks. | You are a Langfuse Legacy Observations V1 specialist. Help users manage and interact with Legacy Observations V1 functionality using the available tools. | legacy_observations_v1 | - | stdio | 0 | 2 | 75 |
| Langfuse Legacy Score V1 Specialist | Expert specialist for legacy_score_v1 domain tasks. | You are a Langfuse Legacy Score V1 specialist. Help users manage and interact with Legacy Score V1 functionality using the available tools. | legacy_score_v1 | - | stdio | 0 | 2 | 75 |
| Langfuse Llm Connections Specialist | Expert specialist for llm_connections domain tasks. | You are a Langfuse Llm Connections specialist. Help users manage and interact with Llm Connections functionality using the available tools. | llm_connections | - | stdio | 0 | 2 | 70 |
| Langfuse Media Specialist | Expert specialist for media domain tasks. | You are a Langfuse Media specialist. Help users manage and interact with Media functionality using the available tools. | media | - | stdio | 0 | 3 | 53 |
| Langfuse Metrics Specialist | Expert specialist for metrics domain tasks. | You are a Langfuse Metrics specialist. Help users manage and interact with Metrics functionality using the available tools. | metrics | - | stdio | 0 | 1 | 85 |
| Langfuse Models Specialist | Expert specialist for models domain tasks. | You are a Langfuse Models specialist. Help users manage and interact with Models functionality using the available tools. | models | - | stdio | 0 | 4 | 51 |
| Langfuse Observations Specialist | Expert specialist for observations domain tasks. | You are a Langfuse Observations specialist. Help users manage and interact with Observations functionality using the available tools. | observations | - | stdio | 0 | 1 | 85 |
| Langfuse Opentelemetry Specialist | Expert specialist for opentelemetry domain tasks. | You are a Langfuse Opentelemetry specialist. Help users manage and interact with Opentelemetry functionality using the available tools. | opentelemetry | - | stdio | 0 | 1 | 85 |
| Langfuse Prompt Version Specialist | Expert specialist for prompt_version domain tasks. | You are a Langfuse Prompt Version specialist. Help users manage and interact with Prompt Version functionality using the available tools. | prompt_version | - | stdio | 0 | 1 | 65 |
| Langfuse Prompts Specialist | Expert specialist for prompts domain tasks. | You are a Langfuse Prompts specialist. Help users manage and interact with Prompts functionality using the available tools. | prompts | - | stdio | 0 | 4 | 68 |
| Langfuse Scim Specialist | Expert specialist for scim domain tasks. | You are a Langfuse Scim specialist. Help users manage and interact with Scim functionality using the available tools. | scim | - | stdio | 0 | 7 | 66 |
| Langfuse Score Configs Specialist | Expert specialist for score_configs domain tasks. | You are a Langfuse Score Configs specialist. Help users manage and interact with Score Configs functionality using the available tools. | score_configs | - | stdio | 0 | 4 | 67 |
| Langfuse Scores Specialist | Expert specialist for scores domain tasks. | You are a Langfuse Scores specialist. Help users manage and interact with Scores functionality using the available tools. | scores | - | stdio | 0 | 2 | 62 |
| Langfuse Sessions Specialist | Expert specialist for sessions domain tasks. | You are a Langfuse Sessions specialist. Help users manage and interact with Sessions functionality using the available tools. | sessions | - | stdio | 0 | 2 | 67 |
| Langfuse Trace Specialist | Expert specialist for trace domain tasks. | You are a Langfuse Trace specialist. Help users manage and interact with Trace functionality using the available tools. | trace | - | stdio | 0 | 4 | 51 |
| Leanix Poll Specialist | Expert specialist for leanix_poll domain tasks. | You are a Leanix Poll specialist. Help users manage and interact with Leanix Poll functionality using the available tools. | leanix_poll | - | stdio | 0 | 1 | 50 |
| Leanix Discovery Linking V2 Specialist | Expert specialist for leanix_discovery_linking_v2 domain tasks. | You are a Leanix Discovery Linking V2 specialist. Help users manage and interact with Leanix Discovery Linking V2 functionality using the available tools. | leanix_discovery_linking_v2 | - | stdio | 0 | 1 | 50 |
| Leanix Reference Data Catalog Specialist | Expert specialist for leanix_reference_data_catalog domain tasks. | You are a Leanix Reference Data Catalog specialist. Help users manage and interact with Leanix Reference Data Catalog functionality using the available tools. | leanix_reference_data_catalog | - | stdio | 0 | 1 | 50 |
| Leanix Metrics Specialist | Expert specialist for leanix_metrics domain tasks. | You are a Leanix Metrics specialist. Help users manage and interact with Leanix Metrics functionality using the available tools. | leanix_metrics | - | stdio | 0 | 1 | 50 |
| Leanix Discovery Saas Specialist | Expert specialist for leanix_discovery_saas domain tasks. | You are a Leanix Discovery Saas specialist. Help users manage and interact with Leanix Discovery Saas functionality using the available tools. | leanix_discovery_saas | - | stdio | 0 | 1 | 50 |
| Leanix Mtm Specialist | Expert specialist for leanix_mtm domain tasks. | You are a Leanix Mtm specialist. Help users manage and interact with Leanix Mtm functionality using the available tools. | leanix_mtm | - | stdio | 0 | 1 | 50 |
| Leanix Webhooks Specialist | Expert specialist for leanix_webhooks domain tasks. | You are a Leanix Webhooks specialist. Help users manage and interact with Leanix Webhooks functionality using the available tools. | leanix_webhooks | - | stdio | 0 | 1 | 50 |
| Leanix Storage Specialist | Expert specialist for leanix_storage domain tasks. | You are a Leanix Storage specialist. Help users manage and interact with Leanix Storage functionality using the available tools. | leanix_storage | - | stdio | 0 | 1 | 50 |
| Leanix Transformations Specialist | Expert specialist for leanix_transformations domain tasks. | You are a Leanix Transformations specialist. Help users manage and interact with Leanix Transformations functionality using the available tools. | leanix_transformations | - | stdio | 0 | 1 | 50 |
| Leanix Integration Collibra Specialist | Expert specialist for leanix_integration_collibra domain tasks. | You are a Leanix Integration Collibra specialist. Help users manage and interact with Leanix Integration Collibra functionality using the available tools. | leanix_integration_collibra | - | stdio | 0 | 1 | 50 |
| Leanix Discovery Sap Extension Specialist | Expert specialist for leanix_discovery_sap_extension domain tasks. | You are a Leanix Discovery Sap Extension specialist. Help users manage and interact with Leanix Discovery Sap Extension functionality using the available tools. | leanix_discovery_sap_extension | - | stdio | 0 | 1 | 50 |
| Leanix Impacts Specialist | Expert specialist for leanix_impacts domain tasks. | You are a Leanix Impacts specialist. Help users manage and interact with Leanix Impacts functionality using the available tools. | leanix_impacts | - | stdio | 0 | 1 | 50 |
| Leanix Technology Discovery Specialist | Expert specialist for leanix_technology_discovery domain tasks. | You are a Leanix Technology Discovery specialist. Help users manage and interact with Leanix Technology Discovery functionality using the available tools. | leanix_technology_discovery | - | stdio | 0 | 1 | 50 |
| Leanix Ai Inventory Builder Specialist | Expert specialist for leanix_ai_inventory_builder domain tasks. | You are a Leanix Ai Inventory Builder specialist. Help users manage and interact with Leanix Ai Inventory Builder functionality using the available tools. | leanix_ai_inventory_builder | - | stdio | 0 | 1 | 50 |
| Leanix Managed Code Execution Specialist | Expert specialist for leanix_managed_code_execution domain tasks. | You are a Leanix Managed Code Execution specialist. Help users manage and interact with Leanix Managed Code Execution functionality using the available tools. | leanix_managed_code_execution | - | stdio | 0 | 1 | 50 |
| Leanix Graphql Specialist | Expert specialist for graphql domain tasks. | You are a Leanix Graphql specialist. Help users manage and interact with Graphql functionality using the available tools. | graphql | - | stdio | 0 | 1 | 50 |
| Leanix Reference Data Specialist | Expert specialist for leanix_reference_data domain tasks. | You are a Leanix Reference Data specialist. Help users manage and interact with Leanix Reference Data functionality using the available tools. | leanix_reference_data | - | stdio | 0 | 1 | 50 |
| Leanix Survey Specialist | Expert specialist for leanix_survey domain tasks. | You are a Leanix Survey specialist. Help users manage and interact with Leanix Survey functionality using the available tools. | leanix_survey | - | stdio | 0 | 1 | 50 |
| Leanix Navigation Specialist | Expert specialist for leanix_navigation domain tasks. | You are a Leanix Navigation specialist. Help users manage and interact with Leanix Navigation functionality using the available tools. | leanix_navigation | - | stdio | 0 | 1 | 50 |
| Leanix Integration Signavio Specialist | Expert specialist for leanix_integration_signavio domain tasks. | You are a Leanix Integration Signavio specialist. Help users manage and interact with Leanix Integration Signavio functionality using the available tools. | leanix_integration_signavio | - | stdio | 0 | 1 | 50 |
| Leanix Pathfinder Specialist | Expert specialist for leanix_pathfinder domain tasks. | You are a Leanix Pathfinder specialist. Help users manage and interact with Leanix Pathfinder functionality using the available tools. | leanix_pathfinder | - | stdio | 0 | 1 | 50 |
| Leanix Todo Specialist | Expert specialist for leanix_todo domain tasks. | You are a Leanix Todo specialist. Help users manage and interact with Leanix Todo functionality using the available tools. | leanix_todo | - | stdio | 0 | 1 | 50 |
| Leanix Discovery Ai Agents Specialist | Expert specialist for leanix_discovery_ai_agents domain tasks. | You are a Leanix Discovery Ai Agents specialist. Help users manage and interact with Leanix Discovery Ai Agents functionality using the available tools. | leanix_discovery_ai_agents | - | stdio | 0 | 1 | 50 |
| Leanix Integration Servicenow Specialist | Expert specialist for leanix_integration_servicenow domain tasks. | You are a Leanix Integration Servicenow specialist. Help users manage and interact with Leanix Integration Servicenow functionality using the available tools. | leanix_integration_servicenow | - | stdio | 0 | 1 | 50 |
| Leanix Automations Specialist | Expert specialist for leanix_automations domain tasks. | You are a Leanix Automations specialist. Help users manage and interact with Leanix Automations functionality using the available tools. | leanix_automations | - | stdio | 0 | 1 | 50 |
| Leanix Discovery Linking V1 Specialist | Expert specialist for leanix_discovery_linking_v1 domain tasks. | You are a Leanix Discovery Linking V1 specialist. Help users manage and interact with Leanix Discovery Linking V1 functionality using the available tools. | leanix_discovery_linking_v1 | - | stdio | 0 | 1 | 50 |
| Leanix Discovery Sap Specialist | Expert specialist for leanix_discovery_sap domain tasks. | You are a Leanix Discovery Sap specialist. Help users manage and interact with Leanix Discovery Sap functionality using the available tools. | leanix_discovery_sap | - | stdio | 0 | 1 | 50 |
| Leanix Synclog Specialist | Expert specialist for leanix_synclog domain tasks. | You are a Leanix Synclog specialist. Help users manage and interact with Leanix Synclog functionality using the available tools. | leanix_synclog | - | stdio | 0 | 1 | 50 |
| Leanix Integration Api Specialist | Expert specialist for leanix_integration_api domain tasks. | You are a Leanix Integration Api specialist. Help users manage and interact with Leanix Integration Api functionality using the available tools. | leanix_integration_api | - | stdio | 0 | 1 | 50 |
| Leanix Inventory Data Quality Specialist | Expert specialist for leanix_inventory_data_quality domain tasks. | You are a Leanix Inventory Data Quality specialist. Help users manage and interact with Leanix Inventory Data Quality functionality using the available tools. | leanix_inventory_data_quality | - | stdio | 0 | 1 | 50 |
| Leanix Documents Specialist | Expert specialist for leanix_documents domain tasks. | You are a Leanix Documents specialist. Help users manage and interact with Leanix Documents functionality using the available tools. | leanix_documents | - | stdio | 0 | 1 | 50 |
| Leanix Apptio Connector Specialist | Expert specialist for leanix_apptio_connector domain tasks. | You are a Leanix Apptio Connector specialist. Help users manage and interact with Leanix Apptio Connector functionality using the available tools. | leanix_apptio_connector | - | stdio | 0 | 1 | 50 |
| Mealie App Specialist | Expert specialist for app domain tasks. | You are a Mealie App specialist. Help users manage and interact with App functionality using the available tools. | app | - | stdio | 0 | 9 | 50 |
| Mealie Households Specialist | Expert specialist for households domain tasks. | You are a Mealie Households specialist. Help users manage and interact with Households functionality using the available tools. | households | - | stdio | 0 | 64 | 59 |
| Mealie Recipes Specialist | Expert specialist for recipes domain tasks. | You are a Mealie Recipes specialist. Help users manage and interact with Recipes functionality using the available tools. | recipes | - | stdio | 0 | 64 | 59 |
| Mealie Organizer Specialist | Expert specialist for organizer domain tasks. | You are a Mealie Organizer specialist. Help users manage and interact with Organizer functionality using the available tools. | organizer | - | stdio | 0 | 20 | 58 |
| Mealie Shared Specialist | Expert specialist for shared domain tasks. | You are a Mealie Shared specialist. Help users manage and interact with Shared functionality using the available tools. | shared | - | stdio | 0 | 4 | 48 |
| Mealie Admin Specialist | Expert specialist for admin domain tasks. | You are a Mealie Admin specialist. Help users manage and interact with Admin functionality using the available tools. | admin | - | stdio | 0 | 44 | 54 |
| Mealie Explore Specialist | Expert specialist for explore domain tasks. | You are a Mealie Explore specialist. Help users manage and interact with Explore functionality using the available tools. | explore | - | stdio | 0 | 15 | 58 |
| Mealie Utils Specialist | Expert specialist for utils domain tasks. | You are a Mealie Utils specialist. Help users manage and interact with Utils functionality using the available tools. | utils | - | stdio | 0 | 1 | 45 |
| Media-Downloader Collection Management Specialist | Expert specialist for collection_management domain tasks. | You are a Media-Downloader Collection Management specialist. Help users manage and interact with Collection Management functionality using the available tools. | collection_management | - | stdio | 0 | 6 | 65 |
| Media-Downloader Files Specialist | Expert specialist for files domain tasks. | You are a Media-Downloader Files specialist. Help users manage and interact with Files functionality using the available tools. | files | - | stdio | 0 | 60 | 71 |
| Media-Downloader Text Editor Specialist | Expert specialist for text_editor domain tasks. | You are a Media-Downloader Text Editor specialist. Help users manage and interact with Text Editor functionality using the available tools. | text_editor | - | stdio | 0 | 2 | 70 |
| Microsoft Auth Specialist | Expert specialist for auth domain tasks. | You are a Microsoft Auth specialist. Help users manage and interact with Auth functionality using the available tools. | auth | - | stdio | 0 | 5 | 50 |
| Microsoft Meta Specialist | Expert specialist for meta domain tasks. | You are a Microsoft Meta specialist. Help users manage and interact with Meta functionality using the available tools. | meta | - | stdio | 0 | 1 | 50 |
| Microsoft Mail Specialist | Expert specialist for mail domain tasks. | You are a Microsoft Mail specialist. Help users manage and interact with Mail functionality using the available tools. | mail | - | stdio | 0 | 27 | 82 |
| Microsoft User Specialist | Expert specialist for user domain tasks. | You are a Microsoft User specialist. Help users manage and interact with User functionality using the available tools. | user | - | stdio | 0 | 33 | 79 |
| Microsoft Chat Specialist | Expert specialist for chat domain tasks. | You are a Microsoft Chat specialist. Help users manage and interact with Chat functionality using the available tools. | chat | - | stdio | 0 | 9 | 75 |
| Microsoft Notes Specialist | Expert specialist for notes domain tasks. | You are a Microsoft Notes specialist. Help users manage and interact with Notes functionality using the available tools. | notes | - | stdio | 0 | 5 | 71 |
| Microsoft Tasks Specialist | Expert specialist for tasks domain tasks. | You are a Microsoft Tasks specialist. Help users manage and interact with Tasks functionality using the available tools. | tasks | - | stdio | 0 | 13 | 64 |
| Microsoft Contacts Specialist | Expert specialist for contacts domain tasks. | You are a Microsoft Contacts specialist. Help users manage and interact with Contacts functionality using the available tools. | contacts | - | stdio | 0 | 8 | 62 |
| Microsoft Sites Specialist | Expert specialist for sites domain tasks. | You are a Microsoft Sites specialist. Help users manage and interact with Sites functionality using the available tools. | sites | - | stdio | 0 | 14 | 71 |
| Microsoft Search Specialist | Expert specialist for search domain tasks. | You are a Microsoft Search specialist. Help users manage and interact with Search functionality using the available tools. | search | - | stdio | 0 | 15 | 55 |
| Microsoft Organization Specialist | Expert specialist for organization domain tasks. | You are a Microsoft Organization specialist. Help users manage and interact with Organization functionality using the available tools. | organization | - | stdio | 0 | 5 | 73 |
| Microsoft Domains Specialist | Expert specialist for domains domain tasks. | You are a Microsoft Domains specialist. Help users manage and interact with Domains functionality using the available tools. | domains | - | stdio | 0 | 6 | 70 |
| Microsoft Subscriptions Specialist | Expert specialist for subscriptions domain tasks. | You are a Microsoft Subscriptions specialist. Help users manage and interact with Subscriptions functionality using the available tools. | subscriptions | - | stdio | 0 | 5 | 71 |
| Microsoft Communications Specialist | Expert specialist for communications domain tasks. | You are a Microsoft Communications specialist. Help users manage and interact with Communications functionality using the available tools. | communications | - | stdio | 0 | 10 | 77 |
| Microsoft Identity Specialist | Expert specialist for identity domain tasks. | You are a Microsoft Identity specialist. Help users manage and interact with Identity functionality using the available tools. | identity | - | stdio | 0 | 10 | 82 |
| Microsoft Security Specialist | Expert specialist for security domain tasks. | You are a Microsoft Security specialist. Help users manage and interact with Security functionality using the available tools. | security | - | stdio | 0 | 17 | 77 |
| Microsoft Audit Specialist | Expert specialist for audit domain tasks. | You are a Microsoft Audit specialist. Help users manage and interact with Audit functionality using the available tools. | audit | - | stdio | 0 | 5 | 68 |
| Microsoft Reports Specialist | Expert specialist for reports domain tasks. | You are a Microsoft Reports specialist. Help users manage and interact with Reports functionality using the available tools. | reports | - | stdio | 0 | 6 | 85 |
| Microsoft Applications Specialist | Expert specialist for applications domain tasks. | You are a Microsoft Applications specialist. Help users manage and interact with Applications functionality using the available tools. | applications | - | stdio | 0 | 12 | 70 |
| Microsoft Directory Specialist | Expert specialist for directory domain tasks. | You are a Microsoft Directory specialist. Help users manage and interact with Directory functionality using the available tools. | directory | - | stdio | 0 | 12 | 77 |
| Microsoft Policies Specialist | Expert specialist for policies domain tasks. | You are a Microsoft Policies specialist. Help users manage and interact with Policies functionality using the available tools. | policies | - | stdio | 0 | 5 | 78 |
| Microsoft Devices Specialist | Expert specialist for devices domain tasks. | You are a Microsoft Devices specialist. Help users manage and interact with Devices functionality using the available tools. | devices | - | stdio | 0 | 9 | 74 |
| Microsoft Education Specialist | Expert specialist for education domain tasks. | You are a Microsoft Education specialist. Help users manage and interact with Education functionality using the available tools. | education | - | stdio | 0 | 6 | 71 |
| Microsoft Agreements Specialist | Expert specialist for agreements domain tasks. | You are a Microsoft Agreements specialist. Help users manage and interact with Agreements functionality using the available tools. | agreements | - | stdio | 0 | 4 | 65 |
| Microsoft Places Specialist | Expert specialist for places domain tasks. | You are a Microsoft Places specialist. Help users manage and interact with Places functionality using the available tools. | places | - | stdio | 0 | 4 | 56 |
| Microsoft Print Specialist | Expert specialist for print domain tasks. | You are a Microsoft Print specialist. Help users manage and interact with Print functionality using the available tools. | print | - | stdio | 0 | 5 | 58 |
| Microsoft Privacy Specialist | Expert specialist for privacy domain tasks. | You are a Microsoft Privacy specialist. Help users manage and interact with Privacy functionality using the available tools. | privacy | - | stdio | 0 | 3 | 85 |
| Microsoft Solutions Specialist | Expert specialist for solutions domain tasks. | You are a Microsoft Solutions specialist. Help users manage and interact with Solutions functionality using the available tools. | solutions | - | stdio | 0 | 5 | 74 |
| Microsoft Storage Specialist | Expert specialist for storage domain tasks. | You are a Microsoft Storage specialist. Help users manage and interact with Storage functionality using the available tools. | storage | - | stdio | 0 | 3 | 81 |
| Microsoft Employee Experience Specialist | Expert specialist for employee_experience domain tasks. | You are a Microsoft Employee Experience specialist. Help users manage and interact with Employee Experience functionality using the available tools. | employee_experience | - | stdio | 0 | 3 | 78 |
| Microsoft Connections Specialist | Expert specialist for connections domain tasks. | You are a Microsoft Connections specialist. Help users manage and interact with Connections functionality using the available tools. | connections | - | stdio | 0 | 4 | 72 |
| Nextcloud Sharing Specialist | Expert specialist for sharing domain tasks. | You are a Nextcloud Sharing specialist. Help users manage and interact with Sharing functionality using the available tools. | sharing | - | stdio | 0 | 3 | 53 |
| Owncast External Specialist | Expert specialist for external domain tasks. | You are a Owncast External specialist. Help users manage and interact with External functionality using the available tools. | external | - | stdio | 0 | 11 | 65 |
| Owncast Internal Specialist | Expert specialist for internal domain tasks. | You are a Owncast Internal specialist. Help users manage and interact with Internal functionality using the available tools. | internal | - | stdio | 0 | 107 | 65 |
| Owncast Objects Specialist | Expert specialist for objects domain tasks. | You are a Owncast Objects specialist. Help users manage and interact with Objects functionality using the available tools. | objects | - | stdio | 0 | 3 | 65 |
| Plane Work Items Specialist | Expert specialist for work_items domain tasks. | You are a Plane Work Items specialist. Help users manage and interact with Work Items functionality using the available tools. | work_items | - | stdio | 0 | 16 | 64 |
| Plane Cycles Specialist | Expert specialist for cycles domain tasks. | You are a Plane Cycles specialist. Help users manage and interact with Cycles functionality using the available tools. | cycles | - | stdio | 0 | 7 | 48 |
| Plane Epics Specialist | Expert specialist for epics domain tasks. | You are a Plane Epics specialist. Help users manage and interact with Epics functionality using the available tools. | epics | - | stdio | 0 | 5 | 46 |
| Plane Initiatives Specialist | Expert specialist for initiatives domain tasks. | You are a Plane Initiatives specialist. Help users manage and interact with Initiatives functionality using the available tools. | initiatives | - | stdio | 0 | 2 | 55 |
| Plane Intake Specialist | Expert specialist for intake domain tasks. | You are a Plane Intake specialist. Help users manage and interact with Intake functionality using the available tools. | intake | - | stdio | 0 | 2 | 55 |
| Plane Labels Specialist | Expert specialist for labels domain tasks. | You are a Plane Labels specialist. Help users manage and interact with Labels functionality using the available tools. | labels | - | stdio | 0 | 2 | 45 |
| Plane Pages Specialist | Expert specialist for pages domain tasks. | You are a Plane Pages specialist. Help users manage and interact with Pages functionality using the available tools. | pages | - | stdio | 0 | 2 | 52 |
| Plane Milestones Specialist | Expert specialist for milestones domain tasks. | You are a Plane Milestones specialist. Help users manage and interact with Milestones functionality using the available tools. | milestones | - | stdio | 0 | 5 | 56 |
| Plane Modules Specialist | Expert specialist for modules domain tasks. | You are a Plane Modules specialist. Help users manage and interact with Modules functionality using the available tools. | modules | - | stdio | 0 | 5 | 56 |
| Plane Workspaces Specialist | Expert specialist for workspaces domain tasks. | You are a Plane Workspaces specialist. Help users manage and interact with Workspaces functionality using the available tools. | workspaces | - | stdio | 0 | 4 | 58 |
| Portainer Auth Specialist | Expert specialist for Auth domain tasks. | You are a Portainer Auth specialist. Help users manage and interact with Auth functionality using the available tools. | Auth | - | stdio | 0 | 3 | 53 |
| Portainer Docker Specialist | Expert specialist for Docker domain tasks. | You are a Portainer Docker specialist. Help users manage and interact with Docker functionality using the available tools. | Docker | - | stdio | 0 | 29 | 54 |
| Portainer Stack Specialist | Expert specialist for Stack domain tasks. | You are a Portainer Stack specialist. Help users manage and interact with Stack functionality using the available tools. | Stack | - | stdio | 0 | 11 | 55 |
| Portainer Kubernetes Specialist | Expert specialist for Kubernetes domain tasks. | You are a Portainer Kubernetes specialist. Help users manage and interact with Kubernetes functionality using the available tools. | Kubernetes | - | stdio | 0 | 14 | 63 |
| Portainer Edge Specialist | Expert specialist for Edge domain tasks. | You are a Portainer Edge specialist. Help users manage and interact with Edge functionality using the available tools. | Edge | - | stdio | 0 | 11 | 50 |
| Portainer Template Specialist | Expert specialist for Template domain tasks. | You are a Portainer Template specialist. Help users manage and interact with Template functionality using the available tools. | Template | - | stdio | 0 | 7 | 61 |
| Portainer Registry Specialist | Expert specialist for Registry domain tasks. | You are a Portainer Registry specialist. Help users manage and interact with Registry functionality using the available tools. | Registry | - | stdio | 0 | 4 | 60 |
| Postiz Integrations Specialist | Expert specialist for integrations domain tasks. | You are a Postiz Integrations specialist. Help users manage and interact with Integrations functionality using the available tools. | integrations | - | stdio | 0 | 5 | 67 |
| Postiz Posts Specialist | Expert specialist for posts domain tasks. | You are a Postiz Posts specialist. Help users manage and interact with Posts functionality using the available tools. | posts | - | stdio | 0 | 6 | 56 |
| Postiz Uploads Specialist | Expert specialist for uploads domain tasks. | You are a Postiz Uploads specialist. Help users manage and interact with Uploads functionality using the available tools. | uploads | - | stdio | 0 | 2 | 65 |
| Postiz Analytics Specialist | Expert specialist for analytics domain tasks. | You are a Postiz Analytics specialist. Help users manage and interact with Analytics functionality using the available tools. | analytics | - | stdio | 0 | 2 | 67 |
| Postiz Notifications Specialist | Expert specialist for notifications domain tasks. | You are a Postiz Notifications specialist. Help users manage and interact with Notifications functionality using the available tools. | notifications | - | stdio | 0 | 1 | 60 |
| Postiz Video Specialist | Expert specialist for video domain tasks. | You are a Postiz Video specialist. Help users manage and interact with Video functionality using the available tools. | video | - | stdio | 0 | 2 | 60 |
| Qbittorrent Torrents Specialist | Expert specialist for torrents domain tasks. | You are a Qbittorrent Torrents specialist. Help users manage and interact with Torrents functionality using the available tools. | torrents | - | stdio | 0 | 46 | 62 |
| Qbittorrent Transfer Specialist | Expert specialist for transfer domain tasks. | You are a Qbittorrent Transfer specialist. Help users manage and interact with Transfer functionality using the available tools. | transfer | - | stdio | 0 | 8 | 66 |
| Qbittorrent Rss Specialist | Expert specialist for rss domain tasks. | You are a Qbittorrent Rss specialist. Help users manage and interact with Rss functionality using the available tools. | rss | - | stdio | 0 | 12 | 55 |
| Qbittorrent Sync Specialist | Expert specialist for sync domain tasks. | You are a Qbittorrent Sync specialist. Help users manage and interact with Sync functionality using the available tools. | sync | - | stdio | 0 | 2 | 57 |
| Repository-Manager Devops Engineer Specialist | Expert specialist for devops_engineer domain tasks. | You are a Repository-Manager Devops Engineer specialist. Help users manage and interact with Devops Engineer functionality using the available tools. | devops_engineer | - | stdio | 0 | 4 | 80 |
| Repository-Manager Project Manager Specialist | Expert specialist for project_manager domain tasks. | You are a Repository-Manager Project Manager specialist. Help users manage and interact with Project Manager functionality using the available tools. | project_manager | - | stdio | 0 | 3 | 78 |
| Repository-Manager Workspace Management Specialist | Expert specialist for workspace_management domain tasks. | You are a Repository-Manager Workspace Management specialist. Help users manage and interact with Workspace Management functionality using the available tools. | workspace_management | - | stdio | 0 | 9 | 70 |
| Repository-Manager Git Operations Specialist | Expert specialist for git_operations domain tasks. | You are a Repository-Manager Git Operations specialist. Help users manage and interact with Git Operations functionality using the available tools. | git_operations | - | stdio | 0 | 3 | 81 |
| Repository-Manager Project Management Specialist | Expert specialist for project_management domain tasks. | You are a Repository-Manager Project Management specialist. Help users manage and interact with Project Management functionality using the available tools. | project_management | - | stdio | 0 | 1 | 85 |
| Repository-Manager Graph Intelligence Specialist | Expert specialist for graph_intelligence domain tasks. | You are a Repository-Manager Graph Intelligence specialist. Help users manage and interact with Graph Intelligence functionality using the available tools. | graph_intelligence | - | stdio | 0 | 6 | 68 |
| Repository-Manager Visualization Specialist | Expert specialist for visualization domain tasks. | You are a Repository-Manager Visualization specialist. Help users manage and interact with Visualization functionality using the available tools. | visualization | - | stdio | 0 | 3 | 71 |
| Servicenow-Api Flows Specialist | Expert specialist for flows domain tasks. | You are a Servicenow-Api Flows specialist. Help users manage and interact with Flows functionality using the available tools. | flows | - | stdio | 0 | 1 | 70 |
| Servicenow-Api Application Specialist | Expert specialist for application domain tasks. | You are a Servicenow-Api Application specialist. Help users manage and interact with Application functionality using the available tools. | application | - | stdio | 0 | 1 | 65 |
| Servicenow-Api Cmdb Specialist | Expert specialist for cmdb domain tasks. | You are a Servicenow-Api Cmdb specialist. Help users manage and interact with Cmdb functionality using the available tools. | cmdb | - | stdio | 0 | 9 | 59 |
| Servicenow-Api Cicd Specialist | Expert specialist for cicd domain tasks. | You are a Servicenow-Api Cicd specialist. Help users manage and interact with Cicd functionality using the available tools. | cicd | - | stdio | 0 | 12 | 61 |
| Servicenow-Api Plugins Specialist | Expert specialist for plugins domain tasks. | You are a Servicenow-Api Plugins specialist. Help users manage and interact with Plugins functionality using the available tools. | plugins | - | stdio | 0 | 2 | 70 |
| Servicenow-Api Source Control Specialist | Expert specialist for source_control domain tasks. | You are a Servicenow-Api Source Control specialist. Help users manage and interact with Source Control functionality using the available tools. | source_control | - | stdio | 0 | 2 | 72 |
| Servicenow-Api Testing Specialist | Expert specialist for testing domain tasks. | You are a Servicenow-Api Testing specialist. Help users manage and interact with Testing functionality using the available tools. | testing | - | stdio | 0 | 1 | 70 |
| Servicenow-Api Update Sets Specialist | Expert specialist for update_sets domain tasks. | You are a Servicenow-Api Update Sets specialist. Help users manage and interact with Update Sets functionality using the available tools. | update_sets | - | stdio | 0 | 6 | 65 |
| Servicenow-Api Batch Specialist | Expert specialist for batch domain tasks. | You are a Servicenow-Api Batch specialist. Help users manage and interact with Batch functionality using the available tools. | batch | - | stdio | 0 | 1 | 50 |
| Servicenow-Api Change Management Specialist | Expert specialist for change_management domain tasks. | You are a Servicenow-Api Change Management specialist. Help users manage and interact with Change Management functionality using the available tools. | change_management | - | stdio | 0 | 25 | 73 |
| Servicenow-Api Cilifecycle Specialist | Expert specialist for cilifecycle domain tasks. | You are a Servicenow-Api Cilifecycle specialist. Help users manage and interact with Cilifecycle functionality using the available tools. | cilifecycle | - | stdio | 0 | 3 | 68 |
| Servicenow-Api Devops Specialist | Expert specialist for devops domain tasks. | You are a Servicenow-Api Devops specialist. Help users manage and interact with Devops functionality using the available tools. | devops | - | stdio | 0 | 2 | 65 |
| Servicenow-Api Import Sets Specialist | Expert specialist for import_sets domain tasks. | You are a Servicenow-Api Import Sets specialist. Help users manage and interact with Import Sets functionality using the available tools. | import_sets | - | stdio | 0 | 3 | 70 |
| Servicenow-Api Incidents Specialist | Expert specialist for incidents domain tasks. | You are a Servicenow-Api Incidents specialist. Help users manage and interact with Incidents functionality using the available tools. | incidents | - | stdio | 0 | 2 | 65 |
| Servicenow-Api Knowledge Management Specialist | Expert specialist for knowledge_management domain tasks. | You are a Servicenow-Api Knowledge Management specialist. Help users manage and interact with Knowledge Management functionality using the available tools. | knowledge_management | - | stdio | 0 | 5 | 73 |
| Servicenow-Api Table Api Specialist | Expert specialist for table_api domain tasks. | You are a Servicenow-Api Table Api specialist. Help users manage and interact with Table Api functionality using the available tools. | table_api | - | stdio | 0 | 6 | 70 |
| Servicenow-Api Custom Api Specialist | Expert specialist for custom_api domain tasks. | You are a Servicenow-Api Custom Api specialist. Help users manage and interact with Custom Api functionality using the available tools. | custom_api | - | stdio | 0 | 1 | 70 |
| Servicenow-Api Email Specialist | Expert specialist for email domain tasks. | You are a Servicenow-Api Email specialist. Help users manage and interact with Email functionality using the available tools. | email | - | stdio | 0 | 1 | 50 |
| Servicenow-Api Data Classification Specialist | Expert specialist for data_classification domain tasks. | You are a Servicenow-Api Data Classification specialist. Help users manage and interact with Data Classification functionality using the available tools. | data_classification | - | stdio | 0 | 1 | 60 |
| Servicenow-Api Attachment Specialist | Expert specialist for attachment domain tasks. | You are a Servicenow-Api Attachment specialist. Help users manage and interact with Attachment functionality using the available tools. | attachment | - | stdio | 0 | 3 | 56 |
| Servicenow-Api Aggregate Specialist | Expert specialist for aggregate domain tasks. | You are a Servicenow-Api Aggregate specialist. Help users manage and interact with Aggregate functionality using the available tools. | aggregate | - | stdio | 0 | 1 | 55 |
| Servicenow-Api Activity Subscriptions Specialist | Expert specialist for activity_subscriptions domain tasks. | You are a Servicenow-Api Activity Subscriptions specialist. Help users manage and interact with Activity Subscriptions functionality using the available tools. | activity_subscriptions | - | stdio | 0 | 1 | 60 |
| Servicenow-Api Account Specialist | Expert specialist for account domain tasks. | You are a Servicenow-Api Account specialist. Help users manage and interact with Account functionality using the available tools. | account | - | stdio | 0 | 1 | 55 |
| Servicenow-Api Hr Specialist | Expert specialist for hr domain tasks. | You are a Servicenow-Api Hr specialist. Help users manage and interact with Hr functionality using the available tools. | hr | - | stdio | 0 | 1 | 45 |
| Servicenow-Api Metricbase Specialist | Expert specialist for metricbase domain tasks. | You are a Servicenow-Api Metricbase specialist. Help users manage and interact with Metricbase functionality using the available tools. | metricbase | - | stdio | 0 | 1 | 60 |
| Servicenow-Api Service Qualification Specialist | Expert specialist for service_qualification domain tasks. | You are a Servicenow-Api Service Qualification specialist. Help users manage and interact with Service Qualification functionality using the available tools. | service_qualification | - | stdio | 0 | 3 | 63 |
| Servicenow-Api Ppm Specialist | Expert specialist for ppm domain tasks. | You are a Servicenow-Api Ppm specialist. Help users manage and interact with Ppm functionality using the available tools. | ppm | - | stdio | 0 | 2 | 55 |
| Servicenow-Api Product Inventory Specialist | Expert specialist for product_inventory domain tasks. | You are a Servicenow-Api Product Inventory specialist. Help users manage and interact with Product Inventory functionality using the available tools. | product_inventory | - | stdio | 0 | 2 | 60 |
| Stirlingpdf Pdf Specialist | Expert specialist for PDF domain tasks. | You are a Stirlingpdf Pdf specialist. Help users manage and interact with Pdf functionality using the available tools. | PDF | - | stdio | 0 | 1 | 50 |
| Systems-Manager System Management Specialist | Expert specialist for system_management domain tasks. | You are a Systems-Manager System Management specialist. Help users manage and interact with System Management functionality using the available tools. | system_management | - | stdio | 0 | 6 | 81 |
| Systems-Manager Windows Specialist | Expert specialist for windows domain tasks. | You are a Systems-Manager Windows specialist. Help users manage and interact with Windows functionality using the available tools. | windows | - | stdio | 0 | 3 | 80 |
| Systems-Manager Linux Specialist | Expert specialist for linux domain tasks. | You are a Systems-Manager Linux specialist. Help users manage and interact with Linux functionality using the available tools. | linux | - | stdio | 0 | 3 | 83 |
| Systems-Manager Service Specialist | Expert specialist for service domain tasks. | You are a Systems-Manager Service specialist. Help users manage and interact with Service functionality using the available tools. | service | - | stdio | 0 | 7 | 60 |
| Systems-Manager Process Specialist | Expert specialist for process domain tasks. | You are a Systems-Manager Process specialist. Help users manage and interact with Process functionality using the available tools. | process | - | stdio | 0 | 3 | 68 |
| Systems-Manager Disk Specialist | Expert specialist for disk domain tasks. | You are a Systems-Manager Disk specialist. Help users manage and interact with Disk functionality using the available tools. | disk | - | stdio | 0 | 3 | 56 |
| Systems-Manager Cron Specialist | Expert specialist for cron domain tasks. | You are a Systems-Manager Cron specialist. Help users manage and interact with Cron functionality using the available tools. | cron | - | stdio | 0 | 3 | 56 |
| Systems-Manager Firewall Management Specialist | Expert specialist for firewall_management domain tasks. | You are a Systems-Manager Firewall Management specialist. Help users manage and interact with Firewall Management functionality using the available tools. | firewall_management | - | stdio | 0 | 4 | 70 |
| Systems-Manager Ssh Management Specialist | Expert specialist for ssh_management domain tasks. | You are a Systems-Manager Ssh Management specialist. Help users manage and interact with Ssh Management functionality using the available tools. | ssh_management | - | stdio | 0 | 3 | 63 |
| Systems-Manager Filesystem Specialist | Expert specialist for filesystem domain tasks. | You are a Systems-Manager Filesystem specialist. Help users manage and interact with Filesystem functionality using the available tools. | filesystem | - | stdio | 0 | 4 | 61 |
| Systems-Manager Shell Specialist | Expert specialist for shell domain tasks. | You are a Systems-Manager Shell specialist. Help users manage and interact with Shell functionality using the available tools. | shell | - | stdio | 0 | 1 | 55 |
| Systems-Manager Python Specialist | Expert specialist for python domain tasks. | You are a Systems-Manager Python specialist. Help users manage and interact with Python functionality using the available tools. | python | - | stdio | 0 | 3 | 50 |
| Systems-Manager Nodejs Specialist | Expert specialist for nodejs domain tasks. | You are a Systems-Manager Nodejs specialist. Help users manage and interact with Nodejs functionality using the available tools. | nodejs | - | stdio | 0 | 3 | 50 |
| Tunnel-Manager Host Management Specialist | Expert specialist for host_management domain tasks. | You are a Tunnel-Manager Host Management specialist. Help users manage and interact with Host Management functionality using the available tools. | host_management | - | stdio | 0 | 3 | 58 |
| Tunnel-Manager Remote Access Specialist | Expert specialist for remote_access domain tasks. | You are a Tunnel-Manager Remote Access specialist. Help users manage and interact with Remote Access functionality using the available tools. | remote_access | - | stdio | 0 | 15 | 74 |
| Uptime Specialist | Expert specialist for uptime domain tasks. | You are a Uptime specialist. Help users manage and interact with Uptime functionality using the available tools. | uptime | - | stdio | 0 | 9 | 39 |
| Wger Routine Specialist | Expert specialist for Routine domain tasks. | You are a Wger Routine specialist. Help users manage and interact with Routine functionality using the available tools. | Routine | - | stdio | 0 | 12 | 57 |
| Wger Routineconfig Specialist | Expert specialist for RoutineConfig domain tasks. | You are a Wger Routineconfig specialist. Help users manage and interact with Routineconfig functionality using the available tools. | RoutineConfig | - | stdio | 0 | 7 | 68 |
| Wger Exercise Specialist | Expert specialist for Exercise domain tasks. | You are a Wger Exercise specialist. Help users manage and interact with Exercise functionality using the available tools. | Exercise | - | stdio | 0 | 8 | 66 |
| Wger Workout Specialist | Expert specialist for Workout domain tasks. | You are a Wger Workout specialist. Help users manage and interact with Workout functionality using the available tools. | Workout | - | stdio | 0 | 7 | 62 |
| Wger Nutrition Specialist | Expert specialist for Nutrition domain tasks. | You are a Wger Nutrition specialist. Help users manage and interact with Nutrition functionality using the available tools. | Nutrition | - | stdio | 0 | 10 | 62 |
| Wger Body Specialist | Expert specialist for Body domain tasks. | You are a Wger Body specialist. Help users manage and interact with Body functionality using the available tools. | Body | - | stdio | 0 | 8 | 51 |

## Tool Inventory Table

| Tool Name | Description | Tag | Source | Score | Approval |
|-----------|-------------|-----|--------|-------|----------|
| get_version | Get AdGuard Home version. | system | adguard-home-agent | 45 | No |
| set_protection | Set protection state. | system | adguard-home-agent | 45 | Yes |
| clear_cache | Clear DNS cache. | system | adguard-home-agent | 50 | Yes |
| get_access_list | List current access list (allowed/disallowed clients, blocked hosts). | access | adguard-home-agent | 55 | No |
| set_access_list | Set access list. | access | adguard-home-agent | 45 | Yes |
| get_blocked_services_list | List blocked services. | blocked-services | adguard-home-agent | 60 | No |
| get_all_blocked_services | Get all available blocked services. | blocked-services | adguard-home-agent | 65 | No |
| update_blocked_services | Update blocked services list. | blocked-services | adguard-home-agent | 60 | Yes |
| set_filtering_rules | Set user-defined filtering rules. | filtering | adguard-home-agent | 60 | Yes |
| check_host_filtering | Check if a host is filtered. | filtering | adguard-home-agent | 65 | No |
| set_filter_url_params | Set filter URL parameters. | filtering | adguard-home-agent | 65 | Yes |
| get_filtering_status | Get filtering status. | filtering | adguard-home-agent | 60 | No |
| set_filtering_config | Set filtering configuration. | filtering | adguard-home-agent | 60 | Yes |
| add_filter_url | Add a filter URL. | filtering | adguard-home-agent | 65 | Yes |
| remove_filter_url | Remove a filter URL. | filtering | adguard-home-agent | 65 | Yes |
| refresh_filters | Refresh all filters. | filtering | adguard-home-agent | 60 | No |
| list_clients | List clients. | clients | adguard-home-agent | 50 | No |
| search_clients | Search for clients. | clients | adguard-home-agent | 60 | No |
| add_client | Add a new client. | clients | adguard-home-agent | 60 | Yes |
| update_client | Update a client. | clients | adguard-home-agent | 55 | Yes |
| delete_client | Delete a client. | clients | adguard-home-agent | 55 | Yes |
| get_profile | Get current user profile info. | profile | adguard-home-agent | 55 | No |
| update_profile | Update current user profile info. | profile | adguard-home-agent | 55 | Yes |
| get_dhcp_status | Get DHCP status. | dhcp | adguard-home-agent | 50 | No |
| get_dhcp_interfaces | Get available network interfaces for DHCP. | dhcp | adguard-home-agent | 50 | No |
| set_dhcp_config | Set DHCP configuration. | dhcp | adguard-home-agent | 50 | Yes |
| find_active_dhcp | Search for an active DHCP server on the network. | dhcp | adguard-home-agent | 55 | No |
| add_dhcp_static_lease | Add a static DHCP lease. | dhcp | adguard-home-agent | 55 | Yes |
| remove_dhcp_static_lease | Remove a static DHCP lease. | dhcp | adguard-home-agent | 55 | Yes |
| update_dhcp_static_lease | Update a static DHCP lease. | dhcp | adguard-home-agent | 55 | Yes |
| reset_dhcp | Reset DHCP configuration. | dhcp | adguard-home-agent | 50 | Yes |
| reset_dhcp_leases | Reset DHCP leases. | dhcp | adguard-home-agent | 55 | Yes |
| get_parental_status | Get parental control status. | settings | adguard-home-agent | 60 | No |
| enable_parental_control | Enable parental control. | settings | adguard-home-agent | 65 | Yes |
| disable_parental_control | Disable parental control. | settings | adguard-home-agent | 65 | Yes |
| get_safebrowsing_status | Get safe browsing status. | settings | adguard-home-agent | 60 | No |
| enable_safebrowsing | Enable safe browsing. | settings | adguard-home-agent | 60 | Yes |
| disable_safebrowsing | Disable safe browsing. | settings | adguard-home-agent | 60 | Yes |
| get_safesearch_status | Get safe search status. | settings | adguard-home-agent | 60 | No |
| get_query_log | Get query log. | query-log | adguard-home-agent | 55 | No |
| clear_query_log | Clear query log. | query-log | adguard-home-agent | 65 | Yes |
| list_rewrites | List DNS rewrites. | rewrites | adguard-home-agent | 55 | Yes |
| add_rewrite | Add a DNS rewrite. | rewrites | adguard-home-agent | 60 | Yes |
| delete_rewrite | Delete a DNS rewrite. | rewrites | adguard-home-agent | 55 | Yes |
| update_rewrite | Update a DNS rewrite. | rewrites | adguard-home-agent | 55 | Yes |
| get_rewrite_settings | Get rewrite settings. | rewrites | adguard-home-agent | 60 | Yes |
| update_rewrite_settings | Update rewrite settings. | rewrites | adguard-home-agent | 60 | Yes |
| get_tls_status | Get TLS status. | tls | adguard-home-agent | 45 | No |
| configure_tls | Configure TLS. | tls | adguard-home-agent | 45 | No |
| validate_tls | Validate TLS configuration. | tls | adguard-home-agent | 50 | No |
| get_doh_mobile_config | Get DNS over HTTPS .mobileconfig. | mobile | adguard-home-agent | 55 | No |
| get_dot_mobile_config | Get DNS over TLS .mobileconfig. | mobile | adguard-home-agent | 55 | No |
| get_stats | Get overall statistics. | stats | adguard-home-agent | 45 | No |
| reset_stats | Reset statistics. | stats | adguard-home-agent | 50 | Yes |
| get_stats_config | Get statistics configuration. | stats | adguard-home-agent | 50 | No |
| set_stats_config | Set statistics configuration. | stats | adguard-home-agent | 50 | Yes |
| get_dns_info | Get general DNS parameters. | dns | adguard-home-agent | 50 | No |
| set_dns_config | Set general DNS parameters. | dns | adguard-home-agent | 50 | Yes |
| test_upstream_dns | Test upstream configuration. | dns | adguard-home-agent | 55 | No |
| list_inventories | Retrieves a paginated list of inventories from Ansible Tower. Returns a list of dictionaries, each containing inventory details like id, name, and description. Display results in a markdown table for clarity. | inventory | ansible-tower-mcp | 75 | No |
| get_inventory | Fetches details of a specific inventory by ID from Ansible Tower. Returns a dictionary with inventory information such as name, description, and hosts count. | inventory | ansible-tower-mcp | 75 | No |
| create_inventory | Creates a new inventory in Ansible Tower. Returns a dictionary with the created inventory's details, including its ID. | inventory | ansible-tower-mcp | 75 | Yes |
| update_inventory | Updates an existing inventory in Ansible Tower. Returns a dictionary with the updated inventory's details. | inventory | ansible-tower-mcp | 75 | Yes |
| delete_inventory | Deletes a specific inventory by ID from Ansible Tower. Returns a dictionary confirming the deletion status. | inventory | ansible-tower-mcp | 75 | Yes |
| list_hosts | Retrieves a paginated list of hosts from Ansible Tower, optionally filtered by inventory. Returns a list of dictionaries, each with host details like id, name, and variables. Display in a markdown table. | hosts | ansible-tower-mcp | 65 | No |
| get_host | Fetches details of a specific host by ID from Ansible Tower. Returns a dictionary with host information such as name, variables, and inventory. | hosts | ansible-tower-mcp | 65 | No |
| create_host | Creates a new host in a specified inventory in Ansible Tower. Returns a dictionary with the created host's details, including its ID. | hosts | ansible-tower-mcp | 65 | Yes |
| update_host | Updates an existing host in Ansible Tower. Returns a dictionary with the updated host's details. | hosts | ansible-tower-mcp | 55 | Yes |
| delete_host | Deletes a specific host by ID from Ansible Tower. Returns a dictionary confirming the deletion status. | hosts | ansible-tower-mcp | 65 | Yes |
| list_groups | Retrieves a paginated list of groups in a specified inventory from Ansible Tower. Returns a list of dictionaries, each with group details like id, name, and variables. Display in a markdown table. | groups | ansible-tower-mcp | 65 | No |
| get_group | Fetches details of a specific group by ID from Ansible Tower. Returns a dictionary with group information such as name, variables, and inventory. | groups | ansible-tower-mcp | 65 | No |
| create_group | Creates a new group in a specified inventory in Ansible Tower. Returns a dictionary with the created group's details, including its ID. | groups | ansible-tower-mcp | 65 | Yes |
| update_group | Updates an existing group in Ansible Tower. Returns a dictionary with the updated group's details. | groups | ansible-tower-mcp | 55 | Yes |
| delete_group | Deletes a specific group by ID from Ansible Tower. Returns a dictionary confirming the deletion status. | groups | ansible-tower-mcp | 65 | Yes |
| add_host_to_group | Adds a host to a group in Ansible Tower. Returns a dictionary confirming the association. | groups | ansible-tower-mcp | 65 | Yes |
| remove_host_from_group | Removes a host from a group in Ansible Tower. Returns a dictionary confirming the disassociation. | groups | ansible-tower-mcp | 65 | Yes |
| list_job_templates | Retrieves a paginated list of job templates from Ansible Tower. Returns a list of dictionaries, each with template details like id, name, and playbook. Display in a markdown table. | job-templates | ansible-tower-mcp | 80 | No |
| get_job_template | Fetches details of a specific job template by ID from Ansible Tower. Returns a dictionary with template information such as name, inventory, and extra_vars. | job-templates | ansible-tower-mcp | 80 | No |
| create_job_template | Creates a new job template in Ansible Tower. Returns a dictionary with the created template's details, including its ID. | job-templates | ansible-tower-mcp | 80 | Yes |
| update_job_template | Updates an existing job template in Ansible Tower. Returns a dictionary with the updated template's details. | job-templates | ansible-tower-mcp | 80 | Yes |
| delete_job_template | Deletes a specific job template by ID from Ansible Tower. Returns a dictionary confirming the deletion status. | job-templates | ansible-tower-mcp | 80 | Yes |
| launch_job | Launches a job from a template in Ansible Tower, optionally with extra variables. Returns a dictionary with the launched job's details, including its ID. | job-templates | ansible-tower-mcp | 80 | No |
| list_jobs | Retrieves a paginated list of jobs from Ansible Tower, optionally filtered by status. Returns a list of dictionaries, each with job details like id, status, and elapsed time. Display in a markdown table. | jobs | ansible-tower-mcp | 65 | No |
| get_job | Fetches details of a specific job by ID from Ansible Tower. Returns a dictionary with job information such as status, start time, and artifacts. | jobs | ansible-tower-mcp | 65 | No |
| cancel_job | Cancels a running job in Ansible Tower. Returns a dictionary confirming the cancellation status. | jobs | ansible-tower-mcp | 60 | No |
| relaunch_job | Relaunches a job by getting its details and launching the same job template with the same variables. Returns a dictionary with the results of the new job. | jobs | ansible-tower-mcp | 70 | No |
| get_job_events | Retrieves a paginated list of events for a specific job from Ansible Tower. Returns a list of dictionaries, each with event details like type, host, and stdout. Display in a markdown table. | jobs | ansible-tower-mcp | 70 | No |
| get_job_stdout | Fetches the stdout output of a job in the specified format from Ansible Tower. Returns a dictionary with the output content. | jobs | ansible-tower-mcp | 70 | No |
| list_projects | Retrieves a paginated list of projects from Ansible Tower. Returns a list of dictionaries, each with project details like id, name, and scm_type. Display in a markdown table. | projects | ansible-tower-mcp | 75 | No |
| get_project | Fetches details of a specific project by ID from Ansible Tower. Returns a dictionary with project information such as name, scm_url, and status. | projects | ansible-tower-mcp | 75 | No |
| create_project | Creates a new project in Ansible Tower. Returns a dictionary with the created project's details, including its ID. | projects | ansible-tower-mcp | 75 | Yes |
| update_project | Updates an existing project in Ansible Tower. Returns a dictionary with the updated project's details. | projects | ansible-tower-mcp | 75 | Yes |
| delete_project | Deletes a specific project by ID from Ansible Tower. Returns a dictionary confirming the deletion status. | projects | ansible-tower-mcp | 75 | Yes |
| sync_project | Syncs (updates from SCM) a project in Ansible Tower. Returns a dictionary with the sync job's details. | projects | ansible-tower-mcp | 80 | No |
| list_credentials | Retrieves a paginated list of credentials from Ansible Tower. Returns a list of dictionaries, each with credential details like id, name, and type. Display in a markdown table. | credentials | ansible-tower-mcp | 75 | No |
| get_credential | Fetches details of a specific credential by ID from Ansible Tower. Returns a dictionary with credential information such as name and inputs (masked). | credentials | ansible-tower-mcp | 75 | No |
| list_credential_types | Retrieves a paginated list of credential types from Ansible Tower. Returns a list of dictionaries, each with type details like id and name. Display in a markdown table. | credentials | ansible-tower-mcp | 80 | No |
| create_credential | Creates a new credential in Ansible Tower. Returns a dictionary with the created credential's details, including its ID. | credentials | ansible-tower-mcp | 75 | Yes |
| update_credential | Updates an existing credential in Ansible Tower. Returns a dictionary with the updated credential's details. | credentials | ansible-tower-mcp | 75 | Yes |
| delete_credential | Deletes a specific credential by ID from Ansible Tower. Returns a dictionary confirming the deletion status. | credentials | ansible-tower-mcp | 75 | Yes |
| list_organizations | Retrieves a paginated list of organizations from Ansible Tower. Returns a list of dictionaries, each with organization details like id and name. Display in a markdown table. | organizations | ansible-tower-mcp | 75 | No |
| get_organization | Fetches details of a specific organization by ID from Ansible Tower. Returns a dictionary with organization information such as name and description. | organizations | ansible-tower-mcp | 75 | No |
| create_organization | Creates a new organization in Ansible Tower. Returns a dictionary with the created organization's details, including its ID. | organizations | ansible-tower-mcp | 75 | Yes |
| update_organization | Updates an existing organization in Ansible Tower. Returns a dictionary with the updated organization's details. | organizations | ansible-tower-mcp | 75 | Yes |
| delete_organization | Deletes a specific organization by ID from Ansible Tower. Returns a dictionary confirming the deletion status. | organizations | ansible-tower-mcp | 75 | Yes |
| list_teams | Retrieves a paginated list of teams from Ansible Tower, optionally filtered by organization. Returns a list of dictionaries, each with team details like id and name. Display in a markdown table. | teams | ansible-tower-mcp | 65 | No |
| get_team | Fetches details of a specific team by ID from Ansible Tower. Returns a dictionary with team information such as name and organization. | teams | ansible-tower-mcp | 65 | No |
| create_team | Creates a new team in a specified organization in Ansible Tower. Returns a dictionary with the created team's details, including its ID. | teams | ansible-tower-mcp | 65 | Yes |
| update_team | Updates an existing team in Ansible Tower. Returns a dictionary with the updated team's details. | teams | ansible-tower-mcp | 55 | Yes |
| delete_team | Deletes a specific team by ID from Ansible Tower. Returns a dictionary confirming the deletion status. | teams | ansible-tower-mcp | 65 | Yes |
| list_users | Retrieves a paginated list of users from Ansible Tower. Returns a list of dictionaries, each with user details like id, username, and email. Display in a markdown table. | users | ansible-tower-mcp | 65 | No |
| get_user | Fetches details of a specific user by ID from Ansible Tower. Returns a dictionary with user information such as username, email, and roles. | users | ansible-tower-mcp | 65 | No |
| create_user | Creates a new user in Ansible Tower. Returns a dictionary with the created user's details, including its ID. | users | ansible-tower-mcp | 65 | Yes |
| update_user | Updates an existing user in Ansible Tower. Returns a dictionary with the updated user's details. | users | ansible-tower-mcp | 55 | Yes |
| delete_user | Deletes a specific user by ID from Ansible Tower. Returns a dictionary confirming the deletion status. | users | ansible-tower-mcp | 65 | Yes |
| run_ad_hoc_command | Runs an ad hoc command on hosts in Ansible Tower. Returns a dictionary with the command job's details, including its ID. | ad_hoc_commands | ansible-tower-mcp | 80 | No |
| get_ad_hoc_command | Fetches details of a specific ad hoc command by ID from Ansible Tower. Returns a dictionary with command information such as status and module_args. | ad_hoc_commands | ansible-tower-mcp | 80 | No |
| cancel_ad_hoc_command | Cancels a running ad hoc command in Ansible Tower. Returns a dictionary confirming the cancellation status. | ad_hoc_commands | ansible-tower-mcp | 85 | No |
| list_workflow_templates | Retrieves a paginated list of workflow templates from Ansible Tower. Returns a list of dictionaries, each with template details like id and name. Display in a markdown table. | workflow_templates | ansible-tower-mcp | 80 | No |
| get_workflow_template | Fetches details of a specific workflow template by ID from Ansible Tower. Returns a dictionary with template information such as name and extra_vars. | workflow_templates | ansible-tower-mcp | 80 | No |
| launch_workflow | Launches a workflow from a template in Ansible Tower, optionally with extra variables. Returns a dictionary with the launched workflow job's details, including its ID. | workflow_templates | ansible-tower-mcp | 80 | No |
| list_workflow_jobs | Retrieves a paginated list of workflow jobs from Ansible Tower, optionally filtered by status. Returns a list of dictionaries, each with job details like id and status. Display in a markdown table. | workflow_jobs | ansible-tower-mcp | 80 | No |
| get_workflow_job | Fetches details of a specific workflow job by ID from Ansible Tower. Returns a dictionary with job information such as status and start time. | workflow_jobs | ansible-tower-mcp | 80 | No |
| cancel_workflow_job | Cancels a running workflow job in Ansible Tower. Returns a dictionary confirming the cancellation status. | workflow_jobs | ansible-tower-mcp | 85 | No |
| list_schedules | Retrieves a paginated list of schedules from Ansible Tower, optionally filtered by template. Returns a list of dictionaries, each with schedule details like id, name, and rrule. Display in a markdown table. | schedules | ansible-tower-mcp | 75 | No |
| get_schedule | Fetches details of a specific schedule by ID from Ansible Tower. Returns a dictionary with schedule information such as name and rrule. | schedules | ansible-tower-mcp | 75 | No |
| create_schedule | Creates a new schedule for a template in Ansible Tower. Returns a dictionary with the created schedule's details, including its ID. | schedules | ansible-tower-mcp | 75 | Yes |
| update_schedule | Updates an existing schedule in Ansible Tower. Returns a dictionary with the updated schedule's details. | schedules | ansible-tower-mcp | 75 | Yes |
| delete_schedule | Deletes a specific schedule by ID from Ansible Tower. Returns a dictionary confirming the deletion status. | schedules | ansible-tower-mcp | 75 | Yes |
| get_ansible_version | Retrieves the Ansible version information from Ansible Tower. Returns a dictionary with version details. | system | ansible-tower-mcp | 70 | No |
| get_dashboard_stats | Fetches dashboard statistics from Ansible Tower. Returns a dictionary with stats like host counts and recent jobs. | system | ansible-tower-mcp | 70 | No |
| get_metrics | Retrieves system metrics from Ansible Tower. Returns a dictionary with performance and usage metrics. | system | ansible-tower-mcp | 65 | No |
| get_api_token | Generate an API token for a given username & password. | authentication | archivebox-mcp | 70 | No |
| check_api_token | Validate an API token to make sure it's valid and non-expired. | authentication | archivebox-mcp | 75 | No |
| get_snapshots | Retrieve list of snapshots. | core | archivebox-mcp | 45 | No |
| get_snapshot | Get a specific Snapshot by abid or id. | core | archivebox-mcp | 45 | No |
| get_archiveresults | List all ArchiveResult entries matching these filters. | core | archivebox-mcp | 55 | No |
| get_tag | Get a specific Tag by id or abid. | core | archivebox-mcp | 45 | No |
| get_any | Get a specific Snapshot, ArchiveResult, or Tag by abid. | core | archivebox-mcp | 55 | No |
| cli_add | Execute archivebox add command. | cli | archivebox-mcp | 50 | Yes |
| cli_update | Execute archivebox update command. | cli | archivebox-mcp | 45 | Yes |
| cli_schedule | Execute archivebox schedule command. | cli | archivebox-mcp | 50 | No |
| cli_list | Execute archivebox list command. | cli | archivebox-mcp | 45 | No |
| cli_remove | Execute archivebox remove command. | cli | archivebox-mcp | 50 | Yes |
| bazarr_download_movie_subtitle | Download a subtitle for a movie. | bazarr | arr-mcp | 40 | No |
| bazarr_download_series_subtitle | Download a subtitle for an episode. | bazarr | arr-mcp | 40 | No |
| bazarr_get_episode_subtitles | Get subtitle information for a specific episode. | bazarr | arr-mcp | 40 | No |
| bazarr_get_movie_subtitles | Get subtitle information for a specific movie. | bazarr | arr-mcp | 40 | No |
| bazarr_get_movies | Get all movies managed by Bazarr. | bazarr | arr-mcp | 35 | No |
| bazarr_get_series | Get all series managed by Bazarr. | bazarr | arr-mcp | 35 | No |
| bazarr_get_series_subtitles | Get subtitle information for a specific series. | bazarr | arr-mcp | 40 | No |
| bazarr_get_wanted_movies | Get movies with wanted/missing subtitles. | bazarr | arr-mcp | 40 | No |
| bazarr_get_wanted_series | Get series episodes with wanted/missing subtitles. | bazarr | arr-mcp | 40 | No |
| bazarr_search_movie_subtitles | Search for subtitles for a movie. | bazarr | arr-mcp | 40 | No |
| bazarr_search_series_subtitles | Search for subtitles for a series or episode. | bazarr | arr-mcp | 40 | No |
| bazarr_get_history | Get subtitle download history. | bazarr | arr-mcp | 35 | No |
| chaptarr_delete_notification_id | Delete notification id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_delete_remotepathmapping_id | Delete remotepathmapping id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_delete_rootfolder_id | Delete rootfolder id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_get_notification_id | Get specific notification. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_remotepathmapping_id | Get specific remotepathmapping. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_rootfolder_id | Get specific rootfolder. | chaptarr | arr-mcp | 35 | No |
| chaptarr_post_notification | Add a new notification. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_notification_action_name | Add a new notification action name. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_notification_test | Test notification. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_remotepathmapping | Add a new remotepathmapping. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_rootfolder | Add a new rootfolder. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_notification_id | Update notification id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_remotepathmapping_id | Update remotepathmapping id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_rootfolder_id | Update rootfolder id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_delete_downloadclient_bulk | Delete downloadclient bulk. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_delete_downloadclient_id | Delete downloadclient id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_delete_importlist_bulk | Delete importlist bulk. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_delete_importlist_id | Delete importlist id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_delete_importlistexclusion_id | Delete importlistexclusion id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_get_config_downloadclient_id | Get specific config downloadclient. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_downloadclient_id | Get specific downloadclient. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_importlist_id | Get specific importlist. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_importlistexclusion_id | Get specific importlistexclusion. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_manualimport | Get manualimport. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_release | Get release. | chaptarr | arr-mcp | 30 | No |
| chaptarr_post_downloadclient | Add a new downloadclient. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_downloadclient_action_name | Add a new downloadclient action name. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_downloadclient_test | Test downloadclient. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_importlist | Add a new importlist. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_importlist_action_name | Add a new importlist action name. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_importlist_test | Test importlist. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_importlistexclusion | Add a new importlistexclusion. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_manualimport | Add a new manualimport. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_release | Add a new release. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_release_push | Add a new release push. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_config_downloadclient_id | Update config downloadclient id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_downloadclient_bulk | Update downloadclient bulk. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_downloadclient_id | Update downloadclient id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_importlist_bulk | Update importlist bulk. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_importlist_id | Update importlist id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_importlistexclusion_id | Update importlistexclusion id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_get_history | Get history. | chaptarr | arr-mcp | 30 | No |
| chaptarr_get_history_author | Get history author. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_history_since | Get history since. | chaptarr | arr-mcp | 40 | No |
| chaptarr_post_history_failed_id | Add a new history failed id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_delete_indexer_bulk | Delete indexer bulk. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_delete_indexer_id | Delete indexer id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_get_config_indexer_id | Get specific config indexer. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_indexer_id | Get specific indexer. | chaptarr | arr-mcp | 35 | No |
| chaptarr_post_indexer | Add a new indexer. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_indexer_action_name | Add a new indexer action name. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_indexer_test | Test indexer. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_put_config_indexer_id | Update config indexer id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_indexer_bulk | Update indexer bulk. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_indexer_id | Update indexer id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_delete_command_id | Delete command id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_get_calendar | Get calendar. | chaptarr | arr-mcp | 30 | No |
| chaptarr_get_calendar_id | Get specific calendar. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_command_id | Get specific command. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_feed_v1_calendar_readarrics | Get feed v1 calendar readarrics. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_parse | Get parse. | chaptarr | arr-mcp | 30 | No |
| chaptarr_post_command | Add a new command. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_delete_customfilter_id | Delete customfilter id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_delete_customformat_id | Delete customformat id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_delete_delayprofile_id | Delete delayprofile id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_delete_metadataprofile_id | Delete metadataprofile id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_delete_qualityprofile_id | Delete qualityprofile id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_delete_releaseprofile_id | Delete releaseprofile id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_get_config_mediamanagement_id | Get specific config mediamanagement. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_config_metadataprovider_id | Get specific config metadataprovider. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_config_naming_examples | Get config naming examples. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_config_naming_id | Get specific config naming. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_customfilter_id | Get specific customfilter. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_customformat_id | Get specific customformat. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_delayprofile_id | Get specific delayprofile. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_language_id | Get specific language. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_metadataprofile_id | Get specific metadataprofile. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_qualitydefinition_id | Get specific qualitydefinition. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_qualityprofile_id | Get specific qualityprofile. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_releaseprofile_id | Get specific releaseprofile. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_wanted_cutoff | Get wanted cutoff. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_wanted_cutoff_id | Get specific wanted cutoff. | chaptarr | arr-mcp | 40 | No |
| chaptarr_post_customfilter | Add a new customfilter. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_customformat | Add a new customformat. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_delayprofile | Add a new delayprofile. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_metadataprofile | Add a new metadataprofile. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_qualityprofile | Add a new qualityprofile. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_releaseprofile | Add a new releaseprofile. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_config_mediamanagement_id | Update config mediamanagement id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_config_metadataprovider_id | Update config metadataprovider id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_config_naming_id | Update config naming id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_customfilter_id | Update customfilter id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_customformat_id | Update customformat id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_delayprofile_id | Update delayprofile id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_delayprofile_reorder_id | Update delayprofile reorder id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_metadataprofile_id | Update metadataprofile id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_qualitydefinition_id | Update qualitydefinition id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_qualitydefinition_update | Update qualitydefinition update. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_qualityprofile_id | Update qualityprofile id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_releaseprofile_id | Update releaseprofile id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_delete_blocklist_bulk | Delete blocklist bulk. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_delete_blocklist_id | Delete blocklist id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_delete_queue_bulk | Delete queue bulk. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_delete_queue_id | Delete queue id. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_get_blocklist | Get blocklist. | chaptarr | arr-mcp | 30 | No |
| chaptarr_get_queue | Get queue. | chaptarr | arr-mcp | 30 | No |
| chaptarr_get_queue_details | Get queue details. | chaptarr | arr-mcp | 40 | No |
| chaptarr_post_queue_grab_bulk | Add a new queue grab bulk. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_queue_grab_id | Add a new queue grab id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_get_search | Get search. | chaptarr | arr-mcp | 30 | No |
| chaptarr_delete_system_backup_id | Delete system backup id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_delete_tag_id | Delete tag id. | chaptarr | arr-mcp | 30 | Yes |
| chaptarr_get_ | Get . | chaptarr | arr-mcp | 25 | No |
| chaptarr_get_config_development_id | Get specific config development. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_config_host_id | Get specific config host. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_config_ui_id | Get specific config ui. | chaptarr | arr-mcp | 35 | No |
| chaptarr_get_content_path | Get content path. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_filesystem | Get filesystem. | chaptarr | arr-mcp | 30 | No |
| chaptarr_get_filesystem_mediafiles | Get filesystem mediafiles. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_filesystem_type | Get filesystem type. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_log | Get log. | chaptarr | arr-mcp | 30 | No |
| chaptarr_get_log_file_filename | Get log file filename. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_log_file_update_filename | Get log file update filename. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_get_path | Get path. | chaptarr | arr-mcp | 30 | No |
| chaptarr_get_system_task_id | Get specific system task. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_tag_detail_id | Get specific tag detail. | chaptarr | arr-mcp | 40 | No |
| chaptarr_get_tag_id | Get specific tag. | chaptarr | arr-mcp | 35 | No |
| chaptarr_post_login | Log in to the Chaptarr instance. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_system_backup_restore_id | Add a new system backup restore id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_post_tag | Add a new tag. | chaptarr | arr-mcp | 35 | Yes |
| chaptarr_put_config_development_id | Update config development id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_config_host_id | Update config host id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_config_ui_id | Update config ui id. | chaptarr | arr-mcp | 40 | Yes |
| chaptarr_put_tag_id | Update tag id. | chaptarr | arr-mcp | 35 | Yes |
| lidarr_delete_album_id | Delete an album and optionally its files and add exclusion. | lidarr | arr-mcp | 45 | Yes |
| lidarr_delete_artist_editor | Delete multiple artists using the artist editor. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_artist_id | Delete artist id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_delete_metadata_id | Delete metadata id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_delete_trackfile_bulk | Delete trackfile bulk. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_trackfile_id | Delete trackfile id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_get_album | Get albums managed by Lidarr. | lidarr | arr-mcp | 35 | No |
| lidarr_get_album_id | Get details for a specific album by ID. | lidarr | arr-mcp | 35 | No |
| lidarr_get_album_lookup | Search for new albums to add to Lidarr. | lidarr | arr-mcp | 40 | No |
| lidarr_get_artist | Get all artists managed by Lidarr. | lidarr | arr-mcp | 35 | No |
| lidarr_get_artist_id | Get details for a specific artist by ID. | lidarr | arr-mcp | 35 | No |
| lidarr_get_artist_lookup | Search for new artists to add to Lidarr. | lidarr | arr-mcp | 40 | No |
| lidarr_get_mediacover_album_album_id_filename | Get specific mediacover album album filename. | lidarr | arr-mcp | 40 | No |
| lidarr_get_mediacover_artist_artist_id_filename | Get specific mediacover artist artist filename. | lidarr | arr-mcp | 40 | No |
| lidarr_get_metadata_id | Get specific metadata. | lidarr | arr-mcp | 35 | No |
| lidarr_get_rename | Get rename. | lidarr | arr-mcp | 30 | Yes |
| lidarr_get_retag | Get retag. | lidarr | arr-mcp | 30 | No |
| lidarr_get_track | Get track. | lidarr | arr-mcp | 30 | No |
| lidarr_get_track_id | Get specific track. | lidarr | arr-mcp | 35 | No |
| lidarr_get_trackfile | Get trackfile. | lidarr | arr-mcp | 30 | No |
| lidarr_get_trackfile_id | Get specific trackfile. | lidarr | arr-mcp | 35 | No |
| lidarr_get_wanted_missing | Get wanted missing. | lidarr | arr-mcp | 40 | No |
| lidarr_get_wanted_missing_id | Get specific wanted missing. | lidarr | arr-mcp | 40 | No |
| lidarr_post_album | Add a new album to Lidarr. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_albumstudio | Perform studio operations on albums. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_artist | Add a new artist to Lidarr. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_metadata | Add a new metadata. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_metadata_action_name | Add a new metadata action name. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_metadata_test | Test metadata. | lidarr | arr-mcp | 35 | Yes |
| lidarr_put_album_id | Update an existing album by ID. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_album_monitor | Update monitoring status for multiple albums. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_artist_editor | Update monitoring or tagging for multiple artists. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_artist_id | Update an existing artist configuration. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_metadata_id | Update metadata id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_trackfile_editor | Update trackfile editor. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_trackfile_id | Update trackfile id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_notification_id | Delete notification id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_delete_remotepathmapping_id | Delete remotepathmapping id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_delete_rootfolder_id | Delete rootfolder id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_get_notification_id | Get specific notification. | lidarr | arr-mcp | 35 | No |
| lidarr_get_remotepathmapping_id | Get specific remotepathmapping. | lidarr | arr-mcp | 35 | No |
| lidarr_get_rootfolder_id | Get specific rootfolder. | lidarr | arr-mcp | 35 | No |
| lidarr_post_notification | Add a new notification. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_notification_action_name | Add a new notification action name. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_notification_test | Test notification. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_remotepathmapping | Add a new remotepathmapping. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_rootfolder | Add a new root folder. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_notification_id | Update notification id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_remotepathmapping_id | Update remotepathmapping id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_rootfolder_id | Update rootfolder id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_downloadclient_bulk | Delete downloadclient bulk. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_downloadclient_id | Delete downloadclient id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_delete_importlist_bulk | Delete multiple import lists in bulk. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_importlist_id | Delete an import list configuration. | lidarr | arr-mcp | 35 | Yes |
| lidarr_delete_importlistexclusion_id | Delete importlistexclusion id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_get_config_downloadclient_id | Get specific config downloadclient. | lidarr | arr-mcp | 40 | No |
| lidarr_get_downloadclient_id | Get specific downloadclient. | lidarr | arr-mcp | 35 | No |
| lidarr_get_importlist_id | Get details for a specific import list by ID. | lidarr | arr-mcp | 35 | No |
| lidarr_get_importlistexclusion_id | Get specific importlistexclusion. | lidarr | arr-mcp | 35 | No |
| lidarr_get_manualimport | Get manualimport. | lidarr | arr-mcp | 35 | No |
| lidarr_get_release | Get release. | lidarr | arr-mcp | 30 | No |
| lidarr_post_downloadclient | Add a new downloadclient. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_downloadclient_action_name | Add a new downloadclient action name. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_downloadclient_test | Test downloadclient. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_importlist | Add a new import list. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_importlist_action_name | Perform a specific action on import lists. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_importlist_test | Test an import list configuration. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_importlistexclusion | Add a new importlistexclusion. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_manualimport | Add a new manualimport. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_release | Add a new release. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_release_push | Add a new release push. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_config_downloadclient_id | Update config downloadclient id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_downloadclient_bulk | Update downloadclient bulk. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_downloadclient_id | Update downloadclient id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_importlist_bulk | Update multiple import lists in bulk. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_importlist_id | Update an existing import list configuration. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_importlistexclusion_id | Update importlistexclusion id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_get_history | Get history. | lidarr | arr-mcp | 30 | No |
| lidarr_get_history_artist | Get history artist. | lidarr | arr-mcp | 40 | No |
| lidarr_get_history_since | Get history since. | lidarr | arr-mcp | 40 | No |
| lidarr_post_history_failed_id | Add a new history failed id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_indexer_bulk | Delete indexer bulk. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_indexer_id | Delete an indexer configuration by ID. | lidarr | arr-mcp | 35 | Yes |
| lidarr_get_config_indexer_id | Get specific config indexer. | lidarr | arr-mcp | 40 | No |
| lidarr_get_indexer_id | Get details for a specific indexer by ID. | lidarr | arr-mcp | 35 | No |
| lidarr_post_indexer | Add a new indexer configuration. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_indexer_action_name | Add a new indexer action name. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_indexer_test | Test indexer. | lidarr | arr-mcp | 35 | Yes |
| lidarr_put_config_indexer_id | Update config indexer id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_indexer_bulk | Update indexer bulk. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_indexer_id | Update an existing indexer configuration. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_autotagging_id | Delete autotagging id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_delete_command_id | Delete command id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_get_autotagging_id | Get specific autotagging. | lidarr | arr-mcp | 35 | No |
| lidarr_get_calendar | Get calendar. | lidarr | arr-mcp | 30 | No |
| lidarr_get_calendar_id | Get specific calendar. | lidarr | arr-mcp | 35 | No |
| lidarr_get_command_id | Get specific command. | lidarr | arr-mcp | 35 | No |
| lidarr_get_feed_v1_calendar_lidarrics | Get feed v1 calendar lidarrics. | lidarr | arr-mcp | 40 | No |
| lidarr_get_parse | Get parse. | lidarr | arr-mcp | 30 | No |
| lidarr_post_autotagging | Add a new autotagging. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_command | Add a new command. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_autotagging_id | Update autotagging id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_customfilter_id | Delete customfilter id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_delete_customformat_bulk | Delete customformat bulk. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_customformat_id | Delete customformat id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_delete_delayprofile_id | Delete delayprofile id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_delete_metadataprofile_id | Delete metadataprofile id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_delete_qualityprofile_id | Delete qualityprofile id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_delete_releaseprofile_id | Delete releaseprofile id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_get_config_mediamanagement_id | Get specific config mediamanagement. | lidarr | arr-mcp | 40 | No |
| lidarr_get_config_metadataprovider_id | Get specific config metadataprovider. | lidarr | arr-mcp | 40 | No |
| lidarr_get_config_naming_examples | Get config naming examples. | lidarr | arr-mcp | 40 | No |
| lidarr_get_config_naming_id | Get specific config naming. | lidarr | arr-mcp | 40 | No |
| lidarr_get_customfilter_id | Get specific customfilter. | lidarr | arr-mcp | 35 | No |
| lidarr_get_customformat_id | Get specific customformat. | lidarr | arr-mcp | 35 | No |
| lidarr_get_delayprofile_id | Get specific delayprofile. | lidarr | arr-mcp | 35 | No |
| lidarr_get_language_id | Get specific language. | lidarr | arr-mcp | 35 | No |
| lidarr_get_metadataprofile_id | Get specific metadataprofile. | lidarr | arr-mcp | 35 | No |
| lidarr_get_qualitydefinition_id | Get specific qualitydefinition. | lidarr | arr-mcp | 35 | No |
| lidarr_get_qualityprofile_id | Get specific qualityprofile. | lidarr | arr-mcp | 35 | No |
| lidarr_get_releaseprofile_id | Get specific releaseprofile. | lidarr | arr-mcp | 35 | No |
| lidarr_get_wanted_cutoff | Get wanted cutoff. | lidarr | arr-mcp | 40 | No |
| lidarr_get_wanted_cutoff_id | Get specific wanted cutoff. | lidarr | arr-mcp | 40 | No |
| lidarr_post_customfilter | Add a new customfilter. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_customformat | Add a new customformat. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_delayprofile | Add a new delayprofile. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_metadataprofile | Add a new metadataprofile. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_qualityprofile | Add a new qualityprofile. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_releaseprofile | Add a new release profile configuration. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_config_mediamanagement_id | Update config mediamanagement id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_config_metadataprovider_id | Update config metadataprovider id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_config_naming_id | Update config naming id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_customfilter_id | Update customfilter id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_customformat_bulk | Update customformat bulk. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_customformat_id | Update customformat id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_delayprofile_id | Update delayprofile id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_delayprofile_reorder_id | Update delayprofile reorder id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_metadataprofile_id | Update metadataprofile id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_qualitydefinition_id | Update qualitydefinition id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_qualitydefinition_update | Update qualitydefinition update. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_qualityprofile_id | Update qualityprofile id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_releaseprofile_id | Update releaseprofile id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_blocklist_bulk | Delete blocklist bulk. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_blocklist_id | Delete blocklist id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_delete_queue_bulk | Delete queue bulk. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_queue_id | Delete queue id. | lidarr | arr-mcp | 35 | Yes |
| lidarr_get_blocklist | Get blocklist. | lidarr | arr-mcp | 30 | No |
| lidarr_get_queue | Get queue. | lidarr | arr-mcp | 30 | No |
| lidarr_get_queue_details | Get queue details. | lidarr | arr-mcp | 40 | No |
| lidarr_post_queue_grab_bulk | Add a new queue grab bulk. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_queue_grab_id | Add a new queue grab id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_get_search | Get search. | lidarr | arr-mcp | 30 | No |
| lidarr_delete_system_backup_id | Delete system backup id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_delete_tag_id | Delete tag id. | lidarr | arr-mcp | 30 | Yes |
| lidarr_get_ | Get . | lidarr | arr-mcp | 25 | No |
| lidarr_get_config_host_id | Get specific config host. | lidarr | arr-mcp | 40 | No |
| lidarr_get_config_ui_id | Get specific config ui. | lidarr | arr-mcp | 35 | No |
| lidarr_get_content_path | Get content path. | lidarr | arr-mcp | 40 | No |
| lidarr_get_filesystem | Get filesystem. | lidarr | arr-mcp | 30 | No |
| lidarr_get_filesystem_mediafiles | Get filesystem mediafiles. | lidarr | arr-mcp | 40 | No |
| lidarr_get_filesystem_type | Get filesystem type. | lidarr | arr-mcp | 40 | No |
| lidarr_get_log | Get log. | lidarr | arr-mcp | 30 | No |
| lidarr_get_log_file_filename | Get log file filename. | lidarr | arr-mcp | 40 | No |
| lidarr_get_log_file_update_filename | Get log file update filename. | lidarr | arr-mcp | 40 | Yes |
| lidarr_get_path | Get path. | lidarr | arr-mcp | 30 | No |
| lidarr_get_system_task_id | Get specific system task. | lidarr | arr-mcp | 40 | No |
| lidarr_get_tag_detail_id | Get specific tag detail. | lidarr | arr-mcp | 40 | No |
| lidarr_get_tag_id | Get specific tag. | lidarr | arr-mcp | 35 | No |
| lidarr_post_login | Add a new login. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_system_backup_restore_id | Add a new system backup restore id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_post_tag | Add a new tag. | lidarr | arr-mcp | 35 | Yes |
| lidarr_put_config_host_id | Update config host id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_config_ui_id | Update config ui id. | lidarr | arr-mcp | 40 | Yes |
| lidarr_put_tag_id | Update tag id. | lidarr | arr-mcp | 35 | Yes |
| prowlarr_delete_notification_id | Delete notification id. | prowlarr | arr-mcp | 35 | Yes |
| prowlarr_get_notification_id | Get specific notification. | prowlarr | arr-mcp | 35 | No |
| prowlarr_post_notification | Add a new notification. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_notification_action_name | Add a new notification action name. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_notification_test | Test notification. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_put_notification_id | Update notification id. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_delete_downloadclient_bulk | Delete downloadclient bulk. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_delete_downloadclient_id | Delete downloadclient id. | prowlarr | arr-mcp | 35 | Yes |
| prowlarr_get_config_downloadclient_id | Get specific config downloadclient. | prowlarr | arr-mcp | 40 | No |
| prowlarr_get_downloadclient_id | Get specific downloadclient. | prowlarr | arr-mcp | 35 | No |
| prowlarr_post_downloadclient | Add a new downloadclient. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_downloadclient_action_name | Add a new downloadclient action name. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_downloadclient_test | Test downloadclient. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_put_config_downloadclient_id | Update config downloadclient id. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_put_downloadclient_bulk | Update downloadclient bulk. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_put_downloadclient_id | Update downloadclient id. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_get_history | Get history. | prowlarr | arr-mcp | 30 | No |
| prowlarr_get_history_indexer | Get history indexer. | prowlarr | arr-mcp | 40 | No |
| prowlarr_get_history_since | Get history since. | prowlarr | arr-mcp | 40 | No |
| prowlarr_delete_indexer_bulk | Delete indexer bulk. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_delete_indexer_id | Delete indexer id. | prowlarr | arr-mcp | 35 | Yes |
| prowlarr_delete_indexerproxy_id | Delete indexerproxy id. | prowlarr | arr-mcp | 35 | Yes |
| prowlarr_get_id_api | Get results for a specific indexer endpoint in Newznab format. | prowlarr | arr-mcp | 45 | No |
| prowlarr_get_id_download | Get specific id download. | prowlarr | arr-mcp | 35 | No |
| prowlarr_get_indexer_id | Get specific indexer. | prowlarr | arr-mcp | 35 | No |
| prowlarr_get_indexer_id_download | Download a release from a specific indexer. | prowlarr | arr-mcp | 40 | No |
| prowlarr_get_indexer_id_newznab | Get specific indexer newznab. | prowlarr | arr-mcp | 40 | No |
| prowlarr_get_indexerproxy_id | Get specific indexerproxy. | prowlarr | arr-mcp | 35 | No |
| prowlarr_get_indexerstats | Get indexerstats. | prowlarr | arr-mcp | 35 | No |
| prowlarr_post_indexer | Add a new indexer. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_indexer_action_name | Add a new indexer action name. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_indexer_test | Test indexer. | prowlarr | arr-mcp | 35 | Yes |
| prowlarr_post_indexerproxy | Add a new indexerproxy. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_indexerproxy_action_name | Add a new indexerproxy action name. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_indexerproxy_test | Test indexerproxy. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_put_indexer_bulk | Update indexer bulk. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_put_indexer_id | Update indexer id. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_put_indexerproxy_id | Update indexerproxy id. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_delete_command_id | Delete command id. | prowlarr | arr-mcp | 35 | Yes |
| prowlarr_get_command_id | Get specific command. | prowlarr | arr-mcp | 35 | No |
| prowlarr_post_command | Add a new command. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_delete_customfilter_id | Delete customfilter id. | prowlarr | arr-mcp | 35 | Yes |
| prowlarr_get_customfilter_id | Get specific customfilter. | prowlarr | arr-mcp | 35 | No |
| prowlarr_post_customfilter | Add a new customfilter. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_put_customfilter_id | Update customfilter id. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_get_search | Get search. | prowlarr | arr-mcp | 30 | No |
| prowlarr_post_search | Perform a bulk search across multiple indexers. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_search_bulk | Add a new search bulk. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_search | Search for indexers using the search endpoint. | prowlarr | arr-mcp | 35 | No |
| prowlarr_delete_applications_bulk | Delete applications bulk. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_delete_applications_id | Delete an application configuration. | prowlarr | arr-mcp | 35 | Yes |
| prowlarr_delete_appprofile_id | Delete appprofile id. | prowlarr | arr-mcp | 35 | Yes |
| prowlarr_delete_system_backup_id | Delete system backup id. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_delete_tag_id | Delete tag id. | prowlarr | arr-mcp | 30 | Yes |
| prowlarr_get_ | Get . | prowlarr | arr-mcp | 25 | No |
| prowlarr_get_applications_id | Get details for a specific application by ID. | prowlarr | arr-mcp | 35 | No |
| prowlarr_get_appprofile_id | Get specific appprofile. | prowlarr | arr-mcp | 35 | No |
| prowlarr_get_config_development_id | Get specific config development. | prowlarr | arr-mcp | 40 | No |
| prowlarr_get_config_host_id | Get specific config host. | prowlarr | arr-mcp | 40 | No |
| prowlarr_get_config_ui_id | Get specific config ui. | prowlarr | arr-mcp | 35 | No |
| prowlarr_get_content_path | Get content path. | prowlarr | arr-mcp | 40 | No |
| prowlarr_get_filesystem | Get filesystem. | prowlarr | arr-mcp | 30 | No |
| prowlarr_get_filesystem_type | Get filesystem type. | prowlarr | arr-mcp | 40 | No |
| prowlarr_get_log | Get log. | prowlarr | arr-mcp | 30 | No |
| prowlarr_get_log_file_filename | Get log file filename. | prowlarr | arr-mcp | 40 | No |
| prowlarr_get_log_file_update_filename | Get log file update filename. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_get_path | Get path. | prowlarr | arr-mcp | 30 | No |
| prowlarr_get_system_task_id | Get specific system task. | prowlarr | arr-mcp | 40 | No |
| prowlarr_get_tag_detail_id | Get specific tag detail. | prowlarr | arr-mcp | 40 | No |
| prowlarr_get_tag_id | Get specific tag. | prowlarr | arr-mcp | 35 | No |
| prowlarr_post_applications | Add a new applications. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_applications_action_name | Add a new applications action name. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_applications_test | Test applications. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_appprofile | Add a new appprofile. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_login | Add a new login. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_system_backup_restore_id | Add a new system backup restore id. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_post_tag | Add a new tag. | prowlarr | arr-mcp | 35 | Yes |
| prowlarr_put_applications_bulk | Update applications bulk. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_put_applications_id | Update an existing application configuration. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_put_appprofile_id | Update appprofile id. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_put_config_development_id | Update config development id. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_put_config_host_id | Update config host id. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_put_config_ui_id | Update config ui id. | prowlarr | arr-mcp | 40 | Yes |
| prowlarr_put_tag_id | Update tag id. | prowlarr | arr-mcp | 35 | Yes |
| radarr_add_movie | Lookup a movie by term, pick the first result, and add it to Radarr. | radarr | arr-mcp | 50 | Yes |
| radarr_delete_metadata_id | Delete metadata id. | radarr | arr-mcp | 35 | Yes |
| radarr_delete_movie_editor | Delete movie editor. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_movie_id | Delete movie id. | radarr | arr-mcp | 35 | Yes |
| radarr_delete_moviefile_bulk | Delete moviefile bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_moviefile_id | Delete moviefile id. | radarr | arr-mcp | 35 | Yes |
| radarr_get_alttitle | Get alternative titles for movies. | radarr | arr-mcp | 35 | No |
| radarr_get_alttitle_id | Get a specific alternative title by ID. | radarr | arr-mcp | 35 | No |
| radarr_get_collection | Get collection. | radarr | arr-mcp | 30 | No |
| radarr_get_collection_id | Get specific collection. | radarr | arr-mcp | 35 | No |
| radarr_get_credit | Get credit. | radarr | arr-mcp | 30 | No |
| radarr_get_credit_id | Get specific credit. | radarr | arr-mcp | 35 | No |
| radarr_get_extrafile | Get extrafile. | radarr | arr-mcp | 30 | No |
| radarr_get_importlist_movie | Get importlist movie. | radarr | arr-mcp | 40 | No |
| radarr_get_mediacover_movie_id_filename | Get specific mediacover movie filename. | radarr | arr-mcp | 40 | No |
| radarr_get_metadata_id | Get specific metadata. | radarr | arr-mcp | 35 | No |
| radarr_get_movie | Get movie. | radarr | arr-mcp | 30 | No |
| radarr_get_movie_id | Get specific movie. | radarr | arr-mcp | 35 | No |
| radarr_get_movie_id_folder | Get specific movie folder. | radarr | arr-mcp | 40 | No |
| radarr_get_movie_lookup | Get movie lookup. | radarr | arr-mcp | 40 | No |
| radarr_get_movie_lookup_imdb | Get movie lookup imdb. | radarr | arr-mcp | 40 | No |
| radarr_get_movie_lookup_tmdb | Get movie lookup tmdb. | radarr | arr-mcp | 40 | No |
| radarr_get_moviefile | Get moviefile. | radarr | arr-mcp | 30 | No |
| radarr_get_moviefile_id | Get specific moviefile. | radarr | arr-mcp | 35 | No |
| radarr_get_rename | Get rename. | radarr | arr-mcp | 30 | Yes |
| radarr_get_wanted_missing | Get wanted missing. | radarr | arr-mcp | 40 | No |
| radarr_lookup_movie | Search for a movie using the lookup endpoint. | radarr | arr-mcp | 40 | No |
| radarr_post_importlist_movie | Add a new importlist movie. | radarr | arr-mcp | 40 | Yes |
| radarr_post_metadata | Add a new metadata. | radarr | arr-mcp | 40 | Yes |
| radarr_post_metadata_action_name | Add a new metadata action name. | radarr | arr-mcp | 40 | Yes |
| radarr_post_metadata_test | Test metadata. | radarr | arr-mcp | 35 | Yes |
| radarr_post_movie | Add a new movie to Radarr. | radarr | arr-mcp | 40 | Yes |
| radarr_post_movie_import | Add a new movie import. | radarr | arr-mcp | 40 | Yes |
| radarr_put_collection | Update collection. | radarr | arr-mcp | 40 | Yes |
| radarr_put_collection_id | Update collection id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_metadata_id | Update metadata id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_movie_editor | Update movie editor. | radarr | arr-mcp | 40 | Yes |
| radarr_put_movie_id | Update an existing movie configuration. | radarr | arr-mcp | 40 | Yes |
| radarr_put_moviefile_bulk | Update moviefile bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_put_moviefile_editor | Update moviefile editor. | radarr | arr-mcp | 40 | Yes |
| radarr_put_moviefile_id | Update moviefile id. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_notification_id | Delete notification id. | radarr | arr-mcp | 35 | Yes |
| radarr_delete_remotepathmapping_id | Delete remotepathmapping id. | radarr | arr-mcp | 35 | Yes |
| radarr_delete_rootfolder_id | Delete rootfolder id. | radarr | arr-mcp | 35 | Yes |
| radarr_get_notification_id | Get specific notification. | radarr | arr-mcp | 35 | No |
| radarr_get_remotepathmapping_id | Get specific remotepathmapping. | radarr | arr-mcp | 35 | No |
| radarr_get_rootfolder_id | Get specific rootfolder. | radarr | arr-mcp | 35 | No |
| radarr_post_notification | Add a new notification. | radarr | arr-mcp | 40 | Yes |
| radarr_post_notification_action_name | Add a new notification action name. | radarr | arr-mcp | 40 | Yes |
| radarr_post_notification_test | Test notification. | radarr | arr-mcp | 40 | Yes |
| radarr_post_remotepathmapping | Add a new remotepathmapping. | radarr | arr-mcp | 40 | Yes |
| radarr_post_rootfolder | Add a new rootfolder. | radarr | arr-mcp | 40 | Yes |
| radarr_put_notification_id | Update notification id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_remotepathmapping_id | Update remotepathmapping id. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_downloadclient_bulk | Delete downloadclient bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_downloadclient_id | Delete downloadclient id. | radarr | arr-mcp | 35 | Yes |
| radarr_delete_exclusions_bulk | Delete exclusions bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_exclusions_id | Delete exclusions id. | radarr | arr-mcp | 35 | Yes |
| radarr_delete_importlist_bulk | Delete importlist bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_importlist_id | Delete importlist id. | radarr | arr-mcp | 35 | Yes |
| radarr_get_config_downloadclient_id | Get specific config downloadclient. | radarr | arr-mcp | 40 | No |
| radarr_get_config_importlist_id | Get specific config importlist. | radarr | arr-mcp | 40 | No |
| radarr_get_downloadclient_id | Get specific downloadclient. | radarr | arr-mcp | 35 | No |
| radarr_get_exclusions_id | Get specific exclusions. | radarr | arr-mcp | 35 | No |
| radarr_get_exclusions_paged | Get exclusions paged. | radarr | arr-mcp | 40 | No |
| radarr_get_importlist_id | Get specific importlist. | radarr | arr-mcp | 35 | No |
| radarr_get_manualimport | Get manualimport. | radarr | arr-mcp | 35 | No |
| radarr_get_release | Get release. | radarr | arr-mcp | 30 | No |
| radarr_post_downloadclient | Add a new downloadclient. | radarr | arr-mcp | 40 | Yes |
| radarr_post_downloadclient_action_name | Add a new downloadclient action name. | radarr | arr-mcp | 40 | Yes |
| radarr_post_downloadclient_test | Test downloadclient. | radarr | arr-mcp | 40 | Yes |
| radarr_post_exclusions | Add a new exclusions. | radarr | arr-mcp | 40 | Yes |
| radarr_post_exclusions_bulk | Add a new exclusions bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_post_importlist | Add a new importlist. | radarr | arr-mcp | 40 | Yes |
| radarr_post_importlist_action_name | Add a new importlist action name. | radarr | arr-mcp | 40 | Yes |
| radarr_post_importlist_test | Test importlist. | radarr | arr-mcp | 40 | Yes |
| radarr_post_manualimport | Add a new manualimport. | radarr | arr-mcp | 40 | Yes |
| radarr_post_release | Add a new release. | radarr | arr-mcp | 40 | Yes |
| radarr_post_release_push | Add a new release push. | radarr | arr-mcp | 40 | Yes |
| radarr_put_config_downloadclient_id | Update config downloadclient id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_config_importlist_id | Update config importlist id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_downloadclient_bulk | Update downloadclient bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_put_downloadclient_id | Update downloadclient id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_exclusions_id | Update exclusions id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_importlist_bulk | Update importlist bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_put_importlist_id | Update importlist id. | radarr | arr-mcp | 40 | Yes |
| radarr_get_history | Get history. | radarr | arr-mcp | 30 | No |
| radarr_get_history_movie | Get history movie. | radarr | arr-mcp | 40 | No |
| radarr_get_history_since | Get history since. | radarr | arr-mcp | 40 | No |
| radarr_post_history_failed_id | Add a new history failed id. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_indexer_bulk | Delete indexer bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_indexer_id | Delete indexer id. | radarr | arr-mcp | 35 | Yes |
| radarr_get_config_indexer_id | Get specific config indexer. | radarr | arr-mcp | 40 | No |
| radarr_get_indexer_id | Get specific indexer. | radarr | arr-mcp | 35 | No |
| radarr_post_indexer | Add a new indexer configuration. | radarr | arr-mcp | 40 | Yes |
| radarr_post_indexer_action_name | Add a new indexer action name. | radarr | arr-mcp | 40 | Yes |
| radarr_post_indexer_test | Test indexer. | radarr | arr-mcp | 35 | Yes |
| radarr_put_config_indexer_id | Update config indexer id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_indexer_bulk | Update indexer bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_put_indexer_id | Update an existing indexer configuration by ID. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_autotagging_id | Delete autotagging id. | radarr | arr-mcp | 35 | Yes |
| radarr_delete_command_id | Delete command id. | radarr | arr-mcp | 35 | Yes |
| radarr_get_autotagging_id | Get specific autotagging. | radarr | arr-mcp | 35 | No |
| radarr_get_calendar | Get calendar. | radarr | arr-mcp | 30 | No |
| radarr_get_command_id | Get specific command. | radarr | arr-mcp | 35 | No |
| radarr_get_feed_v3_calendar_radarrics | Get feed v3 calendar radarrics. | radarr | arr-mcp | 40 | No |
| radarr_get_parse | Get parse. | radarr | arr-mcp | 30 | No |
| radarr_post_autotagging | Add a new autotagging. | radarr | arr-mcp | 40 | Yes |
| radarr_post_command | Add a new command. | radarr | arr-mcp | 40 | Yes |
| radarr_put_autotagging_id | Update autotagging id. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_customfilter_id | Delete customfilter id. | radarr | arr-mcp | 35 | Yes |
| radarr_delete_customformat_bulk | Delete customformat bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_customformat_id | Delete customformat id. | radarr | arr-mcp | 35 | Yes |
| radarr_delete_delayprofile_id | Delete delayprofile id. | radarr | arr-mcp | 35 | Yes |
| radarr_delete_qualityprofile_id | Delete qualityprofile id. | radarr | arr-mcp | 35 | Yes |
| radarr_delete_releaseprofile_id | Delete releaseprofile id. | radarr | arr-mcp | 35 | Yes |
| radarr_get_config_mediamanagement_id | Get specific config mediamanagement. | radarr | arr-mcp | 40 | No |
| radarr_get_config_metadata_id | Get specific config metadata. | radarr | arr-mcp | 40 | No |
| radarr_get_config_naming_examples | Get config naming examples. | radarr | arr-mcp | 40 | No |
| radarr_get_config_naming_id | Get specific config naming. | radarr | arr-mcp | 40 | No |
| radarr_get_customfilter_id | Get specific customfilter. | radarr | arr-mcp | 35 | No |
| radarr_get_customformat_id | Get specific customformat. | radarr | arr-mcp | 35 | No |
| radarr_get_delayprofile_id | Get specific delayprofile. | radarr | arr-mcp | 35 | No |
| radarr_get_language_id | Get specific language. | radarr | arr-mcp | 35 | No |
| radarr_get_qualitydefinition_id | Get specific qualitydefinition. | radarr | arr-mcp | 35 | No |
| radarr_get_qualityprofile_id | Get specific qualityprofile. | radarr | arr-mcp | 35 | No |
| radarr_get_releaseprofile_id | Get specific releaseprofile. | radarr | arr-mcp | 35 | No |
| radarr_get_wanted_cutoff | Get wanted cutoff. | radarr | arr-mcp | 40 | No |
| radarr_post_customfilter | Add a new customfilter. | radarr | arr-mcp | 40 | Yes |
| radarr_post_customformat | Add a new customformat. | radarr | arr-mcp | 40 | Yes |
| radarr_post_delayprofile | Add a new delayprofile. | radarr | arr-mcp | 40 | Yes |
| radarr_post_qualityprofile | Add a new qualityprofile. | radarr | arr-mcp | 40 | Yes |
| radarr_post_releaseprofile | Add a new releaseprofile. | radarr | arr-mcp | 40 | Yes |
| radarr_put_config_mediamanagement_id | Update config mediamanagement id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_config_metadata_id | Update config metadata id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_config_naming_id | Update config naming id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_customfilter_id | Update customfilter id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_customformat_bulk | Update customformat bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_put_customformat_id | Update customformat id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_delayprofile_id | Update delayprofile id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_delayprofile_reorder_id | Update delayprofile reorder id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_qualitydefinition_id | Update qualitydefinition id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_qualitydefinition_update | Update qualitydefinition update. | radarr | arr-mcp | 40 | Yes |
| radarr_put_qualityprofile_id | Update qualityprofile id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_releaseprofile_id | Update releaseprofile id. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_blocklist_bulk | Delete blocklist bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_blocklist_id | Delete blocklist id. | radarr | arr-mcp | 35 | Yes |
| radarr_delete_queue_bulk | Delete queue bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_queue_id | Delete an item from the download queue. | radarr | arr-mcp | 35 | Yes |
| radarr_get_blocklist | Get blocklist. | radarr | arr-mcp | 30 | No |
| radarr_get_blocklist_movie | Get blocklisted items for a specific movie. | radarr | arr-mcp | 40 | No |
| radarr_get_queue | Get queue. | radarr | arr-mcp | 30 | No |
| radarr_get_queue_details | Get queue details. | radarr | arr-mcp | 40 | No |
| radarr_post_queue_grab_bulk | Add a new queue grab bulk. | radarr | arr-mcp | 40 | Yes |
| radarr_post_queue_grab_id | Add a new queue grab id. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_system_backup_id | Delete system backup id. | radarr | arr-mcp | 40 | Yes |
| radarr_delete_tag_id | Delete tag id. | radarr | arr-mcp | 30 | Yes |
| radarr_get_ | Get . | radarr | arr-mcp | 25 | No |
| radarr_get_config_host_id | Get specific config host. | radarr | arr-mcp | 40 | No |
| radarr_get_config_ui_id | Get specific config ui. | radarr | arr-mcp | 35 | No |
| radarr_get_content_path | Get content path. | radarr | arr-mcp | 40 | No |
| radarr_get_filesystem | Get filesystem. | radarr | arr-mcp | 30 | No |
| radarr_get_filesystem_mediafiles | Get filesystem mediafiles. | radarr | arr-mcp | 40 | No |
| radarr_get_filesystem_type | Get filesystem type. | radarr | arr-mcp | 40 | No |
| radarr_get_log | Get log. | radarr | arr-mcp | 30 | No |
| radarr_get_log_file_filename | Get log file filename. | radarr | arr-mcp | 40 | No |
| radarr_get_log_file_update_filename | Get log file update filename. | radarr | arr-mcp | 40 | Yes |
| radarr_get_path | Get path. | radarr | arr-mcp | 30 | No |
| radarr_get_system_task_id | Get specific system task. | radarr | arr-mcp | 40 | No |
| radarr_get_tag_detail_id | Get specific tag detail. | radarr | arr-mcp | 40 | No |
| radarr_get_tag_id | Get specific tag. | radarr | arr-mcp | 35 | No |
| radarr_post_login | Log in to the Radarr web interface. | radarr | arr-mcp | 40 | Yes |
| radarr_post_system_backup_restore_id | Add a new system backup restore id. | radarr | arr-mcp | 40 | Yes |
| radarr_post_tag | Add a new tag. | radarr | arr-mcp | 35 | Yes |
| radarr_put_config_host_id | Update config host id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_config_ui_id | Update config ui id. | radarr | arr-mcp | 40 | Yes |
| radarr_put_tag_id | Update tag id. | radarr | arr-mcp | 35 | Yes |
| seerr_get_movie_id | Get movie details | seerr | arr-mcp | 35 | No |
| seerr_get_tv_id | Get TV details | seerr | arr-mcp | 25 | No |
| seerr_delete_request_id | Delete a request | seerr | arr-mcp | 35 | Yes |
| seerr_get_request | Get all requests | seerr | arr-mcp | 35 | No |
| seerr_get_request_id | Get a specific request | seerr | arr-mcp | 35 | No |
| seerr_get_search | Search for content | seerr | arr-mcp | 35 | No |
| seerr_post_request | Create a new request | seerr | arr-mcp | 40 | Yes |
| seerr_post_request_id_approve | Approve a request | seerr | arr-mcp | 40 | Yes |
| seerr_post_request_id_decline | Decline a request | seerr | arr-mcp | 40 | Yes |
| seerr_put_request_id | Update a request | seerr | arr-mcp | 40 | Yes |
| seerr_get_user | Get all users | seerr | arr-mcp | 30 | No |
| seerr_get_user_id | Get user details | seerr | arr-mcp | 35 | No |
| sonarr_add_series | Lookup a series by term, pick the first result, and add it to Sonarr. | sonarr | arr-mcp | 50 | Yes |
| sonarr_delete_episodefile_bulk | Delete episodefile bulk. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_episodefile_id | Delete episodefile id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_delete_metadata_id | Delete metadata id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_delete_series_editor | Delete series editor. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_series_id | Delete series. | sonarr | arr-mcp | 30 | Yes |
| sonarr_get_episode | Get episode. | sonarr | arr-mcp | 30 | No |
| sonarr_get_episode_id | Get specific episode. | sonarr | arr-mcp | 35 | No |
| sonarr_get_episodefile | Get episodefile. | sonarr | arr-mcp | 35 | No |
| sonarr_get_episodefile_id | Get specific episodefile. | sonarr | arr-mcp | 35 | No |
| sonarr_get_mediacover_series_id_filename | Get specific mediacover series filename. | sonarr | arr-mcp | 40 | No |
| sonarr_get_metadata_id | Get specific metadata. | sonarr | arr-mcp | 35 | No |
| sonarr_get_rename | Get rename. | sonarr | arr-mcp | 30 | Yes |
| sonarr_get_series | Get series. | sonarr | arr-mcp | 30 | No |
| sonarr_get_series_id | Get specific series. | sonarr | arr-mcp | 35 | No |
| sonarr_get_series_id_folder | Get series folder. | sonarr | arr-mcp | 40 | No |
| sonarr_get_series_lookup | Lookup series. | sonarr | arr-mcp | 35 | No |
| sonarr_get_wanted_missing | Get wanted missing. | sonarr | arr-mcp | 40 | No |
| sonarr_get_wanted_missing_id | Get specific wanted missing. | sonarr | arr-mcp | 40 | No |
| sonarr_lookup_series | Search for a series using the lookup endpoint. | sonarr | arr-mcp | 40 | No |
| sonarr_post_metadata | Add a new metadata. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_metadata_action_name | Add a new metadata action name. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_metadata_test | Test metadata. | sonarr | arr-mcp | 35 | Yes |
| sonarr_post_seasonpass | Add a new seasonpass. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_series | Add a new series. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_series_import | Import series. | sonarr | arr-mcp | 35 | Yes |
| sonarr_put_episode_id | Update episode id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_episode_monitor | Update episode monitor. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_episodefile_bulk | Update episodefile bulk. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_episodefile_editor | Update episodefile editor. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_episodefile_id | Update episodefile id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_metadata_id | Update metadata id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_series_editor | Update series editor. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_series_id | Update series id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_notification_id | Delete notification id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_delete_remotepathmapping_id | Delete remotepathmapping id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_delete_rootfolder_id | Delete rootfolder id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_get_notification_id | Get specific notification. | sonarr | arr-mcp | 35 | No |
| sonarr_get_remotepathmapping_id | Get specific remotepathmapping. | sonarr | arr-mcp | 35 | No |
| sonarr_get_rootfolder_id | Get specific rootfolder. | sonarr | arr-mcp | 35 | No |
| sonarr_post_notification | Add a new notification. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_notification_action_name | Add a new notification action name. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_notification_test | Test notification. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_remotepathmapping | Add a new remotepathmapping. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_rootfolder | Add a new rootfolder. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_notification_id | Update notification id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_remotepathmapping_id | Update remotepathmapping id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_downloadclient_bulk | Delete downloadclient bulk. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_downloadclient_id | Delete downloadclient id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_delete_importlist_bulk | Delete importlist bulk. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_importlist_id | Delete an import list configuration by ID. | sonarr | arr-mcp | 35 | Yes |
| sonarr_delete_importlistexclusion_bulk | Delete importlistexclusion bulk. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_importlistexclusion_id | Delete importlistexclusion id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_get_config_downloadclient_id | Get specific config downloadclient. | sonarr | arr-mcp | 40 | No |
| sonarr_get_config_importlist_id | Get specific config importlist. | sonarr | arr-mcp | 40 | No |
| sonarr_get_downloadclient_id | Get specific downloadclient. | sonarr | arr-mcp | 35 | No |
| sonarr_get_importlist_id | Get details for a specific import list by ID. | sonarr | arr-mcp | 35 | No |
| sonarr_get_importlistexclusion_id | Get specific importlistexclusion. | sonarr | arr-mcp | 35 | No |
| sonarr_get_importlistexclusion_paged | Get importlistexclusion paged. | sonarr | arr-mcp | 40 | No |
| sonarr_get_manualimport | Get manualimport. | sonarr | arr-mcp | 35 | No |
| sonarr_get_release | Get release. | sonarr | arr-mcp | 30 | No |
| sonarr_post_downloadclient | Add a new downloadclient. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_downloadclient_action_name | Add a new downloadclient action name. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_downloadclient_test | Test downloadclient. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_importlist | Add a new import list configuration. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_importlist_action_name | Add a new importlist action name. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_importlist_test | Test importlist. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_importlistexclusion | Add a new importlistexclusion. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_manualimport | Add a new manualimport. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_release | Add a new release. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_release_push | Add a new release push. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_config_downloadclient_id | Update config downloadclient id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_config_importlist_id | Update config importlist id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_downloadclient_bulk | Update downloadclient bulk. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_downloadclient_id | Update downloadclient id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_importlist_bulk | Update importlist bulk. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_importlist_id | Update an existing import list configuration. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_importlistexclusion_id | Update importlistexclusion id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_get_history | Get history. | sonarr | arr-mcp | 30 | No |
| sonarr_get_history_series | Get history series. | sonarr | arr-mcp | 40 | No |
| sonarr_get_history_since | Get history since. | sonarr | arr-mcp | 40 | No |
| sonarr_post_history_failed_id | Add a new history failed id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_indexer_bulk | Delete indexer bulk. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_indexer_id | Delete an indexer configuration by ID. | sonarr | arr-mcp | 35 | Yes |
| sonarr_get_config_indexer_id | Get specific config indexer. | sonarr | arr-mcp | 40 | No |
| sonarr_get_indexer_id | Get specific indexer. | sonarr | arr-mcp | 35 | No |
| sonarr_post_indexer | Add a new indexer configuration. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_indexer_action_name | Add a new indexer action name. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_indexer_test | Test indexer. | sonarr | arr-mcp | 35 | Yes |
| sonarr_put_config_indexer_id | Update config indexer id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_indexer_bulk | Update indexer bulk. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_indexer_id | Update an existing indexer configuration by ID. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_autotagging_id | Delete an auto-tagging rule by ID. | sonarr | arr-mcp | 35 | Yes |
| sonarr_delete_command_id | Delete command id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_get_autotagging_id | Get details for a specific auto-tagging rule by ID. | sonarr | arr-mcp | 45 | No |
| sonarr_get_calendar | Get calendar. | sonarr | arr-mcp | 30 | No |
| sonarr_get_calendar_id | Get specific calendar. | sonarr | arr-mcp | 35 | No |
| sonarr_get_command_id | Get specific command. | sonarr | arr-mcp | 35 | No |
| sonarr_get_feed_v3_calendar_sonarrics | Get feed v3 calendar sonarrics. | sonarr | arr-mcp | 40 | No |
| sonarr_get_parse | Get parse. | sonarr | arr-mcp | 30 | No |
| sonarr_post_autotagging | Add a new auto-tagging rule. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_command | Add a new command. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_autotagging_id | Update an existing auto-tagging rule by ID. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_customfilter_id | Delete customfilter id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_delete_customformat_bulk | Delete customformat bulk. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_customformat_id | Delete customformat id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_delete_delayprofile_id | Delete delayprofile id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_delete_languageprofile_id | Delete languageprofile id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_delete_qualityprofile_id | Delete qualityprofile id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_delete_releaseprofile_id | Delete releaseprofile id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_get_config_mediamanagement_id | Get specific config mediamanagement. | sonarr | arr-mcp | 40 | No |
| sonarr_get_config_naming_examples | Get config naming examples. | sonarr | arr-mcp | 40 | No |
| sonarr_get_config_naming_id | Get specific config naming. | sonarr | arr-mcp | 40 | No |
| sonarr_get_customfilter_id | Get specific customfilter. | sonarr | arr-mcp | 35 | No |
| sonarr_get_customformat_id | Get specific customformat. | sonarr | arr-mcp | 35 | No |
| sonarr_get_delayprofile_id | Get specific delayprofile. | sonarr | arr-mcp | 35 | No |
| sonarr_get_language_id | Get specific language. | sonarr | arr-mcp | 35 | No |
| sonarr_get_languageprofile_id | Get specific languageprofile. | sonarr | arr-mcp | 35 | No |
| sonarr_get_qualitydefinition_id | Get a specific quality definition by ID. | sonarr | arr-mcp | 35 | No |
| sonarr_get_qualityprofile_id | Get specific qualityprofile. | sonarr | arr-mcp | 35 | No |
| sonarr_get_releaseprofile_id | Get specific releaseprofile. | sonarr | arr-mcp | 35 | No |
| sonarr_get_wanted_cutoff | Get wanted cutoff. | sonarr | arr-mcp | 40 | No |
| sonarr_get_wanted_cutoff_id | Get specific wanted cutoff. | sonarr | arr-mcp | 40 | No |
| sonarr_post_customfilter | Add a new customfilter. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_customformat | Add a new customformat. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_delayprofile | Add a new delayprofile. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_languageprofile | Add a new languageprofile. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_qualityprofile | Add a new qualityprofile. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_releaseprofile | Add a new releaseprofile. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_config_mediamanagement_id | Update config mediamanagement id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_config_naming_id | Update config naming id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_customfilter_id | Update customfilter id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_customformat_bulk | Update customformat bulk. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_customformat_id | Update customformat id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_delayprofile_id | Update delayprofile id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_delayprofile_reorder_id | Update delayprofile reorder id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_languageprofile_id | Update languageprofile id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_qualitydefinition_id | Update qualitydefinition id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_qualitydefinition_update | Update qualitydefinition update. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_qualityprofile_id | Update qualityprofile id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_releaseprofile_id | Update releaseprofile id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_blocklist_bulk | Delete blocklist bulk. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_blocklist_id | Delete a blocklisted item by ID. | sonarr | arr-mcp | 35 | Yes |
| sonarr_delete_queue_bulk | Delete queue bulk. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_queue_id | Delete queue id. | sonarr | arr-mcp | 35 | Yes |
| sonarr_get_blocklist | Get blocklist. | sonarr | arr-mcp | 30 | No |
| sonarr_get_queue | Get queue. | sonarr | arr-mcp | 30 | No |
| sonarr_get_queue_details | Get queue details. | sonarr | arr-mcp | 40 | No |
| sonarr_post_queue_grab_bulk | Add a new queue grab bulk. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_queue_grab_id | Add a new queue grab id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_system_backup_id | Delete a system backup file by ID. | sonarr | arr-mcp | 40 | Yes |
| sonarr_delete_tag_id | Delete a tag. | sonarr | arr-mcp | 30 | Yes |
| sonarr_get_ | Get resource by path. | sonarr | arr-mcp | 30 | No |
| sonarr_get_config_host_id | Get specific config host. | sonarr | arr-mcp | 40 | No |
| sonarr_get_config_ui_id | Get specific UI configuration. | sonarr | arr-mcp | 35 | No |
| sonarr_get_content_path | Get content path. | sonarr | arr-mcp | 40 | No |
| sonarr_get_filesystem | Get filesystem. | sonarr | arr-mcp | 30 | No |
| sonarr_get_filesystem_mediafiles | Get filesystem mediafiles. | sonarr | arr-mcp | 40 | No |
| sonarr_get_filesystem_type | Get filesystem type. | sonarr | arr-mcp | 40 | No |
| sonarr_get_localization_id | Get specific localization. | sonarr | arr-mcp | 35 | No |
| sonarr_get_log | Get log. | sonarr | arr-mcp | 30 | No |
| sonarr_get_log_file_filename | Get log file filename. | sonarr | arr-mcp | 40 | No |
| sonarr_get_log_file_update_filename | Get log file update content. | sonarr | arr-mcp | 40 | Yes |
| sonarr_get_path | Get system routes. | sonarr | arr-mcp | 35 | No |
| sonarr_get_system_task_id | Get specific system task. | sonarr | arr-mcp | 40 | No |
| sonarr_get_tag_detail_id | Get specific tag usage details. | sonarr | arr-mcp | 40 | No |
| sonarr_get_tag_id | Get specific tag. | sonarr | arr-mcp | 35 | No |
| sonarr_post_login | Log in to the Sonarr web interface. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_system_backup_restore_id | Add a new system backup restore id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_post_tag | Add a new tag. | sonarr | arr-mcp | 35 | Yes |
| sonarr_put_config_host_id | Update config host id. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_config_ui_id | Update UI configuration. | sonarr | arr-mcp | 40 | Yes |
| sonarr_put_tag_id | Update a tag. | sonarr | arr-mcp | 35 | Yes |
| atlassian_general_tools | General tools for atlassian (offline extraction). | atlassian | atlassian | 40 | No |
| transcribe_audio | Transcribes audio from a provided file or by recording from the microphone. | audio_processing | audio-transcriber-mcp | 70 | No |
| get_version | Retrieves the version information of the container manager (Docker or Podman).<br/>Returns: A dictionary with keys like 'version', 'api_version', etc., detailing the manager's version. | info | container-manager-mcp | 65 | No |
| get_info | Retrieves detailed information about the container manager system.<br/>Returns: A dictionary containing system info such as OS, architecture, storage driver, and more. | info | container-manager-mcp | 65 | No |
| list_images | Lists all container images available on the system.<br/>Returns: A list of dictionaries, each with image details like 'id', 'tags', 'created', 'size'. | image | container-manager-mcp | 65 | No |
| pull_image | Pulls a container image from a registry.<br/>Returns: A dictionary with the pull status, including 'id' of the pulled image and any error messages. | image | container-manager-mcp | 70 | No |
| remove_image | Removes a specified container image.<br/>Returns: A dictionary indicating success or failure, with details like removed image ID. | image | container-manager-mcp | 70 | Yes |
| prune_images | Prunes unused container images.<br/>Returns: A dictionary with prune results, including space reclaimed and list of deleted images. | image | container-manager-mcp | 70 | Yes |
| list_containers | Lists containers on the system.<br/>Returns: A list of dictionaries, each with container details like 'id', 'name', 'status', 'image'. | container | container-manager-mcp | 75 | No |
| run_container | Runs a new container from the specified image.<br/>Returns: A dictionary with the container's ID and status after starting. | container | container-manager-mcp | 75 | No |
| stop_container | Stops a running container.<br/>Returns: A dictionary confirming the stop action, with container ID and any errors. | container | container-manager-mcp | 80 | Yes |
| remove_container | Removes a container.<br/>Returns: A dictionary with removal status, including deleted container ID. | container | container-manager-mcp | 70 | Yes |
| prune_containers | Prunes stopped containers.<br/>Returns: A dictionary with prune results, including space reclaimed and deleted containers. | container | container-manager-mcp | 80 | Yes |
| exec_in_container | Executes a command inside a running container.<br/>Returns: A dictionary with execution results, including 'exit_code' and 'output' as string. | container | container-manager-mcp | 80 | No |
| get_container_logs | Retrieves logs from a container.<br/>Returns: A string containing the log output, parse as plain text lines. | container, debug, log | container-manager-mcp | 95 | No |
| compose_logs | Retrieves logs for services in a Docker Compose project.<br/>Returns: A string containing combined log output, prefixed by service names; parse as text lines. | compose, log | container-manager-mcp | 90 | No |
| list_volumes | Lists all volumes.<br/>Returns: A dictionary with 'volumes' as a list of dicts containing name, driver, mountpoint, etc. | volume | container-manager-mcp | 65 | No |
| create_volume | Creates a new volume.<br/>Returns: A dictionary with details of the created volume, like 'name' and 'mountpoint'. | volume | container-manager-mcp | 65 | Yes |
| remove_volume | Removes a volume.<br/>Returns: A dictionary confirming removal, with deleted volume name. | volume | container-manager-mcp | 60 | Yes |
| prune_volumes | Prunes unused volumes.<br/>Returns: A dictionary with prune results, including space reclaimed and deleted volumes. | volume | container-manager-mcp | 70 | Yes |
| list_networks | Lists all networks.<br/>Returns: A list of dictionaries, each with network details like 'id', 'name', 'driver', 'scope'. | network | container-manager-mcp | 75 | No |
| create_network | Creates a new network.<br/>Returns: A dictionary with the created network's ID and details. | network | container-manager-mcp | 65 | Yes |
| remove_network | Removes a network.<br/>Returns: A dictionary confirming removal, with deleted network ID. | network | container-manager-mcp | 70 | Yes |
| prune_networks | Prunes unused networks.<br/>Returns: A dictionary with prune results, including deleted networks. | network | container-manager-mcp | 70 | Yes |
| prune_system | Prunes all unused system resources (containers, images, volumes, networks).<br/>Returns: A dictionary summarizing the prune operation across resources. | system | container-manager-mcp | 70 | Yes |
| init_swarm | Initializes a Docker Swarm cluster.<br/>Returns: A dictionary with swarm info, including join tokens for manager and worker. | swarm | container-manager-mcp | 70 | No |
| leave_swarm | Leaves the Docker Swarm cluster.<br/>Returns: A dictionary confirming the leave action. | swarm | container-manager-mcp | 60 | No |
| list_nodes | Lists nodes in the Docker Swarm cluster.<br/>Returns: A list of dictionaries, each with node details like 'id', 'hostname', 'status', 'role'. | swarm | container-manager-mcp | 65 | No |
| list_services | Lists services in the Docker Swarm.<br/>Returns: A list of dictionaries, each with service details like 'id', 'name', 'replicas', 'image'. | swarm | container-manager-mcp | 65 | No |
| create_service | Creates a new service in Docker Swarm.<br/>Returns: A dictionary with the created service's ID and details. | swarm | container-manager-mcp | 65 | Yes |
| remove_service | Removes a service from Docker Swarm.<br/>Returns: A dictionary confirming the removal. | swarm | container-manager-mcp | 60 | Yes |
| compose_up | Starts services defined in a Docker Compose file.<br/>Returns: A string with the output of the compose up command, parse for status messages. | compose | container-manager-mcp | 75 | No |
| compose_down | Stops and removes services from a Docker Compose file.<br/>Returns: A string with the output of the compose down command, parse for status messages. | compose | container-manager-mcp | 80 | No |
| compose_ps | Lists containers for a Docker Compose project.<br/>Returns: A string in table format listing name, command, state, ports; parse as text table. | compose | container-manager-mcp | 75 | No |
| binary_version | Get the binary version of the server (using buildInfo). | system | documentdb-mcp | 60 | No |
| list_databases | List all databases in the connected DocumentDB/MongoDB instance. | system | documentdb-mcp | 55 | No |
| run_command | Run a raw command against the database. | system | documentdb-mcp | 45 | Yes |
| list_collections | List all collections in a specific database. | collections | documentdb-mcp | 55 | No |
| create_collection | Create a new collection in the specified database. | collections | documentdb-mcp | 55 | Yes |
| drop_collection | Drop a collection from the specified database. | collections | documentdb-mcp | 60 | Yes |
| create_database | Explicitly create a database by creating a collection in it (MongoDB creates DBs lazily). | collections | documentdb-mcp | 65 | Yes |
| drop_database | Drop a database. | collections | documentdb-mcp | 60 | Yes |
| rename_collection | Rename a collection. | collections | documentdb-mcp | 60 | Yes |
| create_user | Create a new user on the specified database. | users | documentdb-mcp | 45 | Yes |
| drop_user | Drop a user from the specified database. | users | documentdb-mcp | 50 | Yes |
| update_user | Update a user's password or roles. | users | documentdb-mcp | 45 | Yes |
| users_info | Get information about a user. | users | documentdb-mcp | 50 | No |
| insert_one | Insert a single document into a collection. | crud | documentdb-mcp | 50 | Yes |
| insert_many | Insert multiple documents into a collection. | crud | documentdb-mcp | 50 | Yes |
| find_one | Find a single document matching the filter. | crud | documentdb-mcp | 50 | No |
| find | Find documents matching the filter.<br/>'sort' should be a list of [key, direction] pairs, e.g. [["name", 1], ["date", -1]]. | crud | documentdb-mcp | 65 | No |
| replace_one | Replace a single document matching the filter. | crud | documentdb-mcp | 50 | Yes |
| update_one | Update a single document matching the filter. 'update' must contain update operators like $set. | crud | documentdb-mcp | 55 | Yes |
| update_many | Update multiple documents matching the filter. | crud | documentdb-mcp | 45 | Yes |
| delete_one | Delete a single document matching the filter. | crud | documentdb-mcp | 45 | Yes |
| delete_many | Delete multiple documents matching the filter. | crud | documentdb-mcp | 45 | Yes |
| count_documents | Count documents matching the filter. | crud | documentdb-mcp | 50 | No |
| find_one_and_update | Finds a single document and updates it. return_document: 'before' or 'after'. | crud | documentdb-mcp | 65 | Yes |
| find_one_and_replace | Finds a single document and replaces it. return_document: 'before' or 'after'. | crud | documentdb-mcp | 65 | Yes |
| find_one_and_delete | Finds a single document and deletes it. | crud | documentdb-mcp | 55 | Yes |
| distinct | Find distinct values for a key. | analysis | documentdb-mcp | 55 | No |
| aggregate | Run an aggregation pipeline. | analysis | documentdb-mcp | 55 | No |
| github_list_repos | List repositories for the authenticated user. | repos | github-mcp | 50 | No |
| github_get_repo | Get details for a specific repository. | repos | github-mcp | 50 | No |
| github_list_issues | List issues for a repository. | issues | github-mcp | 50 | No |
| github_list_pull_requests | List pull requests for a repository. | pulls | github-mcp | 55 | No |
| github_get_contents | Get contents of a file or directory. | contents | github-mcp | 60 | No |
| get_branches | Get branches in a GitLab project, optionally filtered. | branches | gitlab-api | 65 | No |
| create_branch | Create a new branch in a GitLab project from a reference. | branches | gitlab-api | 65 | Yes |
| delete_branch | Delete a branch or all merged branches in a GitLab project.<br/><br/>- If delete_merged_branches=True, deletes all merged branches (excluding protected).<br/>- Otherwise, deletes the specified branch. | branches | gitlab-api | 75 | Yes |
| get_commits | Get commits in a GitLab project, optionally filtered. | commits | gitlab-api | 65 | No |
| create_commit | Create a new commit in a GitLab project. | commits | gitlab-api | 55 | Yes |
| get_commit_diff | Get the diff of a specific commit in a GitLab project. | commits | gitlab-api | 70 | No |
| revert_commit | Revert a commit in a target branch in a GitLab project.<br/><br/>- If dry_run=True, simulates the revert without applying changes.<br/>- Returns the revert commit details or simulation result. | commits | gitlab-api | 80 | Yes |
| get_commit_comments | Retrieve comments on a specific commit in a GitLab project. | commits | gitlab-api | 70 | No |
| create_commit_comment | Create a new comment on a specific commit in a GitLab project. | commits | gitlab-api | 70 | Yes |
| get_commit_discussions | Retrieve discussions (threaded comments) on a specific commit in a GitLab project. | commits | gitlab-api | 70 | No |
| get_commit_statuses | Retrieve build/CI statuses for a specific commit in a GitLab project. | commits | gitlab-api | 70 | No |
| post_build_status_to_commit | Post a build/CI status to a specific commit in a GitLab project. | commits | gitlab-api | 75 | Yes |
| get_commit_merge_requests | Retrieve merge requests associated with a specific commit in a GitLab project. | commits | gitlab-api | 75 | No |
| get_commit_gpg_signature | Retrieve the GPG signature for a specific commit in a GitLab project. | commits | gitlab-api | 75 | No |
| get_deploy_tokens | Retrieve a list of all deploy tokens for the GitLab instance. | deploy_tokens | gitlab-api | 70 | No |
| get_project_deploy_tokens | Retrieve a list of deploy tokens for a specific GitLab project. | deploy_tokens | gitlab-api | 75 | No |
| create_project_deploy_token | Create a deploy token for a GitLab project with specified name and scopes. | deploy_tokens | gitlab-api | 75 | Yes |
| delete_project_deploy_token | Delete a specific deploy token for a GitLab project. | deploy_tokens | gitlab-api | 75 | Yes |
| get_group_deploy_tokens | Retrieve deploy tokens for a GitLab group (list or single by ID). | deploy_tokens | gitlab-api | 75 | No |
| create_group_deploy_token | Create a deploy token for a GitLab group with specified name and scopes. | deploy_tokens | gitlab-api | 75 | Yes |
| delete_group_deploy_token | Delete a specific deploy token for a GitLab group. | deploy_tokens | gitlab-api | 65 | Yes |
| get_environments | Retrieve a list of environments for a GitLab project, optionally filtered by name, search, or states or a single environment by id. | environments | gitlab-api | 75 | No |
| create_environment | Create a new environment in a GitLab project with a specified name and optional external URL. | environments | gitlab-api | 65 | Yes |
| update_environment | Update an existing environment in a GitLab project with new name or external URL. | environments | gitlab-api | 65 | Yes |
| delete_environment | Delete a specific environment in a GitLab project. | environments | gitlab-api | 55 | Yes |
| stop_environment | Stop a specific environment in a GitLab project. | environments | gitlab-api | 60 | Yes |
| stop_stale_environments | Stop stale environments in a GitLab project, optionally filtered by older_than timestamp. | environments | gitlab-api | 75 | Yes |
| delete_stopped_environments | Delete stopped review app environments in a GitLab project. | environments | gitlab-api | 70 | Yes |
| get_protected_environments | Retrieve protected environments in a GitLab project (list or single by name). | environments | gitlab-api | 70 | No |
| protect_environment | Protect an environment in a GitLab project with optional approval count. | environments | gitlab-api | 70 | No |
| update_protected_environment | Update a protected environment in a GitLab project with new approval count. | environments | gitlab-api | 70 | Yes |
| unprotect_environment | Unprotect a specific environment in a GitLab project. | environments | gitlab-api | 70 | No |
| get_groups | Retrieve a list of groups, optionally filtered by search, sort, ownership, or access level or retrieve a single group by id. | groups | gitlab-api | 65 | No |
| edit_group | Edit a specific GitLab group's details (name, path, description, or visibility). | groups | gitlab-api | 60 | No |
| get_group_subgroups | Retrieve a list of subgroups for a specific GitLab group, optionally filtered. | groups | gitlab-api | 60 | No |
| get_group_descendant_groups | Retrieve a list of all descendant groups for a specific GitLab group, optionally filtered. | groups | gitlab-api | 65 | No |
| get_group_projects | Retrieve a list of projects associated with a specific GitLab group, optionally including subgroups. | groups | gitlab-api | 60 | No |
| get_group_merge_requests | Retrieve a list of merge requests associated with a specific GitLab group, optionally filtered. | groups | gitlab-api | 65 | No |
| get_project_jobs | Retrieve a list of jobs for a specific GitLab project, optionally filtered by scope or a single job by id. | jobs | gitlab-api | 70 | No |
| get_project_job_log | Retrieve the log (trace) of a specific job in a GitLab project. | jobs | gitlab-api | 65 | No |
| cancel_project_job | Cancel a specific job in a GitLab project. | jobs | gitlab-api | 55 | No |
| retry_project_job | Retry a specific job in a GitLab project. | jobs | gitlab-api | 55 | No |
| erase_project_job | Erase (delete artifacts and logs of) a specific job in a GitLab project. | jobs | gitlab-api | 65 | No |
| run_project_job | Run (play) a specific manual job in a GitLab project. | jobs | gitlab-api | 60 | No |
| get_pipeline_jobs | Retrieve a list of jobs for a specific pipeline in a GitLab project, optionally filtered by scope. | jobs | gitlab-api | 60 | No |
| get_group_members | Retrieve a list of members in a specific GitLab group, optionally filtered by query or user IDs. | members | gitlab-api | 70 | No |
| get_project_members | Retrieve a list of members in a specific GitLab project, optionally filtered by query or user IDs. | members | gitlab-api | 70 | No |
| create_merge_request | Create a new merge request in a GitLab project with specified source and target branches. | merge-requests | gitlab-api | 70 | Yes |
| get_merge_requests | Retrieve a list of merge requests across all projects, optionally filtered by state, scope, or labels. | merge-requests | gitlab-api | 80 | No |
| get_project_merge_requests | Retrieve a list of merge requests for a specific GitLab project, optionally filtered or a single merge request or a single merge request by merge id | merge-requests | gitlab-api | 85 | No |
| get_project_level_merge_request_approval_rules | Retrieve project-level merge request approval rules for a GitLab project details of a specific project-level merge request approval rule. | merge_rules | gitlab-api | 85 | No |
| create_project_level_rule | Create a new project-level merge request approval rule. | merge_rules | gitlab-api | 75 | Yes |
| update_project_level_rule | Update an existing project-level merge request approval rule. | merge_rules | gitlab-api | 75 | Yes |
| delete_project_level_rule | Delete a project-level merge request approval rule. | merge_rules | gitlab-api | 75 | Yes |
| merge_request_level_approvals | Retrieve approvals for a specific merge request in a GitLab project. | merge_rules | gitlab-api | 75 | No |
| get_approval_state_merge_requests | Retrieve the approval state of a specific merge request in a GitLab project. | merge_rules | gitlab-api | 75 | No |
| get_merge_request_level_rules | Retrieve merge request-level approval rules for a specific merge request in a GitLab project. | merge_rules | gitlab-api | 75 | No |
| approve_merge_request | Approve a specific merge request in a GitLab project. | merge_rules | gitlab-api | 75 | Yes |
| unapprove_merge_request | Unapprove a specific merge request in a GitLab project. | merge_rules | gitlab-api | 75 | Yes |
| get_group_level_rule | Retrieve merge request approval settings for a specific GitLab group. | merge_rules | gitlab-api | 75 | No |
| edit_group_level_rule | Edit merge request approval settings for a specific GitLab group. | merge_rules | gitlab-api | 75 | No |
| get_project_level_rule | Retrieve merge request approval settings for a specific GitLab project. | merge_rules | gitlab-api | 75 | No |
| edit_project_level_rule | Edit merge request approval settings for a specific GitLab project. | merge_rules | gitlab-api | 75 | No |
| get_repository_packages | Retrieve a list of repository packages for a specific GitLab project, optionally filtered by package type. | packages | gitlab-api | 80 | No |
| publish_repository_package | Publish a repository package to a specific GitLab project. | packages | gitlab-api | 75 | No |
| download_repository_package | Download a repository package from a specific GitLab project. | packages | gitlab-api | 75 | No |
| get_pipelines | Retrieve a list of pipelines for a specific GitLab project, optionally filtered by scope, status, or ref or details of a specific pipeline in a GitLab project.. | pipelines | gitlab-api | 75 | No |
| run_pipeline | Run a pipeline for a specific GitLab project with a given reference (e.g., branch or tag). | pipelines | gitlab-api | 65 | No |
| get_pipeline_schedules | Retrieve a list of pipeline schedules for a specific GitLab project. | pipeline_schedules | gitlab-api | 70 | No |
| get_pipeline_schedule | Retrieve details of a specific pipeline schedule in a GitLab project. | pipeline_schedules | gitlab-api | 70 | No |
| get_pipelines_triggered_from_schedule | Retrieve pipelines triggered by a specific pipeline schedule in a GitLab project. | pipeline_schedules | gitlab-api | 75 | No |
| create_pipeline_schedule | Create a pipeline schedule for a specific GitLab project. | pipeline_schedules | gitlab-api | 70 | Yes |
| edit_pipeline_schedule | Edit a pipeline schedule in a GitLab project. | pipeline_schedules | gitlab-api | 65 | No |
| take_pipeline_schedule_ownership | Take ownership of a pipeline schedule in a GitLab project. | pipeline_schedules | gitlab-api | 75 | No |
| delete_pipeline_schedule | Delete a pipeline schedule in a GitLab project. | pipeline_schedules | gitlab-api | 60 | Yes |
| run_pipeline_schedule | Run a pipeline schedule immediately in a GitLab project. | pipeline_schedules | gitlab-api | 70 | No |
| create_pipeline_schedule_variable | Create a variable for a pipeline schedule in a GitLab project. | pipeline_schedules | gitlab-api | 75 | Yes |
| delete_pipeline_schedule_variable | Delete a variable from a pipeline schedule in a GitLab project. | pipeline_schedules | gitlab-api | 75 | Yes |
| get_projects | Retrieve a list of projects, optionally filtered by ownership, search, sort, or visibility or Retrieve details of a specific GitLab project. | projects | gitlab-api | 75 | No |
| get_nested_projects_by_group | Retrieve a list of nested projects within a GitLab group, including descendant groups. | projects | gitlab-api | 75 | No |
| get_project_contributors | Retrieve a list of contributors to a specific GitLab project. | projects | gitlab-api | 70 | No |
| get_project_statistics | Retrieve statistics for a specific GitLab project. | projects | gitlab-api | 60 | No |
| edit_project | Edit a specific GitLab project's details (name, description, or visibility). | projects | gitlab-api | 70 | No |
| get_project_groups | Retrieve a list of groups associated with a specific GitLab project, optionally filtered. | projects | gitlab-api | 70 | No |
| archive_project | Archive a specific GitLab project. | projects | gitlab-api | 60 | No |
| unarchive_project | Unarchive a specific GitLab project. | projects | gitlab-api | 60 | No |
| delete_project | Delete a specific GitLab project. | projects | gitlab-api | 55 | Yes |
| share_project | Share a specific GitLab project with a group, specifying access level. | projects | gitlab-api | 70 | No |
| get_protected_branches | Retrieve a list of protected branches in a specific GitLab project or Retrieve details of a specific protected branch in a GitLab project.. | protected_branches | gitlab-api | 80 | No |
| protect_branch | Protect a specific branch in a GitLab project with specified access levels. | protected_branches | gitlab-api | 70 | No |
| unprotect_branch | Unprotect a specific branch in a GitLab project. | protected_branches | gitlab-api | 60 | No |
| require_code_owner_approvals_single_branch | Require or disable code owner approvals for a specific branch in a GitLab project. | protected_branches | gitlab-api | 75 | No |
| get_releases | Retrieve a list of releases for a specific GitLab project, optionally filtered. | releases | gitlab-api | 65 | No |
| get_latest_release | Retrieve details of the latest release in a GitLab project. | releases | gitlab-api | 70 | No |
| get_latest_release_evidence | Retrieve evidence for the latest release in a GitLab project. | releases | gitlab-api | 75 | No |
| get_latest_release_asset | Retrieve a specific asset for the latest release in a GitLab project. | releases | gitlab-api | 75 | Yes |
| get_group_releases | Retrieve a list of releases for a specific GitLab group, optionally filtered. | releases | gitlab-api | 70 | No |
| download_release_asset | Download a release asset from a group's release in GitLab. | releases | gitlab-api | 75 | Yes |
| get_release_by_tag | Retrieve details of a release by its tag in a GitLab project. | releases | gitlab-api | 70 | No |
| create_release | Create a new release in a GitLab project. | releases | gitlab-api | 55 | Yes |
| create_release_evidence | Create evidence for a release in a GitLab project. | releases | gitlab-api | 60 | Yes |
| update_release | Update a release in a GitLab project. | releases | gitlab-api | 55 | Yes |
| delete_release | Delete a release in a GitLab project. | releases | gitlab-api | 55 | Yes |
| get_runners | Retrieve a list of runners in GitLab, optionally filtered by scope, type, status, or tags or Retrieve details of a specific GitLab runner.. | runners | gitlab-api | 75 | No |
| update_runner_details | Update details for a specific GitLab runner. | runners | gitlab-api | 60 | Yes |
| pause_runner | Pause or unpause a specific GitLab runner. | runners | gitlab-api | 60 | Yes |
| get_runner_jobs | Retrieve jobs for a specific GitLab runner, optionally filtered by status or sorted. | runners | gitlab-api | 70 | No |
| get_project_runners | Retrieve a list of runners in a specific GitLab project, optionally filtered by scope. | runners | gitlab-api | 70 | No |
| enable_project_runner | Enable a runner in a specific GitLab project. | runners | gitlab-api | 65 | Yes |
| delete_project_runner | Delete a runner from a specific GitLab project. | runners | gitlab-api | 60 | Yes |
| get_group_runners | Retrieve a list of runners in a specific GitLab group, optionally filtered by scope. | runners | gitlab-api | 70 | No |
| register_new_runner | Register a new GitLab runner. | runners | gitlab-api | 65 | No |
| delete_runner | Delete a GitLab runner by ID or token. | runners | gitlab-api | 55 | Yes |
| verify_runner_authentication | Verify authentication for a GitLab runner using its token. | runners | gitlab-api | 75 | No |
| reset_gitlab_runner_token | Reset the GitLab runner registration token. | runners | gitlab-api | 65 | Yes |
| reset_project_runner_token | Reset the registration token for a project's runner in GitLab. | runners | gitlab-api | 75 | Yes |
| reset_group_runner_token | Reset the registration token for a group's runner in GitLab. | runners | gitlab-api | 75 | Yes |
| reset_token | Reset the authentication token for a specific GitLab runner. | runners | gitlab-api | 70 | Yes |
| get_tags | Retrieve a list of tags for a specific GitLab project, optionally filtered or sorted or Retrieve details of a specific tag in a GitLab project. | tags | gitlab-api | 65 | No |
| create_tag | Create a new tag in a GitLab project. | tags | gitlab-api | 45 | Yes |
| delete_tag | Delete a specific tag in a GitLab project. | tags | gitlab-api | 45 | Yes |
| get_protected_tags | Retrieve a list of protected tags in a specific GitLab project, optionally filtered by name. | tags | gitlab-api | 60 | No |
| get_protected_tag | Retrieve details of a specific protected tag in a GitLab project. | tags | gitlab-api | 60 | No |
| protect_tag | Protect a specific tag in a GitLab project with specified access levels. | tags | gitlab-api | 60 | No |
| unprotect_tag | Unprotect a specific tag in a GitLab project. | tags | gitlab-api | 50 | No |
| api_request | Make a custom API request to a GitLab instance. | custom-api | gitlab-api | 60 | No |
| ha-status | Check if Home Assistant API is up and running. | config | home | 45 | No |
| ha-config | Get Home Assistant configuration. | config | home | 45 | No |
| ha-components | List currently loaded components. | config | home | 45 | No |
| ha-check-config | Trigger a check of configuration.yaml. | config | home | 50 | No |
| ha-list-states | Return a list of all entity states. | states | home | 45 | No |
| ha-get-state | Return the state of a specific entity. | states | home | 45 | No |
| ha-update-state | Updates or creates a state for an entity (internal representation). | states | home | 55 | Yes |
| ha-delete-state | Deletes an entity state. | states | home | 45 | Yes |
| ha-list-services | List all available services. | services | home | 55 | No |
| ha-call-service | Call a service (e.g., turn a light on). | services | home | 60 | No |
| ha-list-events | List all event types and listener counts. | events | home | 45 | No |
| ha-fire-event | Fire an event on the Home Assistant event bus. | events | home | 50 | No |
| ha-subscribe-events | Subscribe to events (one-shot check). | events | home | 50 | No |
| ha-get-history | Get history of one or more entities. | history | home | 55 | No |
| ha-get-logbook | Get logbook entries. | logbook | home | 55 | No |
| ha-get-error-log | Retrieve all errors logged during the current session. | logbook | home | 70 | No |
| ha-list-calendars | List calendar entities. | calendar | home | 55 | No |
| ha-get-calendar-events | Get events for a calendar. | calendar | home | 60 | No |
| ha-get-panels | Get registered panels in Home Assistant. | panels | home | 45 | No |
| ha-list-exposed-entities | List exposure status of entities across all assistants. | voice | home | 60 | No |
| ha-expose-entities | Expose or unexpose entities to voice assistants. | voice | home | 50 | No |
| ha-get-entity-registry-display | Get lightweight, optimized list of entity registry entries for UI display. | entities | home | 75 | No |
| ha-extract-from-target | Extract entities, devices, and areas from one or multiple targets. | entities | home | 75 | No |
| ha-get-triggers-for-target | Get applicable triggers for entities of a given target. | entities | home | 75 | No |
| ha-get-conditions-for-target | Get applicable conditions for entities of a given target. | entities | home | 75 | No |
| ha-get-services-for-target | Get applicable services for entities of a given target. | entities | home | 75 | No |
| ha-render-template | Render a Home Assistant template. | system | home | 50 | No |
| ha-ping | Ping the Home Assistant WebSocket API. | system | home | 45 | No |
| ha-handle-intent | Handle an intent in Home Assistant. | system | home | 50 | No |
| ha-validate-config | Validate triggers, conditions, and action configurations. | system | home | 60 | No |
| get_log_entries | Gets activity log entries. | ActivityLog | jellyfin-mcp | 60 | No |
| get_keys | Get all keys. | ApiKey | jellyfin-mcp | 40 | No |
| create_key | Create a new api key. | ApiKey | jellyfin-mcp | 45 | Yes |
| revoke_key | Remove an api key. | ApiKey | jellyfin-mcp | 50 | No |
| get_artists | Gets all artists from a given item, folder, or the entire library. | Artists | jellyfin-mcp | 65 | No |
| get_artist_by_name | Gets an artist by name. | Artists | jellyfin-mcp | 60 | No |
| get_album_artists | Gets all album artists from a given item, folder, or the entire library. | Artists | jellyfin-mcp | 70 | No |
| get_audio_stream | Gets an audio stream. | Audio | jellyfin-mcp | 50 | No |
| get_audio_stream_by_container | Gets an audio stream. | Audio | jellyfin-mcp | 55 | No |
| list_backups | Gets a list of all currently present backups in the backup directory. | Backup | jellyfin-mcp | 55 | No |
| create_backup | Creates a new Backup. | Backup | jellyfin-mcp | 45 | Yes |
| get_backup | Gets the descriptor from an existing archive is present. | Backup | jellyfin-mcp | 55 | No |
| start_restore_backup | Restores to a backup by restarting the server and applying the backup. | Backup | jellyfin-mcp | 65 | Yes |
| get_branding_options | Gets branding configuration. | Branding | jellyfin-mcp | 60 | No |
| get_branding_css | Gets branding css. | Branding | jellyfin-mcp | 60 | No |
| get_branding_css_2 | Gets branding css. | Branding | jellyfin-mcp | 60 | No |
| get_channels | Gets available channels. | Channels | jellyfin-mcp | 55 | No |
| get_channel_features | Get channel features. | Channels | jellyfin-mcp | 60 | No |
| get_channel_items | Get channel items. | Channels | jellyfin-mcp | 60 | No |
| get_all_channel_features | Get all channel features. | Channels | jellyfin-mcp | 65 | No |
| get_latest_channel_items | Gets latest channel items. | Channels | jellyfin-mcp | 65 | No |
| log_file | Upload a document. | ClientLog | jellyfin-mcp | 60 | No |
| create_collection | Creates a new collection. | Collection | jellyfin-mcp | 55 | Yes |
| add_to_collection | Adds items to a collection. | Collection | jellyfin-mcp | 60 | Yes |
| remove_from_collection | Removes items from a collection. | Collection | jellyfin-mcp | 65 | Yes |
| get_configuration | Gets application configuration. | Configuration | jellyfin-mcp | 55 | No |
| update_configuration | Updates application configuration. | Configuration | jellyfin-mcp | 55 | Yes |
| get_named_configuration | Gets a named configuration. | Configuration | jellyfin-mcp | 60 | No |
| update_named_configuration | Updates named configuration. | Configuration | jellyfin-mcp | 60 | Yes |
| update_branding_configuration | Updates branding configuration. | Configuration | jellyfin-mcp | 60 | Yes |
| get_default_metadata_options | Gets a default MetadataOptions object. | Configuration | jellyfin-mcp | 65 | No |
| get_dashboard_configuration_page | Gets a dashboard configuration page. | Dashboard | jellyfin-mcp | 65 | No |
| get_configuration_pages | Gets the configuration pages. | Dashboard | jellyfin-mcp | 60 | No |
| get_devices | Get Devices. | Devices | jellyfin-mcp | 50 | No |
| delete_device | Deletes a device. | Devices | jellyfin-mcp | 55 | Yes |
| get_device_info | Get info for a device. | Devices | jellyfin-mcp | 60 | No |
| get_device_options | Get options for a device. | Devices | jellyfin-mcp | 60 | No |
| update_device_options | Update device options. | Devices | jellyfin-mcp | 60 | Yes |
| get_display_preferences | Get Display Preferences. | DisplayPreferences | jellyfin-mcp | 60 | No |
| update_display_preferences | Update Display Preferences. | DisplayPreferences | jellyfin-mcp | 60 | Yes |
| get_hls_audio_segment | Gets a video stream using HTTP live streaming. | DynamicHls | jellyfin-mcp | 65 | No |
| get_variant_hls_audio_playlist | Gets an audio stream using HTTP live streaming. | DynamicHls | jellyfin-mcp | 65 | No |
| get_master_hls_audio_playlist | Gets an audio hls playlist stream. | DynamicHls | jellyfin-mcp | 65 | No |
| get_hls_video_segment | Gets a video stream using HTTP live streaming. | DynamicHls | jellyfin-mcp | 65 | No |
| get_live_hls_stream | Gets a hls live stream. | DynamicHls | jellyfin-mcp | 65 | No |
| get_variant_hls_video_playlist | Gets a video stream using HTTP live streaming. | DynamicHls | jellyfin-mcp | 65 | No |
| get_master_hls_video_playlist | Gets a video hls playlist stream. | DynamicHls | jellyfin-mcp | 65 | No |
| get_default_directory_browser | Get Default directory browser. | Environment | jellyfin-mcp | 65 | No |
| get_directory_contents | Gets the contents of a given directory in the file system. | Environment | jellyfin-mcp | 70 | No |
| get_drives | Gets available drives from the server's file system. | Environment | jellyfin-mcp | 65 | No |
| get_network_shares | Gets network paths. | Environment | jellyfin-mcp | 60 | No |
| get_parent_path | Gets the parent path of a given path. | Environment | jellyfin-mcp | 60 | No |
| validate_path | Validates path. | Environment | jellyfin-mcp | 55 | No |
| get_query_filters_legacy | Gets legacy query filters. | Filter | jellyfin-mcp | 55 | No |
| get_query_filters | Gets query filters. | Filter | jellyfin-mcp | 50 | No |
| get_genres | Gets all genres from a given item, folder, or the entire library. | Genres | jellyfin-mcp | 55 | No |
| get_genre | Gets a genre, by name. | Genres | jellyfin-mcp | 45 | No |
| get_hls_audio_segment_legacy_aac | Gets the specified audio segment for an audio item. | HlsSegment | jellyfin-mcp | 75 | No |
| get_hls_audio_segment_legacy_mp3 | Gets the specified audio segment for an audio item. | HlsSegment | jellyfin-mcp | 75 | No |
| get_hls_video_segment_legacy | Gets a hls video segment. | HlsSegment | jellyfin-mcp | 65 | No |
| get_hls_playlist_legacy | Gets a hls video playlist. | HlsSegment | jellyfin-mcp | 65 | No |
| stop_encoding_process | Stops an active encoding. | HlsSegment | jellyfin-mcp | 65 | Yes |
| get_artist_image | Get artist image by name. | Image | jellyfin-mcp | 50 | No |
| get_splashscreen | Generates or gets the splashscreen. | Image | jellyfin-mcp | 45 | No |
| upload_custom_splashscreen | Uploads a custom splashscreen. The body is expected to the image contents base64 encoded. | Image | jellyfin-mcp | 65 | Yes |
| delete_custom_splashscreen | Delete a custom splashscreen. | Image | jellyfin-mcp | 50 | Yes |
| get_genre_image | Get genre image by name. | Image | jellyfin-mcp | 50 | No |
| get_genre_image_by_index | Get genre image by name. | Image | jellyfin-mcp | 55 | No |
| get_item_image_infos | Get item image infos. | Image | jellyfin-mcp | 55 | No |
| delete_item_image | Delete an item's image. | Image | jellyfin-mcp | 50 | Yes |
| set_item_image | Set item image. | Image | jellyfin-mcp | 45 | Yes |
| get_item_image | Gets the item's image. | Image | jellyfin-mcp | 50 | No |
| delete_item_image_by_index | Delete an item's image. | Image | jellyfin-mcp | 55 | Yes |
| set_item_image_by_index | Set item image. | Image | jellyfin-mcp | 50 | Yes |
| get_item_image_by_index | Gets the item's image. | Image | jellyfin-mcp | 55 | No |
| get_item_image2 | Gets the item's image. | Image | jellyfin-mcp | 50 | No |
| update_item_image_index | Updates the index for an item image. | Image | jellyfin-mcp | 55 | Yes |
| get_music_genre_image | Get music genre image by name. | Image | jellyfin-mcp | 55 | No |
| get_music_genre_image_by_index | Get music genre image by name. | Image | jellyfin-mcp | 55 | No |
| get_person_image | Get person image by name. | Image | jellyfin-mcp | 50 | No |
| get_person_image_by_index | Get person image by name. | Image | jellyfin-mcp | 55 | No |
| get_studio_image | Get studio image by name. | Image | jellyfin-mcp | 50 | No |
| get_studio_image_by_index | Get studio image by name. | Image | jellyfin-mcp | 55 | No |
| post_user_image | Sets the user image. | Image | jellyfin-mcp | 55 | Yes |
| delete_user_image | Delete the user's image. | Image | jellyfin-mcp | 50 | Yes |
| get_user_image | Get user profile image. | Image | jellyfin-mcp | 50 | No |
| get_instant_mix_from_album | Creates an instant playlist based on a given album. | InstantMix | jellyfin-mcp | 75 | No |
| get_instant_mix_from_artists | Creates an instant playlist based on a given artist. | InstantMix | jellyfin-mcp | 75 | No |
| get_instant_mix_from_artists2 | Creates an instant playlist based on a given artist. | InstantMix | jellyfin-mcp | 75 | No |
| get_instant_mix_from_item | Creates an instant playlist based on a given item. | InstantMix | jellyfin-mcp | 65 | No |
| get_instant_mix_from_music_genre_by_name | Creates an instant playlist based on a given genre. | InstantMix | jellyfin-mcp | 75 | No |
| get_instant_mix_from_music_genre_by_id | Creates an instant playlist based on a given genre. | InstantMix | jellyfin-mcp | 75 | No |
| get_instant_mix_from_playlist | Creates an instant playlist based on a given playlist. | InstantMix | jellyfin-mcp | 75 | No |
| get_instant_mix_from_song | Creates an instant playlist based on a given song. | InstantMix | jellyfin-mcp | 65 | No |
| get_external_id_infos | Get the item's external id info. | ItemLookup | jellyfin-mcp | 60 | No |
| apply_search_criteria | Applies search criteria to an item and refreshes metadata. | ItemLookup | jellyfin-mcp | 75 | No |
| get_book_remote_search_results | Get book remote search. | ItemLookup | jellyfin-mcp | 65 | No |
| get_box_set_remote_search_results | Get box set remote search. | ItemLookup | jellyfin-mcp | 65 | Yes |
| get_movie_remote_search_results | Get movie remote search. | ItemLookup | jellyfin-mcp | 65 | No |
| get_music_album_remote_search_results | Get music album remote search. | ItemLookup | jellyfin-mcp | 65 | No |
| get_music_artist_remote_search_results | Get music artist remote search. | ItemLookup | jellyfin-mcp | 65 | No |
| get_music_video_remote_search_results | Get music video remote search. | ItemLookup | jellyfin-mcp | 65 | No |
| get_person_remote_search_results | Get person remote search. | ItemLookup | jellyfin-mcp | 65 | No |
| get_series_remote_search_results | Get series remote search. | ItemLookup | jellyfin-mcp | 65 | No |
| get_trailer_remote_search_results | Get trailer remote search. | ItemLookup | jellyfin-mcp | 65 | No |
| refresh_item | Refreshes metadata for an item. | ItemRefresh | jellyfin-mcp | 60 | No |
| get_items | Gets items based on a query. | Items | jellyfin-mcp | 45 | No |
| get_item_user_data | Get Item User Data. | Items | jellyfin-mcp | 55 | No |
| update_item_user_data | Update Item User Data. | Items | jellyfin-mcp | 55 | Yes |
| get_resume_items | Gets items based on a query. | Items | jellyfin-mcp | 50 | No |
| delete_items | Deletes items from the library and filesystem. | Library | jellyfin-mcp | 55 | Yes |
| delete_item | Deletes an item from the library and filesystem. | Library | jellyfin-mcp | 55 | Yes |
| get_similar_albums | Gets similar items. | Library | jellyfin-mcp | 60 | No |
| get_similar_artists | Gets similar items. | Library | jellyfin-mcp | 60 | No |
| get_ancestors | Gets all parents of an item. | Library | jellyfin-mcp | 55 | No |
| get_critic_reviews | Gets critic review for an item. | Library | jellyfin-mcp | 60 | No |
| get_download | Downloads item media. | Library | jellyfin-mcp | 55 | No |
| get_file | Get the original file of an item. | Library | jellyfin-mcp | 55 | No |
| get_similar_items | Gets similar items. | Library | jellyfin-mcp | 60 | No |
| get_theme_media | Get theme songs and videos for an item. | Library | jellyfin-mcp | 60 | No |
| get_theme_songs | Get theme songs for an item. | Library | jellyfin-mcp | 60 | No |
| get_theme_videos | Get theme videos for an item. | Library | jellyfin-mcp | 60 | No |
| get_item_counts | Get item counts. | Library | jellyfin-mcp | 60 | No |
| get_library_options_info | Gets the library options info. | Library | jellyfin-mcp | 65 | No |
| post_updated_media | Reports that new movies have been added by an external source. | Library | jellyfin-mcp | 75 | Yes |
| get_media_folders | Gets all user media folders. | Library | jellyfin-mcp | 60 | No |
| post_added_movies | Reports that new movies have been added by an external source. | Library | jellyfin-mcp | 75 | Yes |
| post_updated_movies | Reports that new movies have been added by an external source. | Library | jellyfin-mcp | 75 | Yes |
| get_physical_paths | Gets a list of physical paths from virtual folders. | Library | jellyfin-mcp | 70 | No |
| refresh_library | Starts a library scan. | Library | jellyfin-mcp | 60 | No |
| post_added_series | Reports that new episodes of a series have been added by an external source. | Library | jellyfin-mcp | 75 | Yes |
| post_updated_series | Reports that new episodes of a series have been added by an external source. | Library | jellyfin-mcp | 75 | Yes |
| get_similar_movies | Gets similar items. | Library | jellyfin-mcp | 60 | No |
| get_similar_shows | Gets similar items. | Library | jellyfin-mcp | 60 | No |
| get_similar_trailers | Gets similar items. | Library | jellyfin-mcp | 60 | No |
| update_item | Updates an item. | ItemUpdate | jellyfin-mcp | 55 | Yes |
| update_item_content_type | Updates an item's content type. | ItemUpdate | jellyfin-mcp | 65 | Yes |
| get_metadata_editor_info | Gets metadata editor info for an item. | ItemUpdate | jellyfin-mcp | 65 | No |
| get_item | Gets an item from a user's library. | UserLibrary | jellyfin-mcp | 55 | No |
| get_intros | Gets intros to play before the main media item plays. | UserLibrary | jellyfin-mcp | 65 | No |
| get_local_trailers | Gets local trailers for an item. | UserLibrary | jellyfin-mcp | 60 | No |
| get_special_features | Gets special features for an item. | UserLibrary | jellyfin-mcp | 60 | No |
| get_latest_media | Gets latest media. | UserLibrary | jellyfin-mcp | 60 | No |
| get_root_folder | Gets the root folder from a user's library. | UserLibrary | jellyfin-mcp | 60 | No |
| mark_favorite_item | Marks an item as a favorite. | UserLibrary | jellyfin-mcp | 65 | No |
| unmark_favorite_item | Unmarks item as a favorite. | UserLibrary | jellyfin-mcp | 65 | No |
| delete_user_item_rating | Deletes a user's saved personal rating for an item. | UserLibrary | jellyfin-mcp | 75 | Yes |
| update_user_item_rating | Updates a user's rating for an item. | UserLibrary | jellyfin-mcp | 65 | Yes |
| get_virtual_folders | Gets all virtual folders. | LibraryStructure | jellyfin-mcp | 60 | No |
| add_virtual_folder | Adds a virtual folder. | LibraryStructure | jellyfin-mcp | 65 | Yes |
| remove_virtual_folder | Removes a virtual folder. | LibraryStructure | jellyfin-mcp | 65 | Yes |
| update_library_options | Update library options. | LibraryStructure | jellyfin-mcp | 60 | Yes |
| rename_virtual_folder | Renames a virtual folder. | LibraryStructure | jellyfin-mcp | 65 | Yes |
| add_media_path | Add a media path to a library. | LibraryStructure | jellyfin-mcp | 65 | Yes |
| remove_media_path | Remove a media path. | LibraryStructure | jellyfin-mcp | 65 | Yes |
| update_media_path | Updates a media path. | LibraryStructure | jellyfin-mcp | 60 | Yes |
| get_channel_mapping_options | Get channel mapping options. | LiveTv | jellyfin-mcp | 55 | No |
| set_channel_mapping | Set channel mappings. | LiveTv | jellyfin-mcp | 50 | Yes |
| get_live_tv_channels | Gets available live tv channels. | LiveTv | jellyfin-mcp | 50 | No |
| get_channel | Gets a live tv channel. | LiveTv | jellyfin-mcp | 45 | No |
| get_guide_info | Get guide info. | LiveTv | jellyfin-mcp | 45 | No |
| get_live_tv_info | Gets available live tv services. | LiveTv | jellyfin-mcp | 50 | No |
| add_listing_provider | Adds a listings provider. | LiveTv | jellyfin-mcp | 55 | Yes |
| delete_listing_provider | Delete listing provider. | LiveTv | jellyfin-mcp | 50 | Yes |
| get_default_listing_provider | Gets default listings provider info. | LiveTv | jellyfin-mcp | 55 | No |
| get_lineups | Gets available lineups. | LiveTv | jellyfin-mcp | 45 | No |
| get_schedules_direct_countries | Gets available countries. | LiveTv | jellyfin-mcp | 55 | No |
| get_live_recording_file | Gets a live tv recording stream. | LiveTv | jellyfin-mcp | 55 | No |
| get_live_stream_file | Gets a live tv channel stream. | LiveTv | jellyfin-mcp | 55 | No |
| get_live_tv_programs | Gets available live tv epgs. | LiveTv | jellyfin-mcp | 50 | No |
| get_programs | Gets available live tv epgs. | LiveTv | jellyfin-mcp | 45 | No |
| get_program | Gets a live tv program. | LiveTv | jellyfin-mcp | 45 | No |
| get_recommended_programs | Gets recommended live tv epgs. | LiveTv | jellyfin-mcp | 50 | No |
| get_recordings | Gets live tv recordings. | LiveTv | jellyfin-mcp | 45 | No |
| get_recording | Gets a live tv recording. | LiveTv | jellyfin-mcp | 45 | No |
| delete_recording | Deletes a live tv recording. | LiveTv | jellyfin-mcp | 45 | Yes |
| get_recording_folders | Gets recording folders. | LiveTv | jellyfin-mcp | 50 | No |
| get_recording_groups | Gets live tv recording groups. | LiveTv | jellyfin-mcp | 50 | No |
| get_recording_group | Get recording group. | LiveTv | jellyfin-mcp | 50 | No |
| get_recordings_series | Gets live tv recording series. | LiveTv | jellyfin-mcp | 50 | No |
| get_series_timers | Gets live tv series timers. | LiveTv | jellyfin-mcp | 50 | No |
| create_series_timer | Creates a live tv series timer. | LiveTv | jellyfin-mcp | 50 | Yes |
| get_series_timer | Gets a live tv series timer. | LiveTv | jellyfin-mcp | 50 | No |
| cancel_series_timer | Cancels a live tv series timer. | LiveTv | jellyfin-mcp | 55 | No |
| update_series_timer | Updates a live tv series timer. | LiveTv | jellyfin-mcp | 50 | Yes |
| get_timers | Gets the live tv timers. | LiveTv | jellyfin-mcp | 45 | No |
| create_timer | Creates a live tv timer. | LiveTv | jellyfin-mcp | 45 | Yes |
| get_timer | Gets a timer. | LiveTv | jellyfin-mcp | 40 | No |
| cancel_timer | Cancels a live tv timer. | LiveTv | jellyfin-mcp | 50 | No |
| update_timer | Updates a live tv timer. | LiveTv | jellyfin-mcp | 45 | Yes |
| get_default_timer | Gets the default values for a new timer. | LiveTv | jellyfin-mcp | 50 | No |
| add_tuner_host | Adds a tuner host. | LiveTv | jellyfin-mcp | 55 | Yes |
| delete_tuner_host | Deletes a tuner host. | LiveTv | jellyfin-mcp | 50 | Yes |
| get_tuner_host_types | Get tuner host types. | LiveTv | jellyfin-mcp | 55 | No |
| reset_tuner | Resets a tv tuner. | LiveTv | jellyfin-mcp | 50 | Yes |
| discover_tuners | Discover tuners. | LiveTv | jellyfin-mcp | 50 | No |
| discvover_tuners | Discover tuners. | LiveTv | jellyfin-mcp | 50 | No |
| get_countries | Gets known countries. | Localization | jellyfin-mcp | 55 | No |
| get_cultures | Gets known cultures. | Localization | jellyfin-mcp | 55 | No |
| get_localization_options | Gets localization options. | Localization | jellyfin-mcp | 60 | No |
| get_parental_ratings | Gets known parental ratings. | Localization | jellyfin-mcp | 60 | No |
| get_lyrics | Gets an item's lyrics. | Lyrics | jellyfin-mcp | 45 | No |
| upload_lyrics | Upload an external lyric file. | Lyrics | jellyfin-mcp | 50 | Yes |
| delete_lyrics | Deletes an external lyric file. | Lyrics | jellyfin-mcp | 45 | Yes |
| search_remote_lyrics | Search remote lyrics. | Lyrics | jellyfin-mcp | 55 | No |
| download_remote_lyrics | Downloads a remote lyric. | Lyrics | jellyfin-mcp | 55 | No |
| get_remote_lyrics | Gets the remote lyrics. | Lyrics | jellyfin-mcp | 50 | No |
| get_playback_info | Gets live playback media info for an item. | MediaInfo | jellyfin-mcp | 60 | No |
| get_posted_playback_info | Gets live playback media info for an item. | MediaInfo | jellyfin-mcp | 65 | Yes |
| close_live_stream | Closes a media source. | MediaInfo | jellyfin-mcp | 65 | No |
| open_live_stream | Opens a media source. | MediaInfo | jellyfin-mcp | 65 | No |
| get_bitrate_test_bytes | Tests the network with a request with the size of the bitrate. | MediaInfo | jellyfin-mcp | 75 | No |
| get_item_segments | Gets all media segments based on an itemId. | MediaSegments | jellyfin-mcp | 60 | No |
| get_movie_recommendations | Gets movie recommendations. | Movies | jellyfin-mcp | 50 | No |
| get_music_genres | Gets all music genres from a given item, folder, or the entire library. | MusicGenres | jellyfin-mcp | 70 | No |
| get_music_genre | Gets a music genre, by name. | MusicGenres | jellyfin-mcp | 60 | No |
| get_packages | Gets available packages. | Package | jellyfin-mcp | 55 | No |
| get_package_info | Gets a package by name or assembly GUID. | Package | jellyfin-mcp | 60 | No |
| install_package | Installs a package. | Package | jellyfin-mcp | 60 | Yes |
| cancel_package_installation | Cancels a package installation. | Package | jellyfin-mcp | 65 | Yes |
| get_repositories | Gets all package repositories. | Package | jellyfin-mcp | 55 | No |
| set_repositories | Sets the enabled and existing package repositories. | Package | jellyfin-mcp | 65 | Yes |
| get_persons | Gets all persons. | Persons | jellyfin-mcp | 55 | No |
| get_person | Get person by name. | Persons | jellyfin-mcp | 55 | No |
| create_playlist | Creates a new playlist. | Playlists | jellyfin-mcp | 55 | Yes |
| update_playlist | Updates a playlist. | Playlists | jellyfin-mcp | 55 | Yes |
| get_playlist | Get a playlist. | Playlists | jellyfin-mcp | 50 | No |
| add_item_to_playlist | Adds items to a playlist. | Playlists | jellyfin-mcp | 65 | Yes |
| remove_item_from_playlist | Removes items from a playlist. | Playlists | jellyfin-mcp | 65 | Yes |
| get_playlist_items | Gets the original items of a playlist. | Playlists | jellyfin-mcp | 60 | No |
| move_item | Moves a playlist item. | Playlists | jellyfin-mcp | 60 | Yes |
| get_playlist_users | Get a playlist's users. | Playlists | jellyfin-mcp | 60 | No |
| get_playlist_user | Get a playlist user. | Playlists | jellyfin-mcp | 60 | No |
| update_playlist_user | Modify a user of a playlist's users. | Playlists | jellyfin-mcp | 60 | Yes |
| remove_user_from_playlist | Remove a user from a playlist's users. | Playlists | jellyfin-mcp | 65 | Yes |
| on_playback_start | Reports that a session has begun playing an item. | Playstate | jellyfin-mcp | 60 | Yes |
| on_playback_stopped | Reports that a session has stopped playing an item. | Playstate | jellyfin-mcp | 70 | Yes |
| on_playback_progress | Reports a session's playback progress. | Playstate | jellyfin-mcp | 60 | No |
| report_playback_start | Reports playback has started within a session. | Playstate | jellyfin-mcp | 65 | Yes |
| ping_playback_session | Pings a playback session. | Playstate | jellyfin-mcp | 65 | No |
| report_playback_progress | Reports playback progress within a session. | Playstate | jellyfin-mcp | 65 | No |
| report_playback_stopped | Reports playback has stopped within a session. | Playstate | jellyfin-mcp | 65 | Yes |
| mark_played_item | Marks an item as played for user. | Playstate | jellyfin-mcp | 65 | No |
| mark_unplayed_item | Marks an item as unplayed for user. | Playstate | jellyfin-mcp | 65 | No |
| get_plugins | Gets a list of currently installed plugins. | Plugins | jellyfin-mcp | 55 | No |
| uninstall_plugin | Uninstalls a plugin. | Plugins | jellyfin-mcp | 60 | Yes |
| uninstall_plugin_by_version | Uninstalls a plugin by version. | Plugins | jellyfin-mcp | 65 | Yes |
| disable_plugin | Disable a plugin. | Plugins | jellyfin-mcp | 60 | Yes |
| enable_plugin | Enables a disabled plugin. | Plugins | jellyfin-mcp | 60 | Yes |
| get_plugin_image | Gets a plugin's image. | Plugins | jellyfin-mcp | 60 | No |
| get_plugin_configuration | Gets plugin configuration. | Plugins | jellyfin-mcp | 60 | No |
| update_plugin_configuration | Updates plugin configuration. | Plugins | jellyfin-mcp | 60 | Yes |
| get_plugin_manifest | Gets a plugin's manifest. | Plugins | jellyfin-mcp | 60 | No |
| authorize_quick_connect | Authorizes a pending quick connect request. | QuickConnect | jellyfin-mcp | 65 | No |
| get_quick_connect_state | Attempts to retrieve authentication information. | QuickConnect | jellyfin-mcp | 65 | No |
| get_quick_connect_enabled | Gets the current quick connect state. | QuickConnect | jellyfin-mcp | 65 | Yes |
| initiate_quick_connect | Initiate a new quick connect request. | QuickConnect | jellyfin-mcp | 65 | No |
| get_remote_images | Gets available remote images for an item. | RemoteImage | jellyfin-mcp | 60 | No |
| download_remote_image | Downloads a remote image for an item. | RemoteImage | jellyfin-mcp | 65 | No |
| get_remote_image_providers | Gets available remote image providers for an item. | RemoteImage | jellyfin-mcp | 65 | No |
| get_tasks | Get tasks. | ScheduledTasks | jellyfin-mcp | 50 | No |
| get_task | Get task by id. | ScheduledTasks | jellyfin-mcp | 50 | No |
| update_task | Update specified task triggers. | ScheduledTasks | jellyfin-mcp | 55 | Yes |
| start_task | Start specified task. | ScheduledTasks | jellyfin-mcp | 60 | Yes |
| stop_task | Stop specified task. | ScheduledTasks | jellyfin-mcp | 60 | Yes |
| get_search_hints | Gets the search hint result. | Search | jellyfin-mcp | 50 | No |
| get_password_reset_providers | Get all password reset providers. | Session | jellyfin-mcp | 65 | Yes |
| get_auth_providers | Get all auth providers. | Session | jellyfin-mcp | 60 | No |
| get_sessions | Gets a list of sessions. | Session | jellyfin-mcp | 55 | No |
| send_full_general_command | Issues a full general command to a client. | Session | jellyfin-mcp | 65 | No |
| send_general_command | Issues a general command to a client. | Session | jellyfin-mcp | 65 | No |
| send_message_command | Issues a command to a client to display a message to the user. | Session | jellyfin-mcp | 75 | No |
| play | Instructs a session to play an item. | Session | jellyfin-mcp | 55 | No |
| send_playstate_command | Issues a playstate command to a client. | Session | jellyfin-mcp | 65 | No |
| send_system_command | Issues a system command to a client. | Session | jellyfin-mcp | 65 | No |
| add_user_to_session | Adds an additional user to a session. | Session | jellyfin-mcp | 65 | Yes |
| remove_user_from_session | Removes an additional user from a session. | Session | jellyfin-mcp | 65 | Yes |
| display_content | Instructs a session to browse to an item or view. | Session | jellyfin-mcp | 60 | No |
| post_capabilities | Updates capabilities for a device. | Session | jellyfin-mcp | 60 | Yes |
| post_full_capabilities | Updates capabilities for a device. | Session | jellyfin-mcp | 65 | Yes |
| report_session_ended | Reports that a session has ended. | Session | jellyfin-mcp | 65 | No |
| report_viewing | Reports that a session is viewing an item. | Session | jellyfin-mcp | 60 | No |
| complete_wizard | Completes the startup wizard. | Startup | jellyfin-mcp | 60 | No |
| get_startup_configuration | Gets the initial startup wizard configuration. | Startup | jellyfin-mcp | 60 | Yes |
| update_initial_configuration | Sets the initial startup wizard configuration. | Startup | jellyfin-mcp | 60 | Yes |
| get_first_user_2 | Gets the first user. | Startup | jellyfin-mcp | 60 | No |
| set_remote_access | Sets remote access and UPnP. | Startup | jellyfin-mcp | 60 | Yes |
| get_first_user | Gets the first user. | Startup | jellyfin-mcp | 60 | No |
| update_startup_user | Sets the user name and password. | Startup | jellyfin-mcp | 60 | Yes |
| get_studios | Gets all studios from a given item, folder, or the entire library. | Studios | jellyfin-mcp | 65 | No |
| get_studio | Gets a studio by name. | Studios | jellyfin-mcp | 55 | No |
| get_fallback_font_list | Gets a list of available fallback font files. | Subtitle | jellyfin-mcp | 60 | No |
| get_fallback_font | Gets a fallback font file. | Subtitle | jellyfin-mcp | 60 | No |
| search_remote_subtitles | Search remote subtitles. | Subtitle | jellyfin-mcp | 65 | No |
| download_remote_subtitles | Downloads a remote subtitle. | Subtitle | jellyfin-mcp | 65 | No |
| get_remote_subtitles | Gets the remote subtitles. | Subtitle | jellyfin-mcp | 60 | No |
| get_subtitle_playlist | Gets an HLS subtitle playlist. | Subtitle | jellyfin-mcp | 60 | No |
| upload_subtitle | Upload an external subtitle file. | Subtitle | jellyfin-mcp | 60 | Yes |
| delete_subtitle | Deletes an external subtitle file. | Subtitle | jellyfin-mcp | 55 | Yes |
| get_subtitle_with_ticks | Gets subtitles in a specified format. | Subtitle | jellyfin-mcp | 65 | No |
| get_subtitle | Gets subtitles in a specified format. | Subtitle | jellyfin-mcp | 55 | No |
| get_suggestions | Gets suggestions. | Suggestions | jellyfin-mcp | 55 | No |
| sync_play_get_group | Gets a SyncPlay group by id. | SyncPlay | jellyfin-mcp | 65 | No |
| sync_play_buffering | Notify SyncPlay group that member is buffering. | SyncPlay | jellyfin-mcp | 65 | No |
| sync_play_join_group | Join an existing SyncPlay group. | SyncPlay | jellyfin-mcp | 65 | No |
| sync_play_leave_group | Leave the joined SyncPlay group. | SyncPlay | jellyfin-mcp | 65 | No |
| sync_play_get_groups | Gets all SyncPlay groups. | SyncPlay | jellyfin-mcp | 65 | No |
| sync_play_move_playlist_item | Request to move an item in the playlist in SyncPlay group. | SyncPlay | jellyfin-mcp | 75 | Yes |
| sync_play_create_group | Create a new SyncPlay group. | SyncPlay | jellyfin-mcp | 65 | Yes |
| sync_play_next_item | Request next item in SyncPlay group. | SyncPlay | jellyfin-mcp | 65 | No |
| sync_play_pause | Request pause in SyncPlay group. | SyncPlay | jellyfin-mcp | 65 | Yes |
| sync_play_ping | Update session ping. | SyncPlay | jellyfin-mcp | 65 | No |
| sync_play_previous_item | Request previous item in SyncPlay group. | SyncPlay | jellyfin-mcp | 65 | No |
| sync_play_queue | Request to queue items to the playlist of a SyncPlay group. | SyncPlay | jellyfin-mcp | 75 | No |
| sync_play_ready | Notify SyncPlay group that member is ready for playback. | SyncPlay | jellyfin-mcp | 75 | No |
| sync_play_remove_from_playlist | Request to remove items from the playlist in SyncPlay group. | SyncPlay | jellyfin-mcp | 75 | Yes |
| sync_play_seek | Request seek in SyncPlay group. | SyncPlay | jellyfin-mcp | 65 | No |
| sync_play_set_ignore_wait | Request SyncPlay group to ignore member during group-wait. | SyncPlay | jellyfin-mcp | 75 | Yes |
| sync_play_set_new_queue | Request to set new playlist in SyncPlay group. | SyncPlay | jellyfin-mcp | 65 | Yes |
| sync_play_set_playlist_item | Request to change playlist item in SyncPlay group. | SyncPlay | jellyfin-mcp | 65 | Yes |
| sync_play_set_repeat_mode | Request to set repeat mode in SyncPlay group. | SyncPlay | jellyfin-mcp | 65 | Yes |
| sync_play_set_shuffle_mode | Request to set shuffle mode in SyncPlay group. | SyncPlay | jellyfin-mcp | 65 | Yes |
| sync_play_stop | Request stop in SyncPlay group. | SyncPlay | jellyfin-mcp | 65 | Yes |
| sync_play_unpause | Request unpause in SyncPlay group. | SyncPlay | jellyfin-mcp | 65 | Yes |
| get_endpoint_info | Gets information about the request endpoint. | System | jellyfin-mcp | 50 | No |
| get_system_info | Gets information about the server. | System | jellyfin-mcp | 50 | No |
| get_public_system_info | Gets public information about the server. | System | jellyfin-mcp | 55 | No |
| get_system_storage | Gets information about the server. | System | jellyfin-mcp | 50 | No |
| get_server_logs | Gets a list of available server log files. | System | jellyfin-mcp | 50 | No |
| get_log_file | Gets a log file. | System | jellyfin-mcp | 50 | No |
| get_ping_system | Pings the system. | System | jellyfin-mcp | 50 | No |
| post_ping_system | Pings the system. | System | jellyfin-mcp | 55 | Yes |
| restart_application | Restarts the application. | System | jellyfin-mcp | 50 | Yes |
| shutdown_application | Shuts down the application. | System | jellyfin-mcp | 50 | Yes |
| get_utc_time | Gets the current UTC time. | TimeSync | jellyfin-mcp | 60 | No |
| tmdb_client_configuration | Gets the TMDb image configuration options. | Tmdb | jellyfin-mcp | 55 | No |
| get_trailers | Finds movies and trailers similar to a given trailer. | Trailers | jellyfin-mcp | 65 | No |
| get_trickplay_tile_image | Gets a trickplay tile image. | Trickplay | jellyfin-mcp | 65 | No |
| get_trickplay_hls_playlist | Gets an image tiles playlist for trickplay. | Trickplay | jellyfin-mcp | 65 | No |
| get_episodes | Gets episodes for a tv season. | TvShows | jellyfin-mcp | 55 | No |
| get_seasons | Gets seasons for a tv series. | TvShows | jellyfin-mcp | 55 | No |
| get_next_up | Gets a list of next up episodes. | TvShows | jellyfin-mcp | 55 | No |
| get_upcoming_episodes | Gets a list of upcoming episodes. | TvShows | jellyfin-mcp | 60 | No |
| get_universal_audio_stream | Gets an audio stream. | UniversalAudio | jellyfin-mcp | 65 | No |
| get_users | Gets a list of users. | User | jellyfin-mcp | 45 | No |
| update_user | Updates a user. | User | jellyfin-mcp | 40 | Yes |
| get_user_by_id | Gets a user by Id. | User | jellyfin-mcp | 45 | No |
| delete_user | Deletes a user. | User | jellyfin-mcp | 40 | Yes |
| update_user_policy | Updates a user policy. | User | jellyfin-mcp | 50 | Yes |
| authenticate_user_by_name | Authenticates a user by name. | User | jellyfin-mcp | 55 | No |
| authenticate_with_quick_connect | Authenticates a user with quick connect. | User | jellyfin-mcp | 55 | No |
| update_user_configuration | Updates a user configuration. | User | jellyfin-mcp | 50 | Yes |
| forgot_password | Initiates the forgot password process for a local user. | User | jellyfin-mcp | 60 | No |
| forgot_password_pin | Redeems a forgot password pin. | User | jellyfin-mcp | 55 | No |
| get_current_user | Gets the user based on auth token. | User | jellyfin-mcp | 50 | No |
| create_user_by_name | Creates a user. | User | jellyfin-mcp | 45 | Yes |
| update_user_password | Updates a user's password. | User | jellyfin-mcp | 50 | Yes |
| get_public_users | Gets a list of publicly visible users for display on a login screen. | User | jellyfin-mcp | 60 | No |
| get_user_views | Get user views. | UserViews | jellyfin-mcp | 55 | No |
| get_grouping_options | Get user view grouping options. | UserViews | jellyfin-mcp | 60 | No |
| get_attachment | Get video attachment. | VideoAttachments | jellyfin-mcp | 55 | No |
| get_additional_part | Gets additional parts for a video. | Videos | jellyfin-mcp | 50 | Yes |
| delete_alternate_sources | Removes alternate video sources. | Videos | jellyfin-mcp | 50 | Yes |
| get_video_stream | Gets a video stream. | Videos | jellyfin-mcp | 50 | No |
| get_video_stream_by_container | Gets a video stream. | Videos | jellyfin-mcp | 55 | No |
| merge_versions | Merges videos into a single record. | Videos | jellyfin-mcp | 50 | No |
| get_years | Get years. | Years | jellyfin-mcp | 40 | No |
| get_year | Gets a year. | Years | jellyfin-mcp | 40 | No |
| langfuse-annotation-queues-annotation-queues-list-queues | Get all annotation queues | annotation_queues | langfuse | 65 | No |
| langfuse-annotation-queues-annotation-queues-create-queue | Create an annotation queue | annotation_queues | langfuse | 65 | Yes |
| langfuse-annotation-queues-annotation-queues-get-queue | Get an annotation queue by ID | annotation_queues | langfuse | 65 | No |
| langfuse-annotation-queues-annotation-queues-list-queue-items | Get items for a specific annotation queue | annotation_queues | langfuse | 65 | No |
| langfuse-annotation-queues-annotation-queues-create-queue-item | Add an item to an annotation queue | annotation_queues | langfuse | 65 | Yes |
| langfuse-annotation-queues-annotation-queues-get-queue-item | Get a specific item from an annotation queue | annotation_queues | langfuse | 65 | No |
| langfuse-annotation-queues-annotation-queues-update-queue-item | Update an annotation queue item | annotation_queues | langfuse | 65 | Yes |
| langfuse-annotation-queues-annotation-queues-delete-queue-item | Remove an item from an annotation queue | annotation_queues | langfuse | 65 | Yes |
| langfuse-annotation-queues-annotation-queues-create-queue-assignment | Create an assignment for a user to an annotation queue | annotation_queues | langfuse | 75 | Yes |
| langfuse-annotation-queues-annotation-queues-delete-queue-assignment | Delete an assignment for a user to an annotation queue | annotation_queues | langfuse | 75 | Yes |
| langfuse-blob-storage-integrations-blob-storage-integrations-get-blob-storage-integrations | Get all blob storage integrations for the organization (requires organization-scoped API key) | blob_storage_integrations | langfuse | 75 | No |
| langfuse-blob-storage-integrations-blob-storage-integrations-upsert-blob-storage-integration | Create or update a blob storage integration for a specific project (requires organization-scoped API key). The configuration is validated by performing a test upload to the bucket. | blob_storage_integrations | langfuse | 85 | No |
| langfuse-blob-storage-integrations-blob-storage-integrations-get-blob-storage-integration-status | Get the sync status of a blob storage integration by integration ID (requires organization-scoped API key) | blob_storage_integrations | langfuse | 85 | No |
| langfuse-blob-storage-integrations-blob-storage-integrations-delete-blob-storage-integration | Delete a blob storage integration by ID (requires organization-scoped API key) | blob_storage_integrations | langfuse | 75 | Yes |
| langfuse-comments-create | Create a comment. Comments may be attached to different object types (trace, observation, session, prompt). | comments | langfuse | 80 | Yes |
| langfuse-comments-get | Get all comments | comments | langfuse | 60 | No |
| langfuse-comments-get-by-id | Get a comment by id | comments | langfuse | 60 | No |
| langfuse-dataset-items-dataset-items-create | Create a dataset item | dataset_items | langfuse | 65 | Yes |
| langfuse-dataset-items-dataset-items-list | Get dataset items. Optionally specify a version to get the items as they existed at that point in time. Note: If version parameter is provided, datasetName must also be provided. | dataset_items | langfuse | 85 | Yes |
| langfuse-dataset-items-dataset-items-get | Get a dataset item | dataset_items | langfuse | 65 | Yes |
| langfuse-dataset-items-dataset-items-delete | Delete a dataset item and all its run items. This action is irreversible. | dataset_items | langfuse | 75 | Yes |
| langfuse-dataset-run-items-dataset-run-items-create | Create a dataset run item | dataset_run_items | langfuse | 65 | Yes |
| langfuse-dataset-run-items-dataset-run-items-list | List dataset run items | dataset_run_items | langfuse | 65 | Yes |
| langfuse-datasets-list | Get all datasets | datasets | langfuse | 60 | Yes |
| langfuse-datasets-create | Create a dataset | datasets | langfuse | 60 | Yes |
| langfuse-datasets-get | Get a dataset | datasets | langfuse | 55 | Yes |
| langfuse-datasets-get-run | Get a dataset run and its items | datasets | langfuse | 60 | Yes |
| langfuse-datasets-delete-run | Delete a dataset run and all its run items. This action is irreversible. | datasets | langfuse | 70 | Yes |
| langfuse-datasets-get-runs | Get dataset runs | datasets | langfuse | 65 | Yes |
| langfuse-health-health | Check health of API and database | health | langfuse | 55 | No |
| langfuse-ingestion-batch | **Legacy endpoint for batch ingestion for Langfuse Observability.**  -> Please use the OpenTelemetry endpoint (`/api/public/otel/v1/traces`). Learn more: https://langfuse.com/integrations/native/opentelemetry  Within each batch, there can be multiple events. Each event has a type, an id, a timestamp, metadata and a body. Internally, we refer to this as the "event envelope" as it tells us something about the event but not the trace. We use the event id within this envelope to deduplicate messages to avoid processing the same event twice, i.e. the event id should be unique per request. The event.body.id is the ID of the actual trace and will be used for updates and will be visible within the Langfuse App. I.e. if you want to update a trace, you'd use the same body id, but separate event IDs.  Notes: - Introduction to data model: https://langfuse.com/docs/observability/data-model - Batch sizes are limited to 3.5 MB in total. You need to adjust the number of events per batch accordingly. - The API does not return a 4xx status code for input errors. Instead, it responds with a 207 status code, which includes a list of the encountered errors. | ingestion | langfuse | 85 | Yes |
| langfuse-legacy-metrics-v1-legacy-metrics-v1-metrics | Get metrics from the Langfuse project using a query object.  Consider using the [v2 metrics endpoint](/api-reference#tag/metricsv2/GET/api/public/v2/metrics) for better performance.  For more details, see the [Metrics API documentation](https://langfuse.com/docs/metrics/features/metrics-api). | legacy_metrics_v1 | langfuse | 85 | No |
| langfuse-legacy-observations-v1-legacy-observations-v1-get | Get a observation | legacy_observations_v1 | langfuse | 65 | No |
| langfuse-legacy-observations-v1-legacy-observations-v1-get-many | Get a list of observations.  Consider using the [v2 observations endpoint](/api-reference#tag/observationsv2/GET/api/public/v2/observations) for cursor-based pagination and field selection. | legacy_observations_v1 | langfuse | 85 | No |
| langfuse-legacy-score-v1-legacy-score-v1-create | Create a score (supports both trace and session scores) | legacy_score_v1 | langfuse | 75 | Yes |
| langfuse-legacy-score-v1-legacy-score-v1-delete | Delete a score (supports both trace and session scores) | legacy_score_v1 | langfuse | 75 | Yes |
| langfuse-llm-connections-llm-connections-list | Get all LLM connections in a project | llm_connections | langfuse | 65 | No |
| langfuse-llm-connections-llm-connections-upsert | Create or update an LLM connection. The connection is upserted on provider. | llm_connections | langfuse | 75 | No |
| langfuse-media-get | Get a media record | media | langfuse | 50 | No |
| langfuse-media-patch | Patch a media record | media | langfuse | 55 | Yes |
| langfuse-media-get-upload-url | Get a presigned upload URL for a media record | media | langfuse | 55 | Yes |
| langfuse-metrics-metrics | Get metrics from the Langfuse project using a query object. V2 endpoint with optimized performance.  ## V2 Differences - Supports `observations`, `scores-numeric`, and `scores-categorical` views only (traces view not supported) - Direct access to tags and release fields on observations - Backwards-compatible: traceName, traceRelease, traceVersion dimensions are still available on observations view - High cardinality dimensions are not supported and will return a 400 error (see below)  For more details, see the [Metrics API documentation](https://langfuse.com/docs/metrics/features/metrics-api).  ## Available Views  ### observations Query observation-level data (spans, generations, events).  **Dimensions:** - `environment` - Deployment environment (e.g., production, staging) - `type` - Type of observation (SPAN, GENERATION, EVENT) - `name` - Name of the observation - `level` - Logging level of the observation - `version` - Version of the observation - `tags` - User-defined tags - `release` - Release version - `traceName` - Name of the parent trace (backwards-compatible) - `traceRelease` - Release version of the parent trace (backwards-compatible, maps to release) - `traceVersion` - Version of the parent trace (backwards-compatible, maps to version) - `providedModelName` - Name of the model used - `promptName` - Name of the prompt used - `promptVersion` - Version of the prompt used - `startTimeMonth` - Month of start_time in YYYY-MM format  **Measures:** - `count` - Total number of observations - `latency` - Observation latency (milliseconds) - `streamingLatency` - Generation latency from completion start to end (milliseconds) - `inputTokens` - Sum of input tokens consumed - `outputTokens` - Sum of output tokens produced - `totalTokens` - Sum of all tokens consumed - `outputTokensPerSecond` - Output tokens per second - `tokensPerSecond` - Total tokens per second - `inputCost` - Input cost (USD) - `outputCost` - Output cost (USD) - `totalCost` - Total cost (USD) - `timeToFirstToken` - Time to first token (milliseconds) - `countScores` - Number of scores attached to the observation  ### scores-numeric Query numeric and boolean score data.  **Dimensions:** - `environment` - Deployment environment - `name` - Name of the score (e.g., accuracy, toxicity) - `source` - Origin of the score (API, ANNOTATION, EVAL) - `dataType` - Data type (NUMERIC, BOOLEAN) - `configId` - Identifier of the score config - `timestampMonth` - Month in YYYY-MM format - `timestampDay` - Day in YYYY-MM-DD format - `value` - Numeric value of the score - `traceName` - Name of the parent trace - `tags` - Tags - `traceRelease` - Release version - `traceVersion` - Version - `observationName` - Name of the associated observation - `observationModelName` - Model name of the associated observation - `observationPromptName` - Prompt name of the associated observation - `observationPromptVersion` - Prompt version of the associated observation  **Measures:** - `count` - Total number of scores - `value` - Score value (for aggregations)  ### scores-categorical Query categorical score data. Same dimensions as scores-numeric except uses `stringValue` instead of `value`.  **Measures:** - `count` - Total number of scores  ## High Cardinality Dimensions The following dimensions cannot be used as grouping dimensions in v2 metrics API as they can cause performance issues. Use them in filters instead.  **observations view:** - `id` - Use traceId filter to narrow down results - `traceId` - Use traceId filter instead - `userId` - Use userId filter instead - `sessionId` - Use sessionId filter instead - `parentObservationId` - Use parentObservationId filter instead  **scores-numeric / scores-categorical views:** - `id` - Use specific filters to narrow down results - `traceId` - Use traceId filter instead - `userId` - Use userId filter instead - `sessionId` - Use sessionId filter instead - `observationId` - Use observationId filter instead  ## Aggregations Available aggregation functions: `sum`, `avg`, `count`, `max`, `min`, `p50`, `p75`, `p90`, `p95`, `p99`, `histogram`  ## Time Granularities Available granularities for timeDimension: `auto`, `minute`, `hour`, `day`, `week`, `month` - `auto` bins the data into approximately 50 buckets based on the time range | metrics | langfuse | 85 | No |
| langfuse-models-create | Create a model | models | langfuse | 45 | Yes |
| langfuse-models-list | Get all models | models | langfuse | 45 | No |
| langfuse-models-get | Get a model | models | langfuse | 45 | No |
| langfuse-models-delete | Delete a model. Cannot delete models managed by Langfuse. You can create your own definition with the same modelName to override the definition though. | models | langfuse | 70 | Yes |
| langfuse-observations-get-many | Get a list of observations with cursor-based pagination and flexible field selection.  ## Cursor-based Pagination This endpoint uses cursor-based pagination for efficient traversal of large datasets. The cursor is returned in the response metadata and should be passed in subsequent requests to retrieve the next page of results.  ## Field Selection Use the `fields` parameter to control which observation fields are returned: - `core` - Always included: id, traceId, startTime, endTime, projectId, parentObservationId, type - `basic` - name, level, statusMessage, version, environment, bookmarked, public, userId, sessionId - `time` - completionStartTime, createdAt, updatedAt - `io` - input, output - `metadata` - metadata (truncated to 200 chars by default, use `expandMetadata` to get full values) - `model` - providedModelName, internalModelId, modelParameters - `usage` - usageDetails, costDetails, totalCost - `prompt` - promptId, promptName, promptVersion - `metrics` - latency, timeToFirstToken  If not specified, `core` and `basic` field groups are returned.  ## Filters Multiple filtering options are available via query parameters or the structured `filter` parameter. When using the `filter` parameter, it takes precedence over individual query parameter filters. | observations | langfuse | 85 | No |
| langfuse-opentelemetry-export-traces | **OpenTelemetry Traces Ingestion Endpoint**  This endpoint implements the OTLP/HTTP specification for trace ingestion, providing native OpenTelemetry integration for Langfuse Observability.  **Supported Formats:** - Binary Protobuf: `Content-Type: application/x-protobuf` - JSON Protobuf: `Content-Type: application/json` - Supports gzip compression via `Content-Encoding: gzip` header  **Specification Compliance:** - Conforms to [OTLP/HTTP Trace Export](https://opentelemetry.io/docs/specs/otlp/#otlphttp) - Implements `ExportTraceServiceRequest` message format  **Documentation:** - Integration guide: https://langfuse.com/integrations/native/opentelemetry - Data model: https://langfuse.com/docs/observability/data-model | opentelemetry | langfuse | 85 | No |
| langfuse-organizations-get-organization-memberships | Get all memberships for the organization associated with the API key (requires organization-scoped API key) | organizations | langfuse | 85 | No |
| langfuse-organizations-update-organization-membership | Create or update a membership for the organization associated with the API key (requires organization-scoped API key) | organizations | langfuse | 85 | Yes |
| langfuse-organizations-delete-organization-membership | Delete a membership from the organization associated with the API key (requires organization-scoped API key) | organizations | langfuse | 85 | Yes |
| langfuse-organizations-get-project-memberships | Get all memberships for a specific project (requires organization-scoped API key) | organizations | langfuse | 75 | No |
| langfuse-organizations-update-project-membership | Create or update a membership for a specific project (requires organization-scoped API key). The user must already be a member of the organization. | organizations | langfuse | 85 | Yes |
| langfuse-organizations-delete-project-membership | Delete a membership from a specific project (requires organization-scoped API key). The user must be a member of the organization. | organizations | langfuse | 85 | Yes |
| langfuse-organizations-get-organization-projects | Get all projects for the organization associated with the API key (requires organization-scoped API key) | organizations | langfuse | 85 | No |
| langfuse-organizations-get-organization-api-keys | Get all API keys for the organization associated with the API key (requires organization-scoped API key) | organizations | langfuse | 85 | No |
| langfuse-projects-get | Get Project associated with API key (requires project-scoped API key). You can use GET /api/public/organizations/projects to get all projects with an organization-scoped key. | projects | langfuse | 80 | No |
| langfuse-projects-create | Create a new project (requires organization-scoped API key) | projects | langfuse | 70 | Yes |
| langfuse-projects-update | Update a project by ID (requires organization-scoped API key). | projects | langfuse | 70 | Yes |
| langfuse-projects-delete | Delete a project by ID (requires organization-scoped API key). Project deletion is processed asynchronously. | projects | langfuse | 80 | Yes |
| langfuse-projects-get-api-keys | Get all API keys for a project (requires organization-scoped API key) | projects | langfuse | 75 | No |
| langfuse-projects-create-api-key | Create a new API key for a project (requires organization-scoped API key) | projects | langfuse | 75 | Yes |
| langfuse-projects-delete-api-key | Delete an API key for a project (requires organization-scoped API key) | projects | langfuse | 75 | Yes |
| langfuse-prompt-version-prompt-version-update | Update labels for a specific prompt version | prompt_version | langfuse | 65 | Yes |
| langfuse-prompts-get | Get a prompt | prompts | langfuse | 55 | No |
| langfuse-prompts-delete | Delete prompt versions. If neither version nor label is specified, all versions of the prompt are deleted. | prompts | langfuse | 80 | Yes |
| langfuse-prompts-list | Get a list of prompt names with versions and labels | prompts | langfuse | 70 | No |
| langfuse-prompts-create | Create a new version for the prompt with the given `name` | prompts | langfuse | 70 | Yes |
| langfuse-scim-get-service-provider-config | Get SCIM Service Provider Configuration (requires organization-scoped API key) | scim | langfuse | 65 | No |
| langfuse-scim-get-resource-types | Get SCIM Resource Types (requires organization-scoped API key) | scim | langfuse | 65 | No |
| langfuse-scim-get-schemas | Get SCIM Schemas (requires organization-scoped API key) | scim | langfuse | 65 | No |
| langfuse-scim-list-users | List users in the organization (requires organization-scoped API key) | scim | langfuse | 65 | No |
| langfuse-scim-create-user | Create a new user in the organization (requires organization-scoped API key) | scim | langfuse | 65 | Yes |
| langfuse-scim-get-user | Get a specific user by ID (requires organization-scoped API key) | scim | langfuse | 65 | No |
| langfuse-scim-delete-user | Remove a user from the organization (requires organization-scoped API key). Note that this only removes the user from the organization but does not delete the user entity itself. | scim | langfuse | 75 | Yes |
| langfuse-score-configs-score-configs-create | Create a score configuration (config). Score configs are used to define the structure of scores | score_configs | langfuse | 75 | Yes |
| langfuse-score-configs-score-configs-get | Get all score configs | score_configs | langfuse | 65 | No |
| langfuse-score-configs-score-configs-get-by-id | Get a score config | score_configs | langfuse | 65 | No |
| langfuse-score-configs-score-configs-update | Update a score config | score_configs | langfuse | 65 | Yes |
| langfuse-scores-get-many | Get a list of scores (supports both trace and session scores) | scores | langfuse | 65 | No |
| langfuse-scores-get-by-id | Get a score (supports both trace and session scores) | scores | langfuse | 60 | No |
| langfuse-sessions-list | Get sessions | sessions | langfuse | 55 | No |
| langfuse-sessions-get | Get a session. Please note that `traces` on this endpoint are not paginated, if you plan to fetch large sessions, consider `GET /api/public/traces?sessionId=<sessionId>` | sessions | langfuse | 80 | No |
| langfuse-trace-get | Get a specific trace | trace | langfuse | 50 | No |
| langfuse-trace-delete | Delete a specific trace | trace | langfuse | 50 | Yes |
| langfuse-trace-list | Get list of traces | trace | langfuse | 50 | No |
| langfuse-trace-delete-multiple | Delete multiple traces | trace | langfuse | 55 | Yes |
| leanix-agent_leanix_poll_toolset | Static hint toolset for leanix_poll based on config env. | leanix_poll | leanix-agent | 50 | No |
| leanix-agent_leanix_discovery_linking_v2_toolset | Static hint toolset for leanix_discovery_linking_v2 based on config env. | leanix_discovery_linking_v2 | leanix-agent | 50 | No |
| leanix-agent_leanix_reference_data_catalog_toolset | Static hint toolset for leanix_reference_data_catalog based on config env. | leanix_reference_data_catalog | leanix-agent | 50 | No |
| leanix-agent_leanix_metrics_toolset | Static hint toolset for leanix_metrics based on config env. | leanix_metrics | leanix-agent | 50 | No |
| leanix-agent_leanix_discovery_saas_toolset | Static hint toolset for leanix_discovery_saas based on config env. | leanix_discovery_saas | leanix-agent | 50 | No |
| leanix-agent_leanix_mtm_toolset | Static hint toolset for leanix_mtm based on config env. | leanix_mtm | leanix-agent | 50 | No |
| leanix-agent_leanix_webhooks_toolset | Static hint toolset for leanix_webhooks based on config env. | leanix_webhooks | leanix-agent | 50 | No |
| leanix-agent_leanix_storage_toolset | Static hint toolset for leanix_storage based on config env. | leanix_storage | leanix-agent | 50 | No |
| leanix-agent_leanix_transformations_toolset | Static hint toolset for leanix_transformations based on config env. | leanix_transformations | leanix-agent | 50 | No |
| leanix-agent_leanix_integration_collibra_toolset | Static hint toolset for leanix_integration_collibra based on config env. | leanix_integration_collibra | leanix-agent | 50 | No |
| leanix-agent_leanix_discovery_sap_extension_toolset | Static hint toolset for leanix_discovery_sap_extension based on config env. | leanix_discovery_sap_extension | leanix-agent | 50 | No |
| leanix-agent_leanix_impacts_toolset | Static hint toolset for leanix_impacts based on config env. | leanix_impacts | leanix-agent | 50 | No |
| leanix-agent_leanix_technology_discovery_toolset | Static hint toolset for leanix_technology_discovery based on config env. | leanix_technology_discovery | leanix-agent | 50 | No |
| leanix-agent_leanix_ai_inventory_builder_toolset | Static hint toolset for leanix_ai_inventory_builder based on config env. | leanix_ai_inventory_builder | leanix-agent | 50 | No |
| leanix-agent_leanix_managed_code_execution_toolset | Static hint toolset for leanix_managed_code_execution based on config env. | leanix_managed_code_execution | leanix-agent | 50 | No |
| leanix-agent_graphql_toolset | Static hint toolset for graphql based on config env. | graphql | leanix-agent | 50 | No |
| leanix-agent_leanix_reference_data_toolset | Static hint toolset for leanix_reference_data based on config env. | leanix_reference_data | leanix-agent | 50 | No |
| leanix-agent_leanix_survey_toolset | Static hint toolset for leanix_survey based on config env. | leanix_survey | leanix-agent | 50 | No |
| leanix-agent_leanix_navigation_toolset | Static hint toolset for leanix_navigation based on config env. | leanix_navigation | leanix-agent | 50 | No |
| leanix-agent_leanix_integration_signavio_toolset | Static hint toolset for leanix_integration_signavio based on config env. | leanix_integration_signavio | leanix-agent | 50 | No |
| leanix-agent_leanix_pathfinder_toolset | Static hint toolset for leanix_pathfinder based on config env. | leanix_pathfinder | leanix-agent | 50 | No |
| leanix-agent_leanix_todo_toolset | Static hint toolset for leanix_todo based on config env. | leanix_todo | leanix-agent | 50 | No |
| leanix-agent_leanix_discovery_ai_agents_toolset | Static hint toolset for leanix_discovery_ai_agents based on config env. | leanix_discovery_ai_agents | leanix-agent | 50 | No |
| leanix-agent_leanix_integration_servicenow_toolset | Static hint toolset for leanix_integration_servicenow based on config env. | leanix_integration_servicenow | leanix-agent | 50 | No |
| leanix-agent_leanix_automations_toolset | Static hint toolset for leanix_automations based on config env. | leanix_automations | leanix-agent | 50 | No |
| leanix-agent_leanix_discovery_linking_v1_toolset | Static hint toolset for leanix_discovery_linking_v1 based on config env. | leanix_discovery_linking_v1 | leanix-agent | 50 | No |
| leanix-agent_leanix_discovery_sap_toolset | Static hint toolset for leanix_discovery_sap based on config env. | leanix_discovery_sap | leanix-agent | 50 | No |
| leanix-agent_leanix_synclog_toolset | Static hint toolset for leanix_synclog based on config env. | leanix_synclog | leanix-agent | 50 | No |
| leanix-agent_leanix_integration_api_toolset | Static hint toolset for leanix_integration_api based on config env. | leanix_integration_api | leanix-agent | 50 | No |
| leanix-agent_leanix_inventory_data_quality_toolset | Static hint toolset for leanix_inventory_data_quality based on config env. | leanix_inventory_data_quality | leanix-agent | 50 | No |
| leanix-agent_leanix_documents_toolset | Static hint toolset for leanix_documents based on config env. | leanix_documents | leanix-agent | 50 | No |
| leanix-agent_leanix_apptio_connector_toolset | Static hint toolset for leanix_apptio_connector based on config env. | leanix_apptio_connector | leanix-agent | 50 | No |
| get_startup_info | Get Startup Info | app | mealie-mcp | 50 | Yes |
| get_app_theme | Get App Theme | app | mealie-mcp | 45 | No |
| get_token | Get Token | users | mealie-mcp | 40 | No |
| oauth_login | Oauth Login | users | mealie-mcp | 45 | No |
| oauth_callback | Oauth Callback | users | mealie-mcp | 45 | No |
| refresh_token | Refresh Token | users | mealie-mcp | 45 | No |
| logout | Logout | users | mealie-mcp | 40 | No |
| register_new_user | Register New User | users | mealie-mcp | 55 | No |
| get_logged_in_user | Get Logged In User | users | mealie-mcp | 50 | No |
| get_logged_in_user_ratings | Get Logged In User Ratings | users | mealie-mcp | 55 | No |
| get_logged_in_user_rating_for_recipe | Get Logged In User Rating For Recipe | users | mealie-mcp | 55 | No |
| get_logged_in_user_favorites | Get Logged In User Favorites | users | mealie-mcp | 55 | No |
| update_password | Update Password | users | mealie-mcp | 40 | Yes |
| update_user | Update User | users | mealie-mcp | 40 | Yes |
| forgot_password | Forgot Password | users | mealie-mcp | 45 | No |
| reset_password | Reset Password | users | mealie-mcp | 45 | Yes |
| update_user_image | Update User Image | users | mealie-mcp | 50 | Yes |
| create | Create Api Token | users | mealie-mcp | 40 | Yes |
| delete | Delete Api Token | users | mealie-mcp | 40 | Yes |
| get_ratings | Get Ratings | users | mealie-mcp | 40 | No |
| get_favorites | Get Favorites | users | mealie-mcp | 40 | No |
| set_rating | Set Rating | users | mealie-mcp | 40 | Yes |
| add_favorite | Add Favorite | users | mealie-mcp | 45 | Yes |
| remove_favorite | Remove Favorite | users | mealie-mcp | 45 | Yes |
| get_households_cookbooks | Get All | households | mealie-mcp | 55 | No |
| post_households_cookbooks | Create One | households | mealie-mcp | 60 | Yes |
| put_households_cookbooks | Update Many | households | mealie-mcp | 60 | Yes |
| get_households_cookbooks_item_id | Get One | households | mealie-mcp | 60 | No |
| put_households_cookbooks_item_id | Update One | households | mealie-mcp | 60 | Yes |
| delete_households_cookbooks_item_id | Delete One | households | mealie-mcp | 60 | Yes |
| get_households_events_notifications | Get All | households | mealie-mcp | 60 | No |
| post_households_events_notifications | Create One | households | mealie-mcp | 60 | Yes |
| get_households_events_notifications_item_id | Get One | households | mealie-mcp | 60 | No |
| put_households_events_notifications_item_id | Update One | households | mealie-mcp | 60 | Yes |
| delete_households_events_notifications_item_id | Delete One | households | mealie-mcp | 60 | Yes |
| test_notification | Test Notification | households | mealie-mcp | 60 | No |
| get_households_recipe_actions | Get All | households | mealie-mcp | 60 | No |
| post_households_recipe_actions | Create One | households | mealie-mcp | 60 | Yes |
| get_households_recipe_actions_item_id | Get One | households | mealie-mcp | 60 | No |
| put_households_recipe_actions_item_id | Update One | households | mealie-mcp | 60 | Yes |
| delete_households_recipe_actions_item_id | Delete One | households | mealie-mcp | 60 | Yes |
| trigger_action | Trigger Action | households | mealie-mcp | 55 | No |
| get_logged_in_user_household | Get Logged In User Household | households | mealie-mcp | 65 | No |
| get_household_recipe | Get Household Recipe | households | mealie-mcp | 60 | No |
| get_household_members | Get Household Members | households | mealie-mcp | 60 | No |
| get_household_preferences | Get Household Preferences | households | mealie-mcp | 60 | No |
| update_household_preferences | Update Household Preferences | households | mealie-mcp | 60 | Yes |
| set_member_permissions | Set Member Permissions | households | mealie-mcp | 60 | Yes |
| get_statistics | Get Statistics | households | mealie-mcp | 50 | No |
| get_invite_tokens | Get Invite Tokens | households | mealie-mcp | 60 | No |
| create_invite_token | Create Invite Token | households | mealie-mcp | 60 | Yes |
| email_invitation | Email Invitation | households | mealie-mcp | 60 | No |
| get_households_shopping_lists | Get All | households | mealie-mcp | 60 | No |
| post_households_shopping_lists | Create One | households | mealie-mcp | 60 | Yes |
| get_households_shopping_lists_item_id | Get One | households | mealie-mcp | 60 | No |
| put_households_shopping_lists_item_id | Update One | households | mealie-mcp | 60 | Yes |
| delete_households_shopping_lists_item_id | Delete One | households | mealie-mcp | 60 | Yes |
| update_label_settings | Update Label Settings | households | mealie-mcp | 60 | Yes |
| add_recipe_ingredients_to_list | Add Recipe Ingredients To List | households | mealie-mcp | 65 | Yes |
| add_single_recipe_ingredients_to_list | Add Single Recipe Ingredients To List | households | mealie-mcp | 65 | Yes |
| remove_recipe_ingredients_from_list | Remove Recipe Ingredients From List | households | mealie-mcp | 65 | Yes |
| get_households_shopping_items | Get All | households | mealie-mcp | 60 | No |
| post_households_shopping_items | Create One | households | mealie-mcp | 60 | Yes |
| put_households_shopping_items | Update Many | households | mealie-mcp | 60 | Yes |
| delete_households_shopping_items | Delete Many | households | mealie-mcp | 60 | Yes |
| post_households_shopping_items_create_bulk | Create Many | households | mealie-mcp | 60 | Yes |
| get_households_shopping_items_item_id | Get One | households | mealie-mcp | 60 | No |
| put_households_shopping_items_item_id | Update One | households | mealie-mcp | 60 | Yes |
| delete_households_shopping_items_item_id | Delete One | households | mealie-mcp | 60 | Yes |
| get_households_webhooks | Get All | households | mealie-mcp | 55 | No |
| post_households_webhooks | Create One | households | mealie-mcp | 60 | Yes |
| rerun_webhooks | Rerun Webhooks | households | mealie-mcp | 55 | No |
| get_households_webhooks_item_id | Get One | households | mealie-mcp | 60 | No |
| put_households_webhooks_item_id | Update One | households | mealie-mcp | 60 | Yes |
| delete_households_webhooks_item_id | Delete One | households | mealie-mcp | 60 | Yes |
| test_one | Test One | households | mealie-mcp | 55 | No |
| get_households_mealplans_rules | Get All | households | mealie-mcp | 60 | No |
| post_households_mealplans_rules | Create One | households | mealie-mcp | 60 | Yes |
| get_households_mealplans_rules_item_id | Get One | households | mealie-mcp | 60 | No |
| put_households_mealplans_rules_item_id | Update One | households | mealie-mcp | 60 | Yes |
| delete_households_mealplans_rules_item_id | Delete One | households | mealie-mcp | 60 | Yes |
| get_households_mealplans | Get All | households | mealie-mcp | 55 | No |
| post_households_mealplans | Create One | households | mealie-mcp | 60 | Yes |
| get_todays_meals | Get Todays Meals | households | mealie-mcp | 60 | No |
| create_random_meal | Create Random Meal | households | mealie-mcp | 60 | Yes |
| get_households_mealplans_item_id | Get One | households | mealie-mcp | 60 | No |
| put_households_mealplans_item_id | Update One | households | mealie-mcp | 60 | Yes |
| delete_households_mealplans_item_id | Delete One | households | mealie-mcp | 60 | Yes |
| get_all_households | Get All Households | groups | mealie-mcp | 50 | No |
| get_one_household | Get One Household | groups | mealie-mcp | 50 | No |
| get_logged_in_user_group | Get Logged In User Group | groups | mealie-mcp | 55 | No |
| get_group_members | Get Group Members | groups | mealie-mcp | 50 | No |
| get_group_member | Get Group Member | groups | mealie-mcp | 50 | No |
| get_group_preferences | Get Group Preferences | groups | mealie-mcp | 50 | No |
| update_group_preferences | Update Group Preferences | groups | mealie-mcp | 50 | Yes |
| get_storage | Get Storage | groups | mealie-mcp | 40 | No |
| start_data_migration | Start Data Migration | groups | mealie-mcp | 55 | Yes |
| get_groups_reports | Get All | groups | mealie-mcp | 45 | No |
| get_groups_reports_item_id | Get One | groups | mealie-mcp | 50 | No |
| delete_groups_reports_item_id | Delete One | groups | mealie-mcp | 50 | Yes |
| get_groups_labels | Get All | groups | mealie-mcp | 45 | No |
| post_groups_labels | Create One | groups | mealie-mcp | 50 | Yes |
| get_groups_labels_item_id | Get One | groups | mealie-mcp | 50 | No |
| put_groups_labels_item_id | Update One | groups | mealie-mcp | 50 | Yes |
| delete_groups_labels_item_id | Delete One | groups | mealie-mcp | 50 | Yes |
| seed_foods | Seed Foods | groups | mealie-mcp | 45 | No |
| seed_labels | Seed Labels | groups | mealie-mcp | 45 | No |
| seed_units | Seed Units | groups | mealie-mcp | 45 | No |
| get_recipe_formats_and_templates | Get Recipe Formats And Templates | recipes | mealie-mcp | 65 | No |
| get_recipe_as_format | Get Recipe As Format | recipes | mealie-mcp | 60 | No |
| test_parse_recipe_url | Test Parse Recipe Url | recipes | mealie-mcp | 65 | No |
| create_recipe_from_html_or_json | Create Recipe From Html Or Json | recipes | mealie-mcp | 65 | Yes |
| parse_recipe_url | Parse Recipe Url | recipes | mealie-mcp | 65 | No |
| parse_recipe_url_bulk | Parse Recipe Url Bulk | recipes | mealie-mcp | 65 | No |
| create_recipe_from_zip | Create Recipe From Zip | recipes | mealie-mcp | 65 | Yes |
| create_recipe_from_image | Create Recipe From Image | recipes | mealie-mcp | 65 | Yes |
| get_recipes | Get All | recipes | mealie-mcp | 50 | No |
| post_recipes | Create One | recipes | mealie-mcp | 55 | Yes |
| put_recipes | Update Many | recipes | mealie-mcp | 55 | Yes |
| patch_many | Patch Many | recipes | mealie-mcp | 55 | Yes |
| get_recipes_suggestions | Suggest Recipes | recipes | mealie-mcp | 55 | No |
| get_recipes_slug | Get One | recipes | mealie-mcp | 55 | No |
| put_recipes_slug | Update One | recipes | mealie-mcp | 60 | Yes |
| patch_one | Patch One | recipes | mealie-mcp | 55 | Yes |
| delete_recipes_slug | Delete One | recipes | mealie-mcp | 55 | Yes |
| duplicate_one | Duplicate One | recipes | mealie-mcp | 55 | No |
| update_last_made | Update Last Made | recipes | mealie-mcp | 60 | Yes |
| scrape_image_url | Scrape Image Url | recipes | mealie-mcp | 65 | No |
| update_recipe_image | Update Recipe Image | recipes | mealie-mcp | 60 | Yes |
| delete_recipe_image | Delete Recipe Image | recipes | mealie-mcp | 60 | Yes |
| upload_recipe_asset | Upload Recipe Asset | recipes | mealie-mcp | 65 | Yes |
| get_recipe_comments | Get Recipe Comments | recipes | mealie-mcp | 60 | No |
| bulk_tag_recipes | Bulk Tag Recipes | recipes | mealie-mcp | 65 | No |
| bulk_settings_recipes | Bulk Settings Recipes | recipes | mealie-mcp | 65 | Yes |
| bulk_categorize_recipes | Bulk Categorize Recipes | recipes | mealie-mcp | 65 | No |
| bulk_delete_recipes | Bulk Delete Recipes | recipes | mealie-mcp | 60 | Yes |
| bulk_export_recipes | Bulk Export Recipes | recipes | mealie-mcp | 65 | No |
| get_exported_data | Get Exported Data | recipes | mealie-mcp | 60 | No |
| get_exported_data_token | Get Exported Data Token | recipes | mealie-mcp | 65 | No |
| purge_export_data | Purge Export Data | recipes | mealie-mcp | 65 | No |
| get_shared_recipe | Get Shared Recipe | recipes | mealie-mcp | 60 | No |
| get_shared_recipe_as_zip | Get Shared Recipe As Zip | recipes | mealie-mcp | 65 | No |
| get_recipes_timeline_events | Get All | recipes | mealie-mcp | 60 | No |
| post_recipes_timeline_events | Create One | recipes | mealie-mcp | 60 | Yes |
| get_recipes_timeline_events_item_id | Get One | recipes | mealie-mcp | 60 | No |
| put_recipes_timeline_events_item_id | Update One | recipes | mealie-mcp | 60 | Yes |
| delete_recipes_timeline_events_item_id | Delete One | recipes | mealie-mcp | 60 | Yes |
| update_event_image | Update Event Image | recipes | mealie-mcp | 60 | Yes |
| get_comments | Get All | recipes | mealie-mcp | 50 | No |
| post_comments | Create One | recipes | mealie-mcp | 55 | Yes |
| get_comments_item_id | Get One | recipes | mealie-mcp | 55 | No |
| put_comments_item_id | Update One | recipes | mealie-mcp | 60 | Yes |
| post_parser_ingredient | Delete One | recipes | mealie-mcp | 60 | Yes |
| parse_ingredient | Parse Ingredient | recipes | mealie-mcp | 60 | No |
| parse_ingredients | Parse Ingredients | recipes | mealie-mcp | 60 | No |
| get_foods | Get All | recipes | mealie-mcp | 50 | No |
| post_foods | Create One | recipes | mealie-mcp | 55 | Yes |
| put_foods_merge | Merge One | recipes | mealie-mcp | 60 | Yes |
| get_foods_item_id | Get One | recipes | mealie-mcp | 55 | No |
| put_foods_item_id | Update One | recipes | mealie-mcp | 60 | Yes |
| delete_foods_item_id | Delete One | recipes | mealie-mcp | 55 | Yes |
| get_units | Get All | recipes | mealie-mcp | 50 | No |
| post_units | Create One | recipes | mealie-mcp | 55 | Yes |
| put_units_merge | Merge One | recipes | mealie-mcp | 60 | Yes |
| get_units_item_id | Get One | recipes | mealie-mcp | 55 | No |
| put_units_item_id | Update One | recipes | mealie-mcp | 60 | Yes |
| delete_units_item_id | Delete One | recipes | mealie-mcp | 55 | Yes |
| get_recipe_img | Get Recipe Img | recipes | mealie-mcp | 55 | No |
| get_recipe_timeline_event_img | Get Recipe Timeline Event Img | recipes | mealie-mcp | 65 | No |
| get_recipe_asset | Get Recipe Asset | recipes | mealie-mcp | 60 | Yes |
| get_user_image | Get User Image | recipes | mealie-mcp | 55 | No |
| get_validation_text | Get Validation Text | recipes | mealie-mcp | 60 | No |
| get_organizers_categories | Get All | organizer | mealie-mcp | 55 | No |
| post_organizers_categories | Create One | organizer | mealie-mcp | 60 | Yes |
| get_all_empty | Get All Empty | organizer | mealie-mcp | 55 | No |
| get_organizers_categories_item_id | Get One | organizer | mealie-mcp | 60 | No |
| put_organizers_categories_item_id | Update One | organizer | mealie-mcp | 60 | Yes |
| delete_organizers_categories_item_id | Delete One | organizer | mealie-mcp | 60 | Yes |
| get_organizers_categories_slug_category_slug | Get One By Slug | organizer | mealie-mcp | 60 | No |
| get_organizers_tags | Get All | organizer | mealie-mcp | 55 | No |
| post_organizers_tags | Create One | organizer | mealie-mcp | 60 | Yes |
| get_empty_tags | Get Empty Tags | organizer | mealie-mcp | 55 | No |
| get_organizers_tags_item_id | Get One | organizer | mealie-mcp | 60 | No |
| put_organizers_tags_item_id | Update One | organizer | mealie-mcp | 60 | Yes |
| delete_recipe_tag | Delete Recipe Tag | organizer | mealie-mcp | 60 | Yes |
| get_organizers_tags_slug_tag_slug | Get One By Slug | organizer | mealie-mcp | 60 | No |
| get_organizers_tools | Get All | organizer | mealie-mcp | 55 | No |
| post_organizers_tools | Create One | organizer | mealie-mcp | 60 | Yes |
| get_organizers_tools_item_id | Get One | organizer | mealie-mcp | 60 | No |
| put_organizers_tools_item_id | Update One | organizer | mealie-mcp | 60 | Yes |
| delete_organizers_tools_item_id | Delete One | organizer | mealie-mcp | 60 | Yes |
| get_organizers_tools_slug_tool_slug | Get One By Slug | organizer | mealie-mcp | 60 | No |
| get_shared_recipes | Get All | shared | mealie-mcp | 45 | No |
| post_shared_recipes | Create One | shared | mealie-mcp | 50 | Yes |
| get_shared_recipes_item_id | Get One | shared | mealie-mcp | 50 | No |
| delete_shared_recipes_item_id | Delete One | shared | mealie-mcp | 50 | Yes |
| get_app_info | Get App Info | admin | mealie-mcp | 45 | No |
| get_app_statistics | Get App Statistics | admin | mealie-mcp | 50 | No |
| check_app_config | Check App Config | admin | mealie-mcp | 55 | No |
| get_admin_users | Get All | admin | mealie-mcp | 45 | No |
| post_admin_users | Create One | admin | mealie-mcp | 50 | Yes |
| unlock_users | Unlock Users | admin | mealie-mcp | 45 | No |
| get_admin_users_item_id | Get One | admin | mealie-mcp | 50 | No |
| put_admin_users_item_id | Update One | admin | mealie-mcp | 50 | Yes |
| delete_admin_users_item_id | Delete One | admin | mealie-mcp | 50 | Yes |
| generate_token | Generate Token | admin | mealie-mcp | 45 | No |
| get_admin_households | Get All | admin | mealie-mcp | 45 | No |
| post_admin_households | Create One | admin | mealie-mcp | 50 | Yes |
| get_admin_households_item_id | Get One | admin | mealie-mcp | 50 | No |
| put_admin_households_item_id | Update One | admin | mealie-mcp | 50 | Yes |
| delete_admin_households_item_id | Delete One | admin | mealie-mcp | 50 | Yes |
| get_admin_groups | Get All | admin | mealie-mcp | 45 | No |
| post_admin_groups | Create One | admin | mealie-mcp | 50 | Yes |
| get_admin_groups_item_id | Get One | admin | mealie-mcp | 50 | No |
| put_admin_groups_item_id | Update One | admin | mealie-mcp | 50 | Yes |
| delete_admin_groups_item_id | Delete One | admin | mealie-mcp | 50 | Yes |
| check_email_config | Check Email Config | admin | mealie-mcp | 55 | No |
| send_test_email | Send Test Email | admin | mealie-mcp | 50 | No |
| get_admin_backups | Get All | admin | mealie-mcp | 45 | No |
| post_admin_backups | Create One | admin | mealie-mcp | 50 | Yes |
| get_admin_backups_file_name | Get One | admin | mealie-mcp | 50 | No |
| delete_admin_backups_file_name | Delete One | admin | mealie-mcp | 50 | Yes |
| upload_one | Upload One | admin | mealie-mcp | 45 | Yes |
| import_one | Import One | admin | mealie-mcp | 45 | No |
| get_maintenance_summary | Get Maintenance Summary | admin | mealie-mcp | 50 | No |
| get_storage_details | Get Storage Details | admin | mealie-mcp | 50 | No |
| clean_images | Clean Images | admin | mealie-mcp | 45 | No |
| clean_temp | Clean Temp | admin | mealie-mcp | 45 | No |
| clean_recipe_folders | Clean Recipe Folders | admin | mealie-mcp | 55 | No |
| debug_openai | Debug Openai | admin | mealie-mcp | 45 | No |
| get_explore_groups_group_slug_foods | Get All | explore | mealie-mcp | 60 | No |
| get_explore_groups_group_slug_foods_item_id | Get One | explore | mealie-mcp | 60 | No |
| get_explore_groups_group_slug_households | Get All | explore | mealie-mcp | 60 | No |
| get_household | Get Household | explore | mealie-mcp | 50 | No |
| get_explore_groups_group_slug_organizers_categories | Get All | explore | mealie-mcp | 60 | No |
| get_explore_groups_group_slug_organizers_categories_item_id | Get One | explore | mealie-mcp | 60 | No |
| get_explore_groups_group_slug_organizers_tags | Get All | explore | mealie-mcp | 60 | No |
| get_explore_groups_group_slug_organizers_tags_item_id | Get One | explore | mealie-mcp | 60 | No |
| get_explore_groups_group_slug_organizers_tools | Get All | explore | mealie-mcp | 60 | No |
| get_explore_groups_group_slug_organizers_tools_item_id | Get One | explore | mealie-mcp | 60 | No |
| get_explore_groups_group_slug_cookbooks | Get All | explore | mealie-mcp | 60 | No |
| get_explore_groups_group_slug_cookbooks_item_id | Get One | explore | mealie-mcp | 60 | No |
| get_explore_groups_group_slug_recipes | Get All | explore | mealie-mcp | 60 | No |
| get_explore_groups_group_slug_recipes_suggestions | Suggest Recipes | explore | mealie-mcp | 60 | No |
| get_recipe | Get Recipe | explore | mealie-mcp | 50 | No |
| download_file | Download File | utils | mealie-mcp | 45 | No |
| run_command | Run a bash command on the local system. | collection_management | media-downloader-mcp | 55 | Yes |
| download_media | Downloads media from a given URL to the specified directory. Download as a video or audio file.<br/>Returns a Dictionary response with status, download directory, audio only, and other details. | collection_management | media-downloader-mcp | 80 | No |
| text_editor | View and edit files on the local filesystem. | files, text_editor | media-downloader-mcp | 70 | No |
| login | Authenticate with Microsoft using device code flow | auth | microsoft-agent | 45 | No |
| logout | Log out from Microsoft account | auth | microsoft-agent | 45 | No |
| verify_login | Check current Microsoft authentication status | auth | microsoft-agent | 50 | No |
| list_accounts | List all available Microsoft accounts | auth | microsoft-agent | 45 | No |
| search_tools | Search available Microsoft Graph API tools | meta | microsoft-agent | 50 | No |
| list_mail_messages | list_mail_messages: GET /me/messages<br/><br/>TIP: CRITICAL: When searching emails, the $search parameter value MUST be wrapped in double quotes. Format: $search='your search query here'. Use KQL (Keyword Query Language) syntax to search specific properties: 'from:', 'subject:', 'body:', 'to:', 'cc:', 'bcc:', 'attachment:', 'hasAttachments:', 'importance:', 'received:', 'sent:'. Examples: $search='from:john@example.com' \| $search='subject:meeting AND hasAttachments:true' \| $search='body:urgent AND received>=2024-01-01' \| $search='from:john AND importance:high'. Remember: ALWAYS wrap the entire search expression in double quotes! Reference: https://learn.microsoft.com/en-us/graph/search-query-parameter | files, mail, user | microsoft-agent | 95 | No |
| list_mail_folders | list_mail_folders: GET /me/mailFolders | files, mail | microsoft-agent | 70 | No |
| list_mail_folder_messages | list_mail_folder_messages: GET /me/mailFolders/{mailFolder-id}/messages<br/><br/><br/><br/>TIP: CRITICAL: When searching emails, the $search parameter value MUST be wrapped in double quotes. Format: $search='your search query here'. Use KQL (Keyword Query Language) syntax to search specific properties: 'from:', 'subject:', 'body:', 'to:', 'cc:', 'bcc:', 'attachment:', 'hasAttachments:', 'importance:', 'received:', 'sent:'. Examples: $search='from:john@example.com' \| $search='subject:meeting AND hasAttachments:true' \| $search='body:urgent AND received>=2024-01-01' \| $search='from:alice AND importance:high'. Remember: ALWAYS wrap the entire search expression in double quotes! Reference: https://learn.microsoft.com/en-us/graph/search-query-parameter | files, mail, user | microsoft-agent | 100 | No |
| get_mail_message | get_mail_message: GET /me/messages/{message-id} | mail, user | microsoft-agent | 70 | No |
| send_mail | send_mail: POST /me/sendMail<br/><br/><br/><br/>TIP: CRITICAL: Do not try to guess the email address of the recipients. Use the list-users tool to find the email address of the recipients. | mail | microsoft-agent | 70 | No |
| list_shared_mailbox_messages | list_shared_mailbox_messages: GET /users/{user-id}/messages<br/><br/><br/><br/>TIP: CRITICAL: When searching emails, the $search parameter value MUST be wrapped in double quotes. Format: $search='your search query here'. Use KQL (Keyword Query Language) syntax to search specific properties: 'from:', 'subject:', 'body:', 'to:', 'cc:', 'bcc:', 'attachment:', 'hasAttachments:', 'importance:', 'received:', 'sent:'. Examples: $search='from:john@example.com' \| $search='subject:meeting AND hasAttachments:true' \| $search='body:urgent AND received>=2024-01-01' \| $search='from:alice AND importance:high'. Remember: ALWAYS wrap the entire search expression in double quotes! Reference: https://learn.microsoft.com/en-us/graph/search-query-parameter | files, mail, user | microsoft-agent | 100 | No |
| list_shared_mailbox_folder_messages | list_shared_mailbox_folder_messages: GET /users/{user-id}/mailFolders/{mailFolder-id}/messages<br/><br/><br/><br/>TIP: CRITICAL: When searching emails, the $search parameter value MUST be wrapped in double quotes. Format: $search='your search query here'. Use KQL (Keyword Query Language) syntax to search specific properties: 'from:', 'subject:', 'body:', 'to:', 'cc:', 'bcc:', 'attachment:', 'hasAttachments:', 'importance:', 'received:', 'sent:'. Examples: $search='from:john@example.com' \| $search='subject:meeting AND hasAttachments:true' \| $search='body:urgent AND received>=2024-01-01' \| $search='from:alice AND importance:high'. Remember: ALWAYS wrap the entire search expression in double quotes! Reference: https://learn.microsoft.com/en-us/graph/search-query-parameter | files, mail, user | microsoft-agent | 100 | No |
| get_shared_mailbox_message | get_shared_mailbox_message: GET /users/{user-id}/messages/{message-id} | mail, user | microsoft-agent | 85 | No |
| send_shared_mailbox_mail | send_shared_mailbox_mail: POST /users/{user-id}/sendMail<br/><br/><br/><br/>TIP: CRITICAL: Do not try to guess the email address of the recipients. Use the list-users tool to find the email address of the recipients. | mail | microsoft-agent | 75 | No |
| create_draft_email | create_draft_email: POST /me/messages | mail | microsoft-agent | 50 | Yes |
| delete_mail_message | delete_mail_message: DELETE /me/messages/{message-id} | mail, user | microsoft-agent | 80 | Yes |
| move_mail_message | move_mail_message: POST /me/messages/{message-id}/move | mail, user | microsoft-agent | 85 | Yes |
| update_mail_message | update_mail_message: PATCH /me/messages/{message-id} | mail, user | microsoft-agent | 80 | Yes |
| add_mail_attachment | add_mail_attachment: POST /me/messages/{message-id}/attachments | mail, user | microsoft-agent | 85 | Yes |
| list_mail_attachments | list_mail_attachments: GET /me/messages/{message-id}/attachments | files, mail, user | microsoft-agent | 85 | No |
| get_mail_attachment | get_mail_attachment: GET /me/messages/{message-id}/attachments/{attachment-id} | mail, user | microsoft-agent | 80 | No |
| delete_mail_attachment | delete_mail_attachment: DELETE /me/messages/{message-id}/attachments/{attachment-id} | mail, user | microsoft-agent | 80 | Yes |
| get_root_folder | get_root_folder: GET /drives/{drive-id}/root | files, mail | microsoft-agent | 70 | No |
| list_folder_files | list_folder_files: GET /drives/{drive-id}/items/{driveItem-id}/children | files, mail | microsoft-agent | 80 | No |
| list_chat_messages | list_chat_messages: GET /chats/{chat-id}/messages | chat, files, mail, user | microsoft-agent | 75 | No |
| get_chat_message | get_chat_message: GET /chats/{chat-id}/messages/{chatMessage-id} | chat, mail, user | microsoft-agent | 85 | No |
| send_chat_message | send_chat_message: POST /chats/{chat-id}/messages | chat, mail, user | microsoft-agent | 80 | No |
| list_channel_messages | list_channel_messages: GET /teams/{team-id}/channels/{channel-id}/messages | files, mail, teams, user | microsoft-agent | 85 | No |
| get_channel_message | get_channel_message: GET /teams/{team-id}/channels/{channel-id}/messages/{chatMessage-id} | mail, teams, user | microsoft-agent | 85 | No |
| send_channel_message | send_channel_message: POST /teams/{team-id}/channels/{channel-id}/messages | mail, teams, user | microsoft-agent | 90 | No |
| list_chat_message_replies | list_chat_message_replies: GET /chats/{chat-id}/messages/{chatMessage-id}/replies | chat, files, mail, user | microsoft-agent | 90 | No |
| reply_to_chat_message | reply_to_chat_message: POST /chats/{chat-id}/messages/{chatMessage-id}/replies | chat, mail, user | microsoft-agent | 90 | No |
| list_users | list_users: GET /users<br/><br/><br/><br/>TIP: CRITICAL: This request requires the ConsistencyLevel header set to eventual. When searching users, the $search parameter value MUST be wrapped in double quotes. Format: $search='your search query here'. Use KQL (Keyword Query Language) syntax to search specific properties: 'displayName:'. Examples: $search='displayName:john' \| $search='displayName:john' OR 'displayName:jane'. Remember: ALWAYS wrap the entire search expression in double quotes and set the ConsistencyLevel header to eventual! Reference: https://learn.microsoft.com/en-us/graph/search-query-parameter | files, user | microsoft-agent | 85 | No |
| list_drives | list_drives: GET /me/drives | files | microsoft-agent | 45 | No |
| get_drive_root_item | get_drive_root_item: GET /drives/{drive-id}/root | files | microsoft-agent | 55 | No |
| download_onedrive_file_content | download_onedrive_file_content: GET /drives/{drive-id}/items/{driveItem-id}/content | files | microsoft-agent | 65 | No |
| delete_onedrive_file | delete_onedrive_file: DELETE /drives/{drive-id}/items/{driveItem-id} | files | microsoft-agent | 60 | Yes |
| upload_file_content | upload_file_content: PUT /drives/{drive-id}/items/{driveItem-id}/content | files | microsoft-agent | 65 | Yes |
| create_excel_chart | create_excel_chart: POST /drives/{drive-id}/items/{driveItem-id}/workbook/worksheets/{workbookWorksheet-id}/charts/add | files | microsoft-agent | 70 | Yes |
| format_excel_range | format_excel_range: PATCH /drives/{drive-id}/items/{driveItem-id}/workbook/worksheets/{workbookWorksheet-id}/range()/format | files | microsoft-agent | 75 | No |
| sort_excel_range | sort_excel_range: PATCH /drives/{drive-id}/items/{driveItem-id}/workbook/worksheets/{workbookWorksheet-id}/range()/sort | files | microsoft-agent | 75 | No |
| get_excel_range | get_excel_range: GET /drives/{drive-id}/items/{driveItem-id}/workbook/worksheets/{workbookWorksheet-id}/range(address='{address}') | files | microsoft-agent | 70 | No |
| list_excel_worksheets | list_excel_worksheets: GET /drives/{drive-id}/items/{driveItem-id}/workbook/worksheets | files | microsoft-agent | 60 | No |
| list_excel_tables | list_excel_tables: GET /drives/{drive-id}/items/{driveItem-id}/workbook/tables<br/><br/><br/><br/>List Excel tables in a workbook. | files | microsoft-agent | 70 | No |
| get_excel_workbook | get_excel_workbook: GET /drives/{drive-id}/items/{item-id}/workbook | files | microsoft-agent | 60 | No |
| list_onenote_notebooks | list_onenote_notebooks: GET /me/onenote/notebooks | files, notes | microsoft-agent | 70 | No |
| list_onenote_notebook_sections | list_onenote_notebook_sections: GET /me/onenote/notebooks/{notebook-id}/sections | files, notes | microsoft-agent | 85 | No |
| list_onenote_section_pages | list_onenote_section_pages: GET /me/onenote/sections/{onenoteSection-id}/pages | files, notes | microsoft-agent | 85 | No |
| list_todo_task_lists | list_todo_task_lists: GET /me/todo/lists | files, tasks | microsoft-agent | 75 | No |
| list_todo_tasks | list_todo_tasks: GET /me/todo/lists/{todoTaskList-id}/tasks | files, tasks | microsoft-agent | 80 | No |
| list_planner_tasks | list_planner_tasks: GET /me/planner/tasks | files, tasks | microsoft-agent | 70 | No |
| list_plan_tasks | list_plan_tasks: GET /planner/plans/{plannerPlan-id}/tasks | files, tasks | microsoft-agent | 80 | No |
| list_outlook_contacts | list_outlook_contacts: GET /me/contacts | contacts, files | microsoft-agent | 70 | No |
| list_chats | list_chats: GET /me/chats | chat, files | microsoft-agent | 65 | No |
| get_excel_worksheet | get_excel_worksheet: GET /drives/{drive-id}/items/{item-id}/workbook/worksheets/{worksheet-id} | files | microsoft-agent | 60 | No |
| list_joined_teams | list_joined_teams: GET /me/joinedTeams | files, teams | microsoft-agent | 70 | No |
| list_team_channels | list_team_channels: GET /teams/{team-id}/channels | files, teams | microsoft-agent | 70 | No |
| list_team_members | list_team_members: GET /teams/{team-id}/members | files, teams, user | microsoft-agent | 75 | No |
| list_site_drives | list_site_drives: GET /sites/{site-id}/drives | files, sites | microsoft-agent | 70 | No |
| get_site_drive_by_id | get_site_drive_by_id: GET /sites/{site-id}/drives/{drive-id} | files, sites | microsoft-agent | 80 | No |
| list_site_items | list_site_items: GET /sites/{site-id}/items | files, sites | microsoft-agent | 70 | No |
| get_site_item | get_site_item: GET /sites/{site-id}/items/{baseItem-id} | files, sites | microsoft-agent | 80 | No |
| list_site_lists | list_site_lists: GET /sites/{site-id}/lists<br/><br/><br/><br/>List lists for a SharePoint site. | files, sites | microsoft-agent | 80 | No |
| get_site_list | get_site_list: GET /sites/{site-id}/lists/{list-id}<br/><br/><br/><br/>Get a specific SharePoint site list. | files, sites | microsoft-agent | 75 | No |
| list_sharepoint_site_list_items | list_sharepoint_site_list_items: GET /sites/{site-id}/lists/{list-id}/items<br/><br/><br/><br/>List items in a SharePoint site list. | files, sites | microsoft-agent | 95 | No |
| get_sharepoint_site_list_item | get_sharepoint_site_list_item: GET /sites/{site-id}/lists/{list-id}/items/{listItem-id} | files, sites | microsoft-agent | 85 | No |
| get_excel_table | get_excel_table: GET /drives/{drive-id}/items/{item-id}/workbook/tables/{table-id} | files | microsoft-agent | 60 | No |
| list_calendar_events | list_calendar_events: GET /me/events | calendar, files | microsoft-agent | 70 | No |
| get_calendar_event | get_calendar_event: GET /me/events/{event-id} | calendar | microsoft-agent | 60 | No |
| create_calendar_event | create_calendar_event: POST /me/events<br/><br/><br/><br/>TIP: CRITICAL: Do not try to guess the email address of the recipients. Use the list-users tool to find the email address of the recipients. | calendar | microsoft-agent | 80 | Yes |
| update_calendar_event | update_calendar_event: PATCH /me/events/{event-id}<br/><br/><br/><br/>TIP: CRITICAL: Do not try to guess the email address of the recipients. Use the list-users tool to find the email address of the recipients. | calendar | microsoft-agent | 80 | Yes |
| delete_calendar_event | delete_calendar_event: DELETE /me/events/{event-id} | calendar | microsoft-agent | 70 | Yes |
| list_specific_calendar_events | list_specific_calendar_events: GET /me/calendars/{calendar-id}/events | calendar, files | microsoft-agent | 85 | No |
| get_specific_calendar_event | get_specific_calendar_event: GET /me/calendars/{calendar-id}/events/{event-id} | calendar | microsoft-agent | 75 | No |
| create_specific_calendar_event | create_specific_calendar_event: POST /me/calendars/{calendar-id}/events<br/><br/><br/><br/>TIP: CRITICAL: Do not try to guess the email address of the recipients. Use the list-users tool to find the email address of the recipients. | calendar | microsoft-agent | 85 | Yes |
| update_specific_calendar_event | update_specific_calendar_event: PATCH /me/calendars/{calendar-id}/events/{event-id}<br/><br/><br/><br/>TIP: CRITICAL: Do not try to guess the email address of the recipients. Use the list-users tool to find the email address of the recipients. | calendar | microsoft-agent | 85 | Yes |
| delete_specific_calendar_event | delete_specific_calendar_event: DELETE /me/calendars/{calendar-id}/events/{event-id} | calendar | microsoft-agent | 75 | Yes |
| get_calendar_view | get_calendar_view: GET /me/calendarView | calendar | microsoft-agent | 60 | No |
| list_calendars | list_calendars: GET /me/calendars | calendar, files | microsoft-agent | 65 | No |
| find_meeting_times | find_meeting_times: POST /me/findMeetingTimes | calendar, user | microsoft-agent | 75 | No |
| get_onenote_page_content | get_onenote_page_content: GET /me/onenote/pages/{onenotePage-id}/content | notes | microsoft-agent | 65 | No |
| create_onenote_page | create_onenote_page: POST /me/onenote/pages | notes | microsoft-agent | 50 | Yes |
| get_todo_task | get_todo_task: GET /me/todo/lists/{todoTaskList-id}/tasks/{todoTask-id} | tasks | microsoft-agent | 60 | No |
| create_todo_task | create_todo_task: POST /me/todo/lists/{todoTaskList-id}/tasks | tasks | microsoft-agent | 60 | Yes |
| update_todo_task | update_todo_task: PATCH /me/todo/lists/{todoTaskList-id}/tasks/{todoTask-id} | tasks | microsoft-agent | 60 | Yes |
| delete_todo_task | delete_todo_task: DELETE /me/todo/lists/{todoTaskList-id}/tasks/{todoTask-id} | tasks | microsoft-agent | 60 | Yes |
| get_planner_plan | get_planner_plan: GET /planner/plans/{plannerPlan-id} | tasks | microsoft-agent | 60 | No |
| get_planner_task | get_planner_task: GET /planner/tasks/{plannerTask-id} | tasks | microsoft-agent | 60 | No |
| create_planner_task | create_planner_task: POST /planner/tasks | tasks | microsoft-agent | 50 | Yes |
| update_planner_task | update_planner_task: PATCH /planner/tasks/{plannerTask-id} | tasks | microsoft-agent | 60 | Yes |
| update_planner_task_details | update_planner_task_details: PATCH /planner/tasks/{plannerTask-id}/details | tasks | microsoft-agent | 65 | Yes |
| get_outlook_contact | get_outlook_contact: GET /me/contacts/{contact-id} | contacts | microsoft-agent | 60 | No |
| create_outlook_contact | create_outlook_contact: POST /me/contacts | contacts | microsoft-agent | 60 | Yes |
| update_outlook_contact | update_outlook_contact: PATCH /me/contacts/{contact-id} | contacts | microsoft-agent | 70 | Yes |
| delete_outlook_contact | delete_outlook_contact: DELETE /me/contacts/{contact-id} | contacts | microsoft-agent | 70 | Yes |
| get_current_user | get_current_user: GET /me | user | microsoft-agent | 50 | No |
| get_me | get_me: GET /me | user | microsoft-agent | 35 | No |
| get_chat | get_chat: GET /chats/{chat-id} | chat | microsoft-agent | 45 | No |
| get_team | get_team: GET /teams/{team-id} | teams | microsoft-agent | 45 | No |
| get_team_channel | get_team_channel: GET /teams/{team-id}/channels/{channel-id} | teams | microsoft-agent | 60 | No |
| list_sites | list_sites: GET /sites | sites | microsoft-agent | 45 | No |
| get_site | get_site: GET /sites/{site-id} | sites | microsoft-agent | 45 | No |
| get_sharepoint_site_by_path | get_sharepoint_site_by_path: GET /sites/{site-id}/getByPath(path='{path}') | sites | microsoft-agent | 65 | No |
| get_sharepoint_sites_delta | get_sharepoint_sites_delta: GET /sites/delta() | sites | microsoft-agent | 55 | No |
| search_query | search_query: POST /search/query | search | microsoft-agent | 50 | No |
| list_groups | list_groups: GET /groups<br/><br/><br/><br/>List all Microsoft 365 groups and security groups in the organization. Supports $filter, $search, $select, $top, $orderby, $count query parameters. Requires ConsistencyLevel: eventual header for advanced queries. | groups | microsoft-agent | 65 | No |
| get_group | get_group: GET /groups/{group-id}<br/><br/><br/><br/>Get properties and relationships of a group object. | groups | microsoft-agent | 55 | No |
| create_group | create_group: POST /groups<br/><br/><br/><br/>Create a new Microsoft 365 group or security group. Required fields: displayName, mailNickname, mailEnabled, securityEnabled. For M365 groups, set groupTypes=['Unified']. | groups | microsoft-agent | 65 | Yes |
| update_group | update_group: PATCH /groups/{group-id}<br/><br/><br/><br/>Update properties of a group object. | groups | microsoft-agent | 55 | Yes |
| delete_group | delete_group: DELETE /groups/{group-id}<br/><br/><br/><br/>Delete a group. This permanently removes the group and its associated content. | groups | microsoft-agent | 65 | Yes |
| list_group_members | list_group_members: GET /groups/{group-id}/members<br/><br/><br/><br/>Get a list of the group's direct members. | groups, user | microsoft-agent | 80 | No |
| add_group_member | add_group_member: POST /groups/{group-id}/members/$ref<br/><br/><br/><br/>Add a member to a group. Provide memberId or directoryObjectId in the request body. | groups, user | microsoft-agent | 95 | Yes |
| remove_group_member | remove_group_member: DELETE /groups/{group-id}/members/{member-id}/$ref<br/><br/><br/><br/>Remove a member from a group. | groups, user | microsoft-agent | 95 | Yes |
| list_group_owners | list_group_owners: GET /groups/{group-id}/owners<br/><br/><br/><br/>Get owners of a group. | groups, user | microsoft-agent | 80 | No |
| list_group_conversations | list_group_conversations: GET /groups/{group-id}/conversations<br/><br/><br/><br/>List conversations in a Microsoft 365 group. | chat, groups | microsoft-agent | 90 | No |
| list_group_drives | list_group_drives: GET /groups/{group-id}/drives<br/><br/><br/><br/>List drives (document libraries) of a group. | files, groups | microsoft-agent | 80 | No |
| list_service_health | list_service_health: GET /admin/serviceAnnouncement/healthOverviews<br/><br/><br/><br/>Get the service health status for all services in the tenant. | admin | microsoft-agent | 70 | No |
| get_service_health | get_service_health: GET /admin/serviceAnnouncement/healthOverviews/{service-name}<br/><br/><br/><br/>Get the health status for a specific service. | admin | microsoft-agent | 70 | No |
| list_service_health_issues | list_service_health_issues: GET /admin/serviceAnnouncement/issues<br/><br/><br/><br/>List all service health issues for the tenant. | admin | microsoft-agent | 75 | No |
| get_service_health_issue | get_service_health_issue: GET /admin/serviceAnnouncement/issues/{issue-id}<br/><br/><br/><br/>Get a specific service health issue. | admin | microsoft-agent | 75 | No |
| list_service_update_messages | list_service_update_messages: GET /admin/serviceAnnouncement/messages<br/><br/><br/><br/>List service update messages (message center posts) for the tenant. | admin | microsoft-agent | 70 | Yes |
| get_service_update_message | get_service_update_message: GET /admin/serviceAnnouncement/messages/{message-id}<br/><br/><br/><br/>Get a specific service update message. | admin | microsoft-agent | 70 | Yes |
| get_admin_sharepoint | get_admin_sharepoint: GET /admin/sharepoint<br/><br/><br/><br/>Get SharePoint admin settings for the tenant. | admin, sites | microsoft-agent | 80 | No |
| update_admin_sharepoint | update_admin_sharepoint: PATCH /admin/sharepoint<br/><br/><br/><br/>Update SharePoint admin settings for the tenant. | admin, sites | microsoft-agent | 80 | Yes |
| list_delegated_admin_relationships | list_delegated_admin_relationships: GET /tenantRelationships/delegatedAdminRelationships<br/><br/><br/><br/>List delegated admin relationships. | admin | microsoft-agent | 75 | No |
| get_delegated_admin_relationship | get_delegated_admin_relationship: GET /tenantRelationships/delegatedAdminRelationships/{id}<br/><br/><br/><br/>Get a specific delegated admin relationship. | admin | microsoft-agent | 75 | No |
| list_organization | list_organization: GET /organization<br/><br/><br/><br/>Get the properties and relationships of the currently authenticated organization. | organization | microsoft-agent | 75 | No |
| get_organization | get_organization: GET /organization/{org-id}<br/><br/><br/><br/>Get a specific organization by ID. | organization | microsoft-agent | 65 | No |
| update_organization | update_organization: PATCH /organization/{org-id}<br/><br/><br/><br/>Update organization properties. | organization | microsoft-agent | 65 | Yes |
| get_org_branding | get_org_branding: GET /organization/{org-id}/branding<br/><br/><br/><br/>Get organization branding properties (sign-in page customization). | organization | microsoft-agent | 80 | No |
| update_org_branding | update_org_branding: PATCH /organization/{org-id}/branding<br/><br/><br/><br/>Update organization branding properties. | organization | microsoft-agent | 80 | Yes |
| list_domains | list_domains: GET /domains<br/><br/><br/><br/>List domains associated with the tenant. | domains | microsoft-agent | 65 | No |
| get_domain | get_domain: GET /domains/{domain-id}<br/><br/><br/><br/>Get properties of a specific domain. | domains | microsoft-agent | 65 | No |
| create_domain | create_domain: POST /domains<br/><br/><br/><br/>Add a domain to the tenant. Provide the domain name as 'id' in the request body. | domains | microsoft-agent | 75 | Yes |
| delete_domain | delete_domain: DELETE /domains/{domain-id}<br/><br/><br/><br/>Delete a domain from the tenant. | domains | microsoft-agent | 65 | Yes |
| verify_domain | verify_domain: POST /domains/{domain-id}/verify<br/><br/><br/><br/>Verify ownership of a domain. | domains | microsoft-agent | 70 | No |
| list_domain_service_configuration_records | list_domain_service_configuration_records: GET /domains/{domain-id}/serviceConfigurationRecords<br/><br/><br/><br/>List DNS records required by the domain for Microsoft services. | domains | microsoft-agent | 85 | No |
| list_subscriptions | list_subscriptions: GET /subscriptions<br/><br/><br/><br/>List active webhook subscriptions for change notifications. | subscriptions | microsoft-agent | 75 | No |
| get_subscription | get_subscription: GET /subscriptions/{subscription-id}<br/><br/><br/><br/>Get a specific subscription. | subscriptions | microsoft-agent | 65 | No |
| create_subscription | create_subscription: POST /subscriptions<br/><br/><br/><br/>Create a webhook subscription for change notifications. Required fields: changeType, notificationUrl, resource, expirationDateTime. | subscriptions | microsoft-agent | 75 | Yes |
| update_subscription | update_subscription: PATCH /subscriptions/{subscription-id}<br/><br/><br/><br/>Renew a subscription by extending its expiration time. | subscriptions | microsoft-agent | 75 | Yes |
| delete_subscription | delete_subscription: DELETE /subscriptions/{subscription-id}<br/><br/><br/><br/>Delete a webhook subscription. | subscriptions | microsoft-agent | 65 | Yes |
| list_online_meetings | list_online_meetings: GET /me/onlineMeetings<br/><br/><br/><br/>List online meetings for the current user. Returns meeting details including subject, join URL, start/end time, and participants. | communications | microsoft-agent | 80 | No |
| get_online_meeting | get_online_meeting: GET /me/onlineMeetings/{onlineMeeting-id}<br/><br/><br/><br/>Get a specific online meeting by ID. Returns full meeting details including join information, audio conferencing, and lobby settings. | communications | microsoft-agent | 80 | No |
| create_online_meeting | create_online_meeting: POST /me/onlineMeetings<br/><br/><br/><br/>Create a new online meeting. Provide subject, startDateTime, endDateTime, and optional lobby bypass settings. | communications | microsoft-agent | 80 | Yes |
| update_online_meeting | update_online_meeting: PATCH /me/onlineMeetings/{onlineMeeting-id}<br/><br/><br/><br/>Update an existing online meeting. Modify subject, times, or lobby settings. | communications | microsoft-agent | 80 | Yes |
| delete_online_meeting | delete_online_meeting: DELETE /me/onlineMeetings/{onlineMeeting-id}<br/><br/><br/><br/>Delete an online meeting. | communications | microsoft-agent | 70 | Yes |
| list_call_records | list_call_records: GET /communications/callRecords<br/><br/><br/><br/>List call records. Returns information about calls and meetings including participants, modalities, and duration. | communications | microsoft-agent | 80 | No |
| get_call_record | get_call_record: GET /communications/callRecords/{callRecord-id}<br/><br/><br/><br/>Get a specific call record by ID. Returns detailed call information including sessions and segments. | communications | microsoft-agent | 80 | No |
| list_presences | list_presences: GET /communications/presences<br/><br/><br/><br/>List presence information for multiple users. Returns availability and activity status. | communications | microsoft-agent | 75 | No |
| get_presence | get_presence: GET /communications/presences/{presence-id}<br/><br/><br/><br/>Get presence for a specific user by user ID. Returns availability (Available, Busy, Away, etc.) and activity. | communications | microsoft-agent | 75 | No |
| get_my_presence | get_my_presence: GET /me/presence<br/><br/><br/><br/>Get current user's presence status including availability and activity. | communications | microsoft-agent | 75 | No |
| create_invitation | create_invitation: POST /invitations<br/><br/><br/><br/>Create an invitation for an external / guest user. Provide invitedUserEmailAddress and inviteRedirectUrl. Optionally set invitedUserDisplayName and sendInvitationMessage. | identity | microsoft-agent | 75 | Yes |
| list_conditional_access_policies | list_conditional_access_policies: GET /identity/conditionalAccess/policies<br/><br/><br/><br/>List conditional access policies. | identity | microsoft-agent | 85 | No |
| get_conditional_access_policy | get_conditional_access_policy: GET /identity/conditionalAccess/policies/{id}<br/><br/><br/><br/>Get a specific conditional access policy. | identity | microsoft-agent | 85 | No |
| create_conditional_access_policy | create_conditional_access_policy: POST /identity/conditionalAccess/policies<br/><br/><br/><br/>Create a conditional access policy. | identity | microsoft-agent | 85 | Yes |
| update_conditional_access_policy | update_conditional_access_policy: PATCH /identity/conditionalAccess/policies/{id}<br/><br/><br/><br/>Update a conditional access policy. | identity | microsoft-agent | 85 | Yes |
| delete_conditional_access_policy | delete_conditional_access_policy: DELETE /identity/conditionalAccess/policies/{id}<br/><br/><br/><br/>Delete a conditional access policy. | identity | microsoft-agent | 85 | Yes |
| list_access_reviews | list_access_reviews: GET /identityGovernance/accessReviewDefinitions<br/><br/><br/><br/>List access review schedule definitions. | identity | microsoft-agent | 80 | No |
| get_access_review | get_access_review: GET /identityGovernance/accessReviewDefinitions/{id}<br/><br/><br/><br/>Get a specific access review definition. | identity | microsoft-agent | 80 | No |
| list_entitlement_access_packages | list_entitlement_access_packages: GET /identityGovernance/entitlementManagement/accessPackages<br/><br/><br/><br/>List entitlement management access packages. | identity | microsoft-agent | 85 | No |
| list_lifecycle_workflows | list_lifecycle_workflows: GET /identityGovernance/lifecycleWorkflows/workflows<br/><br/><br/><br/>List lifecycle management workflows. | identity | microsoft-agent | 80 | No |
| list_security_alerts | list_security_alerts: GET /security/alerts_v2<br/><br/><br/><br/>List security alerts. Returns alert details including severity, status, and detected threats. | security | microsoft-agent | 80 | No |
| get_security_alert | get_security_alert: GET /security/alerts_v2/{alert-id}<br/><br/><br/><br/>Get a specific security alert by ID. | security | microsoft-agent | 70 | No |
| update_security_alert | update_security_alert: PATCH /security/alerts_v2/{alert-id}<br/><br/><br/><br/>Update a security alert. Change status, assign to user, set classification/determination. | security | microsoft-agent | 80 | Yes |
| list_security_incidents | list_security_incidents: GET /security/incidents<br/><br/><br/><br/>List security incidents. Returns correlated alerts grouped into incidents. | security | microsoft-agent | 80 | No |
| get_security_incident | get_security_incident: GET /security/incidents/{incident-id}<br/><br/><br/><br/>Get a specific security incident by ID. | security | microsoft-agent | 80 | No |
| update_security_incident | update_security_incident: PATCH /security/incidents/{incident-id}<br/><br/><br/><br/>Update a security incident. Change status, assign, classify. | security | microsoft-agent | 80 | Yes |
| list_secure_scores | list_secure_scores: GET /security/secureScores<br/><br/><br/><br/>List tenant secure scores over time. | security | microsoft-agent | 70 | No |
| list_threat_intelligence_hosts | list_threat_intelligence_hosts: GET /security/threatIntelligence/hosts<br/><br/><br/><br/>List threat intelligence hosts. | security | microsoft-agent | 85 | No |
| get_threat_intelligence_host | get_threat_intelligence_host: GET /security/threatIntelligence/hosts/{host-id}<br/><br/><br/><br/>Get a specific threat intelligence host. | security | microsoft-agent | 85 | No |
| run_hunting_query | run_hunting_query: POST /security/runHuntingQuery<br/><br/><br/><br/>Run an advanced hunting query using Kusto Query Language (KQL). | security | microsoft-agent | 80 | No |
| list_risk_detections | list_risk_detections: GET /identityProtection/riskDetections<br/><br/><br/><br/>List risk detections (sign-in anomalies, leaked credentials, etc.). | security | microsoft-agent | 80 | No |
| get_risk_detection | get_risk_detection: GET /identityProtection/riskDetections/{id}<br/><br/><br/><br/>Get a specific risk detection. | security | microsoft-agent | 70 | No |
| list_risky_users | list_risky_users: GET /identityProtection/riskyUsers<br/><br/><br/><br/>List users flagged as risky. | security | microsoft-agent | 70 | No |
| get_risky_user | get_risky_user: GET /identityProtection/riskyUsers/{id}<br/><br/><br/><br/>Get a specific risky user. | security | microsoft-agent | 70 | No |
| dismiss_risky_user | dismiss_risky_user: POST /identityProtection/riskyUsers/dismiss<br/><br/><br/><br/>Dismiss a risky user (mark as safe). | security | microsoft-agent | 85 | No |
| list_sensitivity_labels | list_sensitivity_labels: GET /informationProtection/policy/labels<br/><br/><br/><br/>List sensitivity labels. | security | microsoft-agent | 70 | No |
| get_sensitivity_label | get_sensitivity_label: GET /informationProtection/policy/labels/{id}<br/><br/><br/><br/>Get a specific sensitivity label. | security | microsoft-agent | 80 | No |
| list_directory_audits | list_directory_audits: GET /auditLogs/directoryAudits<br/><br/><br/><br/>List directory audit log entries. Shows changes to directory objects (users, groups, apps). | audit | microsoft-agent | 70 | No |
| get_directory_audit | get_directory_audit: GET /auditLogs/directoryAudits/{directoryAudit-id}<br/><br/><br/><br/>Get a specific directory audit entry. | audit | microsoft-agent | 70 | No |
| list_sign_in_logs | list_sign_in_logs: GET /auditLogs/signIns<br/><br/><br/><br/>List sign-in activity logs. Shows user sign-in events with details. | audit | microsoft-agent | 70 | No |
| get_sign_in_log | get_sign_in_log: GET /auditLogs/signIns/{signIn-id}<br/><br/><br/><br/>Get a specific sign-in log entry. | audit | microsoft-agent | 60 | No |
| list_provisioning_logs | list_provisioning_logs: GET /auditLogs/provisioning<br/><br/><br/><br/>List provisioning logs. Shows automated user/group provisioning events. | audit | microsoft-agent | 70 | No |
| get_email_activity_report | get_email_activity_report: GET /reports/getEmailActivityUserDetail<br/><br/><br/><br/>Get email activity user detail report. Period: D7, D30, D90, D180. | reports | microsoft-agent | 85 | No |
| get_mailbox_usage_report | get_mailbox_usage_report: GET /reports/getMailboxUsageDetail<br/><br/><br/><br/>Get mailbox usage detail report. Period: D7, D30, D90, D180. | reports | microsoft-agent | 85 | No |
| get_office365_active_users | get_office365_active_users: GET /reports/getOffice365ActiveUserDetail<br/><br/><br/><br/>Get Office 365 active user detail report. Period: D7, D30, D90, D180. | reports | microsoft-agent | 85 | No |
| get_sharepoint_activity_report | get_sharepoint_activity_report: GET /reports/getSharePointActivityUserDetail<br/><br/><br/><br/>Get SharePoint activity user detail report. Period: D7, D30, D90, D180. | reports | microsoft-agent | 85 | No |
| get_teams_user_activity | get_teams_user_activity: GET /reports/getTeamsUserActivityUserDetail<br/><br/><br/><br/>Get Teams user activity detail report. Period: D7, D30, D90, D180. | reports | microsoft-agent | 85 | No |
| get_onedrive_usage_report | get_onedrive_usage_report: GET /reports/getOneDriveUsageAccountDetail<br/><br/><br/><br/>Get OneDrive usage account detail report. Period: D7, D30, D90, D180. | reports | microsoft-agent | 85 | No |
| list_applications | list_applications: GET /applications<br/><br/><br/><br/>List app registrations in the tenant. | applications | microsoft-agent | 65 | No |
| get_application | get_application: GET /applications/{id}<br/><br/><br/><br/>Get a specific app registration. | applications | microsoft-agent | 65 | No |
| create_application | create_application: POST /applications<br/><br/><br/><br/>Create an app registration. | applications | microsoft-agent | 65 | Yes |
| update_application | update_application: PATCH /applications/{id}<br/><br/><br/><br/>Update an app registration. | applications | microsoft-agent | 65 | Yes |
| delete_application | delete_application: DELETE /applications/{id}<br/><br/><br/><br/>Delete an app registration. | applications | microsoft-agent | 65 | Yes |
| add_application_password | add_application_password: POST /applications/{id}/addPassword<br/><br/><br/><br/>Add a password credential (client secret) to an app. | applications | microsoft-agent | 85 | Yes |
| remove_application_password | remove_application_password: POST /applications/{id}/removePassword<br/><br/><br/><br/>Remove a password credential from an app. | applications | microsoft-agent | 85 | Yes |
| list_service_principals | list_service_principals: GET /servicePrincipals<br/><br/><br/><br/>List service principals (enterprise apps). | applications | microsoft-agent | 70 | No |
| get_service_principal | get_service_principal: GET /servicePrincipals/{id}<br/><br/><br/><br/>Get a specific service principal. | applications | microsoft-agent | 70 | No |
| create_service_principal | create_service_principal: POST /servicePrincipals<br/><br/><br/><br/>Create a service principal for an app. | applications | microsoft-agent | 70 | Yes |
| update_service_principal | update_service_principal: PATCH /servicePrincipals/{id}<br/><br/><br/><br/>Update a service principal. | applications | microsoft-agent | 70 | Yes |
| delete_service_principal | delete_service_principal: DELETE /servicePrincipals/{id}<br/><br/><br/><br/>Delete a service principal. | applications | microsoft-agent | 70 | Yes |
| list_directory_objects | list_directory_objects: GET /directoryObjects<br/><br/><br/><br/>List directory objects. | directory | microsoft-agent | 70 | No |
| get_directory_object | get_directory_object: GET /directoryObjects/{id}<br/><br/><br/><br/>Get a specific directory object. | directory | microsoft-agent | 70 | No |
| list_directory_roles | list_directory_roles: GET /directoryRoles<br/><br/><br/><br/>List activated directory roles. | directory | microsoft-agent | 70 | No |
| get_directory_role | get_directory_role: GET /directoryRoles/{id}<br/><br/><br/><br/>Get a specific activated directory role. | directory | microsoft-agent | 70 | No |
| list_directory_role_templates | list_directory_role_templates: GET /directoryRoleTemplates<br/><br/><br/><br/>List all directory role templates (built-in role definitions). | directory | microsoft-agent | 85 | No |
| list_deleted_items | list_deleted_items: GET /directory/deletedItems<br/><br/><br/><br/>List recently deleted directory items (users, groups, apps). | directory | microsoft-agent | 80 | Yes |
| restore_deleted_item | restore_deleted_item: POST /directory/deletedItems/{id}/restore<br/><br/><br/><br/>Restore a recently deleted directory item. | directory | microsoft-agent | 85 | Yes |
| list_role_definitions | list_role_definitions: GET /roleManagement/directory/roleDefinitions<br/><br/><br/><br/>List RBAC directory role definitions. | directory | microsoft-agent | 80 | No |
| get_role_definition | get_role_definition: GET /roleManagement/directory/roleDefinitions/{id}<br/><br/><br/><br/>Get a specific RBAC role definition. | directory | microsoft-agent | 80 | No |
| list_role_assignments | list_role_assignments: GET /roleManagement/directory/roleAssignments<br/><br/><br/><br/>List RBAC directory role assignments. | directory | microsoft-agent | 80 | No |
| get_role_assignment | get_role_assignment: GET /roleManagement/directory/roleAssignments/{id}<br/><br/><br/><br/>Get a specific RBAC role assignment. | directory | microsoft-agent | 80 | No |
| create_role_assignment | create_role_assignment: POST /roleManagement/directory/roleAssignments<br/><br/><br/><br/>Create a new RBAC role assignment. | directory | microsoft-agent | 80 | Yes |
| get_authorization_policy | get_authorization_policy: GET /policies/authorizationPolicy<br/><br/><br/><br/>Get the tenant authorization policy. | policies | microsoft-agent | 70 | No |
| list_token_lifetime_policies | list_token_lifetime_policies: GET /policies/tokenLifetimePolicies<br/><br/><br/><br/>List token lifetime policies. | policies | microsoft-agent | 75 | No |
| list_token_issuance_policies | list_token_issuance_policies: GET /policies/tokenIssuancePolicies<br/><br/><br/><br/>List token issuance policies. | policies | microsoft-agent | 75 | No |
| list_permission_grant_policies | list_permission_grant_policies: GET /policies/permissionGrantPolicies<br/><br/><br/><br/>List permission grant policies. | policies | microsoft-agent | 85 | No |
| get_admin_consent_policy | get_admin_consent_policy: GET /policies/adminConsentRequestPolicy<br/><br/><br/><br/>Get the admin consent request policy. | policies | microsoft-agent | 85 | No |
| list_devices | list_devices: GET /devices<br/><br/><br/><br/>List devices registered in the directory. | devices | microsoft-agent | 65 | No |
| get_device | get_device: GET /devices/{id}<br/><br/><br/><br/>Get a specific device. | devices | microsoft-agent | 65 | No |
| delete_device | delete_device: DELETE /devices/{id}<br/><br/><br/><br/>Delete a device. | devices | microsoft-agent | 65 | Yes |
| list_managed_devices | list_managed_devices: GET /deviceManagement/managedDevices<br/><br/><br/><br/>List Intune managed devices. | devices | microsoft-agent | 70 | No |
| get_managed_device | get_managed_device: GET /deviceManagement/managedDevices/{id}<br/><br/><br/><br/>Get a specific managed device. | devices | microsoft-agent | 70 | No |
| list_device_compliance_policies | list_device_compliance_policies: GET /deviceManagement/deviceCompliancePolicies<br/><br/><br/><br/>List device compliance policies. | devices | microsoft-agent | 85 | No |
| list_device_configurations | list_device_configurations: GET /deviceManagement/deviceConfigurations<br/><br/><br/><br/>List device configuration profiles. | devices | microsoft-agent | 80 | No |
| wipe_managed_device | wipe_managed_device: POST /deviceManagement/managedDevices/{id}/wipe<br/><br/><br/><br/>Wipe a managed device (factory reset). | devices | microsoft-agent | 85 | No |
| retire_managed_device | retire_managed_device: POST /deviceManagement/managedDevices/{id}/retire<br/><br/><br/><br/>Retire a managed device (remove company data). | devices | microsoft-agent | 85 | No |
| list_education_classes | list_education_classes: GET /education/classes<br/><br/><br/><br/>List education classes. | education | microsoft-agent | 70 | No |
| get_education_class | get_education_class: GET /education/classes/{id}<br/><br/><br/><br/>Get a specific education class. | education | microsoft-agent | 70 | No |
| list_education_schools | list_education_schools: GET /education/schools<br/><br/><br/><br/>List education schools. | education | microsoft-agent | 70 | No |
| get_education_school | get_education_school: GET /education/schools/{id}<br/><br/><br/><br/>Get a specific education school. | education | microsoft-agent | 70 | No |
| list_education_users | list_education_users: GET /education/users<br/><br/><br/><br/>List education users. | education | microsoft-agent | 70 | No |
| list_education_assignments | list_education_assignments: GET /education/classes/{id}/assignments<br/><br/><br/><br/>List assignments for an education class. | education | microsoft-agent | 80 | No |
| list_agreements | list_agreements: GET /agreements<br/><br/><br/><br/>List terms-of-use agreements. | agreements | microsoft-agent | 65 | No |
| get_agreement | get_agreement: GET /agreements/{id}<br/><br/><br/><br/>Get a specific agreement. | agreements | microsoft-agent | 65 | No |
| create_agreement | create_agreement: POST /agreements<br/><br/><br/><br/>Create a terms-of-use agreement. | agreements | microsoft-agent | 65 | Yes |
| delete_agreement | delete_agreement: DELETE /agreements/{id}<br/><br/><br/><br/>Delete an agreement. | agreements | microsoft-agent | 65 | Yes |
| list_rooms | list_rooms: GET /places/microsoft.graph.room<br/><br/><br/><br/>List conference rooms. | places | microsoft-agent | 55 | No |
| list_room_lists | list_room_lists: GET /places/microsoft.graph.roomList<br/><br/><br/><br/>List room lists. | places | microsoft-agent | 60 | No |
| get_place | get_place: GET /places/{id}<br/><br/><br/><br/>Get a specific place (room or room list). | places | microsoft-agent | 55 | No |
| update_place | update_place: PATCH /places/{id}<br/><br/><br/><br/>Update a place (room). | places | microsoft-agent | 55 | Yes |
| list_printers | list_printers: GET /print/printers<br/><br/><br/><br/>List printers registered in the tenant. | print | microsoft-agent | 55 | No |
| get_printer | get_printer: GET /print/printers/{id}<br/><br/><br/><br/>Get a specific printer. | print | microsoft-agent | 55 | No |
| list_print_jobs | list_print_jobs: GET /print/printers/{id}/jobs<br/><br/><br/><br/>List print jobs for a printer. | print | microsoft-agent | 60 | No |
| create_print_job | create_print_job: POST /print/printers/{id}/jobs<br/><br/><br/><br/>Create a print job. | print | microsoft-agent | 60 | Yes |
| list_print_shares | list_print_shares: GET /print/shares<br/><br/><br/><br/>List printer shares. | print | microsoft-agent | 60 | No |
| list_subject_rights_requests | list_subject_rights_requests: GET /privacy/subjectRightsRequests<br/><br/><br/><br/>List subject rights requests (GDPR/CCPA). | privacy | microsoft-agent | 85 | No |
| get_subject_rights_request | get_subject_rights_request: GET /privacy/subjectRightsRequests/{id}<br/><br/><br/><br/>Get a specific subject rights request. | privacy | microsoft-agent | 85 | No |
| create_subject_rights_request | create_subject_rights_request: POST /privacy/subjectRightsRequests<br/><br/><br/><br/>Create a subject rights request. | privacy | microsoft-agent | 85 | Yes |
| list_booking_businesses | list_booking_businesses: GET /solutions/bookingBusinesses<br/><br/><br/><br/>List booking businesses. | solutions | microsoft-agent | 70 | No |
| get_booking_business | get_booking_business: GET /solutions/bookingBusinesses/{id}<br/><br/><br/><br/>Get a specific booking business. | solutions | microsoft-agent | 70 | No |
| list_booking_appointments | list_booking_appointments: GET /solutions/bookingBusinesses/{id}/appointments<br/><br/><br/><br/>List appointments for a booking business. | solutions | microsoft-agent | 80 | No |
| create_booking_appointment | create_booking_appointment: POST /solutions/bookingBusinesses/{id}/appointments<br/><br/><br/><br/>Create a booking appointment. | solutions | microsoft-agent | 80 | Yes |
| list_virtual_events | list_virtual_events: GET /solutions/virtualEvents/townhalls<br/><br/><br/><br/>List virtual event townhalls. | solutions | microsoft-agent | 70 | No |
| list_file_storage_containers | list_file_storage_containers: GET /storage/fileStorage/containers<br/><br/><br/><br/>List file storage containers. | storage | microsoft-agent | 75 | No |
| get_file_storage_container | get_file_storage_container: GET /storage/fileStorage/containers/{id}<br/><br/><br/><br/>Get a specific file storage container. | storage | microsoft-agent | 85 | No |
| create_file_storage_container | create_file_storage_container: POST /storage/fileStorage/containers<br/><br/><br/><br/>Create a file storage container. | storage | microsoft-agent | 85 | Yes |
| list_learning_providers | list_learning_providers: GET /employeeExperience/learningProviders<br/><br/><br/><br/>List learning providers. | employee_experience | microsoft-agent | 70 | No |
| get_learning_provider | get_learning_provider: GET /employeeExperience/learningProviders/{id}<br/><br/><br/><br/>Get a specific learning provider. | employee_experience | microsoft-agent | 80 | No |
| list_learning_course_activities | list_learning_course_activities: GET /me/employeeExperience/learningCourseActivities<br/><br/><br/><br/>List learning course activities for the current user. | employee_experience | microsoft-agent | 85 | No |
| list_external_connections | list_external_connections: GET /external/connections<br/><br/><br/><br/>List Microsoft Search external connections. | connections | microsoft-agent | 70 | No |
| get_external_connection | get_external_connection: GET /external/connections/{id}<br/><br/><br/><br/>Get a specific external connection. | connections | microsoft-agent | 70 | No |
| create_external_connection | create_external_connection: POST /external/connections<br/><br/><br/><br/>Create an external connection for Microsoft Search. | connections | microsoft-agent | 80 | Yes |
| delete_external_connection | delete_external_connection: DELETE /external/connections/{id}<br/><br/><br/><br/>Delete an external connection. | connections | microsoft-agent | 70 | Yes |
| calendar_today | Show today's calendar events. | calendar | microsoft-agent | 35 | No |
| list_files | List files. | files | nextcloud-agent | 40 | No |
| read_file | Read the contents of a text file from Nextcloud. | files | nextcloud-agent | 50 | No |
| write_file | Write text content to a file in Nextcloud. | files | nextcloud-agent | 50 | Yes |
| create_folder | Create a new directory in Nextcloud. | files | nextcloud-agent | 45 | Yes |
| delete_item | Delete a file or directory in Nextcloud. | files | nextcloud-agent | 45 | Yes |
| move_item | Move a file or directory to a new location. | files | nextcloud-agent | 50 | Yes |
| copy_item | Copy a file or directory to a new location. | files | nextcloud-agent | 50 | No |
| get_properties | Get detailed properties for a file or folder. | files | nextcloud-agent | 45 | No |
| get_user_info | Get information about the current user. | user | nextcloud-agent | 50 | No |
| list_shares | List all shares. | sharing | nextcloud-agent | 55 | No |
| create_share | Create a new share. | sharing | nextcloud-agent | 55 | Yes |
| delete_share | Delete a share. | sharing | nextcloud-agent | 50 | Yes |
| list_calendars | List available calendars. | calendar | nextcloud-agent | 55 | No |
| list_calendar_events | List events in a calendar. | calendar | nextcloud-agent | 60 | No |
| create_calendar_event |  | calendar | nextcloud-agent | 50 | Yes |
| list_address_books | List address books. | contacts | nextcloud-agent | 60 | Yes |
| list_contacts | List contacts in an address book. | contacts | nextcloud-agent | 55 | No |
| create_contact | Create a new contact using raw vCard data. | contacts | nextcloud-agent | 55 | Yes |
| owncast-chat-get-user-details | Get a user's details | chat | owncast | 55 | No |
| owncast-external-send-system-message | Send a system message to the chat | external | owncast | 65 | No |
| owncast-external-send-system-message-to-connected-client | Send a system message to a single client | external | owncast | 65 | No |
| owncast-external-send-user-message | Send a user message to chat | external | owncast | 65 | No |
| owncast-external-send-integration-chat-message | Send a message to chat as a specific 3rd party bot/integration based on its access token | external | owncast | 75 | No |
| owncast-external-send-chat-action | Send a user action to chat | external | owncast | 65 | No |
| owncast-external-update-message-visibility | Hide chat message | external | owncast | 65 | Yes |
| owncast-external-get-status | Get the server's status | external | owncast | 65 | No |
| owncast-external-set-stream-title | Stream title | external | owncast | 60 | Yes |
| owncast-external-get-chat-messages | Get chat history | external | owncast | 65 | No |
| owncast-external-get-connected-chat-clients | Connected clients | external | owncast | 65 | No |
| owncast-external-get-user-details | Get a user's details | external | owncast | 65 | No |
| owncast-internal-get-status | Get the status of the server | internal | owncast | 65 | No |
| owncast-internal-get-custom-emoji-list | Get list of custom emojis supported in the chat | internal | owncast | 65 | No |
| owncast-internal-get-chat-messages | Gets a list of chat messages | internal | owncast | 65 | No |
| owncast-internal-register-anonymous-chat-user | Registers an anonymous chat user | internal | owncast | 65 | No |
| owncast-internal-update-message-visibility | Update chat message visibility | internal | owncast | 65 | Yes |
| owncast-internal-update-user-enabled | Enable/disable a user | internal | owncast | 65 | Yes |
| owncast-internal-get-web-config | Get the web config | internal | owncast | 65 | No |
| owncast-internal-get-ypresponse | Get the YP protocol data | internal | owncast | 65 | No |
| owncast-internal-get-all-social-platforms | Get all social platforms | internal | owncast | 65 | No |
| owncast-internal-get-video-stream-output-variants | Get a list of video variants available | internal | owncast | 65 | Yes |
| owncast-internal-ping | Tell the backend you're an active viewer | internal | owncast | 65 | No |
| owncast-internal-remote-follow | Request remote follow | internal | owncast | 65 | No |
| owncast-internal-get-followers | Gets the list of followers | internal | owncast | 65 | No |
| owncast-internal-report-playback-metrics | Save video playback metrics for future video health recording | internal | owncast | 75 | No |
| owncast-internal-register-for-live-notifications | Register for notifications | internal | owncast | 65 | No |
| owncast-internal-status-admin | Get current inboard broadcaster | internal | owncast | 65 | No |
| owncast-internal-disconnect-inbound-connection | Disconnect inbound stream | internal | owncast | 65 | No |
| owncast-internal-get-server-config | Get the current server config | internal | owncast | 65 | No |
| owncast-internal-get-viewers-over-time | Get viewer count over time | internal | owncast | 65 | No |
| owncast-internal-get-active-viewers | Get active viewers | internal | owncast | 65 | No |
| owncast-internal-get-hardware-stats | Get the current hardware stats | internal | owncast | 65 | No |
| owncast-internal-get-connected-chat-clients | Get a detailed list of currently connected chat clients | internal | owncast | 75 | No |
| owncast-internal-get-chat-messages-admin | Get all chat messages for the admin, unfiltered | internal | owncast | 65 | No |
| owncast-internal-update-message-visibility-admin | Update visibility of chat messages | internal | owncast | 65 | Yes |
| owncast-internal-update-user-enabled-admin | Enable or disable a user | internal | owncast | 65 | Yes |
| owncast-internal-get-disabled-users | Get a list of disabled users | internal | owncast | 65 | Yes |
| owncast-internal-ban-ipaddress | Ban an IP address | internal | owncast | 65 | Yes |
| owncast-internal-unban-ipaddress | Remove an IP ban | internal | owncast | 65 | Yes |
| owncast-internal-get-ipaddress-bans | Get all banned IP addresses | internal | owncast | 65 | Yes |
| owncast-internal-update-user-moderator | Set moderator status for a user | internal | owncast | 65 | Yes |
| owncast-internal-get-moderators | Get a list of moderator users | internal | owncast | 65 | No |
| owncast-internal-get-logs | Get all logs | internal | owncast | 60 | No |
| owncast-internal-get-warnings | Get warning/error logs | internal | owncast | 65 | No |
| owncast-internal-get-followers-admin | Get followers | internal | owncast | 60 | No |
| owncast-internal-get-pending-follow-requests | Get a list of pending follow requests | internal | owncast | 65 | No |
| owncast-internal-get-blocked-and-rejected-followers | Get a list of rejected or blocked follows | internal | owncast | 65 | No |
| owncast-internal-approve-follower | Set the following state of a follower or follow request | internal | owncast | 75 | Yes |
| owncast-internal-upload-custom-emoji | Upload custom emoji | internal | owncast | 65 | Yes |
| owncast-internal-delete-custom-emoji | Delete custom emoji | internal | owncast | 65 | Yes |
| owncast-internal-set-admin-password | Change the current admin password | internal | owncast | 65 | Yes |
| owncast-internal-set-stream-keys | Set an array of valid stream keys | internal | owncast | 65 | Yes |
| owncast-internal-set-extra-page-content | Change the extra page content in memory | internal | owncast | 65 | Yes |
| owncast-internal-set-stream-title | Change the stream title | internal | owncast | 65 | Yes |
| owncast-internal-set-server-welcome-message | Change the welcome message | internal | owncast | 65 | Yes |
| owncast-internal-set-chat-disabled | Disable chat | internal | owncast | 60 | Yes |
| owncast-internal-set-chat-join-messages-enabled | Enable chat for user join messages | internal | owncast | 65 | Yes |
| owncast-internal-set-enable-established-chat-user-mode | Enable/disable chat established user mode | internal | owncast | 65 | Yes |
| owncast-internal-set-forbidden-username-list | Set chat usernames that are not allowed | internal | owncast | 65 | Yes |
| owncast-internal-set-suggested-username-list | Set the suggested chat usernames that will be assigned automatically | internal | owncast | 75 | Yes |
| owncast-internal-set-chat-spam-protection-enabled | Set spam protection enabled | internal | owncast | 65 | Yes |
| owncast-internal-set-chat-slur-filter-enabled | Set slur filter enabled | internal | owncast | 65 | Yes |
| owncast-internal-set-chat-require-authentication | Set require authentication for chat | internal | owncast | 65 | Yes |
| owncast-internal-set-video-codec | Set video codec | internal | owncast | 60 | Yes |
| owncast-internal-set-stream-latency-level | Set the number of video segments and duration per segment in a playlist | internal | owncast | 75 | Yes |
| owncast-internal-set-stream-output-variants | Set an array of video output configurations | internal | owncast | 65 | Yes |
| owncast-internal-set-custom-color-variable-values | Set style/color/css values | internal | owncast | 65 | Yes |
| owncast-internal-set-logo | Update logo | internal | owncast | 60 | Yes |
| owncast-internal-set-favicon | Upload custom favicon | internal | owncast | 65 | Yes |
| owncast-internal-reset-favicon | Reset favicon to default | internal | owncast | 65 | Yes |
| owncast-internal-set-tags | Update server tags | internal | owncast | 65 | Yes |
| owncast-internal-set-ffmpeg-path | Update FFMPEG path | internal | owncast | 65 | Yes |
| owncast-internal-set-web-server-port | Update server port | internal | owncast | 65 | Yes |
| owncast-internal-set-web-server-ip | Update server IP address | internal | owncast | 65 | Yes |
| owncast-internal-set-rtmpserver-port | Update RTMP post | internal | owncast | 65 | Yes |
| owncast-internal-set-socket-host-override | Update websocket host override | internal | owncast | 65 | Yes |
| owncast-internal-set-video-serving-endpoint | Update custom video serving endpoint | internal | owncast | 65 | Yes |
| owncast-internal-set-nsfw | Update NSFW marking | internal | owncast | 65 | Yes |
| owncast-internal-set-directory-enabled | Update directory enabled | internal | owncast | 65 | Yes |
| owncast-internal-set-social-handles | Update social handles | internal | owncast | 65 | Yes |
| owncast-internal-set-s3-configuration | Update S3 configuration | internal | owncast | 65 | Yes |
| owncast-internal-set-server-url | Update server url | internal | owncast | 65 | Yes |
| owncast-internal-set-external-actions | Update external action links | internal | owncast | 65 | Yes |
| owncast-internal-set-custom-styles | Update custom styles | internal | owncast | 65 | Yes |
| owncast-internal-set-custom-javascript | Update custom JavaScript | internal | owncast | 65 | Yes |
| owncast-internal-set-hide-viewer-count | Update hide viewer count | internal | owncast | 65 | Yes |
| owncast-internal-set-disable-search-indexing | Update search indexing | internal | owncast | 65 | Yes |
| owncast-internal-set-federation-enabled | Enable/disable federation features | internal | owncast | 65 | Yes |
| owncast-internal-set-federation-activity-private | Set if federation activities are private | internal | owncast | 65 | Yes |
| owncast-internal-set-federation-show-engagement | Set if fediverse engagement appears in chat | internal | owncast | 65 | Yes |
| owncast-internal-set-federation-username | Set local federated username | internal | owncast | 65 | Yes |
| owncast-internal-set-federation-go-live-message | Set federated go live message | internal | owncast | 65 | Yes |
| owncast-internal-set-federation-block-domains | Set Federation blocked domains | internal | owncast | 65 | Yes |
| owncast-internal-set-discord-notification-configuration | Configure Discord notifications | internal | owncast | 65 | Yes |
| owncast-internal-set-browser-notification-configuration | Configure Browser notifications | internal | owncast | 65 | Yes |
| owncast-internal-get-webhooks | Get all the webhooks | internal | owncast | 65 | No |
| owncast-internal-delete-webhook | Delete a single webhook | internal | owncast | 65 | Yes |
| owncast-internal-create-webhook | Create a single webhook | internal | owncast | 65 | Yes |
| owncast-internal-get-external-apiusers | Get all access tokens | internal | owncast | 65 | No |
| owncast-internal-delete-external-apiuser | Delete a single external API user | internal | owncast | 65 | Yes |
| owncast-internal-create-external-apiuser | Create a single access token | internal | owncast | 65 | Yes |
| owncast-internal-auto-update-options | Return the auto-update features that are supported for this instance | internal | owncast | 75 | Yes |
| owncast-internal-auto-update-start | Begin the auto-update | internal | owncast | 65 | Yes |
| owncast-internal-auto-update-force-quit | Force quit the server and restart it | internal | owncast | 65 | Yes |
| owncast-internal-reset-ypregistration | Reset YP configuration | internal | owncast | 65 | Yes |
| owncast-internal-get-video-playback-metrics | Get video playback metrics | internal | owncast | 65 | No |
| owncast-internal-get-prometheus-api | Endpoint to interface with Prometheus | internal | owncast | 65 | No |
| owncast-internal-post-prometheus-api | Endpoint to interface with Prometheus | internal | owncast | 65 | Yes |
| owncast-internal-put-prometheus-api | Endpoint to interface with Prometheus | internal | owncast | 65 | Yes |
| owncast-internal-delete-prometheus-api | Endpoint to interface with Prometheus | internal | owncast | 65 | Yes |
| owncast-internal-send-federated-message | Send a public message to the Fediverse from the server's user | internal | owncast | 75 | No |
| owncast-internal-get-federated-actions | Get a paginated list of federated activities | internal | owncast | 65 | No |
| owncast-internal-start-indie-auth-flow | Begins auth flow | internal | owncast | 65 | Yes |
| owncast-internal-handle-indie-auth-redirect | Handle the redirect from an IndieAuth server to continue the auth flow | internal | owncast | 75 | No |
| owncast-internal-handle-indie-auth-endpoint-get | Handles the IndieAuth auth endpoint | internal | owncast | 65 | No |
| owncast-internal-handle-indie-auth-endpoint-post | Handles IndieAuth from form submission | internal | owncast | 65 | Yes |
| owncast-internal-register-fediverse-otprequest | Register a Fediverse OTP request | internal | owncast | 65 | No |
| owncast-internal-verify-fediverse-otprequest | Verify Fediverse OTP code | internal | owncast | 65 | No |
| owncast-objects-set-server-name | Change the server name | objects | owncast | 65 | Yes |
| owncast-objects-set-server-summary | Change the server summary | objects | owncast | 65 | Yes |
| owncast-objects-set-custom-offline-message | Change the offline message | objects | owncast | 65 | Yes |
| list_projects | List all projects in the workspace. | projects | plane | 55 | No |
| retrieve_project | Retrieve a project by ID. | projects | plane | 60 | No |
| list_work_items | List work items in a project or search across workspace. | work_items | plane | 70 | No |
| create_work_item | Create a new work item. | work_items | plane | 60 | Yes |
| update_work_item | Update a work item. | work_items | plane | 60 | Yes |
| delete_work_item | Delete a work item. | work_items | plane | 60 | Yes |
| search_work_items | Search work items across workspace. | work_items | plane | 65 | No |
| retrieve_work_item_by_identifier | Retrieve a work item by project identifier and issue sequence number. | work_items | plane | 75 | No |
| retrieve_work_item | Retrieve a work item by ID. | work_items | plane | 65 | No |
| list_work_item_activities | List activities for a work item. | work_items | plane | 65 | No |
| list_work_item_comments | List comments for a work item. | work_items | plane | 65 | No |
| create_work_item_comment | Create a comment for a work item. | work_items | plane | 65 | Yes |
| list_work_item_links | List links for a work item. | work_items | plane | 65 | No |
| create_work_item_link | Create a link for a work item. | work_items | plane | 65 | Yes |
| list_work_item_relations | List relations for a work item. | work_items | plane | 65 | No |
| list_work_item_types | List work item types in a project. | work_items | plane | 65 | No |
| list_work_logs | List work logs for a work item. | work_items | plane | 60 | No |
| create_work_log | Create a work log for a work item. | work_items | plane | 60 | Yes |
| list_cycles | List cycles in a project. | cycles | plane | 45 | No |
| create_cycle | Create a new cycle. | cycles | plane | 45 | Yes |
| retrieve_cycle | Retrieve a cycle by ID. | cycles | plane | 50 | No |
| update_cycle | Update a cycle by ID. | cycles | plane | 45 | Yes |
| delete_cycle | Delete a cycle by ID. | cycles | plane | 45 | Yes |
| list_cycle_work_items | List work items in a cycle. | cycles | plane | 55 | No |
| add_work_items_to_cycle | Add work items to a cycle. | cycles | plane | 55 | Yes |
| list_epics | List epics in a project. | epics | plane | 45 | No |
| create_epic | Create a new epic. | epics | plane | 45 | Yes |
| retrieve_epic | Retrieve an epic by ID. | epics | plane | 50 | No |
| update_epic | Update an epic by ID. | epics | plane | 45 | Yes |
| delete_epic | Delete an epic by ID. | epics | plane | 45 | Yes |
| list_initiatives | List all initiatives in the workspace. | initiatives | plane | 55 | No |
| create_initiative | Create a new initiative. | initiatives | plane | 55 | Yes |
| list_intake_work_items | List all intake work items in a project. | intake | plane | 55 | No |
| create_intake_work_item | Create a new intake work item. | intake | plane | 55 | Yes |
| list_labels | List all labels in a project. | labels | plane | 45 | No |
| create_label | Create a new label. | labels | plane | 45 | Yes |
| retrieve_project_page | Retrieve a project page by ID. | pages | plane | 55 | No |
| create_project_page | Create a project page. | pages | plane | 50 | Yes |
| list_milestones | List milestones in a project. | milestones | plane | 55 | No |
| create_milestone | Create a new milestone. | milestones | plane | 55 | Yes |
| retrieve_milestone | Retrieve a milestone by ID. | milestones | plane | 60 | No |
| update_milestone | Update a milestone by ID. | milestones | plane | 55 | Yes |
| delete_milestone | Delete a milestone by ID. | milestones | plane | 55 | Yes |
| list_modules | List modules in a project. | modules | plane | 55 | No |
| create_module | Create a new module. | modules | plane | 55 | Yes |
| retrieve_module | Retrieve a module by ID. | modules | plane | 60 | No |
| update_module | Update a module by ID. | modules | plane | 55 | Yes |
| delete_module | Delete a module by ID. | modules | plane | 55 | Yes |
| list_states | List states in a project. | states | plane | 45 | No |
| create_state | Create a new state. | states | plane | 45 | Yes |
| list_users | List users in the workspace. | users | plane | 45 | No |
| get_me | Get current user information. | users | plane | 40 | No |
| get_workspace | Get current workspace details. | workspaces | plane | 55 | No |
| get_workspace_members | Get all members of the current workspace. | workspaces | plane | 60 | No |
| get_workspace_features | Get features of the current workspace. | workspaces | plane | 60 | No |
| update_workspace_features | Update features of the current workspace. | workspaces | plane | 60 | Yes |
| authenticate | Authenticate against Portainer with username and password to get a JWT token. | Auth | portainer-agent | 55 | No |
| logout | Logout and invalidate the current authentication token. | Auth | portainer-agent | 55 | No |
| validate_oauth | Validate an OAuth authorization code. | Auth | portainer-agent | 50 | No |
| get_endpoints | List all Portainer environments (endpoints). Each environment represents a Docker host, Swarm cluster, or Kubernetes cluster. | Environment | portainer-agent | 75 | No |
| get_endpoint | Get details of a specific environment (endpoint) by ID. | Environment | portainer-agent | 65 | No |
| create_endpoint | Create a new environment. Types: 1=Docker, 2=AgentOnDocker, 3=Azure, 4=EdgeAgent, 5=KubernetesLocal, 6=AgentOnKubernetes, 7=EdgeAgentOnKubernetes. | Environment | portainer-agent | 75 | Yes |
| update_endpoint | Update an existing environment's configuration. | Environment | portainer-agent | 55 | Yes |
| delete_endpoint | Delete an environment (endpoint). | Environment | portainer-agent | 55 | Yes |
| snapshot_endpoint | Take a snapshot of an environment to refresh its state. | Environment | portainer-agent | 70 | No |
| snapshot_all_endpoints | Take a snapshot of all environments. | Environment | portainer-agent | 65 | No |
| get_endpoint_groups | List all environment groups. | Environment | portainer-agent | 60 | No |
| create_endpoint_group | Create a new environment group. | Environment | portainer-agent | 60 | Yes |
| delete_endpoint_group | Delete an environment group. | Environment | portainer-agent | 60 | Yes |
| get_docker_dashboard | Get Docker dashboard data (containers, images, volumes, networks summary) for an environment. | Docker | portainer-agent | 60 | No |
| get_container_gpus | Get GPU information for a Docker container. | Docker | portainer-agent | 50 | No |
| docker_list_containers | List containers in a Docker environment. | Docker | portainer-agent | 50 | No |
| docker_inspect_container | Return low-level information about a container. | Docker | portainer-agent | 55 | No |
| docker_get_container_logs | Get stdout and stderr logs from a container. | Docker | portainer-agent | 55 | No |
| docker_get_container_stats | Get resource usage statistics for a container. | Docker | portainer-agent | 55 | No |
| docker_start_container | Start a container. | Docker | portainer-agent | 55 | Yes |
| docker_stop_container | Stop a container. | Docker | portainer-agent | 55 | Yes |
| docker_restart_container | Restart a container. | Docker | portainer-agent | 55 | Yes |
| docker_remove_container | Remove a container. | Docker | portainer-agent | 55 | Yes |
| docker_list_services | List Swarm services in a Docker environment. | Docker | portainer-agent | 50 | No |
| docker_inspect_service | Return low-level information about a Swarm service. | Docker | portainer-agent | 65 | No |
| docker_get_service_logs | Get stdout and stderr logs from a Swarm service. | Docker | portainer-agent | 55 | No |
| docker_list_images | List images in a Docker environment. | Docker | portainer-agent | 50 | No |
| docker_inspect_image | Return low-level information about an image. | Docker | portainer-agent | 55 | No |
| docker_list_networks | List networks in a Docker environment. | Docker | portainer-agent | 50 | No |
| docker_inspect_network | Return low-level information about a network. | Docker | portainer-agent | 55 | No |
| docker_list_volumes | List volumes in a Docker environment. | Docker | portainer-agent | 50 | No |
| docker_inspect_volume | Return low-level information about a volume. | Docker | portainer-agent | 55 | No |
| docker_get_info | Get system-wide information for the Docker host. | Docker | portainer-agent | 50 | No |
| docker_get_version | Get Docker version information. | Docker | portainer-agent | 50 | No |
| docker_get_system_df | Get Docker data usage information. | Docker | portainer-agent | 50 | No |
| docker_create_container | Create a new container. | Docker | portainer-agent | 50 | Yes |
| docker_create_network | Create a new network. | Docker | portainer-agent | 50 | Yes |
| docker_create_volume | Create a new volume. | Docker | portainer-agent | 50 | Yes |
| docker_create_exec | Create an exec instance in a container. | Docker | portainer-agent | 50 | Yes |
| docker_start_exec | Start an exec instance. | Docker | portainer-agent | 55 | Yes |
| docker_inspect_exec | Inspect an exec instance. | Docker | portainer-agent | 55 | No |
| docker_get_stack_logs | Get aggregated logs for all containers or services in a Portainer stack. | Docker, Stack | portainer-agent | 85 | No |
| get_stacks | List all stacks across all environments. | Stack | portainer-agent | 45 | No |
| get_stack | Get details of a specific stack by ID. | Stack | portainer-agent | 45 | No |
| get_stack_file | Get the Docker Compose/manifest file content for a stack. | Stack | portainer-agent | 60 | No |
| create_standalone_stack | Create a standalone Docker Compose stack from compose file content. | Stack | portainer-agent | 60 | Yes |
| create_standalone_stack_from_repo | Create a standalone Docker Compose stack from a Git repository. | Stack | portainer-agent | 65 | Yes |
| update_stack | Update a stack's configuration. | Stack | portainer-agent | 45 | Yes |
| delete_stack | Delete a stack. | Stack | portainer-agent | 40 | Yes |
| start_stack | Start a stopped stack. | Stack | portainer-agent | 50 | Yes |
| stop_stack | Stop a running stack. | Stack | portainer-agent | 50 | Yes |
| redeploy_stack_git | Redeploy a stack from its Git repository (pull latest and redeploy). | Stack | portainer-agent | 65 | Yes |
| get_kubernetes_dashboard | Get Kubernetes dashboard data for an environment (pods, services, deployments summary). | Kubernetes | portainer-agent | 70 | No |
| get_kubernetes_namespaces | List Kubernetes namespaces in an environment. | Kubernetes | portainer-agent | 60 | No |
| get_kubernetes_applications | List Kubernetes applications (deployments, statefulsets, daemonsets) in an environment. | Kubernetes | portainer-agent | 70 | No |
| get_kubernetes_services | List Kubernetes services in an environment. | Kubernetes | portainer-agent | 60 | No |
| get_kubernetes_ingresses | List Kubernetes ingresses in an environment. | Kubernetes | portainer-agent | 60 | No |
| get_kubernetes_configmaps | List Kubernetes configmaps in an environment. | Kubernetes | portainer-agent | 60 | No |
| get_kubernetes_secrets | List Kubernetes secrets in an environment. | Kubernetes | portainer-agent | 60 | No |
| get_kubernetes_volumes | List Kubernetes persistent volume claims in an environment. | Kubernetes | portainer-agent | 70 | No |
| get_kubernetes_events | List Kubernetes events in an environment. | Kubernetes | portainer-agent | 60 | No |
| get_kubernetes_nodes_limits | Get Kubernetes node resource limits for capacity planning. | Kubernetes | portainer-agent | 75 | No |
| get_kubernetes_metrics_nodes | Get resource metrics for Kubernetes nodes. | Kubernetes | portainer-agent | 65 | No |
| get_helm_releases | List Helm releases installed in an environment. | Kubernetes | portainer-agent | 60 | No |
| install_helm_chart | Install a Helm chart in an environment. | Kubernetes | portainer-agent | 65 | Yes |
| delete_helm_release | Delete (uninstall) a Helm release. | Kubernetes | portainer-agent | 60 | Yes |
| get_edge_groups | List all edge groups. | Edge | portainer-agent | 50 | No |
| create_edge_group | Create an edge group for organizing edge devices. | Edge | portainer-agent | 50 | Yes |
| delete_edge_group | Delete an edge group. | Edge | portainer-agent | 50 | Yes |
| get_edge_stacks | List all edge stacks deployed to edge groups. | Edge | portainer-agent | 50 | No |
| get_edge_stack | Get details of a specific edge stack. | Edge | portainer-agent | 50 | No |
| create_edge_stack | Create an edge stack from compose file content. | Edge | portainer-agent | 50 | Yes |
| delete_edge_stack | Delete an edge stack. | Edge | portainer-agent | 50 | Yes |
| get_edge_jobs | List all edge jobs. | Edge | portainer-agent | 50 | No |
| get_edge_job | Get details of a specific edge job. | Edge | portainer-agent | 50 | No |
| create_edge_job | Create an edge job to execute scripts on edge devices. | Edge | portainer-agent | 60 | Yes |
| delete_edge_job | Delete an edge job. | Edge | portainer-agent | 50 | Yes |
| get_templates | List available app templates. | Template | portainer-agent | 55 | No |
| get_custom_templates | List custom templates created by users. | Template | portainer-agent | 60 | No |
| get_custom_template | Get details of a specific custom template. | Template | portainer-agent | 60 | No |
| create_custom_template | Create a custom template from compose file content. Types: 1=swarm, 2=compose, 3=kubernetes. | Template | portainer-agent | 70 | Yes |
| delete_custom_template | Delete a custom template. | Template | portainer-agent | 60 | Yes |
| get_custom_template_file | Get the compose file content of a custom template. | Template | portainer-agent | 65 | No |
| get_helm_templates | List available Helm chart templates. | Template | portainer-agent | 60 | No |
| get_users | List all Portainer users. | User | portainer-agent | 45 | No |
| get_user | Get details of a specific user. | User | portainer-agent | 45 | No |
| get_current_user | Get the currently authenticated user's profile. | User | portainer-agent | 50 | No |
| create_user | Create a new Portainer user. Roles: 1=admin, 2=standard. | User | portainer-agent | 55 | Yes |
| delete_user | Delete a Portainer user. | User | portainer-agent | 45 | Yes |
| get_teams | List all teams. | User | portainer-agent | 40 | No |
| create_team | Create a new team. | User | portainer-agent | 45 | Yes |
| delete_team | Delete a team. | User | portainer-agent | 40 | Yes |
| get_roles | List all available roles. | User | portainer-agent | 45 | No |
| get_user_tokens | List API tokens for a user. | User | portainer-agent | 50 | No |
| get_registries | List all configured Docker registries. | Registry | portainer-agent | 55 | No |
| get_registry | Get details of a specific registry. | Registry | portainer-agent | 55 | No |
| create_registry | Add a Docker registry. Types: 1=Quay, 2=Azure, 3=Custom, 4=GitLab, 5=ProGet, 6=DockerHub, 7=ECR, 8=GitHub. | Registry | portainer-agent | 75 | Yes |
| delete_registry | Delete a Docker registry. | Registry | portainer-agent | 55 | Yes |
| get_status | Get Portainer instance status (version, uptime, etc.). | System | portainer-agent | 55 | No |
| get_system_info | Get detailed system information (build info, dependencies, runtime). | System | portainer-agent | 60 | No |
| get_system_version | Get Portainer version information. | System | portainer-agent | 50 | No |
| get_settings | Get Portainer settings (authentication, templates URL, edge agent, etc.). | System | portainer-agent | 55 | Yes |
| update_settings | Update Portainer settings. | System | portainer-agent | 45 | Yes |
| get_tags | List all tags used for organizing environments. | System | portainer-agent | 45 | No |
| create_tag | Create a tag for organizing environments. | System | portainer-agent | 45 | Yes |
| delete_tag | Delete a tag. | System | portainer-agent | 40 | Yes |
| get_motd | Get the Portainer message of the day. | System | portainer-agent | 45 | No |
| backup_portainer | Create a backup of all Portainer data. | System | portainer-agent | 50 | No |
| postiz-list-integrations | List all connected social media channels. | integrations | postiz | 60 | Yes |
| postiz-get-integration-url | Generate an OAuth authorization URL for a given integration. | integrations | postiz | 75 | Yes |
| postiz-delete-channel | Delete a connected channel by its integration ID. | integrations | postiz | 60 | Yes |
| postiz-check-connection | Verify if your API key is valid and connected. | integrations | postiz | 65 | Yes |
| postiz-find-slot | Get the next available time slot for posting to a specific channel. | integrations | postiz | 75 | Yes |
| postiz-list-posts | Get posts within a date range. | posts | postiz | 50 | Yes |
| postiz-create-post | Create or schedule a new post. | posts | postiz | 50 | Yes |
| postiz-delete-post | Delete a post by its ID. | posts | postiz | 50 | Yes |
| postiz-delete-post-by-group | Delete all posts in a group by the group identifier. | posts | postiz | 65 | Yes |
| postiz-get-missing-content | Fetch recent content from the provider to match and connect to a post with 'missing' releaseId. | posts | postiz | 65 | Yes |
| postiz-update-release-id | Update the releaseId of a post that currently has its release ID set to 'missing'. | posts | postiz | 60 | Yes |
| postiz-upload-file | Upload a media file using multipart form data. | uploads | postiz | 65 | Yes |
| postiz-upload-from-url | Upload a file from an existing URL. | uploads | postiz | 65 | Yes |
| postiz-get-analytics | Get analytics data for a specific integration/channel. | analytics | postiz | 70 | Yes |
| postiz-get-post-analytics | Get analytics data for a specific published post. | analytics | postiz | 65 | Yes |
| postiz-list-notifications | Get paginated notifications for your organization. | notifications | postiz | 60 | Yes |
| postiz-generate-video | Create AI-generated videos for your posts. | video | postiz | 55 | Yes |
| postiz-video-function | Execute video-related functions like loading available voices. | video | postiz | 65 | Yes |
| get_application_version | Get qBittorrent application version. | app | qbittorrent | 50 | No |
| get_api_version | Get qBittorrent WebAPI version. | app | qbittorrent | 50 | No |
| get_build_info | Get qBittorrent build information (QT, libtorrent, boost, openssl versions, etc.). | app | qbittorrent | 60 | No |
| shutdown_application | Shutdown the qBittorrent application. | app | qbittorrent | 50 | Yes |
| get_preferences | Get all application preferences/settings. | app | qbittorrent | 45 | No |
| set_preferences | Set application preferences/settings. | app | qbittorrent | 45 | Yes |
| get_default_save_path | Get the default save path for torrents. | app | qbittorrent | 55 | No |
| get_torrent_list | Get list of torrents and their information. | torrents | qbittorrent | 55 | No |
| get_torrent_properties | Get generic properties of a torrent. | torrents | qbittorrent | 60 | No |
| get_torrent_trackers | Get trackers for a torrent. | torrents | qbittorrent | 60 | No |
| get_torrent_webseeds | Get web seeds for a torrent. | torrents | qbittorrent | 60 | No |
| get_torrent_contents | Get contents (files) of a torrent. | torrents | qbittorrent | 60 | No |
| get_torrent_piece_states | Get states of all pieces of a torrent (0:not downloaded, 1:downloading, 2:downloaded). | torrents | qbittorrent | 75 | No |
| get_torrent_piece_hashes | Get hashes of all pieces of a torrent. | torrents | qbittorrent | 65 | No |
| pause_torrents | Pause one or more torrents. | torrents | qbittorrent | 60 | Yes |
| resume_torrents | Resume one or more torrents. | torrents | qbittorrent | 60 | No |
| delete_torrents | Delete one or more torrents. | torrents | qbittorrent | 55 | Yes |
| recheck_torrents | Recheck one or more torrents. | torrents | qbittorrent | 60 | No |
| reannounce_torrents | Reannounce one or more torrents. | torrents | qbittorrent | 60 | No |
| edit_tracker | Edit a tracker URL for a torrent. | torrents | qbittorrent | 60 | No |
| remove_trackers | Remove trackers from a torrent. | torrents | qbittorrent | 60 | Yes |
| add_peers | Add peers to one or more torrents. | torrents | qbittorrent | 60 | Yes |
| add_new_torrent | Add a new torrent from URLs. | torrents | qbittorrent | 65 | Yes |
| add_trackers_to_torrent | Add trackers to a torrent. | torrents | qbittorrent | 65 | Yes |
| increase_torrent_priority | Increase priority of one or more torrents. | torrents | qbittorrent | 65 | No |
| decrease_torrent_priority | Decrease priority of one or more torrents. | torrents | qbittorrent | 65 | No |
| top_torrent_priority | Set one or more torrents to maximum priority. | torrents | qbittorrent | 65 | No |
| bottom_torrent_priority | Set one or more torrents to minimum priority. | torrents | qbittorrent | 65 | No |
| set_file_priority | Set priority for one or more files in a torrent. | torrents | qbittorrent | 60 | Yes |
| get_torrent_download_limit | Get download limit for one or more torrents. | torrents | qbittorrent | 65 | No |
| set_torrent_download_limit | Set download limit for one or more torrents. | torrents | qbittorrent | 65 | Yes |
| set_torrent_share_limit | Set share limits for one or more torrents. | torrents | qbittorrent | 65 | Yes |
| get_torrent_upload_limit | Get upload limit for one or more torrents. | torrents | qbittorrent | 65 | Yes |
| set_torrent_upload_limit | Set upload limit for one or more torrents. | torrents | qbittorrent | 65 | Yes |
| set_torrent_location | Set download location for one or more torrents. | torrents | qbittorrent | 60 | Yes |
| set_torrent_name | Rename a torrent. | torrents | qbittorrent | 60 | Yes |
| set_torrent_category | Assign a category to one or more torrents. | torrents | qbittorrent | 60 | Yes |
| get_all_categories | Get all defined categories. | torrents | qbittorrent | 60 | No |
| add_new_category | Add a new category. | torrents | qbittorrent | 65 | Yes |
| edit_category | Edit an existing category. | torrents | qbittorrent | 60 | No |
| remove_categories | Remove one or more categories. | torrents | qbittorrent | 60 | Yes |
| add_torrent_tags | Add tags to one or more torrents. | torrents | qbittorrent | 65 | Yes |
| remove_torrent_tags | Remove tags from one or more torrents. Empty list removes all tags. | torrents | qbittorrent | 75 | Yes |
| get_all_tags | Get all defined tags. | torrents | qbittorrent | 60 | No |
| create_tags | Create new tags. | torrents | qbittorrent | 55 | Yes |
| delete_tags | Delete tags. | torrents | qbittorrent | 50 | Yes |
| set_auto_management | Set automatic torrent management for one or more torrents. | torrents | qbittorrent | 70 | Yes |
| toggle_sequential_download | Toggle sequential download for one or more torrents. | torrents | qbittorrent | 75 | No |
| toggle_first_last_piece_priority | Toggle prioritization of first/last pieces for one or more torrents. | torrents | qbittorrent | 75 | No |
| set_force_start | Set force start for one or more torrents. | torrents | qbittorrent | 60 | Yes |
| set_super_seeding | Set super seeding for one or more torrents. | torrents | qbittorrent | 60 | Yes |
| rename_file | Rename a file within a torrent. | torrents | qbittorrent | 60 | Yes |
| rename_folder | Rename a folder within a torrent. | torrents | qbittorrent | 60 | Yes |
| get_global_transfer_info | Get global transfer info (speeds, total data, DHT nodes, connection status). | transfer | qbittorrent | 75 | No |
| get_speed_limits_mode | Get alternative speed limits state (1 if enabled, 0 otherwise). | transfer | qbittorrent | 75 | No |
| toggle_speed_limits_mode | Toggle alternative speed limits. | transfer | qbittorrent | 65 | No |
| get_global_download_limit | Get global download limit in bytes/second. | transfer | qbittorrent | 65 | No |
| set_global_download_limit | Set global download limit in bytes/second. | transfer | qbittorrent | 65 | Yes |
| get_global_upload_limit | Get global upload limit in bytes/second. | transfer | qbittorrent | 65 | Yes |
| set_global_upload_limit | Set global upload limit in bytes/second. | transfer | qbittorrent | 65 | Yes |
| ban_peers | Ban specific peers. | transfer | qbittorrent | 60 | No |
| add_rss_folder | Add an RSS folder. | rss | qbittorrent | 55 | Yes |
| add_rss_feed | Add an RSS feed. | rss | qbittorrent | 55 | Yes |
| remove_rss_item | Remove an RSS item (folder or feed). | rss | qbittorrent | 55 | Yes |
| move_rss_item | Move or rename an RSS item. | rss | qbittorrent | 55 | Yes |
| get_all_rss_items | Get all RSS items (folders and feeds). | rss | qbittorrent | 55 | No |
| mark_rss_as_read | Mark RSS articles or feeds as read. | rss | qbittorrent | 55 | No |
| refresh_rss_item | Refresh an RSS item (folder or feed). | rss | qbittorrent | 55 | No |
| set_rss_auto_downloading_rule | Set or update an RSS auto-downloading rule. | rss | qbittorrent | 55 | Yes |
| rename_rss_auto_downloading_rule | Rename an RSS auto-downloading rule. | rss | qbittorrent | 55 | Yes |
| remove_rss_auto_downloading_rule | Remove an RSS auto-downloading rule. | rss | qbittorrent | 55 | Yes |
| get_all_rss_auto_downloading_rules | Get all RSS auto-downloading rules. | rss | qbittorrent | 55 | No |
| get_all_rss_articles_matching_rule | Get all articles matching an RSS rule. | rss | qbittorrent | 55 | No |
| start_search | Start a search job. | search | qbittorrent | 50 | Yes |
| stop_search | Stop a running search job. | search | qbittorrent | 50 | Yes |
| get_search_status | Get status of search jobs. | search | qbittorrent | 50 | No |
| get_search_results | Get results of a search job. | search | qbittorrent | 50 | No |
| delete_search | Delete a search job. | search | qbittorrent | 45 | Yes |
| get_search_plugins | Get all search plugins. | search | qbittorrent | 50 | No |
| install_search_plugin | Install one or more search plugins. | search | qbittorrent | 55 | Yes |
| uninstall_search_plugin | Uninstall one or more search plugins. | search | qbittorrent | 55 | Yes |
| enable_search_plugin | Enable or disable one or more search plugins. | search | qbittorrent | 55 | Yes |
| update_search_plugins | Update all installed search plugins. | search | qbittorrent | 50 | Yes |
| get_main_log | Get the main qBittorrent log. | log | qbittorrent | 50 | No |
| get_peer_log | Get the peer log. | log | qbittorrent | 50 | No |
| get_main_data | Get main sync data (torrents, categories, tags, server state). | sync | qbittorrent | 60 | No |
| get_torrent_peers_data | Get sync data for torrent peers. | sync | qbittorrent | 55 | No |
| git_action | Executes an arbitrary Git command. | devops_engineer, project_manager, workspace_management | repository-manager | 75 | Yes |
| get_workspace_projects | Returns a list of project URLs defined in the workspace. | devops_engineer, git_operations, project_management, workspace_management | repository-manager | 85 | No |
| clone_projects | Clones repositories. Defaults to all in workspace.yml. | devops_engineer, git_operations, project_manager | repository-manager | 85 | No |
| pull_projects | Pulls updates for all projects in the workspace. | devops_engineer, git_operations, project_manager | repository-manager | 75 | No |
| setup_workspace | Sets up the entire workspace, clones repos, and organizes subdirectories. | workspace_management | repository-manager | 70 | Yes |
| install_projects | Bulk installs Python projects defined in the workspace. | workspace_management | repository-manager | 70 | Yes |
| build_projects | Bulk builds Python projects defined in the workspace. | workspace_management | repository-manager | 70 | No |
| validate_projects | Bulk validates agent/MCP servers in the workspace. | workspace_management | repository-manager | 60 | No |
| generate_workspace_template | Generates a new workspace.yml template. | workspace_management | repository-manager | 65 | No |
| save_workspace_config | Saves a WorkspaceConfig to YAML. | workspace_management | repository-manager | 65 | No |
| maintain_workspace | Runs the maintenance lifecycle across all projects in the workspace. | workspace_management | repository-manager | 70 | No |
| graph_build | Builds or synchronizes the Hybrid Workspace Graph (NetworkX + Ladybug). | graph_intelligence | repository-manager | 70 | No |
| graph_query | Queries the Hybrid Graph using vector similarity or Cypher structure. | graph_intelligence | repository-manager | 70 | No |
| graph_path | Finds the shortest path between two symbols across the workspace graph. | graph_intelligence | repository-manager | 70 | No |
| graph_status | Returns the current status of the workspace graph. | graph_intelligence | repository-manager | 60 | No |
| graph_reset | Purges the graph database and forces a clean rebuild. | graph_intelligence | repository-manager | 70 | Yes |
| graph_impact | Calculates multi-repo impact for a symbol using the GraphEngine. | graph_intelligence | repository-manager | 70 | No |
| get_workspace_tree | Generates an ASCII tree of the workspace structure. | visualization | repository-manager | 70 | No |
| get_workspace_mermaid | Generates a Mermaid diagram of the workspace structure. | visualization | repository-manager | 70 | No |
| generate_agents_documentation | Generates an AGENTS.md catalog of discovered projects. | visualization | repository-manager | 75 | No |
| web_search | Perform web searches using SearXNG, a privacy-respecting metasearch engine. Returns relevant web content with customizable parameters.<br/>Returns a Dictionary response with status, message, data (search results), and error if any. | search | searxng-mcp | 70 | No |
| workflow_to_mermaid | Generate a UNIFIED Mermaid diagram + rich Markdown report for multiple ServiceNow flows. Optional: leave flow_identifiers empty to fetch ALL active flows up to 1000 limit. Unrelated flow groups are split into separate safe-to-render diagram blocks. By default saves a polished .md file. | flows | servicenow-api | 70 | No |
| get_application | Retrieves details of a specific application from a ServiceNow instance by its unique identifier. | application | servicenow-api | 65 | No |
| get_cmdb | Fetches a specific Configuration Management Database (CMDB) record from a ServiceNow instance using its unique identifier. | cmdb | servicenow-api | 65 | No |
| delete_cmdb_relation | Deletes the relation for the specified configuration item (CI). | cmdb | servicenow-api | 60 | Yes |
| get_cmdb_instances | Returns the available configuration items (CI) for a specified CMDB class. | cmdb | servicenow-api | 60 | No |
| get_cmdb_instance | Returns attributes and relationship information for a specified CI record. | cmdb | servicenow-api | 60 | No |
| create_cmdb_instance | Creates a single configuration item (CI). | cmdb | servicenow-api | 50 | Yes |
| update_cmdb_instance | Updates the specified CI record (PUT). | cmdb | servicenow-api | 50 | Yes |
| patch_cmdb_instance | Replaces attributes in the specified CI record (PATCH). | cmdb | servicenow-api | 65 | Yes |
| create_cmdb_relation | Adds an inbound and/or outbound relation to the specified CI. | cmdb | servicenow-api | 60 | Yes |
| ingest_cmdb_data | Inserts records into the Import Set table associated with the data source. | cmdb | servicenow-api | 65 | Yes |
| batch_install_result | Retrieves the result of a batch installation process in ServiceNow by result ID. | cicd | servicenow-api | 65 | Yes |
| instance_scan_progress | Gets the progress status of an instance scan in ServiceNow by progress ID. | cicd | servicenow-api | 65 | No |
| progress | Retrieves the progress status of a specified process in ServiceNow by progress ID. | cicd | servicenow-api | 55 | No |
| batch_install | Initiates a batch installation of specified packages in ServiceNow with optional notes. | cicd | servicenow-api | 60 | Yes |
| batch_rollback | Performs a rollback of a batch installation in ServiceNow using the rollback ID. | cicd | servicenow-api | 60 | No |
| app_repo_install | Installs an application from a repository in ServiceNow with specified parameters. | cicd | servicenow-api | 65 | Yes |
| app_repo_publish | Publishes an application to a repository in ServiceNow with development notes and version. | cicd | servicenow-api | 65 | No |
| app_repo_rollback | Rolls back an application to a previous version in ServiceNow by sys_id, scope, and version. | cicd | servicenow-api | 65 | No |
| full_scan | Initiates a full scan of the ServiceNow instance. | cicd | servicenow-api | 50 | No |
| point_scan | Performs a targeted scan on a specific instance and table in ServiceNow. | cicd | servicenow-api | 60 | No |
| combo_suite_scan | Executes a scan on a combination of suites in ServiceNow by combo sys_id. | cicd | servicenow-api | 65 | No |
| suite_scan | Runs a scan on a specified suite with a list of sys_ids and scan type in ServiceNow. | cicd | servicenow-api | 60 | No |
| activate_plugin | Activates a specified plugin in ServiceNow by plugin ID. | plugins | servicenow-api | 70 | Yes |
| rollback_plugin | Rolls back a specified plugin in ServiceNow to its previous state by plugin ID. | plugins | servicenow-api | 70 | No |
| apply_remote_source_control_changes | Applies changes from a remote source control branch to a ServiceNow application. | source_control | servicenow-api | 75 | No |
| import_repository | Imports a repository into ServiceNow with specified credentials and branch. | source_control | servicenow-api | 70 | No |
| run_test_suite | Executes a test suite in ServiceNow with specified browser and OS configurations. | testing | servicenow-api | 70 | No |
| update_set_create | Creates a new update set in ServiceNow with a given name, scope, and description. | update_sets | servicenow-api | 60 | Yes |
| update_set_retrieve | Retrieves an update set from a source instance in ServiceNow with optional preview and cleanup. | update_sets | servicenow-api | 65 | Yes |
| update_set_preview | Previews an update set in ServiceNow by its remote sys_id. | update_sets | servicenow-api | 65 | Yes |
| update_set_commit | Commits an update set in ServiceNow with an option to force commit. | update_sets | servicenow-api | 65 | Yes |
| update_set_commit_multiple | Commits multiple update sets in ServiceNow in the specified order. | update_sets | servicenow-api | 70 | Yes |
| update_set_back_out | Backs out an update set in ServiceNow with an option to rollback installations. | update_sets | servicenow-api | 70 | Yes |
| batch_request | Sends multiple REST API requests in a single call. | batch | servicenow-api | 50 | No |
| get_change_requests | Retrieves change requests from ServiceNow with optional filtering and pagination. | change_management | servicenow-api | 70 | No |
| get_change_request_nextstate | Gets the next state for a specific change request in ServiceNow. | change_management | servicenow-api | 75 | No |
| get_change_request_schedule | Retrieves the schedule for a change request based on a Configuration Item (CI) in ServiceNow. | change_management | servicenow-api | 75 | No |
| get_change_request_tasks | Fetches tasks associated with a change request in ServiceNow with optional filtering. | change_management | servicenow-api | 75 | No |
| get_change_request | Retrieves details of a specific change request in ServiceNow by sys_id and type. | change_management | servicenow-api | 70 | No |
| get_change_request_ci | Gets Configuration Items (CIs) associated with a change request in ServiceNow. | change_management | servicenow-api | 70 | No |
| get_change_request_conflict | Checks for conflicts in a change request in ServiceNow. | change_management | servicenow-api | 75 | No |
| get_standard_change_request_templates | Retrieves standard change request templates from ServiceNow with optional filtering. | change_management | servicenow-api | 75 | No |
| get_change_request_models | Fetches change request models from ServiceNow with optional filtering and type. | change_management | servicenow-api | 75 | No |
| get_standard_change_request_model | Retrieves a specific standard change request model in ServiceNow by sys_id. | change_management | servicenow-api | 75 | No |
| get_standard_change_request_template | Gets a specific standard change request template in ServiceNow by sys_id. | change_management | servicenow-api | 75 | No |
| get_change_request_worker | Retrieves details of a change request worker in ServiceNow by sys_id. | change_management | servicenow-api | 75 | No |
| create_change_request | Creates a new change request in ServiceNow with specified details and type. | change_management | servicenow-api | 70 | Yes |
| create_change_request_task | Creates a task for a change request in ServiceNow with provided details. | change_management | servicenow-api | 75 | Yes |
| create_change_request_ci_association | Associates Configuration Items (CIs) with a change request in ServiceNow. | change_management | servicenow-api | 75 | Yes |
| calculate_standard_change_request_risk | Calculates the risk for a standard change request in ServiceNow. | change_management | servicenow-api | 75 | No |
| check_change_request_conflict | Checks for conflicts in a change request in ServiceNow. | change_management | servicenow-api | 75 | No |
| refresh_change_request_impacted_services | Refreshes the impacted services for a change request in ServiceNow. | change_management | servicenow-api | 75 | No |
| approve_change_request | Approves or rejects a change request in ServiceNow by setting its state. | change_management | servicenow-api | 75 | Yes |
| update_change_request | Updates a change request in ServiceNow with new details and type. | change_management | servicenow-api | 70 | Yes |
| update_change_request_first_available | Updates a change request to the first available state in ServiceNow. | change_management | servicenow-api | 75 | Yes |
| update_change_request_task | Updates a task for a change request in ServiceNow with new details. | change_management | servicenow-api | 75 | Yes |
| delete_change_request | Deletes a change request from ServiceNow by sys_id and type. | change_management | servicenow-api | 70 | Yes |
| delete_change_request_task | Deletes a task associated with a change request in ServiceNow. | change_management | servicenow-api | 75 | Yes |
| delete_change_request_conflict_scan | Deletes a conflict scan for a change request in ServiceNow. | change_management | servicenow-api | 75 | Yes |
| check_ci_lifecycle_compat_actions | Determines whether two specified CI actions are compatible. | cilifecycle | servicenow-api | 75 | No |
| register_ci_lifecycle_operator | Registers an operator for a non-workflow user. | cilifecycle | servicenow-api | 65 | No |
| unregister_ci_lifecycle_operator | Unregisters an operator for non-workflow users. | cilifecycle | servicenow-api | 65 | No |
| check_devops_change_control | Checks if the orchestration task is under change control. | devops | servicenow-api | 65 | No |
| register_devops_artifact | Enables orchestration tools to register artifacts into a ServiceNow instance. | devops | servicenow-api | 65 | No |
| get_import_set | Retrieves details of a specific import set record from a ServiceNow instance. | import_sets | servicenow-api | 65 | Yes |
| insert_import_set | Inserts a new record into a specified import set on a ServiceNow instance. | import_sets | servicenow-api | 70 | Yes |
| insert_multiple_import_sets | Inserts multiple records into a specified import set on a ServiceNow instance. | import_sets | servicenow-api | 75 | Yes |
| get_incidents | Retrieves incident records from a ServiceNow instance, optionally by specific incident ID. | incidents | servicenow-api | 65 | No |
| create_incident | Creates a new incident record on a ServiceNow instance with provided details. | incidents | servicenow-api | 65 | Yes |
| get_knowledge_articles | Get all Knowledge Base articles from a ServiceNow instance. | knowledge_management | servicenow-api | 70 | No |
| get_knowledge_article | Get a specific Knowledge Base article from a ServiceNow instance. | knowledge_management | servicenow-api | 70 | No |
| get_knowledge_article_attachment | Get a Knowledge Base article attachment from a ServiceNow instance. | knowledge_management | servicenow-api | 75 | No |
| get_featured_knowledge_article | Get featured Knowledge Base articles from a ServiceNow instance. | knowledge_management | servicenow-api | 75 | No |
| get_most_viewed_knowledge_articles | Get most viewed Knowledge Base articles from a ServiceNow instance. | knowledge_management | servicenow-api | 75 | No |
| delete_table_record | Delete a record from the specified table on a ServiceNow instance. | table_api | servicenow-api | 70 | Yes |
| get_table | Get records from the specified table on a ServiceNow instance. | table_api | servicenow-api | 65 | No |
| get_table_record | Get a specific record from the specified table on a ServiceNow instance. | table_api | servicenow-api | 70 | No |
| patch_table_record | Partially update a record in the specified table on a ServiceNow instance. | table_api | servicenow-api | 75 | Yes |
| update_table_record | Fully update a record in the specified table on a ServiceNow instance. | table_api | servicenow-api | 70 | Yes |
| add_table_record | Add a new record to the specified table on a ServiceNow instance. | table_api | servicenow-api | 75 | Yes |
| refresh_auth_token | Refreshes the authentication token for the ServiceNow client. | auth | servicenow-api | 65 | No |
| api_request | Make a custom API request to a ServiceNow instance. | custom_api | servicenow-api | 70 | No |
| send_email | Sends an email via ServiceNow. | email | servicenow-api | 50 | No |
| get_data_classification | Retrieves data classification information. | data_classification | servicenow-api | 60 | No |
| get_attachment | Retrieves attachment metadata. | attachment | servicenow-api | 55 | No |
| upload_attachment | Uploads an attachment to a record. | attachment | servicenow-api | 60 | Yes |
| delete_attachment | Deletes an attachment. | attachment | servicenow-api | 55 | Yes |
| get_stats | Retrieves aggregate statistics for a table. | aggregate | servicenow-api | 55 | No |
| get_activity_subscriptions | Retrieves activity subscriptions. | activity_subscriptions | servicenow-api | 60 | No |
| get_account | Retrieves CSM account information. | account | servicenow-api | 55 | No |
| get_hr_profile | Retrieves HR profile information. | hr | servicenow-api | 45 | No |
| metricbase_insert | Inserts time series data into MetricBase. | metricbase | servicenow-api | 60 | Yes |
| check_service_qualification | Creates a technical service qualification request. | service_qualification | servicenow-api | 65 | No |
| get_service_qualification | Retrieves a service qualification request. | service_qualification | servicenow-api | 60 | No |
| process_service_qualification_result | Processes a service qualification result. | service_qualification | servicenow-api | 65 | No |
| insert_cost_plans | Creates cost plans. | ppm | servicenow-api | 55 | Yes |
| insert_project_tasks | Creates a project and associated project tasks. | ppm | servicenow-api | 55 | Yes |
| get_product_inventory | Retrieves product inventory. | product_inventory | servicenow-api | 60 | No |
| delete_product_inventory | Deletes a product inventory record. | product_inventory | servicenow-api | 60 | Yes |
| add_watermark | Add a watermark to a PDF file. | PDF | stirlingpdf-agent | 50 | Yes |
| install_applications | Installs applications using the native package manager with Snap fallback. | system | systems-manager | 60 | Yes |
| update | Updates the system and applications. | system | systems-manager | 40 | Yes |
| clean | Cleans system resources (e.g., trash/recycle bin). | system | systems-manager | 45 | No |
| optimize | Optimizes system resources (e.g., autoremove, defrag). | system | systems-manager | 55 | No |
| install_python_modules | Installs Python modules via pip. | system | systems-manager | 55 | Yes |
| install_fonts | Installs specified Nerd Fonts or all available fonts if 'all' is specified. | system | systems-manager | 60 | Yes |
| get_os_statistics | Retrieves operating system statistics. | system | systems-manager | 45 | No |
| get_hardware_statistics | Retrieves hardware statistics. | system | systems-manager | 50 | No |
| search_package | Searches for packages in the system package manager repositories. | system | systems-manager | 60 | No |
| get_package_info | Gets detailed information about a specific package. | system | systems-manager | 60 | No |
| list_installed_packages | Lists all installed packages on the system. | system | systems-manager | 50 | Yes |
| list_upgradable_packages | Lists all packages that have updates available. | system | systems-manager | 50 | No |
| system_health_check | Performs a comprehensive system health check including CPU, memory, disk, swap, and top processes. | system | systems-manager | 65 | No |
| get_uptime | Gets system uptime and boot time. | system | systems-manager | 45 | No |
| list_env_vars | Lists all environment variables on the system. | system | systems-manager | 50 | No |
| get_env_var | Gets the value of a specific environment variable. | system | systems-manager | 50 | No |
| clean_temp_files | Cleans temporary files from system temp directories. | system | systems-manager | 65 | No |
| clean_package_cache | Cleans the package manager cache to free disk space. | system | systems-manager | 65 | No |
| list_windows_features | Lists all Windows features and their status (Windows only). | system_management, windows | systems-manager | 80 | No |
| enable_windows_features | Enables specified Windows features (Windows only). | system_management, windows | systems-manager | 75 | Yes |
| disable_windows_features | Disables specified Windows features (Windows only). | system_management, windows | systems-manager | 85 | Yes |
| add_repository | Adds an upstream repository to the package manager repository list (Linux only). | linux, system_management | systems-manager | 80 | Yes |
| install_local_package | Installs a local Linux package file using the appropriate tool (dpkg/rpm/dnf/zypper/pacman). (Linux only) | linux, system_management | systems-manager | 95 | Yes |
| run_command | Runs a command on the host. Can run elevated for administrator or root privileges. | linux, system_management | systems-manager | 75 | Yes |
| text_editor | View and edit files on the local filesystem. | files, text_editor | systems-manager | 70 | No |
| list_services | Lists all system services with their current status. | service | systems-manager | 65 | No |
| get_service_status | Gets the status of a specific system service. | service | systems-manager | 60 | No |
| start_service | Starts a system service. | service | systems-manager | 60 | Yes |
| stop_service | Stops a system service. | service | systems-manager | 60 | Yes |
| restart_service | Restarts a system service. | service | systems-manager | 60 | Yes |
| enable_service | Enables a system service to start at boot. | service | systems-manager | 60 | Yes |
| disable_service | Disables a system service from starting at boot. | service | systems-manager | 60 | Yes |
| list_processes | Lists all running processes with PID, name, CPU%, memory%, and status. | process | systems-manager | 65 | No |
| get_process_info | Gets detailed information about a specific process by PID. | process | systems-manager | 70 | No |
| kill_process | Kills a process by PID. Default signal is SIGTERM (15), use 9 for SIGKILL. | process | systems-manager | 70 | Yes |
| list_network_interfaces | Lists all network interfaces with IP addresses, speed, and MTU. | network | systems-manager | 70 | No |
| list_open_ports | Lists all open/listening network ports with associated PIDs. | network | systems-manager | 70 | No |
| ping_host | Pings a host and returns the results. | network | systems-manager | 60 | No |
| dns_lookup | Performs a DNS lookup for a hostname and returns resolved IP addresses. | network | systems-manager | 70 | No |
| list_disks | Lists all disk partitions with mount points and usage statistics. | disk | systems-manager | 55 | No |
| get_disk_usage | Gets disk usage statistics for a specific path. | disk | systems-manager | 50 | No |
| get_disk_space_report | Gets a report of the largest directories under a path. | disk | systems-manager | 65 | No |
| list_users | Lists all system users with UID, GID, home directory, and shell. | user | systems-manager | 55 | No |
| list_groups | Lists all system groups with GID and members. | user | systems-manager | 45 | No |
| get_system_logs | Gets system logs from journalctl (Linux) or Event Log (Windows). | log | systems-manager | 60 | No |
| tail_log_file | Reads the last N lines of a log file. | log | systems-manager | 55 | No |
| list_cron_jobs | Lists cron jobs (Linux) or scheduled tasks (Windows). | cron | systems-manager | 60 | No |
| add_cron_job | Adds a new cron job (Linux only). | cron | systems-manager | 55 | Yes |
| remove_cron_job | Removes cron jobs matching a pattern (Linux only). | cron | systems-manager | 55 | Yes |
| get_firewall_status | Gets the current firewall status (ufw/firewalld/iptables on Linux, netsh on Windows). | firewall_management | systems-manager | 70 | No |
| list_firewall_rules | Lists all firewall rules. | firewall_management | systems-manager | 60 | No |
| add_firewall_rule | Adds a firewall rule using the detected firewall tool. | firewall_management | systems-manager | 75 | Yes |
| remove_firewall_rule | Removes a firewall rule using the detected firewall tool. | firewall_management | systems-manager | 75 | Yes |
| list_ssh_keys | Lists all SSH keys in the user's ~/.ssh directory. | ssh_management | systems-manager | 60 | No |
| generate_ssh_key | Generates a new SSH key pair. | ssh_management | systems-manager | 65 | No |
| add_authorized_key | Adds a public key to the authorized_keys file. | ssh_management | systems-manager | 65 | Yes |
| list_files | Lists files and directories in a path. | filesystem | systems-manager | 55 | No |
| search_files | Searches for files matching a pattern. | filesystem | systems-manager | 60 | No |
| grep_files | Searches for text content inside files (like grep). | filesystem | systems-manager | 70 | No |
| manage_file | Creates, updates, deletes, or reads a file. | filesystem | systems-manager | 60 | No |
| add_shell_alias | Adds an alias to the user's shell profile. | shell | systems-manager | 55 | Yes |
| install_uv | Installs uv (Python package manager). | python | systems-manager | 45 | Yes |
| create_python_venv | Creates a Python virtual environment using uv. | python | systems-manager | 50 | Yes |
| install_python_package_uv | Installs a Python package using uv pip. | python | systems-manager | 55 | Yes |
| install_nvm | Installs NVM (Node Version Manager). | nodejs | systems-manager | 50 | Yes |
| install_node | Installs a Node.js version using NVM. | nodejs | systems-manager | 50 | Yes |
| use_node | Switches the active Node.js version using NVM. | nodejs | systems-manager | 50 | No |
| list_hosts | List all managed hosts in the inventory. | host_management | tunnel-manager-mcp | 55 | No |
| add_host | Add a new host to the managed inventory. | host_management | tunnel-manager-mcp | 60 | Yes |
| remove_host | Remove a host from the managed inventory. | host_management | tunnel-manager-mcp | 60 | Yes |
| run_command_on_remote_host | Run shell command on remote host. Expected return object type: dict | remote_access | tunnel-manager-mcp | 75 | Yes |
| send_file_to_remote_host | Upload file to remote host. Expected return object type: dict | remote_access | tunnel-manager-mcp | 75 | No |
| receive_file_from_remote_host | Download file from remote host. Expected return object type: dict | remote_access | tunnel-manager-mcp | 75 | No |
| check_ssh_server | Check SSH server status. Expected return object type: dict | remote_access | tunnel-manager-mcp | 75 | No |
| test_key_auth | Test key-based auth. Expected return object type: dict | remote_access | tunnel-manager-mcp | 75 | No |
| setup_passwordless_ssh | Setup passwordless SSH. Expected return object type: dict | remote_access | tunnel-manager-mcp | 75 | Yes |
| copy_ssh_config | Copy SSH config to remote host. Expected return object type: dict | remote_access | tunnel-manager-mcp | 75 | No |
| rotate_ssh_key | Rotate SSH key on remote host. Expected return object type: dict | remote_access | tunnel-manager-mcp | 75 | Yes |
| remove_host_key | Remove host key from known_hosts. Expected return object type: dict | remote_access | tunnel-manager-mcp | 75 | Yes |
| configure_key_auth_on_inventory | Setup passwordless SSH for all hosts in group. Expected return object type: dict | remote_access | tunnel-manager-mcp | 75 | No |
| run_command_on_inventory | Run command on all hosts in group. Expected return object type: dict | remote_access | tunnel-manager-mcp | 70 | Yes |
| copy_ssh_config_on_inventory | Copy SSH config to all hosts in YAML group. Expected return object type: dict | remote_access | tunnel-manager-mcp | 75 | No |
| rotate_ssh_key_on_inventory | Rotate SSH keys for all hosts in YAML group. Expected return object type: dict | remote_access | tunnel-manager-mcp | 75 | Yes |
| send_file_to_inventory | Upload a file to all hosts in the specified inventory group. Expected return object type: dict | remote_access | tunnel-manager-mcp | 75 | No |
| receive_file_from_inventory | Download a file from all hosts in the specified inventory group. Expected return object type: dict | remote_access | tunnel-manager-mcp | 75 | No |
| uptime-kuma-get-monitors | Get all monitors | uptime | uptime | 40 | No |
| uptime-kuma-get-monitor | Get a specific monitor by ID | uptime | uptime | 40 | No |
| uptime-kuma-add-monitor | Add a new monitor | uptime | uptime | 40 | Yes |
| uptime-kuma-edit-monitor | Edit an existing monitor | uptime | uptime | 40 | No |
| uptime-kuma-delete-monitor | Delete a monitor | uptime | uptime | 40 | Yes |
| uptime-kuma-pause-monitor | Pause a monitor | uptime | uptime | 35 | Yes |
| uptime-kuma-resume-monitor | Resume a monitor | uptime | uptime | 40 | No |
| uptime-kuma-get-status | Get status for a specific monitor | uptime | uptime | 40 | No |
| uptime-kuma-get-uptime | Get uptime percentages for monitors | uptime | uptime | 40 | No |
| create_collection | Creates a new collection or retrieves an existing one in the vector database. | collection_management | vector-mcp | 65 | Yes |
| add_documents | Adds documents to an existing collection in the vector database.<br/>This can be used to extend collections with additional documents | collection_management | vector-mcp | 80 | Yes |
| delete_collection | Deletes a collection from the vector database. | collection_management | vector-mcp | 55 | Yes |
| list_collections | Lists all collections in the vector database. | collection_management | vector-mcp | 55 | No |
| semantic_search | Retrieves and gathers related knowledge from the vector database instance using the question variable.<br/>This can be used as a primary source of knowledge retrieval.<br/>It will return relevant text(s) which should be parsed for the most<br/>relevant information pertaining to the question and summarized as the final output | search | vector-mcp | 70 | No |
| lexical_search | This is a lexical or term based search that retrieves and gathers related knowledge from the database instance using the question variable via BM25.<br/>This provides a complementary search method to vector search, useful for exact keyword matching. | search | vector-mcp | 70 | No |
| search | Performs a hybrid search combining semantic (vector) and lexical (BM25) methods.<br/>Retrieves results from both, merges them using weighted Reciprocal Rank Fusion (RRF),<br/>and returns the top combined results. | search | vector-mcp | 65 | No |
| get_routines | List all workout routines for the authenticated user. | Routine | wger-agent | 65 | No |
| get_routine | Get a specific routine by ID. | Routine | wger-agent | 55 | No |
| create_routine | Create a new workout routine. | Routine | wger-agent | 55 | Yes |
| delete_routine | Delete a routine. | Routine | wger-agent | 55 | Yes |
| get_days | List workout days. Filter by routine with routine=<id>. | Routine | wger-agent | 65 | No |
| create_day | Create a workout day in a routine. | Routine | wger-agent | 55 | Yes |
| delete_day | Delete a workout day. | Routine | wger-agent | 55 | Yes |
| get_slots | List exercise slots (sets) in workout days. | Routine | wger-agent | 55 | No |
| create_slot | Create an exercise slot (set) in a day. | Routine | wger-agent | 55 | Yes |
| create_slot_entry | Add an exercise to a slot. | Routine | wger-agent | 60 | Yes |
| get_templates | List user's workout templates. | Routine | wger-agent | 55 | No |
| get_public_templates | List publicly shared workout templates. | Routine | wger-agent | 60 | No |
| create_weight_config | Create a weight progression config for a slot entry. Controls how weight progresses across iterations. | RoutineConfig | wger-agent | 80 | Yes |
| get_weight_configs | List weight progression configs. | RoutineConfig | wger-agent | 60 | No |
| create_repetitions_config | Create a repetitions progression config for a slot entry. | RoutineConfig | wger-agent | 70 | Yes |
| get_repetitions_configs | List repetitions configs. | RoutineConfig | wger-agent | 60 | No |
| create_sets_config | Create a sets count progression config for a slot entry. | RoutineConfig | wger-agent | 70 | Yes |
| create_rest_config | Create a rest time progression config for a slot entry. | RoutineConfig | wger-agent | 70 | Yes |
| create_rir_config | Create a RiR (Reps in Reserve) progression config for a slot entry. | RoutineConfig | wger-agent | 70 | Yes |
| get_exercises | List exercises from the exercise database. Supports filters: language, category, muscles, equipment. | Exercise | wger-agent | 65 | No |
| get_exercise_info | Get detailed exercise info including translations, images, muscles worked, and equipment. | Exercise | wger-agent | 70 | No |
| search_exercises | Search exercises by name. Returns exercise info entries matching the search term. | Exercise | wger-agent | 70 | No |
| get_exercise_categories | List exercise categories (e.g., Arms, Legs, Chest, Back, etc.). | Exercise | wger-agent | 70 | No |
| get_equipment | List available equipment (e.g., Barbell, Dumbbell, Kettlebell, etc.). | Exercise | wger-agent | 65 | No |
| get_muscles | List muscles (e.g., Biceps, Pectoralis, Quadriceps, etc.). | Exercise | wger-agent | 65 | No |
| get_exercise_images | List exercise images. Filter by exercise with exercise_base=<id>. | Exercise | wger-agent | 70 | No |
| get_variations | List exercise variation groups. | Exercise | wger-agent | 55 | No |
| get_workout_sessions | List workout sessions. | Workout | wger-agent | 60 | No |
| get_workout_session | Get a specific workout session. | Workout | wger-agent | 60 | No |
| create_workout_session | Create a workout session. Impression: 1=Discomfort, 2=Could be better, 3=Neutral, 4=Good, 5=Perfect. | Workout | wger-agent | 70 | Yes |
| delete_workout_session | Delete a workout session. | Workout | wger-agent | 60 | Yes |
| get_workout_logs | List workout log entries. | Workout | wger-agent | 60 | No |
| create_workout_log | Log a set performed during a workout (exercise, weight, reps, date). | Workout | wger-agent | 70 | Yes |
| delete_workout_log | Delete a workout log entry. | Workout | wger-agent | 60 | Yes |
| get_nutrition_plans | List nutrition plans. | Nutrition | wger-agent | 60 | No |
| get_nutrition_plan_info | Get detailed nutrition plan with meals, items, and nutritional totals. | Nutrition | wger-agent | 75 | No |
| create_nutrition_plan | Create a nutrition plan with optional macro goals. | Nutrition | wger-agent | 60 | Yes |
| delete_nutrition_plan | Delete a nutrition plan. | Nutrition | wger-agent | 60 | Yes |
| create_meal | Create a meal in a nutrition plan. | Nutrition | wger-agent | 55 | Yes |
| create_meal_item | Add an ingredient to a meal. | Nutrition | wger-agent | 60 | Yes |
| get_ingredients | List/search ingredients from the food database. | Nutrition | wger-agent | 55 | No |
| get_ingredient_info | Get detailed ingredient info including nutritional values and weight units. | Nutrition | wger-agent | 70 | No |
| get_nutrition_diary | List nutrition diary entries. | Nutrition | wger-agent | 60 | No |
| log_nutrition | Log a nutrition diary entry (what was actually eaten). | Nutrition | wger-agent | 70 | No |
| get_weight_entries | List body weight entries over time. | Body | wger-agent | 50 | No |
| log_body_weight | Log a body weight entry. | Body | wger-agent | 55 | No |
| delete_weight_entry | Delete a body weight entry. | Body | wger-agent | 50 | Yes |
| get_measurements | List body measurements (biceps, chest, waist, etc.). | Body | wger-agent | 55 | No |
| log_measurement | Log a body measurement. | Body | wger-agent | 50 | No |
| get_measurement_categories | List measurement categories (e.g., Biceps, Chest, Waist). | Body | wger-agent | 60 | No |
| create_measurement_category | Create a new measurement category. | Body | wger-agent | 50 | Yes |
| get_gallery | List progress gallery photos. | Body | wger-agent | 45 | No |
| get_user_profile | Get the authenticated user's profile (age, height, gender, etc.). | User | wger-agent | 60 | No |
| get_user_statistics | Get user statistics (workout counts, etc.). | User | wger-agent | 50 | No |
| get_user_trophies | List user's earned trophies/achievements. | User | wger-agent | 50 | No |
| get_languages | List available languages. | User | wger-agent | 45 | No |
| get_repetition_units | List repetition unit settings (e.g., Repetitions, Until failure, etc.). | User | wger-agent | 60 | No |
| get_weight_unit_settings | List weight unit settings (kg, lb, plates, etc.). | User | wger-agent | 55 | Yes |
