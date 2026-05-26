from __future__ import annotations

"""Default specialist package catalog for the Agent Registry.

CONCEPT:ORCH-1.2

Ships 38 packages out-of-the-box:
- 33 Knuckles-Team packages (4 OS subsystems + 2 OS services + 27 adaptive_agent_router)
- 5 curated community MCP servers

OS subsystems are auto-installed on first ``AgentRegistry`` init.
All other packages are placed in ``available/`` for on-demand install.

Install commands default to ``uvx`` and versions are ``*`` (latest).
"""


import functools

from .registry_cli import ContainerConfig, SpecialistPackage


def _kt_package(
    name: str,
    description: str,
    mcp_entry: str,
    tags: list[str],
    *,
    pip_name: str = "",
    env: dict[str, str] | None = None,
    tools: list[str] | None = None,
    image: str = "",
) -> SpecialistPackage:
    """Helper to build a Knuckles-Team package entry."""
    pip_name = pip_name or name
    return SpecialistPackage(
        name=name,
        version="*",
        description=description,
        mcp_config={
            "command": "uvx",
            "args": ["--from", pip_name, mcp_entry],
            "env": env or {},
        },
        tools=tools or [],
        tags=["knuckles-team", *tags],
        container_config=ContainerConfig(
            image=image or f"docker.io/knucklessg1/{name}:latest",
            compose_ref=f"agents/{name}/compose.yml",
        ),
    )


def _community_package(
    name: str,
    description: str,
    command: str,
    args: list[str],
    tags: list[str],
    *,
    env: dict[str, str] | None = None,
) -> SpecialistPackage:
    """Helper to build a community MCP package entry."""
    return SpecialistPackage(
        name=name,
        version="*",
        description=description,
        mcp_config={
            "command": command,
            "args": args,
            "env": env or {},
        },
        tags=["community", *tags],
    )


@functools.lru_cache(maxsize=1)
def get_default_catalog() -> tuple[SpecialistPackage, ...]:
    """Return the full default catalog as an immutable tuple.

    Results are cached so repeated calls are free.

    Returns:
        Tuple of all default ``SpecialistPackage`` definitions.
    """
    packages: list[SpecialistPackage] = []

    # ── OS Subsystems (auto-installed) ────────────────────────────────

    packages.append(
        _kt_package(
            "systems-manager",
            "Systems Manager — host OS operations, package management, and Agent OS MCP wrappers.",
            "systems-manager-mcp",
            ["os_subsystem", "devops", "system"],
            env={"MEMENTO_ENABLED": "true"},
            tools=[
                "identity_issue",
                "identity_verify",
                "policy_list",
                "policy_update",
                "specialist_install",
                "specialist_list",
                "agent_health_stats",
                "watchdog_check",
                "maintenance_run",
            ],
        )
    )

    packages.append(
        _kt_package(
            "container-manager-mcp",
            "Container Manager — Docker, Podman, Compose, and Swarm lifecycle with multi-endpoint support.",
            "container-manager-mcp",
            ["os_subsystem", "devops", "containers"],
            env={"MEMENTO_ENABLED": "true"},
            tools=[
                "list_containers",
                "run_container",
                "stop_container",
                "compose_up",
                "compose_down",
                "list_images",
                "pull_image",
                "init_swarm",
                "deploy_specialist_container",
            ],
        )
    )

    packages.append(
        _kt_package(
            "tunnel-manager",
            "Tunnel Manager — SSH tunnels, remote command execution, file transfer, network topology, and security audit.",
            "tunnel-manager-mcp",
            ["os_subsystem", "network", "ssh", "security"],
            tools=[
                "list_hosts",
                "add_host",
                "run_command_on_remote_host",
                "send_file_to_remote_host",
                "setup_passwordless_ssh",
                "network_topology",
                "security_audit",
                "discover_services",
            ],
        )
    )

    packages.append(
        _kt_package(
            "repository-manager",
            "Repository Manager — git workspace management, project install/build/validate, dependency graphs.",
            "repository-manager-mcp",
            ["os_subsystem", "devops", "git", "workspace"],
            tools=[
                "git_action",
                "get_workspace_projects",
                "clone_projects",
                "pull_projects",
                "push_projects",
                "validate_projects",
                "maintain_workspace",
                "graph_build",
                "graph_query",
            ],
        )
    )

    # ── OS Services (available, deploy-on-demand) ─────────────────────

    packages.append(
        _kt_package(
            "searxng-mcp",
            "SearXNG — privacy-respecting metasearch engine. Defaults to public instance, self-deployable.",
            "searxng-mcp",
            ["os_service", "search", "internet"],
            env={"USE_RANDOM_INSTANCE": "true"},
            tools=["web_search"],
        )
    )

    packages.append(
        _kt_package(
            "langfuse-agent",
            "Langfuse — observability, tracing, prompt management, and evaluation. Deploy via compose template.",
            "langfuse-mcp",
            ["os_service", "observability", "tracing"],
            tools=[
                "ingestion_batch",
                "trace_delete",
                "prompts_delete",
                "datasets_get_runs",
            ],
        )
    )

    # ── DevOps Specialists ────────────────────────────────────────────

    packages.append(
        _kt_package(
            "gitlab-api",
            "GitLab API — projects, merge requests, pipelines, issues, CI/CD, and repository management.",
            "gitlab-mcp",
            ["devops", "git", "cicd", "enterprise"],
            tools=[
                "list_projects",
                "create_merge_request",
                "list_pipelines",
                "list_issues",
                "get_repository_tree",
            ],
        )
    )

    packages.append(
        _kt_package(
            "github-agent",
            "GitHub Agent — repository management, issues, pull requests, and actions.",
            "github-mcp",
            ["devops", "git", "cicd"],
            tools=[
                "list_repos",
                "create_issue",
                "list_pull_requests",
                "get_repo_contents",
            ],
        )
    )

    packages.append(
        _kt_package(
            "ansible-tower-mcp",
            "Ansible Tower/AWX — automation, playbook execution, inventory management.",
            "ansible-tower-mcp",
            ["devops", "automation", "infrastructure"],
        )
    )

    # ── Enterprise Specialists ────────────────────────────────────────

    packages.append(
        _kt_package(
            "servicenow-api",
            "ServiceNow — ITSM, incident management, CMDB, change requests, and knowledge base.",
            "servicenow-mcp",
            ["enterprise", "itsm", "ticketing"],
            tools=[
                "get_incidents",
                "create_incident",
                "get_change_requests",
                "query_cmdb",
            ],
        )
    )

    packages.append(
        _kt_package(
            "atlassian-agent",
            "Atlassian — Jira issues, Confluence pages, and project management.",
            "atlassian-mcp",
            ["enterprise", "ticketing", "collaboration"],
        )
    )

    packages.append(
        _kt_package(
            "leanix-agent",
            "LeanIX — enterprise architecture management via REST and GraphQL.",
            "leanix-mcp",
            ["enterprise", "architecture"],
        )
    )

    packages.append(
        _kt_package(
            "plane-agent",
            "Plane — open-source project management, issues, cycles, and modules.",
            "plane-mcp",
            ["enterprise", "project-management"],
        )
    )

    packages.append(
        _kt_package(
            "microsoft-agent",
            "Microsoft Graph — Outlook, Teams, OneDrive, SharePoint, and Azure AD.",
            "microsoft-mcp",
            ["enterprise", "productivity", "cloud"],
        )
    )

    # ── Media Specialists ─────────────────────────────────────────────

    packages.append(
        _kt_package(
            "jellyfin-mcp",
            "Jellyfin — media server management, libraries, playback, and user administration.",
            "jellyfin-mcp",
            ["media", "streaming"],
        )
    )

    packages.append(
        _kt_package(
            "media-downloader",
            "Media Downloader — download audio/video from the internet via yt-dlp.",
            "media-downloader-mcp",
            ["media", "download"],
        )
    )

    packages.append(
        _kt_package(
            "audio-transcriber",
            "Audio Transcriber — transcribe audio/video files to text via Whisper.",
            "audio-transcriber-mcp",
            ["media", "transcription", "ai"],
        )
    )

    packages.append(
        _kt_package(
            "arr-mcp",
            "Arr Suite — Sonarr, Radarr, Lidarr, and Prowlarr media automation.",
            "arr-mcp",
            ["media", "automation"],
        )
    )

    # ── Home / IoT Specialists ────────────────────────────────────────

    packages.append(
        _kt_package(
            "home-assistant-agent",
            "Home Assistant — smart home control, automations, devices, and entity management.",
            "home-assistant-mcp",
            ["iot", "smart-home", "automation"],
        )
    )

    packages.append(
        _kt_package(
            "uptime-kuma-agent",
            "Uptime Kuma — uptime monitoring, status pages, and alert management.",
            "uptime-mcp",
            ["iot", "monitoring", "observability"],
        )
    )

    # ── Documents / Data Specialists ──────────────────────────────────

    packages.append(
        _kt_package(
            "stirlingpdf-agent",
            "Stirling PDF — PDF manipulation, conversion, merging, splitting, and OCR.",
            "stirlingpdf-mcp",
            ["documents", "pdf"],
        )
    )

    packages.append(
        _kt_package(
            "documentdb-mcp",
            "DocumentDB — MongoDB-compatible document database on PostgreSQL.",
            "documentdb-mcp",
            ["database", "documents", "nosql"],
        )
    )

    packages.append(
        _kt_package(
            "archivebox-api",
            "ArchiveBox — web archiving, bookmarking, and snapshot management.",
            "archivebox-mcp",
            ["documents", "archiving"],
            pip_name="archivebox-api",
        )
    )

    packages.append(
        _kt_package(
            "vector-mcp",
            "Vector MCP — RAG integration with multiple vector database backends.",
            "vector-mcp",
            ["database", "rag", "ai"],
        )
    )

    # ── Cloud / Infrastructure Specialists ────────────────────────────

    packages.append(
        _kt_package(
            "portainer-agent",
            "Portainer — Docker environment management, stacks, registries, and edge devices.",
            "portainer-mcp",
            ["devops", "containers", "cloud"],
        )
    )

    packages.append(
        _kt_package(
            "nextcloud-agent",
            "Nextcloud — file management, sharing, calendars, and collaboration.",
            "nextcloud-mcp",
            ["cloud", "productivity", "files"],
        )
    )

    # ── Social / Communication Specialists ────────────────────────────

    packages.append(
        _kt_package(
            "postiz-agent",
            "Postiz — social media scheduling and management.",
            "postiz-mcp",
            ["social", "marketing"],
        )
    )

    packages.append(
        _kt_package(
            "owncast-agent",
            "Owncast — self-hosted live streaming server management.",
            "owncast-mcp",
            ["social", "streaming"],
        )
    )

    # ── Misc Specialists ──────────────────────────────────────────────

    packages.append(
        _kt_package(
            "mealie-mcp",
            "Mealie — recipe management, meal planning, and shopping lists.",
            "mealie-mcp",
            ["productivity", "food"],
        )
    )

    packages.append(
        _kt_package(
            "qbittorrent-agent",
            "qBittorrent — torrent management, RSS automation, and search.",
            "qbittorrent-mcp",
            ["media", "download"],
        )
    )

    packages.append(
        _kt_package(
            "wger-agent",
            "Wger — workout tracking, exercise database, nutrition plans, and body measurements.",
            "wger-mcp",
            ["fitness", "health"],
        )
    )

    # ── Core (not OS subsystem) ───────────────────────────────────────

    packages.append(
        SpecialistPackage(
            name="genius-agent",
            version="*",
            description="Genius Agent — the orchestrator. Consumes MCP tools from all other adaptive_agent_router.",
            tags=["knuckles-team", "core"],
            container_config=ContainerConfig(
                image="docker.io/knucklessg1/genius-agent:latest",
                compose_ref="agents/genius-agent/compose.yaml",
            ),
        )
    )

    # ── Community MCP Servers ─────────────────────────────────────────

    packages.append(
        _community_package(
            "mcp-playwright",
            "Playwright — browser automation, web scraping, and end-to-end testing.",
            "npx",
            ["@anthropic/mcp-playwright"],
            ["browser", "automation", "testing"],
        )
    )

    packages.append(
        _community_package(
            "mcp-sentry",
            "Sentry — error monitoring, performance tracking, and issue management.",
            "npx",
            ["@sentry/mcp-server-sentry"],
            ["observability", "monitoring", "errors"],
        )
    )

    packages.append(
        _community_package(
            "mcp-cloudflare",
            "Cloudflare — DNS management, Workers, and CDN configuration.",
            "npx",
            ["@cloudflare/mcp-server-cloudflare"],
            ["cloud", "dns", "cdn"],
        )
    )

    packages.append(
        _community_package(
            "mcp-kubernetes",
            "Kubernetes — cluster management, pods, services, and deployments.",
            "npx",
            ["@k8s/mcp-kubernetes"],
            ["devops", "containers", "orchestration"],
        )
    )

    packages.append(
        _community_package(
            "mcp-sqlite",
            "SQLite — local database queries and schema inspection.",
            "npx",
            ["@modelcontextprotocol/server-sqlite"],
            ["database", "sql"],
        )
    )

    return tuple(packages)
