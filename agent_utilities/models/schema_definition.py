#!/usr/bin/python
# coding: utf-8
"""Unified Graph Schema Definition.

This module defines the structural schema for the knowledge graph,
which can be used by different backends (Ladybug, Neo4j, FalkorDB)
to initialize tables or indices.
"""

from .knowledge_graph import GraphSchemaDefinition, TableDefinition, RelDefinition

SCHEMA = GraphSchemaDefinition(
    nodes=[
        # Core Registry Nodes
        TableDefinition(
            name="Agent",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "description": "STRING",
                "agent_type": "STRING",
                "system_prompt": "STRING",
                "capabilities": "STRING[]",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Tool",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "description": "STRING",
                "mcp_server": "STRING",
                "relevance_score": "INT64",
                "tags": "STRING[]",
                "requires_approval": "BOOLEAN",
                "last_sync": "INT64",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Server",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "url": "STRING",
                "status": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Prompt",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "description": "STRING",
                "system_prompt": "STRING",
                "capabilities": "STRING[]",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Skill",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "description": "STRING",
                "version": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Code",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "description": "STRING",
                "file_path": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Memory",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "description": "STRING",
                "category": "STRING",
                "timestamp": "STRING",
                "tags": "STRING[]",
                "importance_score": "FLOAT",
                "metadata": "STRING",
            },
        ),
        # User & Environment Nodes
        TableDefinition(
            name="Client",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "description": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="User",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "role": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Preference",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "category": "STRING",
                "value": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Job",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "schedule": "STRING",
                "command": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Log",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "timestamp": "STRING",
                "status": "STRING",
                "output": "STRING",
                "importance_score": "FLOAT",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Thread",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "title": "STRING",
                "created_at": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Message",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "role": "STRING",
                "content": "STRING",
                "timestamp": "STRING",
                "embedding": "FLOAT[]",
                "importance_score": "FLOAT",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Heartbeat",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "agent_name": "STRING",
                "timestamp": "STRING",
                "status": "STRING",
                "issues": "STRING[]",
                "raw_data": "STRING",
                "importance_score": "FLOAT",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="ChatSummary",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "summary_text": "STRING",
                "original_count": "INT64",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        # Enhanced Memory & Reasoning Nodes (Neo4j/MAGMA style)
        TableDefinition(
            name="ReasoningTrace",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "thought": "STRING",
                "reflection": "STRING",
                "confidence": "FLOAT",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="ToolCall",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "tool_name": "STRING",
                "args": "STRING",
                "result": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Entity",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "entity_type": "STRING",
                "properties": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Event",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "timestamp": "STRING",
                "event_type": "STRING",
                "importance_score": "FLOAT",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Reflection",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "content": "STRING",
                "confidence": "FLOAT",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Goal",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "goal_text": "STRING",
                "status": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Episode",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "timestamp": "STRING",
                "source": "STRING",
                "description": "STRING",
                "importance_score": "FLOAT",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Fact",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "content": "STRING",
                "certainty": "FLOAT",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Concept",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Capability",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        # Callable Resources & Dynamic Agents
        TableDefinition(
            name="ToolMetadata",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "tags": "STRING[]",
                "prompt_template": "STRING",
                "resources": "STRING",
                "capabilities": "STRING[]",
                "auth_requirements": "STRING",
                "version": "STRING",
                "source": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="CallableResource",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "resource_type": "STRING",
                "endpoint": "STRING",
                "agent_card": "STRING",
                "skill_code_path": "STRING",
                "metadata_id": "STRING",
                "embedding": "FLOAT[]",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="SpawnedAgent",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "system_prompt": "STRING",
                "tool_ids": "STRING[]",
                "parent_task_id": "STRING",
                "created_at": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="SystemPrompt",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "content": "STRING",
                "version": "STRING",
                "tags": "STRING[]",
                "parameters": "STRING",
                "embedding": "FLOAT[]",
                "source": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        # Self-Improvement & Learning (Agent Lightning style)
        TableDefinition(
            name="OutcomeEvaluation",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "reward": "FLOAT",
                "success_criteria_met": "STRING[]",
                "feedback_text": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Critique",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "textual_gradient": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="SelfEvaluation",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "confidence_calibration": "FLOAT",
                "task_difficulty": "FLOAT",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Experiment",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "status": "STRING",
                "importance_score": "FLOAT",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="ProposedSkill",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "code_content": "STRING",
                "frontmatter": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        # Knowledge Base Nodes
        TableDefinition(
            name="KnowledgeBase",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "topic": "STRING",
                "description": "STRING",
                "source_type": "STRING",
                "source_count": "INT64",
                "article_count": "INT64",
                "status": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Article",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "summary": "STRING",
                "content": "STRING",
                "word_count": "INT64",
                "tags": "STRING[]",
                "embedding": "FLOAT[]",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="RawSource",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "file_path": "STRING",
                "source_type": "STRING",
                "content_hash": "STRING",
                "file_size": "INT64",
                "status": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="KBConcept",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "description": "STRING",
                "embedding": "FLOAT[]",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="KBFact",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "content": "STRING",
                "certainty": "FLOAT",
                "source_ids": "STRING[]",
                "embedding": "FLOAT[]",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="KBIndex",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "content": "STRING",
                "kb_id": "STRING",
                "article_count": "INT64",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Checkpoint",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "label": "STRING",
                "turn": "INT64",
                "message_count": "INT64",
                "message_data": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Team",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "name": "STRING",
                "status": "STRING",
                "member_count": "INT64",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
        TableDefinition(
            name="Task",
            columns={
                "id": "STRING PRIMARY KEY",
                "type": "STRING",
                "content": "STRING",
                "status": "STRING",
                "assigned_to": "STRING",
                "created_by": "STRING",
                "importance_score": "FLOAT",
                "timestamp": "STRING",
                "metadata": "STRING",
            },
        ),
    ],
    edges=[
        # Core Relationships
        RelDefinition(
            type="PROVIDES",
            connections=[
                {"from": "Agent", "to": "Tool"},
                {"from": "Skill", "to": "Tool"},
                {"from": "Server", "to": "CallableResource"},
            ],
        ),
        RelDefinition(
            type="DEPENDS_ON",
            connections=[
                {"from": "Agent", "to": "Agent"},
                {"from": "Code", "to": "Code"},
                {"from": "Skill", "to": "Skill"},
            ],
        ),
        RelDefinition(type="CONTAINS", connections=[{"from": "Code", "to": "Code"}]),
        RelDefinition(
            type="RELATED_TO",
            connections=[
                {"from": "Memory", "to": "Memory"},
                {"from": "Memory", "to": "Code"},
                {"from": "Memory", "to": "Agent"},
                {"from": "Memory", "to": "User"},
            ],
        ),
        RelDefinition(
            type="BELONGS_TO", connections=[{"from": "User", "to": "Client"}]
        ),
        RelDefinition(
            type="PREFERS", connections=[{"from": "User", "to": "Preference"}]
        ),
        RelDefinition(type="EXECUTED_BY", connections=[{"from": "Log", "to": "Job"}]),
        RelDefinition(
            type="PART_OF",
            connections=[
                {"from": "Message", "to": "Thread"},
                {"from": "ChatSummary", "to": "Thread"},
            ],
        ),
        RelDefinition(
            type="REPLY_TO", connections=[{"from": "Message", "to": "Message"}]
        ),
        RelDefinition(
            type="HEARTBEAT_OF", connections=[{"from": "Heartbeat", "to": "Agent"}]
        ),
        RelDefinition(
            type="USES",
            connections=[
                {"from": "Agent", "to": "Prompt"},
                {"from": "Agent", "to": "Skill"},
                {"from": "SpawnedAgent", "to": "CallableResource"},
                {"from": "SpawnedAgent", "to": "SystemPrompt"},
            ],
        ),
        # Enhanced Memory & Reasoning Relationships
        RelDefinition(
            type="HAS_REASONING",
            connections=[
                {"from": "Message", "to": "ReasoningTrace"},
                {"from": "Episode", "to": "ReasoningTrace"},
            ],
        ),
        RelDefinition(
            type="USED_TOOL", connections=[{"from": "ReasoningTrace", "to": "ToolCall"}]
        ),
        RelDefinition(
            type="AFFECTS", connections=[{"from": "ToolCall", "to": "Entity"}]
        ),
        RelDefinition(
            type="CAUSED_BY",
            connections=[
                {"from": "ReasoningTrace", "to": "ReasoningTrace"},
                {"from": "Event", "to": "Event"},
            ],
        ),
        RelDefinition(
            type="INFLUENCED",
            connections=[{"from": "Reflection", "to": "ReasoningTrace"}],
        ),
        RelDefinition(
            type="CONTRADICTS",
            connections=[
                {"from": "Fact", "to": "Fact"},
                {"from": "ReasoningTrace", "to": "ReasoningTrace"},
            ],
        ),
        RelDefinition(
            type="UPDATED_BELIEF", connections=[{"from": "Reflection", "to": "Fact"}]
        ),
        RelDefinition(
            type="HAS_EVIDENCE",
            connections=[
                {"from": "Fact", "to": "Message"},
                {"from": "Fact", "to": "Episode"},
            ],
        ),
        RelDefinition(
            type="TEMPORALLY_PRECEDES",
            connections=[
                {"from": "Episode", "to": "Episode"},
                {"from": "Event", "to": "Event"},
            ],
        ),
        RelDefinition(
            type="OCCURRED_DURING", connections=[{"from": "Event", "to": "Episode"}]
        ),
        RelDefinition(
            type="EVOLVED_INTO", connections=[{"from": "Concept", "to": "Concept"}]
        ),
        RelDefinition(
            type="ENABLES", connections=[{"from": "Capability", "to": "Tool"}]
        ),
        RelDefinition(type="IMPLIES", connections=[{"from": "Fact", "to": "Fact"}]),
        RelDefinition(type="INDEXES", connections=[{"from": "Concept", "to": "Fact"}]),
        RelDefinition(
            type="CONSOLIDATES_INTO",
            connections=[{"from": "Episode", "to": "ChatSummary"}],
        ),
        RelDefinition(
            type="SELF_REFLECTS_ON",
            connections=[{"from": "Agent", "to": "ReasoningTrace"}],
        ),
        # Callable Resource & Agent Relationships
        RelDefinition(
            type="HAS_METADATA",
            connections=[{"from": "CallableResource", "to": "ToolMetadata"}],
        ),
        RelDefinition(
            type="PROVIDES_CAPABILITY",
            connections=[{"from": "CallableResource", "to": "Capability"}],
        ),
        RelDefinition(
            type="DELEGATES_TO", connections=[{"from": "Agent", "to": "Agent"}]
        ),
        RelDefinition(
            type="DISCOVERED_VIA", connections=[{"from": "Agent", "to": "Server"}]
        ),
        RelDefinition(
            type="USED_RESOURCE",
            connections=[{"from": "ReasoningTrace", "to": "CallableResource"}],
        ),
        RelDefinition(
            type="USES_BASE_PROMPT",
            connections=[{"from": "SpawnedAgent", "to": "SystemPrompt"}],
        ),
        RelDefinition(
            type="EVOLVED_FROM",
            connections=[{"from": "SystemPrompt", "to": "SystemPrompt"}],
        ),
        RelDefinition(
            type="PROVEN_WITH",
            connections=[{"from": "SystemPrompt", "to": "CallableResource"}],
        ),
        RelDefinition(
            type="DERIVED_FROM_PROMPT",
            connections=[{"from": "ToolMetadata", "to": "SystemPrompt"}],
        ),
        # Self-Improvement Relationships
        RelDefinition(
            type="PRODUCED_OUTCOME",
            connections=[
                {"from": "Episode", "to": "OutcomeEvaluation"},
                {"from": "ReasoningTrace", "to": "OutcomeEvaluation"},
            ],
        ),
        RelDefinition(
            type="SCORED_BY",
            connections=[{"from": "OutcomeEvaluation", "to": "SpawnedAgent"}],
        ),
        RelDefinition(
            type="GENERATED_CRITIQUE",
            connections=[{"from": "ReasoningTrace", "to": "Critique"}],
        ),
        RelDefinition(
            type="LED_TO", connections=[{"from": "Critique", "to": "SystemPrompt"}]
        ),
        RelDefinition(
            type="SUPERSEDES",
            connections=[
                {"from": "Fact", "to": "Fact"},
                {"from": "SystemPrompt", "to": "SystemPrompt"},
            ],
        ),
        # Knowledge Base Relationships
        RelDefinition(
            type="BELONGS_TO_KB",
            connections=[
                {"from": "Article", "to": "KnowledgeBase"},
                {"from": "KBConcept", "to": "KnowledgeBase"},
                {"from": "KBFact", "to": "KnowledgeBase"},
                {"from": "RawSource", "to": "KnowledgeBase"},
                {"from": "KBIndex", "to": "KnowledgeBase"},
            ],
        ),
        RelDefinition(
            type="COMPILED_FROM", connections=[{"from": "Article", "to": "RawSource"}]
        ),
        RelDefinition(
            type="ABOUT", connections=[{"from": "Article", "to": "KBConcept"}]
        ),
        RelDefinition(
            type="CITES",
            connections=[
                {"from": "Article", "to": "RawSource"},
                {"from": "KBFact", "to": "RawSource"},
            ],
        ),
        RelDefinition(
            type="BACKLINKS", connections=[{"from": "Article", "to": "Article"}]
        ),
        RelDefinition(
            type="CONTRADICTS_KB", connections=[{"from": "KBFact", "to": "KBFact"}]
        ),
        RelDefinition(
            type="INDEXES_KB", connections=[{"from": "KBIndex", "to": "Article"}]
        ),
        RelDefinition(
            type="SNAPSHOT_OF", connections=[{"from": "Checkpoint", "to": "Episode"}]
        ),
        RelDefinition(
            type="FORKED_FROM", connections=[{"from": "Checkpoint", "to": "Checkpoint"}]
        ),
        RelDefinition(
            type="ASSIGNED_TO_AGENT",
            connections=[
                {"from": "Task", "to": "Agent"},
                {"from": "Task", "to": "SpawnedAgent"},
            ],
        ),
        RelDefinition(
            type="BLOCKED_BY_TASK", connections=[{"from": "Task", "to": "Task"}]
        ),
        RelDefinition(
            type="BELONGS_TO_TEAM",
            connections=[
                {"from": "Task", "to": "Team"},
                {"from": "Agent", "to": "Team"},
                {"from": "SpawnedAgent", "to": "Team"},
            ],
        ),
    ],
)
