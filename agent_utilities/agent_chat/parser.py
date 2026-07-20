import re

from ..knowledge_graph.core.engine import IntelligenceGraphEngine
from ..models.codemap import CodemapArtifact

CODMAP_PATTERN = re.compile(r"@codemap\{([^}]+)\}")


async def parse_codemap_mentions(
    prompt: str,
    kg: IntelligenceGraphEngine,
) -> tuple[str, dict[str, CodemapArtifact]]:
    """
    Replaces @codemap{artifact-id} with the actual artifact
    and returns the cleaned prompt + a dict of resolved codemaps for the agent.
    """
    mentions: dict[str, CodemapArtifact] = {}
    matches = CODMAP_PATTERN.findall(prompt)

    for artifact_id in matches:
        artifact = await kg.get_codemap_by_id(artifact_id)

        if artifact:
            mentions[artifact_id] = artifact
            # replace the mention with a clean reference the agent understands
            prompt = prompt.replace(
                f"@codemap{{{artifact_id}}}",
                f"[Codemap Reference: {artifact.display_label}]",
            )
        else:
            prompt = prompt.replace(
                f"@codemap{{{artifact_id}}}",
                f"[Codemap Reference Not Found: {artifact_id}]",
            )

    return prompt, mentions
