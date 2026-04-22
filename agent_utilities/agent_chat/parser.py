import re

from ..knowledge_graph.engine import IntelligenceGraphEngine
from ..models.codemap import CodemapArtifact

CODMAP_PATTERN = re.compile(r"@codemap\{([^}]+)\}")


async def parse_codemap_mentions(
    prompt: str,
    kg: IntelligenceGraphEngine,
) -> tuple[str, dict[str, CodemapArtifact]]:
    """
    Replaces @codemap{auth-flow} or @codemap{uuid} with the actual artifact
    and returns the cleaned prompt + a dict of resolved codemaps for the agent.
    """
    mentions: dict[str, CodemapArtifact] = {}
    matches = CODMAP_PATTERN.findall(prompt)

    for slug_or_id in matches:
        # Try by ID first, then by prompt slug
        artifact = await kg.get_codemap_by_id(slug_or_id)
        if not artifact:
            artifact = await kg.get_codemap_by_slug(slug_or_id)

        if artifact:
            mentions[slug_or_id] = artifact
            # replace the mention with a clean reference the agent understands
            prompt = prompt.replace(
                f"@codemap{{{slug_or_id}}}", f"[Codemap Reference: {artifact.prompt}]"
            )
        else:
            prompt = prompt.replace(
                f"@codemap{{{slug_or_id}}}",
                f"[Codemap Reference Not Found: {slug_or_id}]",
            )

    return prompt, mentions
