# Media Generation Gateway (CONCEPT:AU-ECO.toolkit.media-gateway-failure-path)

## Overview
Lazy-`httpx` clients for self-hosted media generation: speech synthesis (`xtts`), image
generation (`flux.2` and Stable Diffusion 3.5, selectable backends), and video generation
(`hunyuanvideo`). Exposed as agent tools under the `MEDIA_TOOLS` gate. Endpoints come from
`{SERVICE}_URL` env vars; unreachable services raise a clear `MediaServiceError`.

## Implementation Details
- **Source Code**: `agent_utilities/ecosystem/media/gateway.py`,
  `agent_utilities/tools/media_tools.py`
- **Services**: `services/xtts`, `services/flux.2`, `services/stable-diffusion`,
  `services/hunyuanvideo` (GB10, swarm-launcher pattern)
- **Pillar**: ECO
