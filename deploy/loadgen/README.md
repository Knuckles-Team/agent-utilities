# Canonical loadgen deployment source

This directory is the AU-owned, reviewable source for the loadgen deployment
definitions. It replaces the unowned workspace snapshot under
`/home/apps/workspace/services/loadgen` once an operator has a canonical GitOps
repository and has registered that repository with the workspace authority.

The files here are templates, not directly deployable manifests. They contain
required substitutions for the image, release/topology/workload/source
digests, certification duration, and workload-identity audience. The renderer
rejects mutable image references, missing values, source-authority digest drift,
and non-empty output directories:

```bash
python scripts/release/render_loadgen_assets.py \
  --image registry.example.invalid/agent-utilities@sha256:<64-lowercase-hex> \
  --release-digest sha256:<64-lowercase-hex> \
  --topology-digest sha256:<64-lowercase-hex> \
  --contract-digest sha256:<64-lowercase-hex> \
  --source-repository https://git.example.invalid/containers/loadgen.git \
  --source-revision <full-40-character-commit> \
  --source-manifest-digest sha256:<64-lowercase-hex> \
  --source-authority-digest sha256:<64-lowercase-hex> \
  --workload-identity-audience graphos-loadgen \
  --duration-s 86400 \
  --output ./rendered-loadgen
python scripts/release/check_loadgen_assets.py --directory ./rendered-loadgen
```

The source-authority digest must be produced by
`scripts.scale.loadgen_source_authority` after the canonical GitOps checkout
and both deployment definitions have passed the source gate. This repository
does not contain credentials, Secret objects, or operator-specific repository
coordinates. The rendered production output references the required
`graphos-loadgen-secrets` Secret and the dedicated
`graphos-certification-load` ServiceAccount; the operator must provision those
through the deployment's existing secret/workload-identity control plane.

Production runs invoke the exact live command:

```text
graphos-certification-load --engine live --scale 1.0 \
  --duration-s <campaign-duration> \
  --report-json /var/run/loadgen/load-report.json \
  --release-digest <signed-release-digest>
```

`mock/` is a separate, explicitly non-certifying overlay. It invokes
`--engine mock --scale 0.001 --duration-s 5`, has no live SecretRefs, and is
labelled so certification cannot consume it. The mock image is still rendered
from the same immutable image input so the overlay cannot become an accidental
floating-image deployment.

`source-registration.json` is deterministic metadata for the generated bundle.
It records the exact source repository/revision/manifest digest, authority
digest, template bundle digest, image digest, required secret-reference names,
and the two command vectors. It contains no secret values or live endpoints.
