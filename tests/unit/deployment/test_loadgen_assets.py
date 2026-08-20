"""Static contracts for canonical rendered loadgen deployment assets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.release import check_loadgen_assets, render_loadgen_assets
from scripts.scale.loadgen_source_authority import source_authority_digest

ROOT = Path(__file__).resolve().parents[3]


def _inputs() -> render_loadgen_assets.LoadgenRenderInputs:
    repository = "https://git.example.invalid/containers/loadgen.git"
    revision = "a" * 40
    manifest_digest = "sha256:" + "4" * 64
    return render_loadgen_assets.LoadgenRenderInputs.from_values(
        image="registry.example.invalid/agent-utilities@sha256:" + "1" * 64,
        release_digest="sha256:" + "2" * 64,
        topology_digest="sha256:" + "3" * 64,
        contract_digest="sha256:" + "5" * 64,
        source_repository=repository,
        source_revision=revision,
        source_manifest_digest=manifest_digest,
        source_authority_digest_value=source_authority_digest(
            repository,
            revision,
            manifest_digest,
        ),
        workload_identity_audience="synthetic-loadgen",
        duration_seconds=86_400,
    )


def test_canonical_templates_have_no_deployable_image_or_secret_material() -> None:
    source = ROOT / "deploy" / "loadgen"
    for path in (
        source / "compose.yml",
        source / "mock.compose.yml",
        source / "k8s/kustomization.yaml",
        source / "k8s/loadgen.yaml",
        source / "k8s/mock/kustomization.yaml",
        source / "k8s/mock/loadgen.yaml",
    ):
        text = path.read_text(encoding="utf-8")
        assert "@sha256:" not in text
        assert "latest" not in text.casefold()
        assert "PRIVATE_KEY" not in text
        assert "secretValue" not in text
    assert "      - live" in (source / "compose.yml").read_text(encoding="utf-8")
    assert "      - mock" in (source / "mock.compose.yml").read_text(encoding="utf-8")


def test_renderer_rejects_mutable_images_and_authority_drift(tmp_path: Path) -> None:
    with pytest.raises(render_loadgen_assets.LoadgenRenderError):
        render_loadgen_assets.LoadgenRenderInputs.from_values(
            image="registry.example.invalid/agent-utilities:latest",
            release_digest="sha256:" + "2" * 64,
            topology_digest="sha256:" + "3" * 64,
            contract_digest="sha256:" + "5" * 64,
            source_repository="https://git.example.invalid/containers/loadgen.git",
            source_revision="a" * 40,
            source_manifest_digest="sha256:" + "4" * 64,
            source_authority_digest_value="sha256:" + "6" * 64,
            workload_identity_audience="synthetic-loadgen",
            duration_seconds=86_400,
        )

    output = tmp_path / "non-empty"
    output.mkdir()
    (output / "occupied").write_text("occupied", encoding="utf-8")
    with pytest.raises(render_loadgen_assets.LoadgenRenderError):
        render_loadgen_assets.render(output_dir=output, inputs=_inputs())


def test_rendered_bundle_is_deterministic_and_has_explicit_mock_boundary(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    render_loadgen_assets.render(output_dir=first, inputs=_inputs())
    render_loadgen_assets.render(output_dir=second, inputs=_inputs())

    files = [
        "production/compose.yml",
        "production/k8s/kustomization.yaml",
        "production/k8s/loadgen.yaml",
        "mock/compose.yml",
        "mock/k8s/kustomization.yaml",
        "mock/k8s/loadgen.yaml",
        "source-registration.json",
    ]
    assert [(first / relative).read_bytes() for relative in files] == [
        (second / relative).read_bytes() for relative in files
    ]
    report = check_loadgen_assets.validate(first)
    assert report["ok"] is True
    assert (
        "--engine"
        in json.loads((first / "source-registration.json").read_text(encoding="utf-8"))[
            "commands"
        ]["mock"]
    )
    assert (
        "live"
        not in json.loads(
            (first / "source-registration.json").read_text(encoding="utf-8")
        )["commands"]["mock"]
    )
