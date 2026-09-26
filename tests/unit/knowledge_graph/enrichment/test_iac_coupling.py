"""IaC Resource extraction (KG-2.103) + git change-coupling (KG-2.104)."""

from __future__ import annotations

import subprocess

from agent_utilities.knowledge_graph.enrichment.git_coupling import (
    change_coupling_for_repo,
    parse_change_coupling,
)
from agent_utilities.knowledge_graph.enrichment.iac import (
    extract_iac,
    link_resources_to_service,
)
from agent_utilities.knowledge_graph.enrichment.models import GraphNode


def test_extract_iac_dockerfile_k8s_terraform():
    files = [
        ("svc/Dockerfile", "FROM python:3.12-slim\nEXPOSE 8080\nCMD ['run']\n"),
        (
            "deploy/app.yaml",
            "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: users-api\n",
        ),
        ("infra/main.tf", 'resource "aws_s3_bucket" "logs" {\n  bucket = "x"\n}\n'),
    ]
    nodes, _ = extract_iac(files)
    by_id = {n.id: n for n in nodes}

    img = by_id["resource:container_image:svc/Dockerfile"]
    assert img.type == "Resource" and img.props["kind"] == "container_image"
    assert img.props["base_image"] == "python:3.12-slim"
    assert img.props["ports"] == "8080"

    dep = by_id["resource:deployment:users-api"]
    assert dep.props["kind"] == "deployment" and dep.props["name"] == "users-api"

    bucket = by_id["resource:aws_s3_bucket:logs"]
    assert bucket.props["kind"] == "aws_s3_bucket" and bucket.props["name"] == "logs"


def test_link_resources_to_service():
    res = [
        GraphNode(id="resource:container_image:Dockerfile", type="Resource", props={})
    ]
    edges = link_resources_to_service(res, "service:cluster:users-api")
    assert len(edges) == 1 and edges[0].rel_type == "PROVISIONS"
    assert edges[0].target == "service:cluster:users-api"
    assert link_resources_to_service(res, "") == []


def test_parse_change_coupling_counts_cochanges():
    # a.py & b.py co-change in 3 commits; c.py is independent. A 200-file bulk
    # commit is ignored (would couple everything).
    commits = [
        ["a.py", "b.py"],
        ["a.py", "b.py", "d.py"],
        ["a.py", "b.py"],
        ["c.py"],
        [f"f{i}.py" for i in range(60)],  # bulk → skipped
    ]
    edges = parse_change_coupling(commits, min_support=3)
    pairs = {(e.source, e.target): e.props["support"] for e in edges}
    assert pairs == {("file:a.py", "file:b.py"): "3"}
    assert all(e.rel_type == "FILE_CHANGES_WITH" for e in edges)
    # Below-threshold pairs are dropped.
    assert ("file:a.py", "file:d.py") not in pairs


def test_git_coupling_entrypoint_preserves_newlines_in_paths(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    path = "line\nbreak.py"
    (tmp_path / path).write_text("test")
    (tmp_path / "peer.py").write_text("test")
    git("add", path)
    git("add", "peer.py")
    git("commit", "-qm", "test")

    edges = change_coupling_for_repo(str(tmp_path), min_support=1, max_commits=1)
    assert [(edge.source, edge.target) for edge in edges] == [
        (f"file:{path}", "file:peer.py")
    ]
