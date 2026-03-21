"""Tests for post-planner routing and submission guards."""

from __future__ import annotations

from config.settings import get_settings
from workflows.planner_routing import planner_route_after_decision


def test_submit_without_file_forces_coder(tmp_path):
    ws = tmp_path
    state = {
        "workspace_dir": str(ws),
        "iteration": 1,
        "next_step": "submit",
        "submission_path": "submission.csv",
    }
    node, patch = planner_route_after_decision(state)
    assert node == "coder"
    assert patch == {"next_step": "coder"}

    (ws / "submission.csv").write_text("a,b\n1,2\n")
    node2, patch2 = planner_route_after_decision(state)
    assert node2 == "submit_action"
    assert patch2 == {}


def test_force_coder_after_iteration_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("FORCE_CODER_MIN_PLANNER_ITERATION", "3")
    get_settings.cache_clear()
    ws = tmp_path
    state = {
        "workspace_dir": str(ws),
        "iteration": 3,
        "next_step": "data_worker",
        "submission_path": "submission.csv",
    }
    node, patch = planner_route_after_decision(state)
    assert node == "coder"
    assert patch == {"next_step": "coder"}
