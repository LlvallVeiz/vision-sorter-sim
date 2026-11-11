"""Tests for report (Markdown/JSON) and dashboard (HTML) generation."""

import json

from vision_sorter_sim.conveyor import run_conveyor
from vision_sorter_sim.dashboard import build_dashboard
from vision_sorter_sim.generator import generate_dataset
from vision_sorter_sim.pipeline import evaluate_dataset
from vision_sorter_sim.report import build_report


def _evaluation_and_run(tmp_path):
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, n_frames=4, seed=200, n_objects=3,
                     defect_rate=0.15)
    summary = evaluate_dataset(dataset)
    run = run_conveyor(n_items=24, seed=200)
    return summary, run


def test_build_report_writes_md_and_json(tmp_path):
    summary, run = _evaluation_and_run(tmp_path)
    out = tmp_path / "out"
    md_path, json_path = build_report(summary, run.to_dict(), "dataset", out)
    assert md_path.exists() and json_path.exists()
    text = md_path.read_text()
    assert "# Vision Sorter" in text
    assert "Confusion matrix" in text
    assert f"{summary['accuracy']:.4f}" in text
    payload = json.loads(json_path.read_text())
    assert payload["evaluation"]["accuracy"] == summary["accuracy"]
    assert payload["conveyor"]["total_items"] == 24


def test_build_dashboard_writes_html(tmp_path):
    summary, run = _evaluation_and_run(tmp_path)
    html_path = build_dashboard(summary, run, "dataset",
                                tmp_path / "dash" / "index.html")
    assert html_path.exists()
    text = html_path.read_text()
    assert "chart.js" in text.lower()
    assert "Confusion matrix" in text
    assert "Throughput" in text
    assert "accept" in text


def test_report_accuracy_matches_evaluation(tmp_path):
    summary, run = _evaluation_and_run(tmp_path)
    out = tmp_path / "out"
    build_report(summary, run.to_dict(), "dataset", out)
    text = (out / "report.md").read_text()
    assert f"Accuracy: {summary['accuracy']:.4f}" in text
