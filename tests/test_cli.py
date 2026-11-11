"""Tests for the CLI subcommands (run in-process via main())."""

import json

import pytest

from vision_sorter_sim.cli import main
from vision_sorter_sim.generator import generate_dataset


def _dataset(tmp_path):
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, n_frames=3, seed=300, n_objects=3,
                     defect_rate=0.15)
    return dataset


def test_cli_generate(tmp_path, capsys):
    out = tmp_path / "gen"
    assert main(["generate", "--output", str(out), "--frames", "2",
                 "--seed", "1", "--objects", "2"]) == 0
    assert len(list(out.glob("frame_*.png"))) == 2
    assert "generated 2 frames" in capsys.readouterr().out


def test_cli_detect(tmp_path, capsys):
    dataset = _dataset(tmp_path)
    img = next(dataset.glob("frame_*.png"))
    assert main(["detect", str(img)]) == 0
    assert "object(s)" in capsys.readouterr().out


def test_cli_classify(tmp_path, capsys):
    dataset = _dataset(tmp_path)
    assert main(["classify", str(dataset)]) == 0
    out = capsys.readouterr().out
    assert "accuracy:" in out
    assert "confusion matrix" in out


def test_cli_sort(capsys):
    assert main(["sort", "--items", "10", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "items: 10" in out
    assert "throughput:" in out


def test_cli_report(tmp_path, capsys):
    dataset = _dataset(tmp_path)
    out = tmp_path / "report"
    assert main(["report", str(dataset), "--output", str(out),
                 "--items", "12", "--seed", "1"]) == 0
    assert (out / "report.md").exists()
    assert (out / "report.json").exists()
    assert (out / "dashboard" / "index.html").exists()
    assert "report:" in capsys.readouterr().out


def test_cli_demo(tmp_path, capsys):
    dataset = _dataset(tmp_path)
    assert main(["demo", str(dataset), "--items", "8", "--seed", "1"]) == 0
    assert "accuracy:" in capsys.readouterr().out


def test_cli_detect_writes_json(tmp_path):
    dataset = _dataset(tmp_path)
    img = next(dataset.glob("frame_*.png"))
    out = tmp_path / "det.json"
    assert main(["detect", str(img), "--output", str(out)]) == 0
    data = json.loads(out.read_text())
    assert isinstance(data, list) and len(data) > 0
