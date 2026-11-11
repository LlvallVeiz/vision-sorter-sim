"""Tests for the end-to-end detection pipeline and dataset evaluation."""

from vision_sorter_sim.generator import generate_dataset
from vision_sorter_sim.pipeline import detect, evaluate_dataset, evaluate_frame


def test_detect_recovers_all_objects(frame_3_objects):
    detections = detect(frame_3_objects.image)
    assert len(detections) == 3
    labels = sorted(d["label"] for d in detections)
    truth = sorted(o.label for o in frame_3_objects.objects)
    assert labels == truth


def test_detect_small_single_triangle_is_one_object():
    """Regression test: the frame that used to fragment into 24
    detections (Otsu splitting the belt) must yield exactly one
    triangle detection."""
    from vision_sorter_sim.generator import generate_frame
    frame = generate_frame(seed=20261111 + 10_000 + 188, n_objects=1,
                           defect_rate=0.12)
    detections = detect(frame.image)
    assert len(detections) == 1
    assert detections[0]["label"] == "triangle"


def test_detect_empty_image_finds_nothing(blank_image):
    assert detect(blank_image) == []


def test_detect_returns_features_and_rules(frame_3_objects):
    det = detect(frame_3_objects.image)[0]
    for key in ("centroid", "bbox", "area", "circularity", "extent",
                "aspect_ratio", "solidity", "hu_log", "label",
                "confidence", "rule", "threshold"):
        assert key in det
    assert det["label"] in ("circle", "square", "rectangle", "triangle", "defect")


def test_evaluate_frame_scores_perfectly_on_clean_data(frame_3_objects):
    truth = [o.to_dict() for o in frame_3_objects.objects]
    result = evaluate_frame(frame_3_objects.image, truth)
    assert result["detection"]["matched"] == 3
    assert result["detection"]["recall"] == 1.0
    assert result["elapsed_ms"] > 0


def test_evaluate_dataset_small(tmp_path):
    generate_dataset(tmp_path, n_frames=5, seed=100, n_objects=4,
                     defect_rate=0.15)
    summary = evaluate_dataset(tmp_path)
    assert summary["frames"] == 5
    assert summary["total"] > 0
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert summary["detection"]["recall"] > 0.9
    assert summary["mean_frame_ms"] > 0


def test_evaluate_dataset_rejects_empty_dir(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        evaluate_dataset(tmp_path)
