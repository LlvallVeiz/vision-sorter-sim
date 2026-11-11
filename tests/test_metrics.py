"""Tests for evaluation metrics (confusion matrix, P/R/F1, matching)."""

import pytest

from vision_sorter_sim.metrics import (
    confusion_matrix,
    detection_metrics,
    match_detections,
    per_class_metrics,
    summarise_metrics,
)


def test_confusion_matrix_counts():
    y_true = ["circle", "circle", "square", "defect"]
    y_pred = ["circle", "square", "square", "defect"]
    matrix = confusion_matrix(y_true, y_pred)
    assert matrix["circle"]["circle"] == 1
    assert matrix["circle"]["square"] == 1
    assert matrix["square"]["square"] == 1
    assert matrix["defect"]["defect"] == 1
    assert matrix["triangle"]["triangle"] == 0


def test_per_class_metrics_hand_computed():
    # circle: TP=1, FP=0, FN=1 -> P=1.0, R=0.5, F1=2/3
    matrix = confusion_matrix(["circle", "circle", "square"],
                              ["circle", "square", "square"])
    m = per_class_metrics(matrix)["circle"]
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["support"] == 2


def test_perfect_classification_scores_one():
    labels = ["circle", "square", "triangle", "defect", "rectangle"]
    summary = summarise_metrics(labels, labels)
    assert summary["accuracy"] == 1.0
    assert summary["macro_f1"] == 1.0
    assert summary["macro_precision"] == 1.0
    assert summary["macro_recall"] == 1.0


def test_summarise_accuracy_counts():
    summary = summarise_metrics(["a", "a", "b"], ["a", "b", "b"],
                                labels=("a", "b"))
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["total"] == 3
    assert summary["correct"] == 2


def test_zero_division_guarded():
    # A class never predicted and never present: metrics are 0, not NaN.
    matrix = confusion_matrix(["circle"], ["circle"])
    m = per_class_metrics(matrix)["triangle"]
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_match_detections_nearest_centroid():
    detections = [{"centroid": (10.0, 10.0)}, {"centroid": (100.0, 100.0)}]
    truth = [{"cx": 12, "cy": 11, "label": "circle"},
             {"cx": 105, "cy": 98, "label": "square"}]
    matched, unmatched_det, unmatched_gt = match_detections(detections, truth)
    assert len(matched) == 2
    assert not unmatched_det and not unmatched_gt


def test_match_detections_leaves_far_ones_unmatched():
    detections = [{"centroid": (10.0, 10.0)}]
    truth = [{"cx": 500, "cy": 400, "label": "circle"}]
    matched, unmatched_det, unmatched_gt = match_detections(detections, truth)
    assert matched == []
    assert len(unmatched_det) == 1 and len(unmatched_gt) == 1


def test_match_detections_one_truth_one_detection_only():
    detections = [{"centroid": (10.0, 10.0)}, {"centroid": (11.0, 11.0)}]
    truth = [{"cx": 10, "cy": 10, "label": "circle"}]
    matched, unmatched_det, _ = match_detections(detections, truth)
    assert len(matched) == 1
    assert len(unmatched_det) == 1


def test_detection_metrics_hand_computed():
    m = detection_metrics(n_matched=8, n_detected=10, n_truth=9)
    assert m["precision"] == pytest.approx(0.8)
    assert m["recall"] == pytest.approx(8 / 9)
    assert m["f1"] == pytest.approx(2 * 0.8 * (8 / 9) / (0.8 + 8 / 9))
