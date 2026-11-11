"""Evaluation metrics: confusion matrix, precision, recall, F1.

All definitions are the standard ones (see docs/RESEARCH.md):
for each class treated as "positive" against all others,
precision = TP / (TP + FP), recall = TP / (TP + FN), and
F1 = 2 * precision * recall / (precision + recall). Macro averages
are the unweighted means over classes. Detection metrics compare the
set of detected objects against ground truth by nearest centroid.
"""

from __future__ import annotations

import math

from . import config


def confusion_matrix(y_true: list[str], y_pred: list[str],
                     labels: tuple[str, ...] = config.CLASSES) -> dict[str, dict[str, int]]:
    """Multi-class confusion matrix as ``{true: {pred: count}}``."""
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for true, pred in zip(y_true, y_pred):
        if true in matrix and pred in matrix[true]:
            matrix[true][pred] += 1
    return matrix


def per_class_metrics(matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """Per-class precision/recall/F1/support from a confusion matrix."""
    out: dict[str, dict[str, float]] = {}
    for label, row in matrix.items():
        tp = row[label]
        fp = sum(matrix[other][label] for other in matrix if other != label)
        fn = sum(row[other] for other in row if other != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        out[label] = {"precision": precision, "recall": recall, "f1": f1,
                      "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}
    return out


def summarise_metrics(y_true: list[str], y_pred: list[str],
                      labels: tuple[str, ...] = config.CLASSES) -> dict:
    """Full evaluation summary: matrix, per-class metrics, accuracy,
    macro precision/recall/F1."""
    matrix = confusion_matrix(y_true, y_pred, labels)
    per_class = per_class_metrics(matrix)
    total = len(y_true)
    correct = sum(matrix[t][t] for t in labels)
    accuracy = correct / total if total else 0.0
    n = len(per_class)
    macro = {
        "precision": sum(m["precision"] for m in per_class.values()) / n if n else 0.0,
        "recall": sum(m["recall"] for m in per_class.values()) / n if n else 0.0,
        "f1": sum(m["f1"] for m in per_class.values()) / n if n else 0.0,
    }
    return {"accuracy": accuracy, "total": total, "correct": correct,
            "confusion_matrix": matrix, "per_class": per_class,
            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"], "macro_f1": macro["f1"]}


def match_detections(detections: list[dict], ground_truth: list[dict],
                     max_distance: float = 30.0) -> tuple[list[tuple[dict, dict]],
                                                          list[dict], list[dict]]:
    """Greedy nearest-centroid matching between detections and ground truth.

    Returns ``(matched, unmatched_detections, unmatched_truth)`` where a
    match pairs a detection dict (with ``"centroid"``) with a ground-truth
    dict (with ``"cx"``/``"cy"``). Each ground-truth object matches at
    most one detection.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, det in enumerate(detections):
        cx, cy = det["centroid"]
        for j, gt in enumerate(ground_truth):
            dist = math.hypot(cx - gt["cx"], cy - gt["cy"])
            if dist <= max_distance:
                pairs.append((dist, i, j))
    pairs.sort()
    used_det, used_gt = set(), set()
    matched: list[tuple[dict, dict]] = []
    for _, i, j in pairs:
        if i not in used_det and j not in used_gt:
            used_det.add(i)
            used_gt.add(j)
            matched.append((detections[i], ground_truth[j]))
    unmatched_det = [d for i, d in enumerate(detections) if i not in used_det]
    unmatched_gt = [g for j, g in enumerate(ground_truth) if j not in used_gt]
    return matched, unmatched_det, unmatched_gt


def detection_metrics(n_matched: int, n_detected: int, n_truth: int) -> dict[str, float]:
    """Detection precision/recall/F1 from match counts."""
    precision = n_matched / n_detected if n_detected else 0.0
    recall = n_matched / n_truth if n_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "matched": n_matched, "detected": n_detected, "truth": n_truth}
