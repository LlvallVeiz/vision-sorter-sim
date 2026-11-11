"""Detection pipeline: threshold -> label -> features -> classify.

One function per stage plus :func:`detect` (full frame) and
:func:`evaluate_dataset` (whole directory of frames + ground truth),
which is what the CLI ``classify`` command and the conveyor simulator
both build on.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .classifier import classify
from .features import extract_features
from .labeling import label_components
from .metrics import detection_metrics, match_detections, summarise_metrics
from .threshold import binarise


def detect(image: np.ndarray, min_area: int = config.MIN_COMPONENT_AREA,
           threshold: int | None = None) -> list[dict]:
    """Run the full vision pipeline on one grayscale image.

    Returns a list of detection dicts, each with ``centroid``,
    ``bbox``, the shape features and the classifier output
    (``label``, ``confidence``, ``rule``), sorted by centroid x.
    """
    mask, used_threshold = binarise(image, threshold)
    labels, n_labels = label_components(mask, connectivity=config.CONNECTIVITY)
    detections: list[dict] = []
    for label_id in range(1, n_labels + 1):
        feats = extract_features(labels, label_id)
        if feats["area"] < min_area:
            continue
        result = classify(feats)
        detections.append({
            "label_id": label_id,
            "centroid": feats["centroid"],
            "bbox": feats["bbox"],
            "area": feats["area"],
            "perimeter": feats["perimeter"],
            "circularity": round(feats["circularity"], 4),
            "extent": round(feats["extent"], 4),
            "aspect_ratio": round(feats["aspect_ratio"], 4),
            "solidity": round(feats["solidity"], 4),
            "n_hull_vertices": feats["n_hull_vertices"],
            "min_hull_edge": round(feats["min_hull_edge"], 1),
            "hu_log": [round(float(v), 3) for v in feats["hu_log"]],
            "label": result["label"],
            "confidence": round(result["confidence"], 3),
            "rule": result["rule"],
            "threshold": used_threshold,
        })
    detections.sort(key=lambda d: (d["centroid"][1], d["centroid"][0]))
    return detections


def load_image(path: Path) -> np.ndarray:
    """Load an image file as a uint8 grayscale NumPy array."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def evaluate_frame(image: np.ndarray, ground_truth: list[dict],
                   **kwargs) -> dict:
    """Detect on one frame and score it against ground truth.

    Returns detection metrics, the matched (true, predicted) label
    pairs, and the raw detections.
    """
    t0 = time.perf_counter()
    detections = detect(image, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    matched, unmatched_det, unmatched_gt = match_detections(detections, ground_truth)
    det_metrics = detection_metrics(len(matched), len(detections), len(ground_truth))
    return {
        "detections": detections,
        "matched": [(gt["label"], det["label"]) for det, gt in matched],
        "unmatched_detections": unmatched_det,
        "unmatched_truth": unmatched_gt,
        "detection": det_metrics,
        "elapsed_ms": round(elapsed_ms, 1),
    }


def evaluate_dataset(dataset_dir: Path, **kwargs) -> dict:
    """Evaluate every ``frame_*.png`` + ``frame_*.json`` pair in a directory.

    Returns the classification summary (confusion matrix, per-class
    precision/recall/F1, accuracy), the aggregate detection metrics,
    and per-frame results.
    """
    frames = sorted(dataset_dir.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"no frame_*.png files in {dataset_dir}")
    y_true, y_pred = [], []
    det_matched = det_total = truth_total = 0
    elapsed: list[float] = []
    per_frame: list[dict] = []
    for img_path in frames:
        truth_path = img_path.with_suffix(".json")
        truth = json.loads(truth_path.read_text(encoding="utf-8"))["objects"]
        result = evaluate_frame(load_image(img_path), truth, **kwargs)
        for true_label, pred_label in result["matched"]:
            y_true.append(true_label)
            y_pred.append(pred_label)
        det_matched += result["detection"]["matched"]
        det_total += result["detection"]["detected"]
        truth_total += result["detection"]["truth"]
        elapsed.append(result["elapsed_ms"])
        per_frame.append({"frame": img_path.name,
                          "n_truth": len(truth),
                          "n_detected": len(result["detections"]),
                          "n_matched": result["detection"]["matched"],
                          "elapsed_ms": result["elapsed_ms"]})
    summary = summarise_metrics(y_true, y_pred)
    summary["detection"] = detection_metrics(det_matched, det_total, truth_total)
    summary["mean_frame_ms"] = round(float(np.mean(elapsed)), 1) if elapsed else 0.0
    summary["frames"] = len(frames)
    summary["per_frame"] = per_frame
    return summary
