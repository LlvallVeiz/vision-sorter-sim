"""Calibrate the classifier thresholds from a labelled synthetic set.

Generates N frames (fixed seed), runs detection, matches detections to
ground truth, and prints the per-class feature ranges (circularity,
extent, aspect ratio, solidity, Hu log-1). The printed ranges are the
evidence behind the thresholds in ``src/vision_sorter_sim/config.py``;
the JSON summary is written to ``data/calibration_summary.json`` and
the ranges are copied into ``docs/CALIBRATION.md``.

Usage:  python scripts/calibrate.py [--frames 60] [--seed 20261111]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_sorter_sim import config  # noqa: E402
from vision_sorter_sim.generator import generate_frame  # noqa: E402
from vision_sorter_sim.metrics import match_detections  # noqa: E402
from vision_sorter_sim.pipeline import detect  # noqa: E402

FEATURES = ("circularity", "extent", "aspect_ratio", "solidity", "hu_log_1",
            "n_hull_vertices", "min_hull_edge")


def collect(n_frames: int, seed: int) -> dict[str, list[dict]]:
    """Detect on generated frames and bucket features by true class."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for i in range(n_frames):
        frame = generate_frame(seed=seed + i, n_objects=5, defect_rate=0.15)
        detections = detect(frame.image)
        truth = [o.to_dict() for o in frame.objects]
        matched, _, _ = match_detections(detections, truth)
        for det, gt in matched:
            by_class[gt["label"]].append({
                "circularity": det["circularity"],
                "extent": det["extent"],
                "aspect_ratio": det["aspect_ratio"],
                "solidity": det["solidity"],
                "hu_log_1": det["hu_log"][0],
                "n_hull_vertices": det["n_hull_vertices"],
                "min_hull_edge": det["min_hull_edge"],
            })
    return by_class


def summarise(by_class: dict[str, list[dict]]) -> dict:
    out = {}
    for label, rows in sorted(by_class.items()):
        stats = {}
        for feat in FEATURES:
            vals = np.array([r[feat] for r in rows], dtype=float)
            stats[feat] = {
                "n": len(vals),
                "min": round(float(vals.min()), 3),
                "max": round(float(vals.max()), 3),
                "mean": round(float(vals.mean()), 3),
                "std": round(float(vals.std()), 3),
            }
        out[label] = stats
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()

    by_class = collect(args.frames, args.seed)
    summary = summarise(by_class)
    total = sum(len(v) for v in by_class.values())
    print(f"calibration set: {args.frames} frames, {total} matched objects")
    for label, stats in summary.items():
        print(f"\n{label} (n={stats['circularity']['n']}):")
        for feat in FEATURES:
            s = stats[feat]
            print(f"  {feat:<13} min={s['min']:<7} max={s['max']:<7} "
                  f"mean={s['mean']:<7} std={s['std']}")
    out_path = Path(__file__).resolve().parents[1] / "data" / "calibration_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"frames": args.frames, "seed": args.seed, "total": total,
         "per_class": summary}, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
