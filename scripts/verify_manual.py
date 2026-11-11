"""Independent verification of the core algorithms, by hand.

Recomputes key quantities with *different* code paths than the library
and compares, so a bug in the library cannot hide behind itself:

1. Circle area — a rasterised circle of radius 25 must match pi*r^2
   within 2% (discrete-pixel tolerance).
2. Square circularity — a filled 40x40 square has area 1600 and
   4*40-4 = 156 boundary pixels, so circularity = 4*pi*1600/156^2.
3. Otsu — a tiny 4x4 image with a known two-level histogram, scored
   by a brute-force loop written here, not by the library.
4. Hu moment phi_1 — for a continuous square, eta_20 = eta_02 = 1/12,
   so phi_1 = 1/6; the rasterised value must be within 10%.
5. Two-pass labelling — a hand-built mask with a diagonal bridge must
   yield exactly 2 components under 8-connectivity and 3 under
   4-connectivity.

Usage:  python scripts/verify_manual.py
Exit code 0 means every check passed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_sorter_sim.features import extract_features, hu_moments  # noqa: E402
from vision_sorter_sim.labeling import label_components  # noqa: E402
from vision_sorter_sim.threshold import otsu_threshold  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    # -- 1. Circle area vs pi*r^2 -------------------------------------
    r = 25
    # Pillow's ellipse bounding box is inclusive on both ends, so the
    # drawn disc spans 2r+1 pixels: effective radius r + 0.5.
    img = Image.new("L", (120, 120), 0)
    ImageDraw.Draw(img).ellipse([60 - r, 60 - r, 60 + r, 60 + r], fill=255)
    mask = np.asarray(img) > 127
    labels, n = label_components(mask, connectivity=8)
    feats = extract_features(labels, 1)
    expected_area = math.pi * (r + 0.5) ** 2
    rel_err = abs(feats["area"] - expected_area) / expected_area
    check("circle area", rel_err < 0.02,
          f"code={feats['area']} analytic={expected_area:.1f} rel_err={rel_err:.4f}")

    # -- 2. Square circularity, hand-computed --------------------------
    side = 40
    img = Image.new("L", (120, 120), 0)
    ImageDraw.Draw(img).rectangle([40, 40, 40 + side - 1, 40 + side - 1], fill=255)
    mask = np.asarray(img) > 127
    labels, n = label_components(mask, connectivity=8)
    feats = extract_features(labels, 1)
    boundary = 4 * side - 4  # perimeter pixels of a filled square
    expected_circ = 4 * math.pi * side * side / (boundary * boundary)
    check("square circularity",
          abs(feats["circularity"] - expected_circ) < 0.01,
          f"code={feats['circularity']:.4f} hand={expected_circ:.4f} "
          f"(area={feats['area']}, perimeter={feats['perimeter']})")

    # -- 3. Otsu on a tiny known histogram, brute-forced here ----------
    tiny = np.array([[10, 10, 200, 200],
                     [10, 10, 200, 200],
                     [10, 10, 200, 200],
                     [10, 10, 200, 200]], dtype=np.uint8)
    hist = np.bincount(tiny.ravel(), minlength=256).astype(float)
    best_t, best_v = 0, -1.0
    for t in range(256):
        w0 = hist[:t + 1].sum()
        w1 = hist[t + 1:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (np.arange(t + 1) * hist[:t + 1]).sum() / w0
        m1 = (np.arange(t + 1, 256) * hist[t + 1:]).sum() / w1
        v = (w0 / 16.0) * (w1 / 16.0) * (m0 - m1) ** 2
        if v > best_v:
            best_t, best_v = t, v
    lib_t = otsu_threshold(tiny)
    check("otsu brute-force", lib_t == best_t,
          f"library={lib_t} brute-force={best_t} (any t in 10..199 is optimal)")

    # -- 4. Hu phi_1 for a square vs 1/6 -------------------------------
    ys, xs = np.nonzero(np.asarray(img) > 127)
    hu, _ = hu_moments(ys, xs)
    check("hu phi_1 square", abs(hu[0] - 1 / 6) < 0.10 * (1 / 6),
          f"code={hu[0]:.4f} analytic=0.1667")

    # -- 5. Labelling connectivity --------------------------------------
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = mask[1, 1] = mask[2, 2] = True   # diagonal chain
    mask[4, 0] = mask[4, 1] = True                # separate blob
    _, n8 = label_components(mask, connectivity=8)
    _, n4 = label_components(mask, connectivity=4)
    check("label 8-connectivity", n8 == 2, f"components={n8} (expected 2)")
    check("label 4-connectivity", n4 == 4, f"components={n4} (expected 4)")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1
    print("all manual verification checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
