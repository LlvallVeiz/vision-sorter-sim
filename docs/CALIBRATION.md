# Calibration — how the classifier thresholds were chosen

Procedure: `scripts/calibrate.py` generates 40 frames (seed
20261111, 5 objects each, defect rate 0.15), runs detection, matches
detections to ground truth by nearest centroid, and reports the
per-class feature ranges below. All thresholds in
`src/vision_sorter_sim/config.py` are midpoints (or just outside) the
measured class ranges — nothing was guessed. The full JSON summary is
`data/calibration_summary.json`. Re-run with
`python scripts/calibrate.py` after any change to the generator or
feature code.

## Measured feature ranges (40 frames, 200 matched objects)

| feature | circle (n=47) | square (n=33) | rectangle (n=30) | triangle (n=28) | defect (n=62) |
|---|---|---|---|---|---|
| circularity | 1.248–1.312 | 0.806–1.571 | 0.715–1.251 | 0.758–0.810 | 0.750–1.232 |
| extent | 0.777–0.786 | 0.500–0.906 | 0.441–0.910 | 0.465–0.508 | 0.640–0.805 |
| aspect ratio | 1.000–1.021 | 1.000–1.025 | 1.046–1.833 | 1.016–1.177 | 1.000–1.128 |
| solidity | 0.982–0.991 | 0.951–0.988 | 0.951–0.978 | 0.943–0.974 | 0.839–0.977 |
| Hu log₁ (−log₁₀ φ₁) | 0.798–0.798 | 0.777–0.778 | 0.664–0.743 | 0.715–0.716 | 0.731–0.795 |
| hull vertices | 8–16 | 4–5 | 4–5 | 3 | 6–12 |
| min hull edge (px) | 12.6–28.3 | 4.0–58.7 | 4.0–54.2 | 39.1–69.3 | 7.8–20.2 |

Class means used as fallback prototypes (hu_log₁, extent, solidity):
circle (0.798, 0.783, 0.988); square (0.778, 0.630, 0.968);
rectangle (0.702, 0.598, 0.968); triangle (0.715, 0.479, 0.957);
defect (0.765, 0.695, 0.901).

## Threshold derivation

- **Circle** needs all four tests at once: solidity ≥ 0.980 (circles
  0.982+, defects ≤ 0.977), Hu log₁ ≥ 0.795 (circles 0.798, defects
  ≤ 0.795), circularity ≥ 1.24 (circles 1.248+, defects ≤ 1.232),
  extent 0.76–0.80 (circles 0.777–0.786). Any blobby shape failing
  any test is a defect.
- **Triangle**: exactly 3 hull vertices. A rasterised triangle
  occasionally keeps a clipped 4th corner, so the 4–5 vertex branch
  also checks the triangle signature: Hu log₁ 0.710–0.720 (measured
  0.715–0.716) + extent 0.44–0.54 (measured 0.465–0.508) + shortest
  hull edge < 25 px (clipped corners measure 4–20 px; a real
  rectangle's short side measures ≥ 28 px).
- **Square vs rectangle** (4–5 vertices): Hu log₁ ≥ 0.760 → square
  (0.777–0.778), below → rectangle (0.664–0.743).
- **Circularity is not a primary discriminator**: with a
  boundary-pixel perimeter it overlaps across classes (square
  0.81–1.57 vs defect 0.75–1.23); it is only used as one of the four
  circle tests.

## Why rotation broke the naive features

The first classifier used extent + aspect ratio (square extent ~1,
triangle ~0.5). It failed: parts are drawn at random rotations, so a
square rotated 45° has extent 0.5 and a rectangle's bounding-box
aspect collapses toward 1.0. The fix was rotation-invariant features
(Hu φ₁, hull corner count, solidity) — measured above, the ranges
are tight exactly because these features do not depend on rotation.

## Independent verification

`scripts/verify_manual.py` recomputes key quantities with different
code than the library (all 6 checks pass):

- Circle area (r = 25) vs πr²: code 2025, analytic 2042.8 (Pillow's
  inclusive ellipse box makes the effective radius 25.5), rel. error
  0.87%.
- Square circularity (40×40): code 0.8262, hand-computed
  4π·1600/156² = 0.8262 (exact).
- Otsu on a 4×4 two-level image vs a brute-force loop written in the
  script: both return t = 10.
- Hu φ₁ of a square vs the analytic 1/6: code 0.1666.
- Labelling connectivity: diagonal chain = 2 components under
  8-connectivity, 4 under 4-connectivity.

## Held-out validation (different seeds from calibration)

| set | frames | objects | accuracy | detection P/R |
|---|---|---|---|---|
| `examples/dataset` (seed 777000) | 60 | 300 | 100% (300/300) | 1.000 / 1.000 |
| validation 2 (seed 555000) | 40 | 200 | 100% (200/200) | 1.000 / 1.000 |
| validation 3 (seed 999000) | 100 | 500 | 100% (500/500) | 1.000 / 1.000 |
| **total** | **200** | **1000** | **100% (1000/1000)** | **1.000 / 1.000** |

Conveyor run (seed 20261111, 240 items, 12% defects, 60 items/min):
lanes accept 123 / divert 50 / reject 67; false accepts 0; false
rejects 0; sort accuracy 100%; mean inspection time 12.8 ms/item.

Note on scope: these are synthetic images with exact ground truth.
100% here means "the classifier is consistent with the generator's
geometry", not "it would score 100% on real photographs" — see
`docs/RESEARCH.md` §4 for the limits.
