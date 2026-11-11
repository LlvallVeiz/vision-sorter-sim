# vision-sorter-sim

End-to-end machine-vision conveyor sorting simulator: synthetic
conveyor images → Otsu thresholding → connected-component labelling
→ shape features → rule-based classification → sort decisions →
statistics, reports and a static HTML dashboard.

All core vision algorithms are implemented **from scratch**
(NumPy/Pillow only, no OpenCV): Otsu's method, two-pass labelling
with union-find, shape features (area, perimeter, circularity,
extent, solidity, convex hull) and the seven Hu moment invariants.

## Background

A camera above a conveyor belt photographs each part; the image is
segmented, shape features are measured, and a classifier routes the
part to a lane: **accept** (good circle/square/triangle), **divert**
(rectangle — second packaging line) or **reject** (defect). The
classic pipeline (threshold → components → features → rules) predates
deep learning and is still used where parts are geometrically well
defined and decisions must be auditable. Theory, sources and the
known failure modes are documented in [`docs/RESEARCH.md`](docs/RESEARCH.md);
how every classifier threshold was measured is in
[`docs/CALIBRATION.md`](docs/CALIBRATION.md).

## Project structure

```
vision-sorter-sim/
├── src/vision_sorter_sim/
│   ├── __init__.py  __main__.py  cli.py        # CLI (generate/detect/classify/sort/report/demo)
│   ├── config.py                                # geometry, classes, calibrated thresholds
│   ├── shapes.py                                # synthetic part drawing (circle/square/rect/triangle/defect)
│   ├── generator.py                             # conveyor-frame generator + ground-truth JSON
│   ├── threshold.py                             # Otsu from scratch + robust fallback
│   ├── labeling.py                              # two-pass connected components (union-find)
│   ├── features.py                              # shape features + Hu moments + convex hull
│   ├── classifier.py                            # rule-based decision tree + prototype fallback
│   ├── metrics.py                               # confusion matrix, precision/recall/F1, matching
│   ├── pipeline.py                              # detect() + dataset evaluation
│   ├── conveyor.py                              # conveyor simulation (lanes, throughput, errors)
│   ├── report.py                                # Markdown + JSON report
│   └── dashboard.py                             # static HTML dashboard (Chart.js CDN)
├── tests/            # 77 pytest tests
├── scripts/
│   ├── calibrate.py      # measure per-class feature ranges -> data/calibration_summary.json
│   └── verify_manual.py  # independent hand-computed verification (6 checks)
├── docs/
│   ├── RESEARCH.md       # theory + references
│   └── CALIBRATION.md    # threshold derivation + held-out validation
├── data/
│   └── calibration_summary.json
└── examples/
    ├── dataset/          # 60 generated frames + ground truth (seed 777000)
    ├── output/           # report.md, report.json, dashboard/index.html from a real run
    ├── frame_0000.png / frame_0000_ground_truth.json
    ├── README_examples.md
    └── run_examples.sh
```

## Requirements

- Python 3.10+ (developed on 3.12)
- `numpy >= 1.26`, `Pillow >= 10` (see `requirements.txt`)
- No OpenCV. `pytest` for the test suite.

## Usage

```bash
pip install -r requirements.txt
export PYTHONPATH=src

# 1. Generate a synthetic dataset (20 frames, 5 parts each)
python -m vision_sorter_sim generate --output data/dataset --frames 20 --seed 0

# 2. Detect + classify one image
python -m vision_sorter_sim detect data/dataset/frame_0000.png

# 3. Evaluate a whole dataset (confusion matrix, P/R/F1)
python -m vision_sorter_sim classify data/dataset

# 4. Simulate the conveyor sorter (120 items, 12% defects)
python -m vision_sorter_sim sort --items 120 --seed 0

# 5. Everything at once: evaluation + simulation + report + dashboard
python -m vision_sorter_sim report data/dataset --output out --items 240

# 6. Quick end-to-end demo
python -m vision_sorter_sim demo data/dataset --items 60
```

## Example output (real runs)

### Detection on one frame (`examples/dataset/frame_0000.png`)

```
examples/dataset/frame_0000.png: 5 object(s)
  # 2 rectangle centroid=(483,81)  area= 2003 circ=0.752 extent=0.622 aspect=1.52 solidity=0.962 conf=1.00
  # 1 rectangle centroid=(278,89)  area= 2674 circ=1.037 extent=0.483 aspect=1.13 solidity=0.972 conf=1.00
  # 3 circle    centroid=(76,308)  area= 2553 circ=1.253 extent=0.786 aspect=1.00 solidity=0.986 conf=1.00
  # 4 circle    centroid=(378,339) area= 1733 circ=1.250 extent=0.784 aspect=1.00 solidity=0.982 conf=1.00
  # 5 square    centroid=(550,390)  area= 3394 circ=1.133 extent=0.524 aspect=1.01 solidity=0.976 conf=1.00
```

### Classification — held-out set (60 frames, 300 objects, seed 777000)

```
frames: 60  matched objects: 300  mean frame time: 51.4 ms
accuracy: 1.0000 (300/300)  macro P/R/F1: 1.000 / 1.000 / 1.000
detection: precision 1.000  recall 1.000  F1 1.000  (300/300 matched)
```

Confusion matrix — every class perfect (see `examples/output/report.md`
for the full table). Two further held-out sets (seeds 555000 and
999000) also scored 500/500 and 200/200; total **1000/1000 objects**
across three independent sets.

### Conveyor run (240 items, 12% defects, 60 items/min)

```
items: 240  duration: 240 s  throughput: 60.0 items/min
lanes: {'accept': 123, 'reject': 67, 'divert': 50}
false accepts: 0  false rejects: 0  unknown: 0
sort accuracy: 1.0000  error rate: 0.0000  mean inspect: 12.8 ms/item
```

Open `examples/output/dashboard/index.html` in a browser for the
charts (per-class precision/recall/F1, lane distribution, throughput
over time).

## How the classifier works

A decision tree over rotation-invariant features, calibrated on
synthetic data (see `docs/CALIBRATION.md`):

1. **Triangle** — exactly 3 hull corners; a rasterised triangle that
   keeps a clipped 4th corner is caught by its signature (Hu φ₁ in
   the tight 0.715–0.716 band + extent ~0.5 + a short hull edge
   of 4–20 px).
2. **Square vs rectangle** (4–5 hull corners) — Hu φ₁: squares
   0.777–0.778, rectangles 0.664–0.743.
3. **Circle vs defect** (6+ hull corners) — a circle must pass all
   four tests (solidity ≥ 0.98, Hu φ₁ ≥ 0.795, circularity ≥ 1.24,
   extent 0.76–0.80); anything else blobby is a defect.
4. **Fallback** — nearest class prototype in z-scored
   (Hu φ₁, extent, solidity) space; too far from every prototype →
   `unknown`.

## Tests and verification

```bash
PYTHONPATH=src python -m pytest tests/ -q   # 77 tests
python scripts/verify_manual.py             # 6 hand-computed checks
python scripts/calibrate.py                 # re-measure feature ranges
```

Independent verification (`scripts/verify_manual.py`, different code
paths than the library): circle area vs πr² (0.87% error), square
circularity vs the hand-computed 4π·1600/156² (exact), Otsu vs a
brute-force loop (identical), Hu φ₁ of a square vs the analytic 1/6
(0.1666 vs 0.1667), labelling connectivity (2 vs 4 components for a
diagonal chain under 8- vs 4-connectivity).

## Design decisions

- **Synthetic data with exact ground truth** — every metric is
  checkable; no invented numbers.
- **Rule-based classifier** — auditable, no training set; its failure
  modes are named in `docs/RESEARCH.md`.
- **No OpenCV** — every vision algorithm is implemented from scratch
  and verified against its defining paper.
- **Otsu guard** — plain Otsu fails when the foreground is ~0.3% of
  the image (it splits the background in half; found live: one frame
  produced 6,756 components and a 16 s detection). `binarise()`
  falls back to a robust mode + 6σ threshold when the foreground
  fraction exceeds 40%.

## Limitations

- Synthetic belt only: uniform background, no occlusion, no touching
  parts, no motion blur, no colour. A real line would need
  illumination control and photographed training data.
- The 100% scores are on synthetic geometry — they prove the
  classifier is consistent with the generator, not that it would
  score 100% on real photographs.
- Throughput is simulated from belt speed and spacing, not measured
  on hardware.

## License

MIT — see [LICENSE](LICENSE).
```

## Version

1.0.0 — 2026-11-11
