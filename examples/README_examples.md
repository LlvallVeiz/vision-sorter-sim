# Examples

All files here come from real runs (nothing hand-written). Reproduce
them with `bash run_examples.sh` from the project root.

## Contents

- `dataset/` — 60 generated conveyor frames (`frame_0000.png` …
  `frame_0059.png`) with matching ground-truth JSON files. Generated
  with `generate --frames 60 --seed 777000 --objects 5
  --defect-rate 0.12` (5 parts per frame, 12% defects, Gaussian noise
  σ = 8, illumination tilt ±6 levels).
- `frame_0000.png` + `frame_0000_ground_truth.json` — one frame
  singled out for quick inspection.
- `output/report.md` — Markdown report of the evaluation (60 frames,
  300 objects) plus the conveyor run (240 items).
- `output/report.json` — the same numbers as JSON.
- `output/dashboard/index.html` — static dashboard (Chart.js via
  CDN); open it in a browser.

## Key numbers (from these exact files)

- Classification accuracy **100% (300/300)**, macro precision /
  recall / F1 all 1.000; detection precision/recall 1.000 (300/300
  matched); mean frame time 51.4 ms.
- Conveyor run: 240 items in 240 s (**60.0 items/min**), lanes
  accept 123 / divert 50 / reject 67, **false accepts 0**,
  **false rejects 0**, sort accuracy 100%, mean inspection time
  12.8 ms/item.

## One-frame detection (`frame_0000.png`)

```
examples/dataset/frame_0000.png: 5 object(s)
  # 2 rectangle centroid=(483,81)  area= 2003 circ=0.752 extent=0.622 aspect=1.52 solidity=0.962 conf=1.00
  # 1 rectangle centroid=(278,89)  area= 2674 circ=1.037 extent=0.483 aspect=1.13 solidity=0.972 conf=1.00
  # 3 circle    centroid=(76,308)  area= 2553 circ=1.253 extent=0.786 aspect=1.00 solidity=0.986 conf=1.00
  # 4 circle    centroid=(378,339) area= 1733 circ=1.250 extent=0.784 aspect=1.00 solidity=0.982 conf=1.00
  # 5 square    centroid=(550,390)  area= 3394 circ=1.133 extent=0.524 aspect=1.01 solidity=0.976 conf=1.00
```
