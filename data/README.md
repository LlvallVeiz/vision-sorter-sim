# data/

- `calibration_summary.json` — per-class feature ranges measured by
  `scripts/calibrate.py` (40 frames, seed 20261111, 200 matched
  objects). The thresholds in `src/vision_sorter_sim/config.py` are
  derived from these numbers; see `docs/CALIBRATION.md`.

Generated datasets are not committed here by default — create them
with `python -m vision_sorter_sim generate --output data/dataset`.
