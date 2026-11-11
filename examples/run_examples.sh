#!/usr/bin/env bash
# Regenerate every example output from scratch (run from the project root).
set -euo pipefail
export PYTHONPATH=src
PY=python3

echo "== generate dataset (60 frames, seed 777000) =="
$PY -m vision_sorter_sim generate --output examples/dataset \
    --frames 60 --seed 777000 --objects 5 --defect-rate 0.12

echo "== detect one frame =="
$PY -m vision_sorter_sim detect examples/dataset/frame_0000.png

echo "== classify the dataset =="
$PY -m vision_sorter_sim classify examples/dataset

echo "== conveyor simulation (240 items) =="
$PY -m vision_sorter_sim sort --items 240 --seed 20261111 \
    --defect-rate 0.12 --rate 60

echo "== full report + dashboard =="
$PY -m vision_sorter_sim report examples/dataset \
    --output examples/output --items 240 --seed 20261111 \
    --defect-rate 0.12 --rate 60

echo "== tests =="
$PY -m pytest tests/ -q
$PY scripts/verify_manual.py

echo "done."
