"""Tests for the conveyor simulator and sort-lane logic."""

import pytest

from vision_sorter_sim.conveyor import (
    decide_lane,
    run_conveyor,
    throughput_series,
)


def test_decide_lane_mapping():
    assert decide_lane("circle") == "accept"
    assert decide_lane("square") == "accept"
    assert decide_lane("triangle") == "accept"
    assert decide_lane("rectangle") == "divert"
    assert decide_lane("defect") == "reject"
    assert decide_lane("unknown") == "reject"


def test_run_conveyor_counts_add_up():
    run = run_conveyor(n_items=30, seed=1, defect_rate=0.2)
    assert run.total == 30
    assert sum(run.lane_counts.values()) == 30
    assert 0.0 <= run.accuracy <= 1.0
    assert 0.0 <= run.error_rate <= 1.0
    assert run.false_accepts + run.false_rejects <= 30
    assert run.items_per_min > 0
    assert run.mean_inspect_ms > 0


def test_run_conveyor_is_reproducible():
    a = run_conveyor(n_items=20, seed=5)
    b = run_conveyor(n_items=20, seed=5)
    assert [e.predicted_label for e in a.events] == [e.predicted_label for e in b.events]
    assert a.lane_counts == b.lane_counts


def test_run_conveyor_no_defects_means_no_false_accepts():
    run = run_conveyor(n_items=20, seed=3, defect_rate=0.0)
    assert run.false_accepts == 0


def test_run_conveyor_all_defects_rejected_or_unknown():
    run = run_conveyor(n_items=20, seed=4, defect_rate=1.0)
    assert run.lane_counts["accept"] == 0
    assert run.false_rejects == 0


def test_throughput_series_buckets():
    run = run_conveyor(n_items=120, seed=2)
    series = throughput_series(run, bucket_s=60.0)
    assert sum(s["items"] for s in series) == 120
    assert all(s["items_per_min"] > 0 for s in series)
