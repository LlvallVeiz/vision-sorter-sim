"""Conveyor simulation: a stream of parts passes the camera, each part
is inspected by the vision pipeline, and a sort decision routes it to
one of three lanes:

- ``accept`` — a good part (circle / square / triangle).
- ``divert`` — a rectangle, routed to a second line (different
  packaging in this scenario).
- ``reject`` — a defect, or a part the classifier could not recognise.

The simulator tracks throughput (items/minute), lane counts, and the
two errors that cost money in a real line: false accepts (a defect
shipped as good) and false rejects (a good part thrown away). Timing
is simulated from the belt speed and item spacing, not wall-clock.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import config
from .generator import generate_frame
from .pipeline import detect


@dataclass
class SortEvent:
    """One inspected part and its sort outcome."""

    index: int
    true_label: str
    predicted_label: str
    lane: str
    confidence: float
    correct: bool
    time_s: float  # simulated arrival time at the inspection zone


@dataclass
class ConveyorRun:
    """Aggregate result of a conveyor run."""

    events: list[SortEvent] = field(default_factory=list)
    duration_s: float = 0.0
    items_per_min: float = 0.0
    lane_counts: dict = field(default_factory=dict)
    false_accepts: int = 0   # defect predicted as a good class
    false_rejects: int = 0   # good part sent to reject
    unknown: int = 0
    mean_inspect_ms: float = 0.0

    @property
    def total(self) -> int:
        return len(self.events)

    @property
    def error_rate(self) -> float:
        if not self.events:
            return 0.0
        return (self.false_accepts + self.false_rejects) / len(self.events)

    @property
    def accuracy(self) -> float:
        if not self.events:
            return 0.0
        return sum(1 for e in self.events if e.correct) / len(self.events)

    def to_dict(self) -> dict:
        return {
            "total_items": self.total,
            "duration_s": round(self.duration_s, 1),
            "items_per_min": round(self.items_per_min, 1),
            "lane_counts": self.lane_counts,
            "false_accepts": self.false_accepts,
            "false_rejects": self.false_rejects,
            "unknown": self.unknown,
            "error_rate": round(self.error_rate, 4),
            "accuracy": round(self.accuracy, 4),
            "mean_inspect_ms": round(self.mean_inspect_ms, 1),
            "events": [
                {"index": e.index, "true_label": e.true_label,
                 "predicted_label": e.predicted_label, "lane": e.lane,
                 "confidence": e.confidence, "correct": e.correct,
                 "time_s": round(e.time_s, 2)}
                for e in self.events
            ],
        }


def decide_lane(predicted_label: str) -> str:
    """Map a predicted class to a sort lane."""
    if predicted_label == "rectangle":
        return config.LANE_DIVERT
    if predicted_label in config.GOOD_CLASSES:
        return config.LANE_ACCEPT
    return config.LANE_REJECT  # defect or unknown


def run_conveyor(n_items: int, seed: int = 0,
                 defect_rate: float = config.DEFAULT_DEFECT_RATE,
                 items_per_min: float = config.DEFAULT_ITEMS_PER_MIN,
                 belt_speed: float = config.BELT_SPEED_M_PER_S,
                 spacing_m: float = config.ITEM_SPACING_M) -> ConveyorRun:
    """Simulate ``n_items`` parts travelling past the inspection camera.

    Each part gets its own single-object frame (seeded, reproducible);
    the vision pipeline inspects it and the lane decision is recorded.
    Arrival times follow the belt: spacing / belt speed between parts,
    capped by the nominal arrival rate.
    """
    interval_s = max(spacing_m / belt_speed, 60.0 / items_per_min)
    events: list[SortEvent] = []
    inspect_times: list[float] = []
    false_accepts = false_rejects = unknown = 0
    lane_counts = {config.LANE_ACCEPT: 0, config.LANE_REJECT: 0, config.LANE_DIVERT: 0}

    for i in range(n_items):
        frame = generate_frame(seed=seed + 10_000 + i, n_objects=1,
                               defect_rate=defect_rate)
        truth = frame.objects[0]
        t0 = time.perf_counter()
        detections = detect(frame.image)
        inspect_times.append((time.perf_counter() - t0) * 1000.0)
        if detections:
            det = detections[0]
            predicted = det["label"]
            confidence = det["confidence"]
        else:
            predicted, confidence = "unknown", 0.0
        lane = decide_lane(predicted)
        lane_counts[lane] += 1
        correct = predicted == truth.label
        if truth.label == "defect" and lane == config.LANE_ACCEPT:
            false_accepts += 1
        if truth.label in config.GOOD_CLASSES and lane == config.LANE_REJECT:
            false_rejects += 1
        if predicted == "unknown":
            unknown += 1
        events.append(SortEvent(index=i, true_label=truth.label,
                                predicted_label=predicted, lane=lane,
                                confidence=confidence, correct=correct,
                                time_s=i * interval_s))

    duration_s = max(n_items * interval_s, 1e-9)
    run = ConveyorRun(
        events=events,
        duration_s=duration_s,
        items_per_min=n_items / duration_s * 60.0,
        lane_counts=lane_counts,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        unknown=unknown,
        mean_inspect_ms=float(np.mean(inspect_times)) if inspect_times else 0.0,
    )
    return run


def throughput_series(run: ConveyorRun, bucket_s: float = 60.0) -> list[dict]:
    """Items inspected per time bucket (for the dashboard chart)."""
    if not run.events:
        return []
    n_buckets = max(1, int(math.ceil(run.duration_s / bucket_s)))
    counts = [0] * n_buckets
    for event in run.events:
        counts[min(int(event.time_s // bucket_s), n_buckets - 1)] += 1
    return [{"t_min": round(i * bucket_s / 60.0, 2),
             "items": counts[i],
             "items_per_min": round(counts[i] / (bucket_s / 60.0), 1)}
            for i in range(n_buckets)]
