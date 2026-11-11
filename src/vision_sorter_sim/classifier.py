"""Rule-based shape classifier.

A small, auditable decision tree over rotation-invariant shape
features, with thresholds calibrated on synthetic data
(docs/CALIBRATION.md). The tree exploits three facts measured on the
calibration set (40 frames, 200 objects):

1. **Hull corner count** (Douglas-Peucker simplified hull) separates
   the polygon families: triangles simplify to 3 vertices (a
   rasterised triangle occasionally keeps a clipped 4th corner, caught
   by the triangle Hu/extent signature below), squares/rectangles to
   4-5, circles and defects to 6+.
2. **Hu moment phi_1** (log form) separates square (0.777-0.778) from
   rectangle (0.664-0.743) and identifies circles (0.798, the
   continuous-domain value 1/(2*pi) = 0.159).
3. **Solidity + circularity + extent** jointly identify a circle:
   only a near-perfect disc passes all four circle tests at once, so
   any other blobby shape (6+ hull vertices) is a defect.

Fallback: nearest class prototype in z-scored (hu_log_1, extent,
solidity) space; if even the nearest prototype is too far, the part
is ``"unknown"`` rather than forced into a class.
"""

from __future__ import annotations

import numpy as np

from . import config
from .features import feature_vector


def classify(features: dict) -> dict:
    """Classify one detected part from its feature dict.

    Returns ``{"label": str, "confidence": float, "rule": str}`` where
    ``confidence`` is 1.0 for a decisive rule hit and a distance-based
    score in (0, 1] for the prototype fallback.
    """
    hu1 = float(features["hu_log"][0])
    solidity = features["solidity"]
    circ = features["circularity"]
    extent = features["extent"]
    n_vertices = features["n_hull_vertices"]

    # 1. Triangle: exactly three hull corners.
    if n_vertices == 3:
        return {"label": "triangle", "confidence": 1.0,
                "rule": "triangle: 3 hull vertices"}

    # 2. Quadrilateral: a rasterised triangle sometimes keeps a 4th
    # corner (a clipped vertex), so check the triangle signature first:
    # Hu phi_1 in the very tight triangle band (0.715-0.716) together
    # with a half-filled bounding box AND a short hull edge (the clipped
    # corner, 4-20 px). A real rectangle can share the Hu/extent band
    # (aspect ~1.7, rotated ~45 deg) but its shortest edge is its short
    # side, >= 28 px, so the edge test separates them.
    if n_vertices in (4, 5):
        if (config.HU_TRIANGLE_MIN <= hu1 <= config.HU_TRIANGLE_MAX
                and config.EXTENT_TRIANGLE_MIN <= extent <= config.EXTENT_TRIANGLE_MAX
                and features["min_hull_edge"] < config.MIN_HULL_EDGE_TRIANGLE_MAX):
            return {"label": "triangle", "confidence": 0.9,
                    "rule": "triangle: Hu1 in triangle band + extent ~0.5 + short hull edge"}
        if hu1 >= config.HU_SQUARE_MIN:
            return {"label": "square", "confidence": 1.0,
                    "rule": "square: 4-5 hull vertices, Hu1 >= %.2f"
                            % config.HU_SQUARE_MIN}
        return {"label": "rectangle", "confidence": 1.0,
                "rule": "rectangle: 4-5 hull vertices, Hu1 < %.2f"
                        % config.HU_SQUARE_MIN}

    # 3. Blobby shape (6+ hull vertices): circle only if it passes all
    # four circle tests; anything else is a defect.
    if n_vertices >= 6:
        is_circle = (
            solidity >= config.SOLIDITY_CIRCLE_MIN
            and hu1 >= config.HU_CIRCLE_MIN
            and circ >= config.CIRCULARITY_CIRCLE_MIN
            and config.EXTENT_CIRCLE_MIN <= extent <= config.EXTENT_CIRCLE_MAX
        )
        if is_circle:
            return {"label": "circle", "confidence": 1.0,
                    "rule": "circle: solidity/Hu1/circularity/extent all match"}
        return {"label": "defect", "confidence": 1.0,
                "rule": "defect: blobby shape failing the circle tests"}

    # Fallback: nearest prototype in z-scored feature space.
    vec = feature_vector(features)
    scales = np.array(config.PROTOTYPE_SCALES, dtype=np.float64)
    best_label, best_dist = None, float("inf")
    for label, proto in config.PROTOTYPES.items():
        dist = float(np.linalg.norm((vec - np.array(proto)) / scales))
        if dist < best_dist:
            best_label, best_dist = label, dist
    if best_dist > config.UNKNOWN_DISTANCE_MAX:
        return {"label": "unknown", "confidence": 0.0,
                "rule": "unknown: nearest prototype %.2f away" % best_dist}
    confidence = float(1.0 / (1.0 + best_dist))
    return {"label": best_label, "confidence": confidence,
            "rule": "prototype fallback: nearest %s (d=%.2f)" % (best_label, best_dist)}
