"""Tests for the rule-based shape classifier."""

import numpy as np

from vision_sorter_sim.classifier import classify


def _feats(**overrides) -> dict:
    base = {
        "circularity": 1.27, "extent": 0.783, "aspect_ratio": 1.0,
        "solidity": 0.988, "n_hull_vertices": 12, "min_hull_edge": 30.0,
        "hu_log": np.array([0.798, 1.5, 2.0, 2.5, 5.0, 4.0, 5.0]),
    }
    base.update(overrides)
    return base


def test_circle_passes_all_four_tests():
    result = classify(_feats())
    assert result["label"] == "circle"
    assert result["confidence"] == 1.0


def test_circle_fails_when_solidity_low():
    # Same blob, but a bite taken out -> defect.
    result = classify(_feats(solidity=0.90))
    assert result["label"] == "defect"


def test_triangle_by_three_vertices():
    result = classify(_feats(n_hull_vertices=3, hu_log=np.array([0.715, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                            extent=0.48, solidity=0.96, circularity=0.78))
    assert result["label"] == "triangle"


def test_triangle_four_vertices_short_edge():
    # A rasterised triangle that kept a clipped 4th corner: Hu1 in the
    # triangle band, half-filled bbox, and a short hull edge.
    result = classify(_feats(n_hull_vertices=4, min_hull_edge=6.0,
                             hu_log=np.array([0.715, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                             extent=0.48, solidity=0.96, circularity=0.78))
    assert result["label"] == "triangle"


def test_rectangle_sharing_triangle_hu_band_stays_rectangle():
    # A rectangle at aspect ~1.7 rotated ~45 deg shares the triangle
    # Hu1/extent band, but its shortest hull edge is its short side
    # (>= 28 px), so it must stay a rectangle.
    result = classify(_feats(n_hull_vertices=4, min_hull_edge=41.0,
                             hu_log=np.array([0.712, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                             extent=0.49, solidity=0.97, circularity=0.95))
    assert result["label"] == "rectangle"


def test_square_vs_rectangle_by_hu1():
    square = classify(_feats(n_hull_vertices=4, hu_log=np.array([0.778, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                             extent=0.63, solidity=0.97, circularity=1.0))
    rectangle = classify(_feats(n_hull_vertices=4, hu_log=np.array([0.70, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                                extent=0.60, solidity=0.97, circularity=0.95))
    assert square["label"] == "square"
    assert rectangle["label"] == "rectangle"


def test_defect_blob_not_a_circle():
    # Irregular blob: many vertices but low solidity and low Hu1.
    result = classify(_feats(solidity=0.88, hu_log=np.array([0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                             circularity=0.9, extent=0.70))
    assert result["label"] == "defect"


def test_unknown_when_far_from_all_prototypes():
    weird = _feats(n_hull_vertices=2, hu_log=np.array([0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                   extent=0.1, solidity=0.3, circularity=0.1)
    result = classify(weird)
    assert result["label"] == "unknown"
    assert result["confidence"] == 0.0


def test_prototype_fallback_near_square():
    # 2 vertices (degenerate path) but features close to the square prototype.
    result = classify(_feats(n_hull_vertices=2, hu_log=np.array([0.778, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                             extent=0.63, solidity=0.968, circularity=1.0))
    assert result["label"] == "square"
    assert 0.0 < result["confidence"] <= 1.0
