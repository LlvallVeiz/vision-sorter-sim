"""Tests for shape feature extraction (area, perimeter, circularity,
extent, solidity, hull corners, Hu moments)."""

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vision_sorter_sim.features import (
    convex_hull,
    douglas_peucker,
    extract_features,
    hu_moments,
    polygon_area,
)
from vision_sorter_sim.labeling import label_components
from vision_sorter_sim.shapes import regular_polygon


def _features_of(mask: np.ndarray) -> dict:
    labels, n = label_components(mask)
    assert n == 1
    return extract_features(labels, 1)


def _filled_polygon_image(points, size=120) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).polygon(points, fill=255)
    return np.asarray(img) > 127


def test_circle_area_matches_pi_r_squared(circle_mask):
    feats = _features_of(circle_mask)
    # Pillow ellipse [30,70] spans 41 px -> effective radius 20.5.
    expected = math.pi * 20.5 ** 2
    assert abs(feats["area"] - expected) / expected < 0.02


def test_square_extent_and_hull_vertices():
    pts = [(40, 40), (80, 40), (80, 80), (40, 80)]
    feats = _features_of(_filled_polygon_image(pts))
    assert feats["area"] == 41 * 41  # inclusive Pillow rectangle
    assert feats["extent"] == pytest.approx(1.0, abs=0.01)
    assert feats["n_hull_vertices"] == 4


def test_triangle_hull_vertices_and_extent():
    pts = regular_polygon(60, 60, 30, 3, rotation_deg=0.0)
    feats = _features_of(_filled_polygon_image(pts))
    assert feats["n_hull_vertices"] == 3
    # A triangle fills half its bounding box.
    assert feats["extent"] == pytest.approx(0.5, abs=0.03)


def test_circle_hull_vertices_many(circle_mask):
    feats = _features_of(circle_mask)
    assert feats["n_hull_vertices"] >= 8


def test_solidity_near_one_for_convex_shapes(circle_mask):
    feats = _features_of(circle_mask)
    assert 0.95 < feats["solidity"] <= 1.0


def test_solidity_drops_for_notched_circle():
    img = Image.new("L", (120, 120), 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 30, 90, 90], fill=255)
    # Bite a 60-degree wedge out of the disc.
    draw.polygon([(60, 60), (95, 60), (78, 90)], fill=0)
    mask = np.asarray(img) > 127
    feats = _features_of(mask)
    assert feats["solidity"] < 0.93


def test_hu_phi1_square_is_one_sixth():
    pts = [(40, 40), (80, 40), (80, 80), (40, 80)]
    mask = _filled_polygon_image(pts)
    ys, xs = np.nonzero(mask)
    hu, _ = hu_moments(ys, xs)
    assert hu[0] == pytest.approx(1 / 6, rel=0.05)


def test_hu_moments_rotation_invariant():
    pts_a = regular_polygon(60, 60, 30, 3, rotation_deg=0.0)
    pts_b = regular_polygon(60, 60, 30, 3, rotation_deg=47.0)
    ys_a, xs_a = np.nonzero(_filled_polygon_image(pts_a))
    ys_b, xs_b = np.nonzero(_filled_polygon_image(pts_b))
    hu_a, _ = hu_moments(ys_a, xs_a)
    hu_b, _ = hu_moments(ys_b, xs_b)
    # phi_1 is stable under rotation; higher-order moments of a small
    # rasterised shape are not (discrete sampling), so only phi_1 is
    # asserted here.
    assert hu_a[0] == pytest.approx(hu_b[0], rel=0.05)


def test_convex_hull_of_square():
    pts = np.array([[0, 0], [1, 0], [2, 0], [2, 2], [0, 2], [1, 1]], dtype=float)
    hull = convex_hull(pts)
    assert len(hull) == 4
    assert polygon_area(hull) == pytest.approx(4.0)


def test_douglas_peucker_straight_line_collapses():
    pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    assert len(douglas_peucker(pts, epsilon=0.5)) == 2


def test_extract_features_empty_component_raises():
    labels = np.zeros((5, 5), dtype=np.int32)
    with pytest.raises(ValueError):
        extract_features(labels, 1)
