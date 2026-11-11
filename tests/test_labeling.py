"""Tests for two-pass connected-component labelling."""

import numpy as np
import pytest

from vision_sorter_sim.labeling import component_pixels, label_components


def test_empty_mask_has_zero_labels(blank_image):
    labels, n = label_components(blank_image > 0)
    assert n == 0
    assert (labels == 0).all()


def test_single_pixel_is_one_component():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    labels, n = label_components(mask)
    assert n == 1
    assert labels[2, 2] == 1


def test_diagonal_chain_8_vs_4_connectivity():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = mask[1, 1] = mask[2, 2] = True
    mask[4, 0] = mask[4, 1] = True
    _, n8 = label_components(mask, connectivity=8)
    _, n4 = label_components(mask, connectivity=4)
    assert n8 == 2
    assert n4 == 4


def test_two_separate_blobs(circle_mask):
    mask = circle_mask.copy()
    mask[5:8, 5:8] = True  # small second blob, far from the disc
    labels, n = label_components(mask)
    assert n == 2
    assert len(set(labels[mask])) == 2


def test_labels_are_compact_1_to_n(circle_mask):
    labels, n = label_components(circle_mask)
    assert n == 1
    assert set(np.unique(labels)) == {0, 1}


def test_invalid_connectivity_raises():
    with pytest.raises(ValueError):
        label_components(np.ones((3, 3), dtype=bool), connectivity=6)


def test_non_bool_mask_raises():
    with pytest.raises(ValueError):
        label_components(np.ones((3, 3), dtype=np.uint8))


def test_component_pixels_match_label_area(circle_mask):
    labels, n = label_components(circle_mask)
    comps = component_pixels(labels, n)
    assert len(comps) == 1
    ys, xs = comps[0]
    assert len(ys) == int(circle_mask.sum())
    assert (labels[ys, xs] == 1).all()
