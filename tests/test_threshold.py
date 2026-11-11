"""Tests for Otsu thresholding (implemented from scratch)."""

import numpy as np

from vision_sorter_sim.threshold import (
    binarise,
    histogram,
    otsu_threshold,
    robust_threshold,
)


def test_otsu_two_level_image(two_level_image):
    # Two clean levels 50 and 200: any threshold in [50, 199] is optimal;
    # the implementation returns the first maximum.
    t = otsu_threshold(two_level_image)
    assert 50 <= t < 200


def test_otsu_returns_int_in_range(frame_3_objects):
    t = otsu_threshold(frame_3_objects.image)
    assert isinstance(t, int)
    assert 0 <= t <= 255
    # Bright objects on a dark belt: threshold must sit between them.
    # The belt level itself (45) is a valid Otsu answer: with a clean
    # two-level histogram every t in [45, 187] scores the same, and the
    # implementation returns the first maximum.
    assert 45 <= t < 200


def test_otsu_uniform_image_is_zero_or_max(blank_image):
    # Degenerate histogram: no split exists; must not crash or NaN.
    t = otsu_threshold(blank_image)
    assert t == 0


def test_binarise_uses_otsu_by_default(frame_3_objects):
    mask, t = binarise(frame_3_objects.image)
    assert mask.dtype == bool
    assert mask.sum() > 0  # objects present
    assert mask.sum() < mask.size  # belt present
    assert t == otsu_threshold(frame_3_objects.image)


def test_binarise_fixed_threshold(two_level_image):
    mask, t = binarise(two_level_image, threshold=100)
    assert t == 100
    assert mask[:, :4].sum() == 0
    assert mask[:, 4:].sum() == 32


def test_histogram_sums_to_pixel_count(two_level_image):
    hist = histogram(two_level_image)
    assert len(hist) == 256
    assert hist.sum() == two_level_image.size
    assert hist[50] == 32 and hist[200] == 32


def test_binarise_falls_back_when_otsu_splits_background():
    """Regression test: a single small triangle on a noisy belt.
    Plain Otsu returns 45 here (it splits the belt's own Gaussian in
    half, because the object is only 0.3% of the pixels); binarise
    must fall back to the robust threshold and return only the part."""
    from vision_sorter_sim.generator import generate_frame
    frame = generate_frame(seed=20261111 + 10_000 + 188, n_objects=1,
                           defect_rate=0.12)
    assert otsu_threshold(frame.image) <= 60  # plain Otsu fails on this input
    mask, t = binarise(frame.image)
    assert t > 60
    assert mask.mean() < 0.05
    # The triangle itself must be foreground.
    assert mask[400, 139]


def test_robust_threshold_above_mode():
    rng = np.random.default_rng(1)
    img = np.clip(rng.normal(45, 8, size=(100, 100)), 0, 255).astype(np.uint8)
    t = robust_threshold(img)
    assert 60 < t < 150  # mode 45 + 6 sigma (~48) lands near 90


def test_otsu_rejects_non_uint8():
    import pytest
    with pytest.raises(ValueError):
        otsu_threshold(np.zeros((4, 4), dtype=np.float64))
