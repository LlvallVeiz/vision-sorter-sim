"""Shared fixtures for the vision-sorter-sim test suite."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vision_sorter_sim.generator import generate_frame


@pytest.fixture()
def blank_image() -> np.ndarray:
    return np.zeros((100, 100), dtype=np.uint8)


@pytest.fixture()
def two_level_image() -> np.ndarray:
    """8x8 image: left half gray 50, right half gray 200."""
    img = np.zeros((8, 8), dtype=np.uint8)
    img[:, :4] = 50
    img[:, 4:] = 200
    return img


@pytest.fixture()
def circle_mask() -> np.ndarray:
    """Boolean mask with a single filled disc, radius 20 at (50, 50)."""
    img = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(img).ellipse([30, 30, 70, 70], fill=255)
    return np.asarray(img) > 127


@pytest.fixture()
def frame_3_objects():
    """A seeded 3-object frame (circle, square, triangle)."""
    return generate_frame(seed=7, n_objects=3, defect_rate=0.0,
                          noise_sigma=0.0, illumination_tilt=0.0)
