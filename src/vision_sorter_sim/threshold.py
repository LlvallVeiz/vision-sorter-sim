"""Thresholding: Otsu's method implemented from scratch.

Otsu (1979) picks the gray level that maximises the between-class
variance of the histogram, i.e. the split that makes foreground and
background as compact and as far apart as possible. It assumes a
bimodal histogram — exactly the case for bright parts on a dark belt.
"""

from __future__ import annotations

import numpy as np

from . import config


def otsu_threshold(image: np.ndarray) -> int:
    """Return the Otsu threshold for a uint8 grayscale image.

    Exhaustively scores all 256 candidate levels with the closed form
    sigma_b^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2 and returns the
    level with the largest between-class variance (first maximum wins
    on ties, matching common implementations).
    """
    if image.dtype != np.uint8:
        raise ValueError("otsu_threshold expects a uint8 image")
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0
    prob = hist / total
    levels = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(prob)
    mu = np.cumsum(prob * levels)
    mu_total = mu[-1]
    # Between-class variance, guarding against empty classes.
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_total * w0 - mu) ** 2 / np.where(
            (w0 > 0) & (w0 < 1), w0 * (1.0 - w0), np.nan)
    sigma_b2 = np.nan_to_num(sigma_b2, nan=-1.0)
    return int(np.argmax(sigma_b2))


def robust_threshold(image: np.ndarray, k: float = 6.0) -> int:
    """Fallback threshold: histogram mode + ``k`` background sigmas.

    Used when Otsu splits the background instead of the object (its
    known small-foreground failure mode: with a tiny object the
    between-class variance is larger for cutting the big background
    class in half). The background sigma is estimated from the pixels
    below the mode, corrected for the half-normal truncation
    (std of a half-normal is ~0.60 * sigma).
    """
    hist = histogram(image)
    mode = int(np.argmax(hist))
    below = image.ravel()[image.ravel() <= mode].astype(np.float64)
    sigma = float(below.std()) / 0.6028 if below.size > 1 else 1.0
    threshold = int(round(mode + k * sigma))
    return max(mode + 15, min(250, threshold))


def binarise(image: np.ndarray, threshold: int | None = None) -> tuple[np.ndarray, int]:
    """Binarise an image: foreground (object) pixels are True.

    With ``threshold=None`` the Otsu threshold is computed first. If
    Otsu's split leaves more than ``MAX_FOREGROUND_FRACTION`` of the
    image as foreground, it has split the belt, not the parts, and the
    robust mode + k*sigma threshold is used instead.
    Returns ``(mask, threshold_used)``.
    """
    if threshold is None:
        threshold = otsu_threshold(image)
        if float((image > threshold).mean()) > config.MAX_FOREGROUND_FRACTION:
            threshold = robust_threshold(image)
    mask = image > threshold
    return mask, int(threshold)


def histogram(image: np.ndarray) -> np.ndarray:
    """256-bin histogram of a uint8 image (counts, not probabilities)."""
    return np.bincount(image.ravel(), minlength=256)
