"""Shape feature extraction from a labelled component.

Features (all from scratch, NumPy only):

- ``area``: foreground pixel count.
- ``perimeter``: boundary-pixel count (a foreground pixel with at least
  one 4-neighbour outside the component). This is a discrete estimate of
  the true perimeter; see docs/RESEARCH.md for the bias it introduces.
- ``circularity``: 4*pi*area / perimeter^2 (1.0 for a perfect circle).
- ``extent``: area / bounding-box area (1.0 for an axis-aligned rectangle).
- ``aspect_ratio``: max(bbox_w, bbox_h) / min(bbox_w, bbox_h).
- ``solidity``: area / convex-hull area (1.0 for convex shapes; a
  notched or irregular part scores well below 1). The hull area is
  corrected by Pick's-theorem (+ perimeter/2 + 1) because the hull
  passes through pixel centres.
- ``n_hull_vertices``: corner count of the Douglas-Peucker simplified
  hull (3 = triangle, 4 = square/rectangle, many = circle/blob).
- ``hu``: the seven Hu moment invariants (Hu 1962), plus their
  log-transformed form (``hu_log``) as used for shape matching.

The convex hull uses the monotone-chain (Andrew) algorithm.
"""

from __future__ import annotations

import math

import numpy as np


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull of 2D points; returns hull vertices.

    ``points`` is an (N, 2) array of (x, y). Returns the hull vertices in
    counter-clockwise order without repeating the first point.
    """
    pts = np.unique(points, axis=0)
    if len(pts) <= 1:
        return pts
    # Sort lexicographically by x, then y.
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    return hull


def douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Simplify a polyline with the Douglas-Peucker algorithm.

    Used to reduce a rasterised convex hull to its true corners:
    a triangle simplifies to 3 vertices, a square/rectangle to 4,
    while a circle keeps many. ``epsilon`` is the maximum allowed
    deviation in pixels.
    """
    if len(points) <= 2:
        return points

    def _simplify(pts: np.ndarray) -> list[np.ndarray]:
        if len(pts) <= 2:
            return [pts[0], pts[-1]] if len(pts) == 2 else list(pts)
        start, end = pts[0], pts[-1]
        line = end - start
        length = float(np.linalg.norm(line))
        if length == 0:
            dists = np.linalg.norm(pts - start, axis=1)
        else:
            # 2D cross product magnitude |line x (p - start)| / |line|.
            rel = pts - start
            dists = np.abs(line[0] * rel[:, 1] - line[1] * rel[:, 0]) / length
        idx = int(np.argmax(dists))
        if dists[idx] <= epsilon:
            return [start, end]
        left = _simplify(pts[:idx + 1])
        right = _simplify(pts[idx:])
        return left[:-1] + right

    simplified = _simplify(np.asarray(points, dtype=np.float64))
    return np.array(simplified)


def polygon_area(points: np.ndarray) -> float:
    """Shoelace area of a polygon given as an (N, 2) vertex array."""
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def hu_moments(ys: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Seven Hu moment invariants of a binary component.

    Computed from the raw moments m_pq = sum x^p y^q over the component
    pixels, via central moments mu_pq and scale-normalised moments
    eta_pq = mu_pq / mu_00^(1 + (p+q)/2). Returns ``(hu, hu_log)``
    where ``hu_log[i] = -sign(hu[i]) * log10(|hu[i]|)`` (the OpenCV
    ``log`` convention, which compresses the huge dynamic range).
    """
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    m00 = float(len(x))
    if m00 == 0:
        return np.zeros(7), np.zeros(7)
    x_bar = x.mean()
    y_bar = y.mean()
    xc = x - x_bar
    yc = y - y_bar

    def central(p: int, q: int) -> float:
        return float(np.sum((xc ** p) * (yc ** q)))

    mu20, mu02, mu11 = central(2, 0), central(0, 2), central(1, 1)
    mu30, mu12 = central(3, 0), central(1, 2)
    mu21, mu03 = central(2, 1), central(0, 3)

    def eta(p: int, q: int, mu: float) -> float:
        return mu / (m00 ** (1.0 + (p + q) / 2.0))

    n20, n02, n11 = eta(2, 0, mu20), eta(0, 2, mu02), eta(1, 1, mu11)
    n30, n12 = eta(3, 0, mu30), eta(1, 2, mu12)
    n21, n03 = eta(2, 1, mu21), eta(0, 3, mu03)

    hu = np.empty(7, dtype=np.float64)
    hu[0] = n20 + n02
    hu[1] = (n20 - n02) ** 2 + 4 * n11 ** 2
    hu[2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    hu[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    hu[4] = ((n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
             + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    hu[5] = ((n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2)
             + 4 * n11 * (n30 + n12) * (n21 + n03))
    hu[6] = ((3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
             - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    with np.errstate(divide="ignore"):
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-300)
    return hu, hu_log


def extract_features(labels: np.ndarray, label_id: int) -> dict:
    """Extract the full feature dict for one component of a label image."""
    ys, xs = np.nonzero(labels == label_id)
    area = int(len(xs))
    if area == 0:
        raise ValueError(f"component {label_id} is empty")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    bbox_area = bw * bh

    # Boundary pixels: foreground pixels missing at least one 4-neighbour
    # inside the same component.
    comp = labels == label_id
    interior = comp.copy()
    interior[1:-1, 1:-1] &= (comp[:-2, 1:-1] & comp[2:, 1:-1]
                            & comp[1:-1, :-2] & comp[1:-1, 2:])
    perimeter = int(np.count_nonzero(comp & ~interior))

    circularity = (4.0 * math.pi * area / (perimeter * perimeter)
                   if perimeter > 0 else 0.0)
    extent = area / bbox_area if bbox_area else 0.0
    aspect_ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) else 0.0

    boundary_ys, boundary_xs = np.nonzero(comp & ~interior)
    boundary_pts = np.column_stack([boundary_xs, boundary_ys]).astype(np.float64)
    hull = convex_hull(boundary_pts)
    # The hull passes through pixel centres, so its shoelace area misses
    # the outer half-pixel rim. Pick's-theorem style correction
    # (+ perimeter/2 + 1) recovers the true pixel area for convex shapes.
    hull_area = polygon_area(hull) + 0.5 * perimeter + 1.0
    solidity = area / hull_area if hull_area > 0 else 0.0
    # Simplified hull: true corner count (3 = triangle, 4 = quad,
    # many = circle/blob). Epsilon 3 px absorbs rasterisation staircases.
    closed_hull = np.vstack([hull, hull[:1]]) if len(hull) else hull
    simple_hull = douglas_peucker(closed_hull, epsilon=3.0)
    n_hull_vertices = int(len(simple_hull) - 1) if len(simple_hull) > 1 else 0
    # Shortest edge of the simplified hull: a rasterised triangle that
    # keeps a clipped 4th corner has one very short edge (a few px),
    # while a real rectangle's shortest edge is its short side (>= ~20 px).
    simple_vertices = simple_hull[:-1] if len(simple_hull) > 1 else simple_hull
    if len(simple_vertices) >= 2:
        rolled = np.roll(simple_vertices, -1, axis=0)
        min_hull_edge = float(np.min(np.linalg.norm(simple_vertices - rolled, axis=1)))
    else:
        min_hull_edge = 0.0

    hu, hu_log = hu_moments(ys, xs)

    return {
        "label_id": int(label_id),
        "area": area,
        "perimeter": perimeter,
        "bbox": (x0, y0, x1, y1),
        "bbox_w": bw,
        "bbox_h": bh,
        "centroid": (float(xs.mean()), float(ys.mean())),
        "circularity": float(circularity),
        "extent": float(extent),
        "aspect_ratio": float(aspect_ratio),
        "solidity": float(solidity),
        "hull_area": float(hull_area),
        "n_hull_vertices": n_hull_vertices,
        "min_hull_edge": min_hull_edge,
        "hu": hu,
        "hu_log": hu_log,
    }


def feature_vector(features: dict) -> np.ndarray:
    """Compact vector used by the classifier fallback: Hu log-1,
    extent, solidity — the three most rotation-robust features."""
    return np.array([features["hu_log"][0], features["extent"],
                     features["solidity"]], dtype=np.float64)
