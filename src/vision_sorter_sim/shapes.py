"""Synthetic shape drawing for the conveyor simulator.

Shapes are drawn with Pillow on a grayscale canvas: circles, squares,
rectangles and triangles (the "good" parts) plus two defect types —
a notched circle (a wedge bitten out) and an irregular blob. Every
shape carries its ground-truth parameters so the generator can emit an
exact annotation alongside the image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from . import config


@dataclass
class ShapeSpec:
    """Ground-truth description of one drawn object."""

    label: str
    cx: float
    cy: float
    size: float
    rotation_deg: float = 0.0
    aspect: float = 1.0
    defect_kind: str = ""
    bbox: tuple[int, int, int, int] = field(default=(0, 0, 0, 0))

    def to_dict(self) -> dict:
        """Serialisable ground-truth record."""
        return {
            "label": self.label,
            "cx": round(self.cx, 1),
            "cy": round(self.cy, 1),
            "size": round(self.size, 1),
            "rotation_deg": round(self.rotation_deg, 1),
            "aspect": round(self.aspect, 2),
            "defect_kind": self.defect_kind,
            "bbox": [int(v) for v in self.bbox],
        }


def rotated_polygon(cx: float, cy: float, points: list[tuple[float, float]],
                    angle_deg: float) -> list[tuple[float, float]]:
    """Rotate ``points`` (relative to the origin) by ``angle_deg`` about (cx, cy)."""
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    out = []
    for x, y in points:
        out.append((cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a))
    return out


def regular_polygon(cx: float, cy: float, radius: float, n_sides: int,
                    rotation_deg: float = 0.0) -> list[tuple[float, float]]:
    """Vertices of a regular n-gon centred at (cx, cy)."""
    points = []
    for k in range(n_sides):
        a = 2.0 * math.pi * k / n_sides - math.pi / 2.0
        points.append((radius * math.cos(a), radius * math.sin(a)))
    return rotated_polygon(cx, cy, points, rotation_deg)


def draw_shape(draw: ImageDraw.ImageDraw, spec: ShapeSpec, gray: int) -> None:
    """Draw one object onto a Pillow ``ImageDraw`` canvas.

    The canvas is the object layer (black background); the belt colour is
    handled by the caller when compositing. Defects are drawn by first
    drawing the base shape and then erasing part of it in black.
    """
    cx, cy, s = spec.cx, spec.cy, spec.size
    label = spec.label
    if label == "circle":
        draw.ellipse([cx - s, cy - s, cx + s, cy + s], fill=gray)
    elif label == "square":
        pts = regular_polygon(cx, cy, s, 4, spec.rotation_deg)
        draw.polygon(pts, fill=gray)
    elif label == "rectangle":
        w, h = s, s / spec.aspect
        pts = rotated_polygon(cx, cy, [(-w, -h), (w, -h), (w, h), (-w, h)],
                              spec.rotation_deg)
        draw.polygon(pts, fill=gray)
    elif label == "triangle":
        pts = regular_polygon(cx, cy, s, 3, spec.rotation_deg)
        draw.polygon(pts, fill=gray)
    elif label == "defect":
        _draw_defect(draw, spec, gray)
    else:  # pragma: no cover - guarded by the generator
        raise ValueError(f"unknown shape label: {label!r}")


def _draw_defect(draw: ImageDraw.ImageDraw, spec: ShapeSpec, gray: int) -> None:
    """Draw a defective part: a circle with a bite taken out, or an
    irregular blob. Both keep roughly the same bounding size as a good
    part so size alone cannot separate them."""
    cx, cy, s = spec.cx, spec.cy, spec.size
    if spec.defect_kind == "chip":
        # Circle with a disc-shaped bite taken out of its edge.
        draw.ellipse([cx - s, cy - s, cx + s, cy + s], fill=gray)
        a = math.radians(spec.rotation_deg)
        bx, by = cx + s * math.cos(a), cy + s * math.sin(a)
        r = s * 0.5
        draw.ellipse([bx - r, by - r, bx + r, by + r], fill=0)
        return
    if spec.defect_kind == "irregular":
        # Blob with randomised radial profile (deterministic per spec).
        rng = np.random.default_rng(int(spec.cx * 31 + spec.cy * 17 + s))
        pts = []
        for k in range(14):
            a = 2.0 * math.pi * k / 14
            r = s * rng.uniform(0.72, 1.0)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        draw.polygon(pts, fill=gray)
        return
    # Default: "notch" — a wedge bitten out of a circle.
    draw.ellipse([cx - s, cy - s, cx + s, cy + s], fill=gray)
    a0 = math.radians(spec.rotation_deg)
    a1 = a0 + math.radians(55.0)
    wedge = [(cx, cy),
             (cx + s * 1.05 * math.cos(a0), cy + s * 1.05 * math.sin(a0)),
             (cx + s * 1.05 * math.cos(a1), cy + s * 1.05 * math.sin(a1))]
    draw.polygon(wedge, fill=0)


def shape_bbox(spec: ShapeSpec) -> tuple[int, int, int, int]:
    """Axis-aligned bounding box of the shape, from its analytic extent."""
    s = spec.size
    if spec.label == "rectangle":
        # Long side s, short side s/aspect; worst case diagonal extent.
        half_diag = 0.5 * math.hypot(s * 2, (s / spec.aspect) * 2)
        ext = half_diag
    else:
        ext = s
    return (int(spec.cx - ext), int(spec.cy - ext),
            int(spec.cx + ext), int(spec.cy + ext))
