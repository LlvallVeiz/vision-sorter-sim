"""Synthetic conveyor-frame generator.

Renders a grayscale top-down view of a conveyor belt with a random
non-overlapping set of parts, adds a linear illumination gradient and
Gaussian sensor noise, and returns the image together with exact
ground-truth annotations (one record per part). The generator is fully
seeded, so any dataset can be reproduced exactly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import config
from .shapes import ShapeSpec, draw_shape, shape_bbox


@dataclass
class Frame:
    """One generated conveyor frame plus its ground truth."""

    image: np.ndarray          # uint8 grayscale array (H, W)
    objects: list[ShapeSpec]   # ground-truth records, same order as drawn

    def save(self, image_path: Path, truth_path: Path) -> None:
        """Write the PNG image and the JSON ground truth side by side."""
        image_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.image).save(image_path)
        truth = {"image": image_path.name,
                 "width": int(self.image.shape[1]),
                 "height": int(self.image.shape[0]),
                 "objects": [o.to_dict() for o in self.objects]}
        truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")


def _place_objects(rng: np.random.Generator, n_objects: int,
                   classes: tuple[str, ...], width: int, height: int,
                   defect_rate: float) -> list[ShapeSpec]:
    """Sample non-overlapping object placements with rejection sampling."""
    specs: list[ShapeSpec] = []
    margin = config.SIZE_MAX + 6
    attempts = 0
    while len(specs) < n_objects and attempts < 400:
        attempts += 1
        label = str(rng.choice(list(classes)))
        if rng.random() < defect_rate and label in config.GOOD_CLASSES:
            label = "defect"
        size = float(rng.uniform(config.SIZE_MIN, config.SIZE_MAX))
        cx = float(rng.uniform(margin, width - margin))
        cy = float(rng.uniform(margin, height - margin))
        rotation = float(rng.uniform(0.0, 360.0))
        aspect = float(rng.uniform(config.RECT_ASPECT_MIN,
                                   config.RECT_ASPECT_MAX)) if label == "rectangle" else 1.0
        defect_kind = ""
        if label == "defect":
            defect_kind = str(rng.choice(["notch", "chip", "irregular"]))
        spec = ShapeSpec(label=label, cx=cx, cy=cy, size=size,
                         rotation_deg=rotation, aspect=aspect,
                         defect_kind=defect_kind)
        spec.bbox = shape_bbox(spec)
        # Keep objects apart so connected-component labelling sees one
        # blob per part (a realistic singulated feed on a conveyor).
        ok = True
        for other in specs:
            dist = math.hypot(cx - other.cx, cy - other.cy)
            if dist < (size + other.size) * 1.15 + 12:
                ok = False
                break
        if ok:
            specs.append(spec)
    return specs


def generate_frame(seed: int, n_objects: int = 5,
                   classes: tuple[str, ...] = config.CLASSES,
                   width: int = config.IMAGE_WIDTH,
                   height: int = config.IMAGE_HEIGHT,
                   defect_rate: float = 0.0,
                   noise_sigma: float = config.NOISE_SIGMA,
                   illumination_tilt: float = config.ILLUMINATION_TILT,
                   belt_gray: int = config.BELT_GRAY,
                   object_gray: int = config.OBJECT_GRAY) -> Frame:
    """Generate one synthetic conveyor frame.

    Pipeline: black object layer -> Pillow drawing -> composite over the
    belt background -> linear illumination gradient -> Gaussian noise ->
    clip to uint8. Objects are bright on a dark belt, matching the
    assumption of the Otsu detector (bimodal histogram).
    """
    rng = np.random.default_rng(seed)
    specs = _place_objects(rng, n_objects, classes, width, height, defect_rate)

    layer = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(layer)
    for spec in specs:
        # Small per-object brightness jitter (surface reflectance).
        gray = int(object_gray + rng.uniform(-12, 12))
        draw_shape(draw, spec, gray)
    objects_arr = np.asarray(layer, dtype=np.float64)

    image = np.full((height, width), float(belt_gray), dtype=np.float64)
    image = np.maximum(image, objects_arr)

    # Linear illumination gradient along x (one side of the belt brighter).
    ramp = np.linspace(-illumination_tilt / 2, illumination_tilt / 2, width)
    image = image + ramp[np.newaxis, :]

    if noise_sigma > 0:
        image = image + rng.normal(0.0, noise_sigma, size=image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return Frame(image=image, objects=specs)


def generate_dataset(output_dir: Path, n_frames: int, seed: int = 0,
                     n_objects: int = 5, defect_rate: float = 0.12,
                     **kwargs) -> list[Path]:
    """Generate ``n_frames`` frames into ``output_dir``.

    Each frame is ``frame_XXXX.png`` with a matching ``frame_XXXX.json``
    ground truth. Returns the list of image paths written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i in range(n_frames):
        frame = generate_frame(seed=seed + i, n_objects=n_objects,
                               defect_rate=defect_rate, **kwargs)
        img_path = output_dir / f"frame_{i:04d}.png"
        frame.save(img_path, output_dir / f"frame_{i:04d}.json")
        written.append(img_path)
    return written


def load_ground_truth(truth_path: Path) -> list[dict]:
    """Load a ground-truth JSON file produced by :meth:`Frame.save`."""
    data = json.loads(truth_path.read_text(encoding="utf-8"))
    return data["objects"]
