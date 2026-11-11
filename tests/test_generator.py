"""Tests for the synthetic conveyor-frame generator."""

import json

import numpy as np

from vision_sorter_sim.generator import generate_dataset, generate_frame, load_ground_truth


def test_same_seed_reproduces_frame():
    a = generate_frame(seed=42, n_objects=5)
    b = generate_frame(seed=42, n_objects=5)
    assert np.array_equal(a.image, b.image)
    assert [o.to_dict() for o in a.objects] == [o.to_dict() for o in b.objects]


def test_different_seeds_differ():
    a = generate_frame(seed=1, n_objects=5)
    b = generate_frame(seed=2, n_objects=5)
    assert not np.array_equal(a.image, b.image)


def test_frame_shape_and_dtype():
    frame = generate_frame(seed=3, n_objects=4)
    assert frame.image.shape == (480, 640)
    assert frame.image.dtype == np.uint8
    assert len(frame.objects) == 4


def test_objects_do_not_overlap():
    import math
    frame = generate_frame(seed=11, n_objects=8)
    for i, a in enumerate(frame.objects):
        for b in frame.objects[i + 1:]:
            dist = math.hypot(a.cx - b.cx, a.cy - b.cy)
            assert dist > (a.size + b.size) * 1.0


def test_ground_truth_labels_are_known_classes():
    frame = generate_frame(seed=5, n_objects=6, defect_rate=0.5)
    labels = {o.label for o in frame.objects}
    assert labels <= {"circle", "square", "rectangle", "triangle", "defect"}


def test_no_noise_is_deterministic_and_clean():
    frame = generate_frame(seed=9, n_objects=2, noise_sigma=0.0,
                           illumination_tilt=0.0)
    unique = np.unique(frame.image)
    assert set(unique.tolist()) <= {45, 200} or len(unique) <= 4


def test_generate_dataset_writes_pairs(tmp_path):
    paths = generate_dataset(tmp_path, n_frames=3, seed=0, n_objects=2)
    assert len(paths) == 3
    for img_path in paths:
        assert img_path.exists()
        truth = load_ground_truth(img_path.with_suffix(".json"))
        assert len(truth) == 2
        assert all("label" in obj and "cx" in obj for obj in truth)


def test_save_roundtrip(tmp_path):
    frame = generate_frame(seed=21, n_objects=2)
    img_path = tmp_path / "f.png"
    truth_path = tmp_path / "f.json"
    frame.save(img_path, truth_path)
    data = json.loads(truth_path.read_text())
    assert data["width"] == 640 and data["height"] == 480
    assert len(data["objects"]) == 2
