"""Central configuration: image geometry, object classes, and the
classifier thresholds calibrated from synthetic data.

Calibration procedure and the measured feature ranges behind every
threshold below are documented in ``docs/CALIBRATION.md``; the numbers
here are the values produced by ``scripts/calibrate.py`` (seed 20261111).
"""

from __future__ import annotations

# ---------------------------------------------------------------- image
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
BELT_GRAY = 45          # conveyor belt background level (0-255)
OBJECT_GRAY = 200       # nominal object level before illumination/noise
NOISE_SIGMA = 8.0       # Gaussian sensor noise (gray levels)
ILLUMINATION_TILT = 12.0  # linear brightness ramp across the frame (gray levels)

# ------------------------------------------------------------- classes
CLASSES = ("circle", "square", "rectangle", "triangle", "defect")
GOOD_CLASSES = ("circle", "square", "rectangle", "triangle")

# Object sizes (pixels). ``size`` is radius for circles, side for squares,
# long side for rectangles, circumradius for triangles.
SIZE_MIN = 22
SIZE_MAX = 42
RECT_ASPECT_MIN = 1.5
RECT_ASPECT_MAX = 2.2

# ------------------------------------------------------------ detection
MIN_COMPONENT_AREA = 120   # components smaller than this are speckle/noise
CONNECTIVITY = 8
# Otsu fails when the foreground is a tiny fraction of the image: it
# can split the background in half instead. A foreground fraction
# above this cap means the belt was split, and the robust threshold
# (mode + 6 sigma) is used instead. Real frames stay far below it
# (5 objects cover ~3-7% of the frame).
MAX_FOREGROUND_FRACTION = 0.40

# ---------------------------------------------------------- classification
# Rule-based decision tree thresholds (see docs/CALIBRATION.md).
# Primary features are rotation-invariant: Hu moment phi_1 (log form),
# the Douglas-Peucker hull corner count, and solidity. Circularity uses
# a boundary-pixel perimeter estimate, so a rasterised circle scores
# ~1.27 instead of 1.0 (see docs/RESEARCH.md) — thresholds below use
# the measured values, not the continuous-domain ideals.
#
# Calibration set (40 frames, 200 matched objects, seed 20261111):
#   circle:    hu_log_1 0.798, solidity 0.982-0.991, circ 1.248-1.312,
#              extent 0.777-0.786, hull vertices 8-16
#   square:    hu_log_1 0.777-0.778, solidity 0.951-0.988,
#              hull vertices 4-5
#   rectangle: hu_log_1 0.664-0.743, solidity 0.951-0.978,
#              hull vertices 4-5
#   triangle:  hu_log_1 0.715-0.716, solidity 0.943-0.974,
#              hull vertices exactly 3
#   defect:    hu_log_1 0.731-0.795, solidity 0.839-0.977,
#              hull vertices 6-15
HU_CIRCLE_MIN = 0.795        # log10 Hu phi_1 of a circle (~0.798)
HU_SQUARE_MIN = 0.760        # squares 0.777+, rectangles <= 0.743
HU_TRIANGLE_MIN = 0.710      # triangles 0.715-0.716 (very tight band)
HU_TRIANGLE_MAX = 0.720
EXTENT_TRIANGLE_MIN = 0.44   # triangles 0.465-0.508
EXTENT_TRIANGLE_MAX = 0.54
MIN_HULL_EDGE_TRIANGLE_MAX = 25.0  # clipped-triangle 4th corner: short edge
# 4-20 px; real rectangles: shortest edge >= 28 px (measured)
SOLIDITY_CIRCLE_MIN = 0.980  # circles 0.982+, defects <= 0.977
CIRCULARITY_CIRCLE_MIN = 1.24  # circles 1.248+, defects <= 1.232
EXTENT_CIRCLE_MIN = 0.76     # circles 0.777-0.786 (pi/4 of bbox)
EXTENT_CIRCLE_MAX = 0.80

# Fallback nearest-prototype distance (in z-scored feature space) above
# which a detection is labelled "unknown" instead of forced into a class.
UNKNOWN_DISTANCE_MAX = 3.5

# Class prototypes for the fallback: (hu_log_1, extent, solidity).
# Calibration-set class means (docs/CALIBRATION.md).
PROTOTYPES = {
    "circle": (0.798, 0.783, 0.988),
    "square": (0.778, 0.630, 0.968),
    "rectangle": (0.702, 0.598, 0.968),
    "triangle": (0.715, 0.479, 0.957),
    "defect": (0.765, 0.695, 0.901),
}
# Feature scales (class-pooled standard deviations) used to z-score the
# distance to each prototype.
PROTOTYPE_SCALES = (0.02, 0.08, 0.03)

# ------------------------------------------------------------- conveyor
BELT_SPEED_M_PER_S = 0.5        # belt speed
INSPECTION_ZONE_M = 0.6         # length of the camera inspection zone
ITEM_SPACING_M = 0.35           # nominal centre-to-centre spacing on the belt
DEFAULT_DEFECT_RATE = 0.12      # fraction of items that are defective
DEFAULT_ITEMS_PER_MIN = 60      # nominal arrival rate at the inspection zone

# Sort lanes.
LANE_ACCEPT = "accept"
LANE_REJECT = "reject"
LANE_DIVERT = "divert"
