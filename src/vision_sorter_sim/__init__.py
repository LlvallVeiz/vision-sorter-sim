"""vision-sorter-sim: end-to-end machine-vision conveyor sorting simulator.

Synthetic conveyor images -> Otsu thresholding -> connected-component
labelling -> shape features -> rule-based classification -> sort decisions
-> statistics, reports and a static HTML dashboard.

All core vision algorithms (Otsu, two-pass labelling, shape features,
Hu moments) are implemented from scratch with NumPy/Pillow only;
no OpenCV dependency.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
