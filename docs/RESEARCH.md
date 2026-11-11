# Research — Machine Vision for Conveyor Sorting

## 1. Background

Automated visual inspection on a conveyor line is one of the oldest
deployments of machine vision: a camera above the belt photographs
each part, the image is segmented, shape features are measured, and a
classifier routes the part to a lane (accept / divert / reject). The
classic pipeline — threshold → connected components → shape features
→ rule-based or statistical classification — predates deep learning
and is still used where parts are geometrically well defined, training
data is scarce, and decisions must be auditable (food sorting,
pharmaceutical packaging, and parts inspection; see the patent
literature on multi-sensor conveyor inspection, e.g. Peleg 1989).

This project implements that classic pipeline end to end on synthetic
conveyor images, with every core algorithm written from scratch
(NumPy/Pillow only, no OpenCV), so each stage can be verified against
its defining paper.

## 2. Theory used

### 2.1 Otsu thresholding (Otsu 1979)

Otsu's method selects the gray level that maximises the
between-class variance of the histogram,
σ_b²(t) = w₀(t)·w₁(t)·(μ₀(t) − μ₁(t))², which is equivalent to
minimising the within-class variance. It assumes a bimodal histogram
— bright parts on a dark belt — and needs no tuning. Otsu, N. (1979).
"A threshold selection method from gray-level histograms." *IEEE
Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.
https://ieeexplore.ieee.org/document/4310076

**Known failure mode (found live during this build):** when the
foreground is a tiny fraction of the image (~0.3%), the variance
criterion can prefer splitting the large background class in half
over isolating the small object class — on one single-triangle frame
Otsu returned t = 45 (the belt level itself), producing 6,756
components and a 16-second detection. The pipeline therefore guards
Otsu: if the foreground fraction exceeds 40%, it falls back to a
robust threshold (histogram mode + 6σ of the background). See
`docs/CALIBRATION.md` and the regression test
`test_binarise_falls_back_when_otsu_splits_background`.

### 2.2 Connected-component labelling (Rosenfeld & Pfaltz 1966)

Two-pass labelling with a union-find equivalence table: the first
raster pass assigns provisional labels and records equivalences
between neighbouring labels; the second pass resolves them to compact
labels. Rosenfeld, A. & Pfaltz, J. (1966). "Sequential operations in
digital picture processing." *Journal of the ACM*, 13(4), 471–494.
8-connectivity is used, so diagonally touching pixels belong to the
same part. Reference: https://sbirchfield.github.io/cvintro/lessons/lesson04_floodfill_connected_components.html

### 2.3 Shape features

- **Circularity** 4πA/P² — 1.0 for a perfect circle in the continuous
  domain. Our perimeter is a boundary-pixel count, which
  underestimates the true perimeter of a digitised circle (diagonal
  boundary steps are counted once), so rasterised circles score
  ~1.27, not 1.0. Thresholds are therefore taken from measured
  values, not from the continuous-domain ideal.
- **Extent** A / bbox area — 1.0 for an axis-aligned rectangle,
  π/4 ≈ 0.785 for a circle, ~0.5 for a triangle.
- **Solidity** A / convex-hull area — 1.0 for convex shapes; a notch
  or bite removes area inside the hull and drops solidity. The hull
  area is corrected (+ perimeter/2 + 1, Pick's-theorem style) because
  the hull passes through pixel centres.
- **Hull corner count** — the convex hull simplified with
  Douglas–Peucker (ε = 3 px) has 3 vertices for a triangle, 4–5 for a
  square/rectangle, and 6+ for circles and defects. The shortest
  simplified-hull edge separates a clipped triangle corner (4–20 px)
  from a rectangle's short side (≥ 28 px).

### 2.4 Hu moment invariants (Hu 1962)

Hu derived seven non-linear combinations of normalised central
moments that are invariant to translation, scale and rotation
(the seventh changes sign under reflection). Hu, M.-K. (1962).
"Visual pattern recognition by moment invariants." *IRE Transactions
on Information Theory*, 8(2), 179–187.
https://doi.org/10.1109/TIT.1962.1057692

The first invariant, φ₁ = η₂₀ + η₀₂, is the workhorse here. Its
continuous-domain values are exact and verifiable by hand:

| shape | φ₁ (analytic) | −log₁₀ φ₁ |
|---|---|---|
| disc | 1/(2π) ≈ 0.159 | 0.798 |
| square | 1/6 ≈ 0.167 | 0.778 |
| equilateral triangle | ≈ 0.193 | 0.715 |
| rectangle, aspect a | (a + 1/a)/12 | 0.66–0.74 for a ∈ [1.5, 2.2] |

Measured values on synthetic data match these to three decimals
(see `docs/CALIBRATION.md`), which is why φ₁ separates square
(0.777–0.778) from rectangle (0.664–0.743) and identifies circles
(0.798). Higher-order moments are far less stable on rasterised
shapes (φ₂ of a triangle varied ~60% between two rotations in our
test), so only φ₁ is used for decisions; all seven are still
reported in the detection output.

### 2.5 Classification metrics

For each class treated as positive against all others:
precision = TP/(TP+FP), recall = TP/(TP+FN),
F1 = 2·precision·recall/(precision+recall). Macro averages are the
unweighted means over classes. The confusion matrix is the primary
report; accuracy alone is not trusted on its own. Definitions follow
the standard treatment, e.g.
https://blog.mbedded.ninja/programming/artificial-intelligence/classification-metrics/

## 3. Design decisions

- **Synthetic data, known ground truth.** Real conveyor photos were
  out of scope for a one-day build; the generator (shapes, noise,
  illumination gradient, random rotation/placement) produces labelled
  data whose truth is exact, so every metric is checkable.
- **Rule-based classifier, not a learned one.** With five geometric
  classes the decision tree is auditable, needs no training set, and
  its failure modes can be named (see §4). A prototype-distance
  fallback covers shapes the rules miss, and anything too far from
  every prototype is labelled `unknown` instead of forced.
- **No OpenCV.** All vision algorithms are implemented from scratch
  so the project is self-contained and verifiable; NumPy + Pillow
  are the only runtime dependencies.
- **Defects are shape defects.** Notch (wedge bitten out), chip
  (disc bitten out of the edge) and irregular blob (randomised radial
  profile). A "crack" defect (a line across the part) was removed
  during development: it split the part into two components, which
  is a labelling artefact, not a defect class.

## 4. Limitations (known, measured)

- Otsu's small-foreground failure mode (see §2.1) — guarded, with a
  regression test.
- A rectangle at aspect ~1.7 rotated ~45° shares the triangle's
  Hu/extent band; only the shortest-hull-edge test separates them.
  A rectangle whose hull gains a noise-induced 5th vertex could in
  principle fall into the triangle band — not observed in 1,000
  evaluated objects, but possible.
- Circularity uses a boundary-pixel perimeter (biased high for
  circles, noisy for rotated polygons); it is a supporting feature
  only, never the primary discriminator.
- Synthetic belt only: uniform background, no occlusion, no touching
  parts, no motion blur, no colour. Real lines would need
  illumination control and a larger, photographed dataset.
- Throughput is simulated from belt speed and spacing, not measured
  on hardware; inspection time is wall-clock on this machine.

## 5. References

1. Otsu, N. (1979). "A threshold selection method from gray-level
   histograms." *IEEE Transactions on Systems, Man, and Cybernetics*,
   9(1), 62–66. https://ieeexplore.ieee.org/document/4310076
2. Hu, M.-K. (1962). "Visual pattern recognition by moment
   invariants." *IRE Transactions on Information Theory*, 8(2),
   179–187. https://doi.org/10.1109/TIT.1962.1057692
3. Rosenfeld, A. & Pfaltz, J. (1966). "Sequential operations in
   digital picture processing." *Journal of the ACM*, 13(4), 471–494.
   http://harveycohen.net/papers/93-4_SEGMENT_ONE_PASS.pdf
4. Flusser, J. "Moment Invariants for Object Recognition" (tutorial).
   http://library.utia.cas.cz/separaty/2015/ZOI/flusser-0442976.pdf
5. OpenCV documentation, "Structural Analysis and Shape
   Descriptors" (HuMoments).
   https://docs.opencv.org/3.0-last-rst/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
6. Birchfield, S. "Lesson 4: Flood Fill and Connected Components."
   https://sbirchfield.github.io/cvintro/lessons/lesson04_floodfill_connected_components.html
7. Peleg, K. (1989). US Patent 4,884,696, "Method and apparatus for
   automatically inspecting and classifying different objects"
   (multi-sensor conveyor inspection).
   https://patentimages.storage.googleapis.com/f1/11/34/5e9690045ea4a2/US4884696.pdf
8. MathWorks, "Image binarization – Otsu's method."
   https://blogs.mathworks.com/steve/2016/06/14/image-binarization-otsus-method/
9. "Classification Metrics" (confusion matrix, precision, recall, F1).
   https://blog.mbedded.ninja/programming/artificial-intelligence/classification-metrics/
10. "Image moment" (Wikipedia) — central/normalised moments, Hu set.
    https://en.wikipedia.org/wiki/Image_moment
