# Method Overview

## 1. Feature-Based Image Registration

Given reference image `I_t1` and target image `I_t2`:

1. Detect keypoints and descriptors with `ORB`, `AKAZE`, or `SIFT`.
2. Match descriptors with KNN + Lowe ratio test.
3. Estimate affine and homography candidates with RANSAC.
4. Select the best plausible transform using inliers, overlap ratio, and alignment error.
   If strict inlier thresholds are not met, a near-identity reliability check can still accept small, plausible shifts when alignment error improves enough.
5. If alignment is unreliable, optionally fall back to identity and restrict later change detection to a support region built from match geometry.
6. Warp `I_t2` into reference coordinates:
   `I_t2_aligned = warpPerspective(I_t2, H)`.

Registration quality is reported with inlier ratio from RANSAC.

## 2. Pretrained Semantic Segmentation

Both `I_t1` and `I_t2_aligned` are segmented using a pretrained SegFormer model:

- Input: RGB image
- Output: per-pixel class map + confidence map
- Class gating: keep only user-selected city-relevant classes (building, road, tree, etc.)
- These segmentation maps are internal signals for change filtering (not exported as standalone PNG outputs).

This reduces false positives from clouds, shadows, and texture noise.

## 3. Hybrid Change Detection

Raw change candidates are computed from absolute pixel difference:

1. `D = |I_t1 - I_t2_aligned|`
2. Blur + threshold to build `M_raw` (adaptive thresholding available)
3. Semantic gate with segmentation confidence/class masks -> `M_semantic`
4. Morphological cleanup + connected components filtering
5. Optional structural edge-density filtering to reject broad low-texture regions that usually come from seasonal vegetation/ground tone shifts in map imagery
6. Optional registration support mask to suppress detections outside trusted overlap on weak alignments

If confident semantic transitions are unavailable, a structural `edge_only` fallback is used instead of raw pixel fallback to reduce lighting-driven false positives.
Color-space (Lab chroma) differencing is also used as an additional seed signal to recover object changes that are weak in grayscale intensity.

## 4. Added vs Removed Inference

For each changed pixel/region:

- `added`: appears as relevant class in `t2` but not `t1`
- `removed`: appears as relevant class in `t1` but not `t2`
- unresolved cases use grayscale delta sign
- uncertain leftovers are labeled `ambiguous`

In `edge_only` fallback mode, removed-by-intensity can be disabled to avoid shadow/illumination false removals.
An optional compactness filter can drop very large low-fill added components (typically merged road/open-area blobs) and keep compact building-like additions.

## 5. Region-Level Reporting

Connected components are converted into region objects:

- bounding box (`x, y, w, h`)
- area in pixels
- region type (`added`, `removed`, `ambiguous`)

All metadata is exported in `metrics.json` for reproducibility.
