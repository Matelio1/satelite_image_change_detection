# Project Report Template

## 1. Problem Statement

- Goal: detect city-level changes between two time-separated aerial images.
- Motivation: urban expansion tracking, infrastructure monitoring, planning.

## 2. Data

- Source: (Google Maps / drone / satellite provider)
- Capture dates: `YYYY-MM-DD` vs `YYYY-MM-DD`
- Resolution and geographic area
- Preprocessing notes (crop, normalization, orientation)

## 3. Method

### 3.1 Registration

- Feature method: `ORB` or `SIFT`
- Match filtering: ratio test threshold
- Transform: homography with RANSAC
- Registration quality metric: inlier ratio

### 3.2 Semantic Segmentation

- Pretrained model name
- Class subset used for gating
- Confidence threshold

### 3.3 Change Detection

- Pixel differencing threshold
- Morphological cleanup
- Added/removed decision logic
- Region extraction threshold

## 4. Experiments

- Baselines:
  - raw differencing (no registration)
  - registration only (no semantic gate)
- Final pipeline
- Runtime comparison (`ORB` vs `SIFT`)

## 5. Results

- Qualitative output panel examples
- Quantitative metrics from `metrics.json`:
  - changed pixels
  - number of regions
  - added vs removed counts
- Failure cases and analysis

## 6. Limitations

- seasonal/lighting differences
- viewpoint mismatch beyond planar homography
- segmentation domain gap for aerial imagery

## 7. Future Work

- use segmentation model trained on aerial datasets
- multi-image temporal stack instead of pairwise comparison
- polygon export to GIS formats

## 8. Reproducibility

- Commit hash / version
- Config file used
- Exact command used to run pipeline
