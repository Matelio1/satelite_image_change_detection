# City Change Detector

City-scale change detection for two time-separated aerial/drone images.

The project does three main things:
1. Aligns `T2` to `T1` using feature registration (`ORB`/`AKAZE`/`SIFT`).
2. Segments both images with a pretrained SegFormer model.
3. Detects and visualizes `added` / `removed` regions with masks, overlays, boxes, and metrics.


## What You Get

After one run, you get:
- registration matches image
- registered target image
- semantic segmentation maps
- binary change/added/removed masks
- final overlay + region boxes
- `metrics.json` with full numeric diagnostics

## Repository Structure

```text
.
|-- data/                         # your input images
|-- docs/
|   |-- method.md
|   `-- project_report_template.md
|-- outputs/                      # run outputs
|-- scripts/
|   `-- generate_synthetic_pair.py
|-- src/city_change/
|   |-- __init__.py
|   |-- change_detection.py
|   |-- config.py                 # single source of configuration
|   |-- io_utils.py
|   |-- pipeline.py
|   |-- registration.py
|   |-- segmentation.py
|   `-- visualization.py
|-- run_pipeline.py               # entry point (no arguments)
|-- pyproject.toml
`-- requirements.txt
```

## Clone And Run (New PC)

### 1) Clone
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2) Create virtual environment

Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 4) Set your input images and output folder
Open `src/city_change/config.py` and edit:
- `RunConfig.reference_path`
- `RunConfig.target_path`
- `RunConfig.output_dir`

Important: by default these paths may not match your files.  
Set them to real files before running.

### 5) Run
```bash
python run_pipeline.py
```

## Model Setup (Important)

This project uses the Hugging Face model:
- `nvidia/segformer-b2-finetuned-ade-512-512`

In `src/city_change/config.py`, this is controlled by:
- `SegmentationConfig.model_name`
- `SegmentationConfig.local_files_only`

Model loading behavior:
- The code first tries local sources (explicit local path or Hugging Face cache snapshot).
- If that succeeds, no network is needed.
- If local load fails and `local_files_only=False`, it then tries online loading.
- If all attempts fail, pipeline stops with an error.

### Option A: Normal online use (recommended for new users)
- Set `local_files_only = False`
- Keep `model_name = "nvidia/segformer-b2-finetuned-ade-512-512"`
- First run downloads model files automatically, then they are cached.

### Option B: Offline/local model folder
- Keep `local_files_only = True`
- Set `model_name` to a local folder path containing the model files.
- You can copy the model snapshot from your HF cache into a project-local folder (for example `models/segformer-b2/...`) and point to that folder.

Note:
- If `local_files_only = True` and model files are not available locally, run will fail.
- If segmentation cannot load or run, pipeline fails immediately because segmentation is required.

## Single Configuration File (`src/city_change/config.py`)

All behavior is controlled through dataclasses in this file.

### `RunConfig`
- `reference_path`: image at time 1 (`T1`)
- `target_path`: image at time 2 (`T2`)
- `output_dir`: where results are written

### `RegistrationConfig`
- `method`: `"auto"` or a fixed method (`"orb"`, `"akaze"`, `"sift"`)
- `methods_to_try`: used only when `method="auto"`
- `ratio_test`, `ransac_reproj_threshold`, `max_features`: feature matching sensitivity
- `min_inlier_matches`, `min_inlier_ratio`: minimum RANSAC quality
- `min_overlap_ratio`: minimum overlap after warp
- `allow_identity_fallback`: allow fallback when no good transform
- `fallback_improvement_margin`: how much better a weak transform must be vs identity
- `fail_on_unreliable`: strict mode switch
- `use_clahe`, `clahe_clip_limit`, `clahe_tile_grid_size`: contrast preprocessing for keypoints

### `SegmentationConfig`
- `enabled`: segmentation on/off
- `model_name`: HF model id or local path
- `device`: `"auto"`, `"cpu"`, `"cuda"`
- `local_files_only`: do not download if model is already cached
- `classes_of_interest`: semantic classes used for change gating
- `confidence_threshold`: pixel confidence threshold for semantic gating

### `ChangeConfig`
- `pixel_diff_threshold`: minimum grayscale difference
- `adaptive_threshold`: enable percentile-based threshold update
- `threshold_percentile`: percentile used when adaptive threshold is enabled
- `max_pixel_diff_threshold`: upper bound for adaptive threshold
- `blur_kernel`: smoothing before thresholding
- `min_region_area`: removes tiny connected components
- `intensity_margin`: resolves ambiguous added/removed by grayscale delta sign
- `semantic_min_kept_fraction`: semantic gate fallback guard
- `use_registration_support_mask`: restrict comparison to trusted overlap if registration is weak

### `VisualizationConfig`
- `overlay_alpha`: overlay intensity
- `save_panel`: save 2x3 summary panel
- `panel_cell_width`: summary tile width

## How Code Works (File By File)

### `run_pipeline.py`
Entry point.  
It:
1. loads app config from `get_app_config()`
2. creates `ChangePipeline`
3. runs pipeline
4. prints key metrics

### `src/city_change/config.py`
Defines dataclasses for all settings and returns one `AppConfig` object.  
This is the only place you edit for paths and behavior.

### `src/city_change/pipeline.py`
Main orchestration:
1. load input images
2. try one or more registration methods
3. select best registration result
4. enforce strict mode (`fail_on_unreliable`)
5. run required segmentation
6. run change detection
7. generate visual outputs
8. save `metrics.json`

### `src/city_change/registration.py`
`FeatureRegistrar` does robust alignment:
1. preprocess grayscale (optional CLAHE)
2. detect features + descriptors
3. KNN matching + Lowe ratio test
4. estimate homography and affine with RANSAC
5. evaluate candidates by inliers, overlap, plausibility, MAE
6. choose best candidate or fallback
7. build support mask for low-trust alignments
8. return `RegistrationResult`

### `src/city_change/segmentation.py`
`SegformerSegmenter` loads and runs Hugging Face SegFormer:
1. tries local path / local HF cache first
2. preprocess image with `AutoImageProcessor`
3. forward pass
4. upsample logits to original size
5. output mask + confidence + label map

Other helpers:
- `classes_to_ids`: maps user class names to model class IDs
- `colorize_mask`: random color visualization for class map

### `src/city_change/change_detection.py`
`detect_changes(...)`:
1. compute absolute difference
2. adaptive/fixed threshold
3. semantic gating using classes and confidence
4. morphology + connected components cleanup
5. infer added/removed using class transitions and intensity fallback
6. extract regions (`bbox`, `area`, counts)
7. return masks + region list + metrics

### `src/city_change/visualization.py`
Visual helpers:
- `overlay_changes`: color blend for added/removed/ambiguous
- `draw_regions`: draw boxes + labels
- `build_summary_panel`: creates final 2x3 panel image

### `src/city_change/io_utils.py`
Small file IO utilities:
- create output folders
- read image safely
- save image/mask/json safely

## Output Files Explained

For each run, output directory contains:
- `00_summary_panel.png`: all key visuals in one image
- `01_matches.png`: feature correspondences (`T1` vs `T2`)
- `02_registered_target.png`: warped `T2` in `T1` coordinate space
- `03_ref_segmentation.png`: segmentation visualization for `T1`
- `04_target_segmentation.png`: segmentation visualization for aligned `T2`
- `05_change_mask.png`: merged change mask
- `06_added_mask.png`: added-only mask
- `07_removed_mask.png`: removed-only mask
- `08_overlay.png`: color overlay on reference image
- `09_regions.png`: overlay + region rectangles + labels
- `metrics.json`: numeric results and full config snapshot
  - check `segmentation.enabled_applied` and `segmentation.model_resolved_name`

## Strict Mode vs Non-Strict Mode

If you see:
`RuntimeError: Registration judged unreliable ...`

It means the input pair cannot be aligned well enough under strict mode.

To continue anyway, set in `src/city_change/config.py`:
- `RegistrationConfig.fail_on_unreliable = False`

This allows output generation on hard pairs, but accuracy can be lower.

## Common Problems

### "Could not load image"
- Path in `RunConfig` is wrong.
- Fix `reference_path` and `target_path`.

### Strict registration failure
- Try better-matched screenshots (same map orientation/zoom/crop).
- Set `fail_on_unreliable=False` if you only want exploratory output.

### Segmentation model download/cache issues
- First run may download model weights.
- If model is already cached, keep `local_files_only=True` to run offline.

### Poor change quality
- Improve pair similarity (overlap, angle, zoom, crop).
- Tune `ChangeConfig` and `RegistrationConfig`.

## Quick Synthetic Demo

Generate a synthetic pair:
```bash
python scripts/generate_synthetic_pair.py --out-dir data/synthetic
```

Then point `RunConfig.reference_path` and `RunConfig.target_path` to:
- `data/synthetic/t1_reference.png`
- `data/synthetic/t2_target.png`

Run:
```bash
python run_pipeline.py
```

## Notes

- Good results depend mostly on registration quality.
- Semantic segmentation improves interpretability but cannot fully fix bad alignment.
- For map screenshots, keep the same zoom level, orientation, and crop as much as possible.
