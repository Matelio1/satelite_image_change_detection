# City Change Detector

Change detection between two time-separated map/aerial images.

Given a reference image (`t1`) and a target image (`t2`), the pipeline:
1. Aligns `t2` to `t1` coordinates (feature registration).
2. Runs semantic segmentation on both images (SegFormer).
3. Builds change masks and classifies regions as `added`, `removed`, or `ambiguous`.

## What The Code Does

### End-to-end flow
`run_pipeline.py` loads config, runs `ChangePipeline`, and writes output images + `metrics.json`.

`src/city_change/pipeline.py` is the orchestrator:
1. Load images.
2. Try multiple registration methods (`orb`, `akaze`, `sift`) and choose best candidate.
3. Handle strict/unreliable registration logic (including small-image fallback).
4. Run segmentation on reference and aligned target.
5. Run change detection.
6. Save visual outputs and metrics.

### Module responsibilities
- `src/city_change/config.py`: all runtime knobs (input paths, thresholds, model options).
- `src/city_change/registration.py`: feature detection/matching + transform estimation + reliability checks.
- `src/city_change/segmentation.py`: Hugging Face model loading and inference.
- `src/city_change/change_detection.py`: mask generation, semantic gating, structural rescue, region extraction.
- `src/city_change/visualization.py`: overlays, summary panel, region drawings.
- `src/city_change/io_utils.py`: read/write helpers.

## Quick Start

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
```

Windows (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
source venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
pip install -e .
```

Run:
```bash
python run_pipeline.py
```

## Using Your Own Images

Edit `src/city_change/config.py`:
- `RunConfig.reference_path`
- `RunConfig.target_path`
- `RunConfig.output_dir`

Example with your own filenames:
- `RunConfig.reference_path = "data/ref.png"`
- `RunConfig.target_path = "data/targ.png"`

Then run:
```bash
python run_pipeline.py
```

## Data Folder Guide

Included files in `data/`:
- `ref1.png ... ref8.png`
- `targ1.png ... targ8.png`

Recommended pairing is by matching index:
- `ref1.png` with `targ1.png`
- `ref2.png` with `targ2.png`
- ...
- `ref8.png` with `targ8.png`

To run one pair, set:
- `RunConfig.reference_path = "data/refN.png"`
- `RunConfig.target_path = "data/targN.png"`

## How Detection Works Internally

`src/city_change/change_detection.py` combines several signals:
- Pixel difference (grayscale + Lab chroma deltas).
- Semantic class transitions from segmentation (classes of interest).
- Structural edges/gradients.
- Morphological cleanup + component filtering.

Normal path:
1. Build initial raw change seed from photometric differences.
2. Gate with semantic transitions.
3. Apply structural rescue when semantic retention is very weak.
4. Split into added/removed/ambiguous and clean masks.
5. Convert components into region objects with confidence scores.

Small-image path (`256x256`-style pairs):
- Automatically used when registration is unreliable and small-image fallback is active.
- Uses semantic add-only fallback (`target_interest & ~reference_interest`) with light morphology.
- Produces stable masks for very low-feature tiny images.

## Small Image Support

Enabled by default:
- `RegistrationConfig.allow_small_image_unreliable_fallback = True`
- `RegistrationConfig.small_image_max_dim = 320`
- `ChangeConfig.small_image_semantic_fallback_enabled = True`
- `ChangeConfig.small_image_semantic_dilate_kernel = 3`
- `ChangeConfig.small_image_semantic_min_region_area = 60`

Behavior:
- For small pairs, unreliable registration does not hard-fail the run.
- `metrics.json` includes reliability flags so you can filter low-trust results.

## Key Config Knobs

### Registration (`RegistrationConfig`)
- `method`, `methods_to_try`: registration algorithms.
- `ratio_test`, `ransac_reproj_threshold`: match/transform strictness.
- `fail_on_unreliable`: stop run when registration is poor.
- `near_identity_*`: relaxed acceptance for small, plausible shifts.

### Segmentation (`SegmentationConfig`)
- `model_name`: segmentation checkpoint.
- `classes_of_interest`: classes considered for change logic.
- `confidence_threshold`: semantic confidence gating.

### Change (`ChangeConfig`)
- `pixel_diff_threshold`, `threshold_percentile`: raw diff thresholding.
- `min_region_area`: base component filter.
- `rescue_semantic_kept_max`: when to trigger structural rescue.
- `small_image_semantic_*`: dedicated fallback for tiny pairs.
- `use_structural_edge_filter`, `structural_min_edge_fraction`: structure filtering.

## Output Files Explained

For each run, `output_dir` contains:
- `00_summary_panel.png`: overview collage (reference, target, alignment, overlay, regions).
- `01_matches.png`: feature matches used in registration.
- `02_registered_target.png`: target warped to reference coordinates.
- `03_change_mask.png`: all detected change pixels (binary).
- `04_added_mask.png`: pixels classified as additions.
- `05_removed_mask.png`: pixels classified as removals.
- `06_overlay.png`: color overlay of changes on reference image.
- `07_regions.png`: connected regions with drawn boxes.
- `metrics.json`: detailed diagnostics and metadata.

### How to read the masks
- `03_change_mask`: union of added/removed/ambiguous.
- `04_added_mask`: preferred mask for "what appeared in target".
- `05_removed_mask`: preferred mask for "what disappeared from reference".

## `metrics.json` fields you should monitor

- `registration.registration_reliable`: registration trust.
- `pair_unreliable`: overall run trust flag.
- `changes.semantic_gate_mode`: which semantic path was used.
- `changes.effective_pixel_diff_threshold`: actual adaptive threshold used.
- `changes.effective_min_region_area`: area filter after scaling.
- `changes.num_regions`, `added_pixels`, `removed_pixels`: output volume.
- `changes.mode == "small_image_semantic_fallback"`: small-image fallback path used.

## Project Layout

```text
.
|-- data/
|-- docs/
|   |-- method.md
|   `-- project_report_template.md
|-- scripts/
|   `-- generate_synthetic_pair.py
|-- src/city_change/
|   |-- change_detection.py
|   |-- config.py
|   |-- io_utils.py
|   |-- pipeline.py
|   |-- registration.py
|   |-- segmentation.py
|   `-- visualization.py
|-- run_pipeline.py
|-- pyproject.toml
`-- requirements.txt
```

## Synthetic Demo

```bash
python scripts/generate_synthetic_pair.py --out-dir data/synthetic
```

Then set:
- `RunConfig.reference_path = "data/synthetic/t1_reference.png"`
- `RunConfig.target_path = "data/synthetic/t2_target.png"`

Run:
```bash
python run_pipeline.py
```

## Troubleshooting

- If model download fails first run:
  - `pip install -U huggingface_hub hf_xet`
- If large false positives appear:
  - lower `rescue_semantic_kept_max`.
- If true changes are missed:
  - raise `rescue_semantic_kept_max` or reduce `min_region_area`.
- If tiny-pair masks are too sparse:
  - increase `small_image_semantic_dilate_kernel`.
- If tiny-pair masks are too broad:
  - increase `small_image_semantic_min_region_area`.
