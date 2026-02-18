# City Change Detector

City-scale change detection between two time-separated aerial or map images.

The pipeline:
1. Registers `T2` to `T1` with feature matching (`ORB`/`AKAZE`/`SIFT`).
2. Runs SegFormer semantic segmentation on both aligned images.
3. Detects and labels `added` / `removed` regions.

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

If LoveDA weights fail to download on first run, reinstall deps to ensure
`hf_xet` is present (used by some Hub-hosted weight files):
```bash
pip install -U huggingface_hub hf_xet
```

Run:
```bash
python run_pipeline.py
```

## Configure Input/Output

Edit `src/city_change/config.py`:
- `RunConfig.reference_path`
- `RunConfig.target_path`
- `RunConfig.output_dir`

Main knobs:
- `RegistrationConfig`: alignment quality and fallback behavior
- `SegmentationConfig`: model, device, classes of interest, confidence threshold
- `ChangeConfig`: pixel thresholding and region cleanup
- `VisualizationConfig`: panel output and overlay blending

Default segmentation model is now:
- `wu-pr-gw/segformer-b2-finetuned-with-LoveDA`

The loader supports both Hugging Face image processors and legacy feature
extractors for checkpoint compatibility.

Default classes of interest:
- `building`
- `road`

Google Earth-style pairs often have low inlier counts but small, real shifts.  
Use the built-in relaxed reliability path in `RegistrationConfig`:
- `near_identity_reliability_enabled`
- `near_identity_min_inlier_matches`
- `near_identity_min_mae_improvement`

For Google Earth screenshots, defaults now focus on structural classes
(`building`, `house`, `bridge`, `tower`) and apply an edge-density
filter to suppress broad low-texture land-cover changes (grass/soil shifts).
You can tune:
- `ChangeConfig.use_structural_edge_filter`
- `ChangeConfig.structural_min_edge_fraction`
- `ChangeConfig.edge_dilate_kernel`
- `ChangeConfig.use_chroma_seed`
- `ChangeConfig.chroma_threshold_percentile`
- `ChangeConfig.unresolved_min_chroma_delta`
- `ChangeConfig.allow_removed_without_semantic_transition`
- `ChangeConfig.drop_sprawling_added_regions`
- `ChangeConfig.sprawling_min_area`
- `ChangeConfig.sprawling_max_fill_ratio`

When semantic transitions are too weak, detector falls back to `edge_only`
mode. In that mode, default behavior is conservative:
- keep structural `added`
- send uncertain negative changes to `ambiguous` instead of `removed`

## Outputs

Per run, `output_dir` contains:
- `00_summary_panel.png`
- `01_matches.png`
- `02_registered_target.png`
- `03_change_mask.png`
- `04_added_mask.png`
- `05_removed_mask.png`
- `06_overlay.png`
- `07_regions.png`
- `metrics.json`

Segmentation is still used in detection, but intermediate segmentation preview images are no longer exported.

## Project Layout

```text
.
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

And run:
```bash
python run_pipeline.py
```

## Notes

- Best results depend on good registration (similar zoom, orientation, crop).
- If strict registration fails, set `RegistrationConfig.fail_on_unreliable = False` for exploratory runs.
- `metrics.json` stores registration diagnostics, detected regions, and full pipeline config for reproducibility.
