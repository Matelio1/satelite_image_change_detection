from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from city_change.config import get_app_config
from city_change.pipeline import ChangePipeline


def main() -> None:
    app_config = get_app_config()
    pipeline = ChangePipeline(app_config.pipeline)
    metrics = pipeline.run(
        app_config.run.reference_path,
        app_config.run.target_path,
        Path(app_config.run.output_dir),
    )

    print("Run completed.")
    print(f"Registration inlier ratio: {metrics['registration']['inlier_ratio']:.3f}")
    print(f"Registration reliable: {metrics['registration']['registration_reliable']}")
    print(f"Registration transform: {metrics['registration']['transform_name']}")
    print(f"Segmentation model source: {metrics['segmentation']['model_resolved_name']}")
    print(f"Changed pixels: {metrics['changes']['changed_pixels']}")
    print(f"Regions detected: {metrics['changes']['num_regions']}")
    print(f"Results saved to: {app_config.run.output_dir}")


if __name__ == "__main__":
    main()
