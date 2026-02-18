from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RegistrationConfig:
    method: str = "auto"
    methods_to_try: list[str] = field(default_factory=lambda: ["orb", "akaze", "sift"])
    ratio_test: float = 0.75
    ransac_reproj_threshold: float = 4.0
    max_features: int = 6000
    min_inlier_matches: int = 12
    min_inlier_ratio: float = 0.25
    min_overlap_ratio: float = 0.45
    allow_identity_fallback: bool = True
    fallback_improvement_margin: float = 1.5
    fail_on_unreliable: bool = True
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    near_identity_reliability_enabled: bool = True
    near_identity_min_inlier_matches: int = 6
    near_identity_min_mae_improvement: float = 1.0
    allow_small_image_unreliable_fallback: bool = True
    small_image_max_dim: int = 320


@dataclass
class SegmentationConfig:
    enabled: bool = True
    model_name: str = "wu-pr-gw/segformer-b2-finetuned-with-LoveDA"
    device: str = "auto"
    local_files_only: bool = False
    classes_of_interest: list[str] = field(
        default_factory=lambda: [
            "building",
            "road",
            "house",
            "bridge",
            "tower",
        ]
    )
    confidence_threshold: float = 0.38


@dataclass
class ChangeConfig:
    pixel_diff_threshold: int = 30
    adaptive_threshold: bool = True
    threshold_percentile: float = 88.0
    max_pixel_diff_threshold: int = 90
    use_chroma_seed: bool = True
    chroma_threshold_percentile: float = 88.0
    min_chroma_diff_threshold: float = 10.0
    max_chroma_diff_threshold: float = 30.0
    blur_kernel: int = 5
    min_region_area: int = 120
    intensity_margin: int = 8
    semantic_min_kept_fraction: float = 0.05
    use_registration_support_mask: bool = True
    use_structural_edge_filter: bool = True
    structural_min_edge_fraction: float = 0.01
    edge_dilate_kernel: int = 5
    unresolved_min_chroma_delta: float = 6.0
    rescue_semantic_kept_max: float = 0.12
    small_image_semantic_fallback_enabled: bool = True
    small_image_semantic_dilate_kernel: int = 3
    small_image_semantic_min_region_area: int = 60
    allow_removed_without_semantic_transition: bool = False
    drop_sprawling_added_regions: bool = False
    sprawling_min_area: int = 6000
    sprawling_max_fill_ratio: float = 0.42


@dataclass
class VisualizationConfig:
    overlay_alpha: float = 0.45
    save_panel: bool = True
    panel_cell_width: int = 520


@dataclass
class PipelineConfig:
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    change: ChangeConfig = field(default_factory=ChangeConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


@dataclass
class RunConfig:
    # Input pair
    reference_path: str = "data/ref1.png"
    target_path: str = "data/targ1.png"
    # Output folder
    output_dir: str = "outputs/results"


@dataclass
class AppConfig:
    run: RunConfig = field(default_factory=RunConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def get_app_config() -> AppConfig:
    return AppConfig()
