from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .change_detection import detect_changes
from .config import PipelineConfig
from .io_utils import ensure_dir, load_image, save_binary_mask, save_image, save_json
from .registration import FeatureRegistrar, RegistrationResult
from .segmentation import SegformerSegmenter, classes_to_ids
from .visualization import build_summary_panel, draw_regions, overlay_changes


class ChangePipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def _make_registrar(self, method: str) -> FeatureRegistrar:
        return FeatureRegistrar(
            method=method,
            ratio_test=self.config.registration.ratio_test,
            ransac_reproj_threshold=self.config.registration.ransac_reproj_threshold,
            max_features=self.config.registration.max_features,
            min_inlier_matches=self.config.registration.min_inlier_matches,
            min_inlier_ratio=self.config.registration.min_inlier_ratio,
            allow_identity_fallback=self.config.registration.allow_identity_fallback,
            min_overlap_ratio=self.config.registration.min_overlap_ratio,
            fallback_improvement_margin=self.config.registration.fallback_improvement_margin,
            use_clahe=self.config.registration.use_clahe,
            clahe_clip_limit=self.config.registration.clahe_clip_limit,
            clahe_tile_grid_size=self.config.registration.clahe_tile_grid_size,
            near_identity_reliability_enabled=self.config.registration.near_identity_reliability_enabled,
            near_identity_min_inlier_matches=self.config.registration.near_identity_min_inlier_matches,
            near_identity_min_mae_improvement=self.config.registration.near_identity_min_mae_improvement,
        )

    @staticmethod
    def _registration_score(result: RegistrationResult) -> tuple[int, int, int, float, float]:
        stats = result.stats
        return (
            1 if bool(stats.get("registration_reliable", False)) else 0,
            1 if stats.get("transform_name") != "identity" else 0,
            int(stats.get("inlier_matches", 0)),
            float(stats.get("inlier_ratio", 0.0)),
            -float(stats.get("alignment_mae", 1e12)),
        )

    def run(self, reference_path: str | Path, target_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        output_dir = ensure_dir(output_dir)

        reference = load_image(reference_path)
        target = load_image(target_path)

        requested_method = self.config.registration.method.lower()
        methods_to_try: list[str]
        if requested_method == "auto":
            methods_to_try = [m.lower() for m in self.config.registration.methods_to_try]
        else:
            methods_to_try = [requested_method]
        methods_to_try = [m for m in methods_to_try if m in {"orb", "sift", "akaze"}]
        if not methods_to_try:
            raise ValueError(
                "No valid registration methods configured. Use registration.method=orb|sift|akaze|auto "
                "or registration.methods_to_try containing orb/sift/akaze."
            )

        registration_candidates: list[RegistrationResult] = []
        registration_errors: dict[str, str] = {}
        for method in methods_to_try:
            try:
                registrar = self._make_registrar(method)
                registration_candidates.append(registrar.register(reference, target))
            except Exception as exc:
                registration_errors[method] = str(exc)

        if not registration_candidates:
            raise RuntimeError(
                "All registration methods failed. "
                + "; ".join(f"{m}: {msg}" for m, msg in registration_errors.items())
            )

        reg_result = max(registration_candidates, key=self._registration_score)
        registration_candidates_stats = [candidate.stats for candidate in registration_candidates]

        if self.config.registration.fail_on_unreliable and not reg_result.stats.get("registration_reliable", False):
            min_dim = min(reference.shape[0], reference.shape[1], target.shape[0], target.shape[1])
            small_image_fallback = bool(
                self.config.registration.allow_small_image_unreliable_fallback
                and min_dim <= int(self.config.registration.small_image_max_dim)
            )
            if not small_image_fallback:
                raise RuntimeError(
                    "Registration judged unreliable. "
                    f"Selected method={reg_result.stats.get('method')}, "
                    f"transform={reg_result.stats.get('transform_name')}, "
                    f"inliers={reg_result.stats.get('inlier_matches')}, "
                    f"inlier_ratio={reg_result.stats.get('inlier_ratio'):.3f}. "
                    "Use better-aligned image pairs, or disable strict mode with "
                    "registration.fail_on_unreliable=false."
                )
            reg_result.stats["small_image_unreliable_fallback"] = True
            reg_result.stats["small_image_min_dim"] = int(min_dim)

        if not self.config.segmentation.enabled:
            raise RuntimeError(
                "Segmentation is disabled in config, but this project requires segmentation. "
                "Set segmentation.enabled=true."
            )

        try:
            segmenter = SegformerSegmenter(
                model_name=self.config.segmentation.model_name,
                device=self.config.segmentation.device,
                local_files_only=self.config.segmentation.local_files_only,
            )
            ref_seg = segmenter.segment(reference)
            tgt_seg = segmenter.segment(reg_result.aligned_target)
            class_ids = classes_to_ids(self.config.segmentation.classes_of_interest, ref_seg.id2label)
            resolved_model_name = segmenter.resolved_model_name
            segmentation_load_source = segmenter.load_source
            segmentation_processor_kind = segmenter.processor_kind
        except Exception as exc:
            raise RuntimeError(
                "Segmentation model is required but could not be loaded or executed. "
                f"Model={self.config.segmentation.model_name}. Error: {exc}"
            ) from exc

        change_result = detect_changes(
            reference_bgr=reference,
            aligned_target_bgr=reg_result.aligned_target,
            reference_seg=ref_seg,
            target_seg=tgt_seg,
            classes_of_interest=class_ids,
            segmentation_confidence_threshold=self.config.segmentation.confidence_threshold,
            pixel_diff_threshold=self.config.change.pixel_diff_threshold,
            use_chroma_seed=self.config.change.use_chroma_seed,
            chroma_threshold_percentile=self.config.change.chroma_threshold_percentile,
            min_chroma_diff_threshold=self.config.change.min_chroma_diff_threshold,
            max_chroma_diff_threshold=self.config.change.max_chroma_diff_threshold,
            blur_kernel=self.config.change.blur_kernel,
            min_region_area=self.config.change.min_region_area,
            intensity_margin=self.config.change.intensity_margin,
            semantic_min_kept_fraction=self.config.change.semantic_min_kept_fraction,
            valid_mask=reg_result.support_mask if self.config.change.use_registration_support_mask else None,
            adaptive_threshold=self.config.change.adaptive_threshold,
            threshold_percentile=self.config.change.threshold_percentile,
            max_pixel_diff_threshold=self.config.change.max_pixel_diff_threshold,
            use_structural_edge_filter=self.config.change.use_structural_edge_filter,
            structural_min_edge_fraction=self.config.change.structural_min_edge_fraction,
            edge_dilate_kernel=self.config.change.edge_dilate_kernel,
            unresolved_min_chroma_delta=self.config.change.unresolved_min_chroma_delta,
            rescue_semantic_kept_max=self.config.change.rescue_semantic_kept_max,
            small_image_semantic_fallback=bool(
                reg_result.stats.get("small_image_unreliable_fallback", False)
                and self.config.change.small_image_semantic_fallback_enabled
            ),
            small_image_semantic_dilate_kernel=self.config.change.small_image_semantic_dilate_kernel,
            small_image_semantic_min_region_area=self.config.change.small_image_semantic_min_region_area,
            allow_removed_without_semantic_transition=self.config.change.allow_removed_without_semantic_transition,
            drop_sprawling_added_regions=self.config.change.drop_sprawling_added_regions,
            sprawling_min_area=self.config.change.sprawling_min_area,
            sprawling_max_fill_ratio=self.config.change.sprawling_max_fill_ratio,
        )

        overlay = overlay_changes(
            base_bgr=reference,
            added_mask=change_result.added_mask,
            removed_mask=change_result.removed_mask,
            ambiguous_mask=change_result.ambiguous_mask,
            alpha=self.config.visualization.overlay_alpha,
        )
        regions_img = draw_regions(overlay, change_result.regions)

        save_image(output_dir / "01_matches.png", reg_result.match_visualization)
        save_image(output_dir / "02_registered_target.png", reg_result.aligned_target)
        save_binary_mask(output_dir / "03_change_mask.png", change_result.change_mask)
        save_binary_mask(output_dir / "04_added_mask.png", change_result.added_mask)
        save_binary_mask(output_dir / "05_removed_mask.png", change_result.removed_mask)
        save_image(output_dir / "06_overlay.png", overlay)
        save_image(output_dir / "07_regions.png", regions_img)

        panel_path = None
        if self.config.visualization.save_panel:
            panel = build_summary_panel(
                reference=reference,
                target=target,
                aligned_target=reg_result.aligned_target,
                matches=reg_result.match_visualization,
                overlay=overlay,
                regions_image=regions_img,
                tile_width=self.config.visualization.panel_cell_width,
            )
            panel_path = output_dir / "00_summary_panel.png"
            save_image(panel_path, panel)

        metrics = {
            "inputs": {
                "reference_path": str(reference_path),
                "target_path": str(target_path),
            },
            "registration": reg_result.stats,
            "registration_candidates": registration_candidates_stats,
            "registration_errors": registration_errors,
            "segmentation": {
                "model_name": self.config.segmentation.model_name,
                "model_resolved_name": resolved_model_name,
                "load_source": segmentation_load_source,
                "processor_kind": segmentation_processor_kind,
                "local_files_only": self.config.segmentation.local_files_only,
                "selected_class_ids": class_ids,
                "selected_classes": {str(cid): ref_seg.id2label[cid] for cid in class_ids},
            },
            "changes": change_result.metrics,
            "regions": change_result.regions,
            "config": asdict(self.config),
            "outputs": {
                "summary_panel": str(panel_path) if panel_path else None,
                "overlay": str(output_dir / "06_overlay.png"),
                "regions": str(output_dir / "07_regions.png"),
            },
        }
        metrics["pair_unreliable"] = bool(
            (not reg_result.stats.get("registration_reliable", False))
            or change_result.metrics.get("pair_unreliable", False)
        )
        save_json(output_dir / "metrics.json", metrics)
        return metrics
