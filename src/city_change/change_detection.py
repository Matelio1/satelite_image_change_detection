from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .segmentation import SegmentationOutput


@dataclass
class ChangeResult:
    change_mask: np.ndarray
    added_mask: np.ndarray
    removed_mask: np.ndarray
    ambiguous_mask: np.ndarray
    regions: list[dict[str, Any]]
    metrics: dict[str, Any]


def _remove_small_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(binary_mask, dtype=bool)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_idx] = True
    return cleaned


def _clean_mask(mask: np.ndarray, blur_kernel: int, min_region_area: int) -> np.ndarray:
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    cleaned = _remove_small_components(closed > 0, min_area=min_region_area)
    return cleaned


def _extract_regions(
    combined_mask: np.ndarray,
    added_mask: np.ndarray,
    removed_mask: np.ndarray,
    min_region_area: int,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask.astype(np.uint8), connectivity=8)
    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_region_area:
            continue

        region_mask = labels == label_idx
        added_pixels = int(np.count_nonzero(added_mask & region_mask))
        removed_pixels = int(np.count_nonzero(removed_mask & region_mask))
        if added_pixels > removed_pixels:
            region_type = "added"
        elif removed_pixels > added_pixels:
            region_type = "removed"
        else:
            region_type = "ambiguous"

        regions.append(
            {
                "type": region_type,
                "bbox_xywh": [x, y, w, h],
                "area": area,
                "added_pixels": added_pixels,
                "removed_pixels": removed_pixels,
            }
        )
    return regions


def _filter_by_edge_density(
    mask: np.ndarray,
    edge_support: np.ndarray,
    min_edge_fraction: float,
    min_region_area: int,
) -> tuple[np.ndarray, int]:
    if min_edge_fraction <= 0:
        return mask, 0

    cleaned = np.zeros_like(mask, dtype=bool)
    removed_components = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_region_area:
            continue
        region = labels == label_idx
        edge_pixels = int(np.count_nonzero(edge_support & region))
        edge_fraction = float(edge_pixels / max(1, area))
        if edge_fraction >= min_edge_fraction:
            cleaned |= region
        else:
            removed_components += 1
    return cleaned, removed_components


def _build_edge_support(ref_gray_u8: np.ndarray, tgt_gray_u8: np.ndarray, dilate_kernel: int) -> np.ndarray:
    edges_ref = cv2.Canny(ref_gray_u8, 50, 150) > 0
    edges_tgt = cv2.Canny(tgt_gray_u8, 50, 150) > 0
    edge_support = edges_ref | edges_tgt
    k = dilate_kernel if dilate_kernel % 2 == 1 else dilate_kernel + 1
    if k > 1:
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edge_support = cv2.dilate(edge_support.astype(np.uint8), edge_kernel).astype(bool)
    return edge_support


def _drop_sprawling_components(
    mask: np.ndarray,
    min_area: int,
    max_fill_ratio: float,
) -> tuple[np.ndarray, int]:
    cleaned = np.zeros_like(mask, dtype=bool)
    dropped = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        fill_ratio = float(area / max(1, w * h))
        if area >= min_area and fill_ratio <= max_fill_ratio:
            dropped += 1
            continue
        cleaned[labels == label_idx] = True
    return cleaned, dropped


def detect_changes(
    reference_bgr: np.ndarray,
    aligned_target_bgr: np.ndarray,
    reference_seg: SegmentationOutput,
    target_seg: SegmentationOutput,
    classes_of_interest: list[int],
    segmentation_confidence_threshold: float,
    pixel_diff_threshold: int = 30,
    use_chroma_seed: bool = True,
    chroma_threshold_percentile: float = 88.0,
    min_chroma_diff_threshold: float = 10.0,
    max_chroma_diff_threshold: float = 30.0,
    blur_kernel: int = 5,
    min_region_area: int = 300,
    intensity_margin: int = 8,
    semantic_min_kept_fraction: float = 0.05,
    valid_mask: np.ndarray | None = None,
    adaptive_threshold: bool = True,
    threshold_percentile: float = 88.0,
    max_pixel_diff_threshold: int = 90,
    use_structural_edge_filter: bool = True,
    structural_min_edge_fraction: float = 0.015,
    edge_dilate_kernel: int = 5,
    unresolved_min_chroma_delta: float = 6.0,
    allow_removed_without_semantic_transition: bool = False,
    drop_sprawling_added_regions: bool = True,
    sprawling_min_area: int = 6000,
    sprawling_max_fill_ratio: float = 0.42,
) -> ChangeResult:
    diff = cv2.absdiff(reference_bgr, aligned_target_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    tgt_lab = cv2.cvtColor(aligned_target_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    chroma_delta = np.sqrt(
        (tgt_lab[:, :, 1] - ref_lab[:, :, 1]).astype(np.float32) ** 2
        + (tgt_lab[:, :, 2] - ref_lab[:, :, 2]).astype(np.float32) ** 2
    )
    if blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        diff_gray = cv2.GaussianBlur(diff_gray, (k, k), 0)

    if valid_mask is not None:
        valid_mask_bool = valid_mask.astype(bool)
        diff_values = diff_gray[valid_mask_bool]
    else:
        valid_mask_bool = np.ones_like(diff_gray, dtype=bool)
        diff_values = diff_gray.reshape(-1)

    effective_threshold = int(pixel_diff_threshold)
    if adaptive_threshold and diff_values.size > 0:
        percentile_value = float(np.percentile(diff_values, threshold_percentile))
        effective_threshold = int(np.clip(percentile_value, pixel_diff_threshold, max_pixel_diff_threshold))

    effective_chroma_threshold = float(min_chroma_diff_threshold)
    if use_chroma_seed:
        if valid_mask is not None:
            chroma_values = chroma_delta[valid_mask_bool]
        else:
            chroma_values = chroma_delta.reshape(-1)
        if adaptive_threshold and chroma_values.size > 0:
            chroma_percentile = float(np.percentile(chroma_values, chroma_threshold_percentile))
            effective_chroma_threshold = float(
                np.clip(chroma_percentile, min_chroma_diff_threshold, max_chroma_diff_threshold)
            )

    gray_seed = diff_gray > effective_threshold
    if use_chroma_seed:
        chroma_seed = chroma_delta > effective_chroma_threshold
        initial_change_mask = (gray_seed | chroma_seed) & valid_mask_bool
    else:
        initial_change_mask = gray_seed & valid_mask_bool
    initial_changed_pixels = int(np.count_nonzero(initial_change_mask))

    if classes_of_interest:
        ref_interest = np.isin(reference_seg.mask, classes_of_interest)
        tgt_interest = np.isin(target_seg.mask, classes_of_interest)
    else:
        ref_interest = np.ones_like(initial_change_mask, dtype=bool)
        tgt_interest = np.ones_like(initial_change_mask, dtype=bool)

    ref_gray_u8 = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    tgt_gray_u8 = cv2.cvtColor(aligned_target_bgr, cv2.COLOR_BGR2GRAY)
    edge_support_for_seed = _build_edge_support(ref_gray_u8, tgt_gray_u8, edge_dilate_kernel)

    ref_confident = reference_seg.confidence >= segmentation_confidence_threshold
    tgt_confident = target_seg.confidence >= segmentation_confidence_threshold
    same_semantic_label = reference_seg.mask == target_seg.mask
    interest_mask = ref_interest | tgt_interest
    semantic_gate_confident = interest_mask & (ref_confident | tgt_confident) & (~same_semantic_label)
    semantic_gate_relaxed = interest_mask & (~same_semantic_label)
    semantic_gate_interest_only = interest_mask

    confident_change_mask = initial_change_mask & semantic_gate_confident
    relaxed_change_mask = initial_change_mask & semantic_gate_relaxed
    interest_only_change_mask = initial_change_mask & semantic_gate_interest_only
    edge_only_change_mask = initial_change_mask & edge_support_for_seed

    gated_changed_pixels = int(np.count_nonzero(confident_change_mask))
    relaxed_gated_changed_pixels = int(np.count_nonzero(relaxed_change_mask))
    interest_only_changed_pixels = int(np.count_nonzero(interest_only_change_mask))
    edge_only_changed_pixels = int(np.count_nonzero(edge_only_change_mask))

    semantic_kept_fraction = float(gated_changed_pixels / max(1, initial_changed_pixels))
    relaxed_semantic_kept_fraction = float(relaxed_gated_changed_pixels / max(1, initial_changed_pixels))

    semantic_gate_mode = "none"
    if (
        classes_of_interest
        and gated_changed_pixels > 0
        and semantic_kept_fraction >= semantic_min_kept_fraction
    ):
        semantic_gate_mode = "confident_transition"
        seed_change = confident_change_mask
    elif (
        classes_of_interest
        and relaxed_gated_changed_pixels > 0
        and relaxed_semantic_kept_fraction >= semantic_min_kept_fraction
    ):
        semantic_gate_mode = "relaxed_transition"
        seed_change = relaxed_change_mask
    elif classes_of_interest:
        if interest_only_changed_pixels > 0:
            semantic_gate_mode = "interest_only"
            seed_change = interest_only_change_mask
        else:
            semantic_gate_mode = "edge_only"
            seed_change = edge_only_change_mask
    else:
        seed_change = initial_change_mask

    change_mask = _clean_mask(seed_change, blur_kernel, min_region_area)

    added_mask = change_mask & tgt_interest & (~ref_interest)
    removed_mask = change_mask & ref_interest & (~tgt_interest)

    unresolved = change_mask & (~added_mask) & (~removed_mask)
    ref_gray = ref_gray_u8.astype(np.int16)
    tgt_gray = tgt_gray_u8.astype(np.int16)
    delta = tgt_gray - ref_gray

    added_candidates = unresolved & (delta > intensity_margin)
    removed_candidates = unresolved & (delta < -intensity_margin)
    fallback_gate = (chroma_delta >= float(unresolved_min_chroma_delta)) | edge_support_for_seed
    added_candidates &= fallback_gate
    removed_candidates &= fallback_gate

    if semantic_gate_mode == "edge_only" and not allow_removed_without_semantic_transition:
        removed_candidates = np.zeros_like(removed_candidates, dtype=bool)

    added_mask |= added_candidates
    removed_mask |= removed_candidates
    ambiguous_mask = unresolved & (~(delta > intensity_margin)) & (~(delta < -intensity_margin))
    ambiguous_mask |= unresolved & (~added_mask) & (~removed_mask)

    added_mask = _clean_mask(added_mask, blur_kernel, min_region_area)
    removed_mask = _clean_mask(removed_mask, blur_kernel, min_region_area)
    ambiguous_mask = _clean_mask(ambiguous_mask, blur_kernel, min_region_area)

    sprawling_added_regions_dropped = 0
    if drop_sprawling_added_regions and np.any(added_mask):
        added_mask, sprawling_added_regions_dropped = _drop_sprawling_components(
            added_mask,
            sprawling_min_area,
            sprawling_max_fill_ratio,
        )

    combined = added_mask | removed_mask | ambiguous_mask
    edge_filtered_components = 0
    if use_structural_edge_filter and np.any(combined):
        edge_support = edge_support_for_seed
        combined, edge_filtered_components = _filter_by_edge_density(
            combined,
            edge_support,
            structural_min_edge_fraction,
            min_region_area,
        )
        added_mask &= combined
        removed_mask &= combined
        ambiguous_mask &= combined

    regions = _extract_regions(combined, added_mask, removed_mask, min_region_area)

    metrics = {
        "effective_pixel_diff_threshold": effective_threshold,
        "effective_chroma_diff_threshold": float(effective_chroma_threshold) if use_chroma_seed else None,
        "initial_changed_pixels": initial_changed_pixels,
        "gated_changed_pixels": gated_changed_pixels,
        "semantic_gate_mode": semantic_gate_mode,
        "support_fraction": float(np.count_nonzero(valid_mask_bool) / max(1, valid_mask_bool.size)),
        "changed_pixels": int(np.count_nonzero(combined)),
        "added_pixels": int(np.count_nonzero(added_mask)),
        "removed_pixels": int(np.count_nonzero(removed_mask)),
        "ambiguous_pixels": int(np.count_nonzero(ambiguous_mask)),
        "num_regions": len(regions),
        "num_added_regions": int(sum(1 for r in regions if r["type"] == "added")),
        "num_removed_regions": int(sum(1 for r in regions if r["type"] == "removed")),
        "edge_filtered_components": int(edge_filtered_components),
        "sprawling_added_regions_dropped": int(sprawling_added_regions_dropped),
    }

    return ChangeResult(
        change_mask=combined,
        added_mask=added_mask,
        removed_mask=removed_mask,
        ambiguous_mask=ambiguous_mask,
        regions=regions,
        metrics=metrics,
    )
