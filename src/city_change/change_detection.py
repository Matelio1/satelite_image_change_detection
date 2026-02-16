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


def detect_changes(
    reference_bgr: np.ndarray,
    aligned_target_bgr: np.ndarray,
    reference_seg: SegmentationOutput,
    target_seg: SegmentationOutput,
    classes_of_interest: list[int],
    segmentation_confidence_threshold: float,
    pixel_diff_threshold: int = 30,
    blur_kernel: int = 5,
    min_region_area: int = 180,
    intensity_margin: int = 8,
    semantic_min_kept_fraction: float = 0.05,
    valid_mask: np.ndarray | None = None,
    adaptive_threshold: bool = True,
    threshold_percentile: float = 88.0,
    max_pixel_diff_threshold: int = 90,
) -> ChangeResult:
    diff = cv2.absdiff(reference_bgr, aligned_target_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
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

    _, initial_change = cv2.threshold(diff_gray, effective_threshold, 255, cv2.THRESH_BINARY)
    initial_change_mask = (initial_change > 0) & valid_mask_bool
    initial_changed_pixels = int(np.count_nonzero(initial_change_mask))

    if classes_of_interest:
        ref_interest = np.isin(reference_seg.mask, classes_of_interest)
        tgt_interest = np.isin(target_seg.mask, classes_of_interest)
    else:
        ref_interest = np.ones_like(initial_change_mask, dtype=bool)
        tgt_interest = np.ones_like(initial_change_mask, dtype=bool)

    ref_confident = reference_seg.confidence >= segmentation_confidence_threshold
    tgt_confident = target_seg.confidence >= segmentation_confidence_threshold
    same_semantic_label = reference_seg.mask == target_seg.mask
    stable_semantics = same_semantic_label & ref_confident & tgt_confident
    semantic_gate = (ref_interest | tgt_interest) & (ref_confident | tgt_confident) & (~stable_semantics)
    gated_change_mask = initial_change_mask & semantic_gate
    gated_changed_pixels = int(np.count_nonzero(gated_change_mask))
    semantic_kept_fraction = float(gated_changed_pixels / max(1, initial_changed_pixels))

    use_semantic_gate = bool(
        classes_of_interest
        and gated_changed_pixels > 0
        and semantic_kept_fraction >= semantic_min_kept_fraction
    )
    seed_change = gated_change_mask if use_semantic_gate else initial_change_mask
    change_mask = _clean_mask(seed_change, blur_kernel, min_region_area)

    added_mask = change_mask & tgt_interest & (~ref_interest)
    removed_mask = change_mask & ref_interest & (~tgt_interest)

    unresolved = change_mask & (~added_mask) & (~removed_mask)
    ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    tgt_gray = cv2.cvtColor(aligned_target_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    delta = tgt_gray - ref_gray

    added_mask |= unresolved & (delta > intensity_margin)
    removed_mask |= unresolved & (delta < -intensity_margin)
    ambiguous_mask = unresolved & (~(delta > intensity_margin)) & (~(delta < -intensity_margin))

    added_mask = _clean_mask(added_mask, blur_kernel, min_region_area)
    removed_mask = _clean_mask(removed_mask, blur_kernel, min_region_area)
    ambiguous_mask = _clean_mask(ambiguous_mask, blur_kernel, min_region_area)

    combined = added_mask | removed_mask | ambiguous_mask
    regions = _extract_regions(combined, added_mask, removed_mask, min_region_area)

    metrics = {
        "effective_pixel_diff_threshold": effective_threshold,
        "initial_changed_pixels": initial_changed_pixels,
        "gated_changed_pixels": gated_changed_pixels,
        "semantic_kept_fraction": semantic_kept_fraction,
        "semantic_gate_used": use_semantic_gate,
        "support_pixels": int(np.count_nonzero(valid_mask_bool)),
        "support_fraction": float(np.count_nonzero(valid_mask_bool) / max(1, valid_mask_bool.size)),
        "changed_pixels": int(np.count_nonzero(combined)),
        "added_pixels": int(np.count_nonzero(added_mask)),
        "removed_pixels": int(np.count_nonzero(removed_mask)),
        "ambiguous_pixels": int(np.count_nonzero(ambiguous_mask)),
        "num_regions": len(regions),
        "num_added_regions": int(sum(1 for r in regions if r["type"] == "added")),
        "num_removed_regions": int(sum(1 for r in regions if r["type"] == "removed")),
    }

    return ChangeResult(
        change_mask=combined,
        added_mask=added_mask,
        removed_mask=removed_mask,
        ambiguous_mask=ambiguous_mask,
        regions=regions,
        metrics=metrics,
    )
