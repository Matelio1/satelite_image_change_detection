from __future__ import annotations

import cv2
import numpy as np


def overlay_changes(
    base_bgr: np.ndarray,
    added_mask: np.ndarray,
    removed_mask: np.ndarray,
    ambiguous_mask: np.ndarray | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = base_bgr.copy()
    overlay[added_mask] = (60, 220, 20)  # green
    overlay[removed_mask] = (20, 60, 230)  # red
    if ambiguous_mask is not None:
        overlay[ambiguous_mask] = (40, 220, 220)  # yellow/cyan tint
    return cv2.addWeighted(overlay, alpha, base_bgr, 1 - alpha, 0)


def draw_regions(base_bgr: np.ndarray, regions: list[dict]) -> np.ndarray:
    output = base_bgr.copy()
    for region in regions:
        x, y, w, h = region["bbox_xywh"]
        region_type = region["type"]
        if region_type == "added":
            color = (60, 220, 20)
        elif region_type == "removed":
            color = (20, 60, 230)
        else:
            color = (40, 220, 220)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            output,
            region_type,
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def _fit_tile(image: np.ndarray, tile_width: int, tile_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(tile_width / w, tile_height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    x0 = (tile_width - new_w) // 2
    y0 = (tile_height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _tile_with_title(image: np.ndarray, title: str, tile_width: int, tile_height: int) -> np.ndarray:
    tile = _fit_tile(image, tile_width, tile_height)
    top = np.zeros((38, tile_width, 3), dtype=np.uint8)
    cv2.putText(top, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    return cv2.vconcat([top, tile])


def build_summary_panel(
    reference: np.ndarray,
    target: np.ndarray,
    aligned_target: np.ndarray,
    matches: np.ndarray,
    overlay: np.ndarray,
    regions_image: np.ndarray,
    tile_width: int = 520,
) -> np.ndarray:
    tile_height = max(260, int(tile_width * 0.58))
    row1 = cv2.hconcat(
        [
            _tile_with_title(reference, "Reference (T1)", tile_width, tile_height),
            _tile_with_title(target, "Target (T2)", tile_width, tile_height),
            _tile_with_title(matches, "Feature Matches", tile_width, tile_height),
        ]
    )
    row2 = cv2.hconcat(
        [
            _tile_with_title(aligned_target, "Registered Target", tile_width, tile_height),
            _tile_with_title(overlay, "Change Overlay", tile_width, tile_height),
            _tile_with_title(regions_image, "Added/Removed Regions", tile_width, tile_height),
        ]
    )
    return cv2.vconcat([row1, row2])
