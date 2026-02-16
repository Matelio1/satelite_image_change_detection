from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class RegistrationResult:
    aligned_target: np.ndarray
    homography: np.ndarray
    match_visualization: np.ndarray
    support_mask: np.ndarray
    stats: dict[str, Any]


class FeatureRegistrar:
    def __init__(
        self,
        method: str = "orb",
        ratio_test: float = 0.75,
        ransac_reproj_threshold: float = 4.0,
        max_features: int = 6000,
        min_inlier_matches: int = 12,
        min_inlier_ratio: float = 0.25,
        allow_identity_fallback: bool = True,
        min_overlap_ratio: float = 0.45,
        fallback_improvement_margin: float = 1.5,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
    ) -> None:
        self.method = method.lower()
        self.ratio_test = ratio_test
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.max_features = max_features
        self.min_inlier_matches = min_inlier_matches
        self.min_inlier_ratio = min_inlier_ratio
        self.allow_identity_fallback = allow_identity_fallback
        self.min_overlap_ratio = min_overlap_ratio
        self.fallback_improvement_margin = fallback_improvement_margin
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

    def _create_detector(self) -> cv2.Feature2D:
        if self.method == "orb":
            return cv2.ORB_create(nfeatures=self.max_features)
        if self.method == "sift":
            if not hasattr(cv2, "SIFT_create"):
                raise RuntimeError("SIFT is not available in this OpenCV build.")
            return cv2.SIFT_create(nfeatures=self.max_features)
        if self.method == "akaze":
            return cv2.AKAZE_create()
        raise ValueError(f"Unsupported registration method: {self.method}")

    def _create_matcher(self) -> cv2.BFMatcher:
        if self.method == "orb":
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        if self.method == "akaze":
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def _preprocess_gray(self, gray: np.ndarray) -> np.ndarray:
        if not self.use_clahe:
            return gray
        tile = max(2, int(self.clahe_tile_grid_size))
        clahe = cv2.createCLAHE(clipLimit=float(self.clahe_clip_limit), tileGridSize=(tile, tile))
        return clahe.apply(gray)

    @staticmethod
    def _identity_alignment(reference_bgr: np.ndarray, target_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h_ref, w_ref = reference_bgr.shape[:2]
        h_tgt, w_tgt = target_bgr.shape[:2]
        aligned = cv2.resize(target_bgr, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
        scale_x = w_ref / max(1.0, float(w_tgt))
        scale_y = h_ref / max(1.0, float(h_tgt))
        transform = np.array([[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return aligned, transform

    @staticmethod
    def _affine_to_homography(affine_2x3: np.ndarray) -> np.ndarray:
        transform = np.eye(3, dtype=np.float64)
        transform[:2, :] = affine_2x3.astype(np.float64)
        return transform

    @staticmethod
    def _warp_valid_mask(target_shape: tuple[int, int], reference_shape: tuple[int, int], transform: np.ndarray) -> np.ndarray:
        h_tgt, w_tgt = target_shape
        h_ref, w_ref = reference_shape
        ones = np.ones((h_tgt, w_tgt), dtype=np.uint8)
        warped = cv2.warpPerspective(ones, transform, (w_ref, h_ref), flags=cv2.INTER_NEAREST)
        return warped > 0

    @staticmethod
    def _is_plausible_transform(transform: np.ndarray, target_shape: tuple[int, int], reference_shape: tuple[int, int]) -> bool:
        if not np.all(np.isfinite(transform)):
            return False

        h_tgt, w_tgt = target_shape
        h_ref, w_ref = reference_shape
        corners = np.array(
            [[[0, 0]], [[w_tgt - 1, 0]], [[w_tgt - 1, h_tgt - 1]], [[0, h_tgt - 1]]],
            dtype=np.float32,
        )
        warped = cv2.perspectiveTransform(corners, transform.astype(np.float64)).reshape(-1, 2)
        hull = cv2.convexHull(warped.astype(np.float32))
        area = float(cv2.contourArea(hull))
        ref_area = float(max(1, w_ref * h_ref))
        area_ratio = area / ref_area
        if area_ratio < 0.2 or area_ratio > 3.0:
            return False

        min_x, min_y = warped.min(axis=0)
        max_x, max_y = warped.max(axis=0)
        x_lo, x_hi = -0.35 * w_ref, 1.35 * w_ref
        y_lo, y_hi = -0.35 * h_ref, 1.35 * h_ref
        return bool(min_x >= x_lo and max_x <= x_hi and min_y >= y_lo and max_y <= y_hi)

    @staticmethod
    def _build_support_mask(
        reference_shape: tuple[int, int],
        keypoints_ref: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
        inlier_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        h_ref, w_ref = reference_shape
        if not matches:
            return np.ones((h_ref, w_ref), dtype=bool)

        selected_points: list[tuple[float, float]] = []
        if inlier_mask is not None:
            flags = inlier_mask.ravel().astype(np.uint8).tolist()
            for m, keep in zip(matches, flags):
                if keep:
                    selected_points.append(keypoints_ref[m.queryIdx].pt)
        else:
            for m in matches:
                selected_points.append(keypoints_ref[m.queryIdx].pt)

        if len(selected_points) < 8:
            return np.ones((h_ref, w_ref), dtype=bool)

        points = np.array(selected_points, dtype=np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(points)
        mask = np.zeros((h_ref, w_ref), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

        # Expand sparse support so uncertain alignment still compares nearby neighborhoods.
        kernel_size = max(15, int(min(h_ref, w_ref) * 0.12))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        coverage = float(np.count_nonzero(mask) / max(1, h_ref * w_ref))
        if coverage < 0.08:
            return np.ones((h_ref, w_ref), dtype=bool)
        return mask > 0

    def register(self, reference_bgr: np.ndarray, target_bgr: np.ndarray) -> RegistrationResult:
        detector = self._create_detector()
        matcher = self._create_matcher()

        ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
        ref_features = self._preprocess_gray(ref_gray)
        tgt_features = self._preprocess_gray(tgt_gray)

        kp_ref, des_ref = detector.detectAndCompute(ref_features, None)
        kp_tgt, des_tgt = detector.detectAndCompute(tgt_features, None)

        if des_ref is None or des_tgt is None:
            raise RuntimeError("Insufficient features were detected for registration.")

        knn_matches = matcher.knnMatch(des_ref, des_tgt, k=2)
        good_matches: list[cv2.DMatch] = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test * n.distance:
                good_matches.append(m)

        good_matches.sort(key=lambda m: m.distance)

        h_ref, w_ref = reference_bgr.shape[:2]
        h_tgt, w_tgt = target_bgr.shape[:2]

        candidates: list[dict[str, Any]] = []

        def _append_candidate(transform_name: str, transform: np.ndarray, inlier_mask: np.ndarray | None) -> None:
            aligned_target = cv2.warpPerspective(target_bgr, transform, (w_ref, h_ref))
            aligned_gray = cv2.cvtColor(aligned_target, cv2.COLOR_BGR2GRAY)
            inlier_count = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
            inlier_ratio = float(inlier_count / max(1, len(good_matches)))
            overlap_mask = self._warp_valid_mask((h_tgt, w_tgt), (h_ref, w_ref), transform)
            overlap_ratio = float(np.count_nonzero(overlap_mask) / max(1, h_ref * w_ref))
            plausible = self._is_plausible_transform(transform, (h_tgt, w_tgt), (h_ref, w_ref))
            alignment_mae = float(np.mean(cv2.absdiff(ref_gray, aligned_gray)))
            candidates.append(
                {
                    "transform_name": transform_name,
                    "aligned_target": aligned_target,
                    "transform": transform,
                    "inlier_mask": inlier_mask,
                    "inlier_count": inlier_count,
                    "inlier_ratio": inlier_ratio,
                    "plausible": plausible,
                    "overlap_mask": overlap_mask,
                    "overlap_ratio": overlap_ratio,
                    "alignment_mae": alignment_mae,
                }
            )

        if len(good_matches) >= 4:
            src_pts = np.float32([kp_tgt[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            homography, homography_mask = cv2.findHomography(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_threshold,
            )
            if homography is not None:
                _append_candidate("homography", homography.astype(np.float64), homography_mask)

            affine, affine_mask = cv2.estimateAffinePartial2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_threshold,
            )
            if affine is not None:
                _append_candidate("affine", self._affine_to_homography(affine), affine_mask)

        aligned_identity, identity_transform = self._identity_alignment(reference_bgr, target_bgr)
        identity_overlap = np.ones((h_ref, w_ref), dtype=bool)
        identity_candidate = {
            "transform_name": "identity",
            "aligned_target": aligned_identity,
            "transform": identity_transform,
            "inlier_mask": None,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "plausible": True,
            "overlap_mask": identity_overlap,
            "overlap_ratio": 1.0,
            "alignment_mae": float(
                np.mean(cv2.absdiff(ref_gray, cv2.cvtColor(aligned_identity, cv2.COLOR_BGR2GRAY)))
            ),
        }

        selected: dict[str, Any] | None = None
        reliable_candidates = [
            c
            for c in candidates
            if c["inlier_count"] >= self.min_inlier_matches
            and c["inlier_ratio"] >= self.min_inlier_ratio
            and c["plausible"]
            and c["overlap_ratio"] >= self.min_overlap_ratio
        ]
        if reliable_candidates:
            selected = max(
                reliable_candidates,
                key=lambda c: (c["inlier_count"], c["inlier_ratio"], -c["alignment_mae"]),
            )
        elif self.allow_identity_fallback:
            selected = identity_candidate
            plausible_unreliable = [c for c in candidates if c["plausible"]]
            if plausible_unreliable:
                def _fallback_score(candidate: dict[str, Any]) -> float:
                    return float(candidate["alignment_mae"] + 25.0 * (1.0 - float(candidate["overlap_ratio"])))

                best_candidate = min(plausible_unreliable, key=_fallback_score)
                if _fallback_score(best_candidate) + self.fallback_improvement_margin < _fallback_score(identity_candidate):
                    selected = best_candidate
        elif candidates:
            selected = min(candidates, key=lambda c: c["alignment_mae"])
        else:
            raise RuntimeError("Registration failed: could not estimate any transform and fallback is disabled.")

        draw_n = min(len(good_matches), 350)
        draw_matches = good_matches[:draw_n]
        draw_mask: list[int] | None = None
        raw_mask = selected["inlier_mask"]
        if raw_mask is not None:
            draw_mask = raw_mask.ravel().astype(np.uint8).tolist()[:draw_n]

        match_visualization = cv2.drawMatches(
            reference_bgr,
            kp_ref,
            target_bgr,
            kp_tgt,
            draw_matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=draw_mask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        registration_reliable = bool(
            selected["transform_name"] != "identity"
            and selected["inlier_count"] >= self.min_inlier_matches
            and selected["inlier_ratio"] >= self.min_inlier_ratio
            and selected["plausible"]
            and selected["overlap_ratio"] >= self.min_overlap_ratio
        )

        if registration_reliable:
            support_mask = np.ones((h_ref, w_ref), dtype=bool)
        else:
            support_mask = self._build_support_mask((h_ref, w_ref), kp_ref, good_matches, selected["inlier_mask"])
            support_mask &= selected["overlap_mask"]
            support_coverage = float(np.count_nonzero(support_mask) / max(1, h_ref * w_ref))
            if support_coverage < 0.08:
                support_mask = selected["overlap_mask"].copy()

        stats = {
            "method": self.method,
            "transform_name": selected["transform_name"],
            "reference_keypoints": len(kp_ref),
            "target_keypoints": len(kp_tgt),
            "raw_good_matches": len(good_matches),
            "inlier_matches": int(selected["inlier_count"]),
            "inlier_ratio": float(selected["inlier_ratio"]),
            "fallback_used": bool(selected["transform_name"] == "identity"),
            "registration_reliable": registration_reliable,
            "min_inlier_matches_required": self.min_inlier_matches,
            "min_inlier_ratio_required": self.min_inlier_ratio,
            "min_overlap_ratio_required": self.min_overlap_ratio,
            "plausible_transform": bool(selected["plausible"]),
            "overlap_ratio": float(selected["overlap_ratio"]),
            "clahe_enabled": self.use_clahe,
            "alignment_mae": float(selected["alignment_mae"]),
            "support_mask_coverage": float(np.count_nonzero(support_mask) / max(1, h_ref * w_ref)),
        }

        return RegistrationResult(
            aligned_target=selected["aligned_target"],
            homography=selected["transform"],
            match_visualization=match_visualization,
            support_mask=support_mask,
            stats=stats,
        )
