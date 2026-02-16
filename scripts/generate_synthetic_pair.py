from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _draw_city_canvas(width: int, height: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), (225, 228, 230), dtype=np.uint8)

    for _ in range(6):
        y = int(rng.integers(40, height - 40))
        cv2.line(img, (0, y), (width, y), (130, 130, 130), 16)
    for _ in range(6):
        x = int(rng.integers(40, width - 40))
        cv2.line(img, (x, 0), (x, height), (130, 130, 130), 16)

    for _ in range(40):
        x = int(rng.integers(10, width - 90))
        y = int(rng.integers(10, height - 90))
        w = int(rng.integers(25, 90))
        h = int(rng.integers(25, 90))
        color = (
            int(rng.integers(150, 210)),
            int(rng.integers(160, 215)),
            int(rng.integers(165, 220)),
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)

    for _ in range(25):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        r = int(rng.integers(8, 18))
        cv2.circle(img, (x, y), r, (60, 130, 60), -1)

    return img


def _modify_for_t2(image_t1: np.ndarray) -> np.ndarray:
    t2 = image_t1.copy()

    cv2.rectangle(t2, (540, 210), (620, 280), (190, 200, 210), -1)
    cv2.rectangle(t2, (540, 210), (620, 280), (80, 80, 80), 2)

    cv2.rectangle(t2, (220, 420), (300, 500), (130, 130, 130), -1)
    cv2.line(t2, (220, 460), (300, 460), (120, 120, 120), 8)

    cv2.rectangle(t2, (350, 120), (430, 190), (150, 150, 150), -1)
    cv2.line(t2, (350, 160), (430, 160), (120, 120, 120), 8)

    return t2


def _warp_like_drift(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [12, 20],
            [w - 30, 8],
            [w - 5, h - 18],
            [25, h - 4],
        ]
    )
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, H, (w, h))
    warped = cv2.convertScaleAbs(warped, alpha=1.04, beta=8)
    return warped


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic temporal aerial pair.")
    parser.add_argument("--out-dir", default="data/synthetic", help="Output directory.")
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=650)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = _draw_city_canvas(args.width, args.height, seed=7)
    t2_nominal = _modify_for_t2(t1)
    t2 = _warp_like_drift(t2_nominal)

    cv2.imwrite(str(out_dir / "t1_reference.png"), t1)
    cv2.imwrite(str(out_dir / "t2_target.png"), t2)
    print(f"Wrote synthetic pair to {out_dir}")


if __name__ == "__main__":
    main()
