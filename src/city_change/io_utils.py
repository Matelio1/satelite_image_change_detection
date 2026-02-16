from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    out = Path(path)
    ensure_dir(out.parent)
    ok = cv2.imwrite(str(out), image)
    if not ok:
        raise IOError(f"Could not save image: {out}")


def save_binary_mask(path: str | Path, mask: np.ndarray) -> None:
    image = (mask.astype(np.uint8) * 255).clip(0, 255)
    save_image(path, image)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
