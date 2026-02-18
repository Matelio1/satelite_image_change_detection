from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any
import warnings

import cv2
import numpy as np
from PIL import Image


@dataclass
class SegmentationOutput:
    mask: np.ndarray
    confidence: np.ndarray
    id2label: dict[int, str]


class SegformerSegmenter:
    def __init__(self, model_name: str, device: str = "auto", local_files_only: bool = False) -> None:
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from huggingface_hub import snapshot_download
        from transformers import (
            AutoFeatureExtractor,
            AutoImageProcessor,
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
        from transformers.utils import logging as transformers_logging

        self._torch = torch
        transformers_logging.disable_progress_bar()
        warnings.filterwarnings(
            "ignore",
            message="The following named arguments are not valid for `SegformerImageProcessor.__init__`",
            category=UserWarning,
        )
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.resolved_model_name = ""
        self.load_source = ""
        self.processor_kind = ""

        load_plan: list[tuple[str, bool, str]] = []
        model_path = Path(model_name)
        if model_path.exists():
            load_plan.append((str(model_path), True, "local_path"))
        else:
            try:
                cached_snapshot = snapshot_download(repo_id=model_name, local_files_only=True)
                load_plan.append((cached_snapshot, True, "hf_cache_snapshot"))
            except Exception:
                pass

        if local_files_only:
            load_plan.append((model_name, True, "repo_id_offline"))
        else:
            load_plan.append((model_name, False, "repo_id_online"))

        seen: set[tuple[str, bool]] = set()
        unique_plan: list[tuple[str, bool, str]] = []
        for source, local_only, source_name in load_plan:
            key = (source, local_only)
            if key in seen:
                continue
            seen.add(key)
            unique_plan.append((source, local_only, source_name))

        load_errors: list[str] = []
        for source, local_only, source_name in unique_plan:
            try:
                processor_errors: list[str] = []
                try:
                    self.processor = AutoImageProcessor.from_pretrained(source, local_files_only=local_only)
                    self.processor_kind = "AutoImageProcessor"
                    processor_errors = []
                except Exception as processor_exc:
                    processor_errors.append(f"AutoImageProcessor: {processor_exc}")
                    try:
                        self.processor = SegformerImageProcessor.from_pretrained(source, local_files_only=local_only)
                        self.processor_kind = "SegformerImageProcessor"
                        processor_errors = []
                    except Exception as segformer_processor_exc:
                        processor_errors.append(f"SegformerImageProcessor: {segformer_processor_exc}")
                        self.processor = AutoFeatureExtractor.from_pretrained(source, local_files_only=local_only)
                        self.processor_kind = "AutoFeatureExtractor"
                        processor_errors = []
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    source,
                    local_files_only=local_only,
                    use_safetensors=False,
                    low_cpu_mem_usage=False,
                ).to(self.device)
                self.model.eval()
                self.resolved_model_name = str(source)
                self.load_source = source_name
                break
            except Exception as exc:
                if processor_errors:
                    load_errors.append(
                        f"{source_name} ({source}, local_only={local_only}): {exc} | "
                        + " ; ".join(processor_errors)
                    )
                else:
                    load_errors.append(f"{source_name} ({source}, local_only={local_only}): {exc}")
        else:
            hint = ""
            if any(
                "pytorch_model.bin" in err or "model.safetensors" in err or "does not appear to have a file named" in err
                for err in load_errors
            ):
                hint = (
                    "\nHint: model weights were not found in local cache. "
                    "If this is an online run, retry with internet enabled and consider updating "
                    "`huggingface_hub` plus installing `hf_xet` for Xet-backed model files."
                )
            raise RuntimeError(
                "Unable to load segmentation model. Attempts:\n- " + "\n- ".join(load_errors) + hint
            )

    def segment(self, image_bgr: np.ndarray) -> SegmentationOutput:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        with self._torch.no_grad():
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            upsampled_logits = self._torch.nn.functional.interpolate(
                logits,
                size=image_bgr.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            probabilities = self._torch.softmax(upsampled_logits, dim=1)
            pred_mask = probabilities.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
            confidence = probabilities.max(dim=1).values[0].cpu().numpy().astype(np.float32)

        raw_id2label: dict[Any, str] = self.model.config.id2label
        id2label = {int(k): v for k, v in raw_id2label.items()}
        return SegmentationOutput(mask=pred_mask, confidence=confidence, id2label=id2label)


def classes_to_ids(class_names: list[str], id2label: dict[int, str]) -> list[int]:
    if not class_names:
        return []
    wanted = [name.lower().strip() for name in class_names]
    selected: list[int] = []

    def _tokens(text: str) -> set[str]:
        tokens = {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}
        normalized = set(tokens)
        for token in tokens:
            if token.endswith("s") and len(token) > 3:
                normalized.add(token[:-1])
        return normalized

    for class_id, label in id2label.items():
        normalized = label.lower()
        label_tokens = _tokens(normalized)
        if any(w == normalized or w in label_tokens for w in wanted):
            selected.append(class_id)
    return selected
