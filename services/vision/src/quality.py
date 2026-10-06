from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

BLUR_THRESHOLD = 100.0
MIN_WIDTH = 224
MIN_HEIGHT = 224
DARK_CLIP_LIMIT = 0.45
BRIGHT_CLIP_LIMIT = 0.45


@dataclass
class QualityReport:
    is_acceptable: bool
    blur_score: float
    exposure_score: float
    rejection_reason: Optional[str] = None


def inspect_image(image_bytes: bytes) -> QualityReport:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return QualityReport(
            is_acceptable=False,
            blur_score=0.0,
            exposure_score=0.0,
            rejection_reason="image_undecodable",
        )
    h, w = img.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return QualityReport(
            is_acceptable=False,
            blur_score=0.0,
            exposure_score=0.0,
            rejection_reason="image_too_small",
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < BLUR_THRESHOLD:
        return QualityReport(
            is_acceptable=False,
            blur_score=blur_score,
            exposure_score=1.0,
            rejection_reason="image_blurred",
        )
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = float(hist.sum())
    dark_ratio = float(hist[:16].sum() / total)
    bright_ratio = float(hist[240:].sum() / total)
    exposure_score = float(1.0 - max(dark_ratio, bright_ratio))
    if dark_ratio > DARK_CLIP_LIMIT:
        return QualityReport(
            is_acceptable=False,
            blur_score=blur_score,
            exposure_score=exposure_score,
            rejection_reason="image_underexposed",
        )
    if bright_ratio > BRIGHT_CLIP_LIMIT:
        return QualityReport(
            is_acceptable=False,
            blur_score=blur_score,
            exposure_score=exposure_score,
            rejection_reason="image_overexposed",
        )
    return QualityReport(
        is_acceptable=True,
        blur_score=blur_score,
        exposure_score=exposure_score,
        rejection_reason=None,
    )
