from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Protocol

import cv2
import numpy as np

from .schemas import Detection

logger = logging.getLogger(__name__)

MODEL_NAMES = ("yolov12", "yolov13", "rtdetrv3", "rf-detr")
_WEIGHT_ENV = {
    "yolov12": "VISION_YOLOV12_WEIGHTS",
    "yolov13": "VISION_YOLOV13_WEIGHTS",
    "rtdetrv3": "VISION_RTDETRV3_WEIGHTS",
    "rf-detr": "VISION_RFDETR_WEIGHTS",
}
_LABELS = {0: "primordium", 1: "mature_basidiocarp"}
_LABEL_ALIASES = {
    "primordium": "primordium",
    "mature_basidiocarp": "mature_basidiocarp",
    "mature basidiocarp": "mature_basidiocarp",
    "mature-basidiocarp": "mature_basidiocarp",
}


class Detector(Protocol):
    def predict(self, image: np.ndarray) -> list[Detection]: ...


class ModelUnavailableError(RuntimeError):
    """Raised when model weights or an inference backend are unavailable."""


class MockDetector:
    """No-op detector for API/CPU smoke tests when no trained weights exist."""

    def predict(self, image: np.ndarray) -> list[Detection]:
        return []


class UltralyticsDetector:
    def __init__(self, weights: str, device: str | None = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelUnavailableError("Install ultralytics to load this model") from exc
        self.model = YOLO(weights)
        self.device = device

    def predict(self, image: np.ndarray) -> list[Detection]:
        result = self.model.predict(image, verbose=False, device=self.device)[0]
        boxes = result.boxes
        if boxes is None:
            return []
        coordinates = boxes.xyxy.cpu().tolist()
        class_ids = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        return _make_detections(coordinates, class_ids, confidences, result.names)


class RFDETRDetector:
    def __init__(self, weights: str) -> None:
        try:
            from rfdetr import RFDETRBase
        except ImportError as exc:
            raise ModelUnavailableError("Install rfdetr to load this model") from exc
        self.model = RFDETRBase(pretrain_weights=weights)

    def predict(self, image: np.ndarray) -> list[Detection]:
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.model.predict(Image.fromarray(rgb), threshold=0.0)
        return _make_detections(
            result.xyxy.tolist(),
            result.class_id.tolist(),
            result.confidence.tolist(),
            getattr(self.model, "class_names", None),
        )


def _class_name(class_id: int, names: object) -> str | None:
    if isinstance(names, dict):
        value = names.get(class_id, names.get(str(class_id)))
    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        value = names[class_id]
    else:
        value = None

    if value is not None:
        normalized = str(value).strip().lower().replace("_", " ")
        label = _LABEL_ALIASES.get(normalized)
        if label is not None:
            return label
    return _LABELS.get(class_id)


def _make_detections(
    boxes: list[list[float]],
    class_ids: list[float],
    confidences: list[float],
    names: object,
) -> list[Detection]:
    detections = []
    for box, raw_class_id, confidence in zip(boxes, class_ids, confidences, strict=False):
        class_id = int(raw_class_id)
        class_label = _class_name(class_id, names)
        if class_label is None or len(box) != 4:
            continue
        detections.append(
            Detection(
                bbox=tuple(float(value) for value in box),
                class_label=class_label,
                confidence=float(confidence),
            )
        )
    return detections


class ModelRegistry:
    def __init__(
        self,
        *,
        mock_mode: bool | None = None,
        detectors: dict[str, Detector] | None = None,
    ) -> None:
        self.mock_mode = (
            os.getenv("VISION_MOCK_MODELS", "false").lower() in {"1", "true", "yes"}
            if mock_mode is None
            else mock_mode
        )
        self._detectors = dict(detectors or {})
        self._statuses = {name: "loaded" for name in self._detectors}
        self._load_lock = Lock()

    def available_models(self) -> list[dict[str, str]]:
        models = []
        for name in MODEL_NAMES:
            status = self._statuses.get(name)
            if status is None:
                status = "configured" if os.getenv(_WEIGHT_ENV[name]) else (
                    "mock" if self.mock_mode else "not_configured"
                )
            models.append({"model_name": name, "weights_status": status})
        return models

    def detect(self, model_name: str, image_bytes: bytes) -> list[Detection]:
        detector = self._get_detector(model_name)
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image could not be decoded after quality inspection")
        return detector.predict(image)

    def _get_detector(self, model_name: str) -> Detector:
        detector = self._detectors.get(model_name)
        if detector is not None:
            return detector

        with self._load_lock:
            detector = self._detectors.get(model_name)
            if detector is not None:
                return detector
            weights = os.getenv(_WEIGHT_ENV[model_name])
            if not weights:
                if self.mock_mode:
                    detector = MockDetector()
                    self._statuses[model_name] = "mock"
                else:
                    self._statuses[model_name] = "not_configured"
                    raise ModelUnavailableError(f"No weights configured for {model_name}")
            else:
                if not Path(weights).is_file():
                    self._statuses[model_name] = "error"
                    raise ModelUnavailableError(f"Configured weights for {model_name} do not exist")
                try:
                    if model_name == "rf-detr":
                        detector = RFDETRDetector(weights)
                    else:
                        device = os.getenv("VISION_DEVICE") or None
                        detector = UltralyticsDetector(weights, device=device)
                except Exception as exc:
                    self._statuses[model_name] = "error"
                    logger.exception("Could not load vision model %s", model_name)
                    raise ModelUnavailableError(f"Could not load {model_name}") from exc
                self._statuses[model_name] = "loaded"
            self._detectors[model_name] = detector
            return detector
