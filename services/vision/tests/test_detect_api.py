import os
import sys

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detector import ModelRegistry
from src.main import create_app
from src.schemas import Detection


class StubDetector:
    def __init__(self):
        self.calls = 0

    def predict(self, image: np.ndarray) -> list[Detection]:
        self.calls += 1
        return [
            Detection(
                bbox=(12.0, 24.0, 160.0, 190.0),
                class_label="mature_basidiocarp",
                confidence=0.94,
            )
        ]


def encode(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return bytes(buffer)


def sharp_image() -> np.ndarray:
    rng = np.random.default_rng(23)
    image = np.full((320, 320, 3), 128, dtype=np.uint8)
    image[::16, :] = 255
    image[:, ::16] = 0
    return cv2.add(image, rng.integers(0, 30, image.shape, dtype=np.uint8))


def test_detect_returns_boxes_labels_confidence_and_latency():
    detector = StubDetector()
    app = create_app(ModelRegistry(detectors={"yolov12": detector}))
    client = TestClient(app)

    response = client.post(
        "/detect?model_name=yolov12",
        files={"file": ("palm.png", encode(sharp_image()), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "yolov12"
    assert body["detections"] == [
        {
            "bbox": [12.0, 24.0, 160.0, 190.0],
            "class_label": "mature_basidiocarp",
            "confidence": 0.94,
        }
    ]
    assert body["inference_latency_ms"] >= 0
    assert detector.calls == 1


def test_quality_rejection_returns_retake_recommendation_before_inference():
    detector = StubDetector()
    app = create_app(ModelRegistry(detectors={"yolov12": detector}))
    client = TestClient(app)
    dark = np.full((320, 320, 3), 5, dtype=np.uint8)

    response = client.post(
        "/detect?model_name=yolov12",
        files={"file": ("dark.png", encode(dark), "image/png")},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "image_quality_rejected"
    assert detail["reason"] == "image_underexposed"
    assert detail["retake_recommendation"]
    assert detector.calls == 0


def test_models_lists_architectures_and_weight_status():
    client = TestClient(create_app(ModelRegistry(mock_mode=True)))

    response = client.get("/models")

    assert response.status_code == 200
    models = response.json()["models"]
    assert [model["model_name"] for model in models] == [
        "yolov12",
        "yolov13",
        "rtdetrv3",
        "rf-detr",
    ]
    assert all(model["weights_status"] == "mock" for model in models)
