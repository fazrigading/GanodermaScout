from typing import Literal

from pydantic import BaseModel, Field

ClassLabel = Literal["primordium", "mature_basidiocarp"]
ModelName = Literal["yolov12", "yolov13", "rtdetrv3", "rf-detr"]


class Detection(BaseModel):
    bbox: tuple[float, float, float, float]
    class_label: ClassLabel
    confidence: float = Field(ge=0.0, le=1.0)


class DetectResponse(BaseModel):
    model_name: ModelName
    detections: list[Detection]
    inference_latency_ms: float = Field(ge=0.0)


class ModelStatus(BaseModel):
    model_name: ModelName
    weights_status: Literal["loaded", "mock", "configured", "not_configured", "error"]


class ModelsResponse(BaseModel):
    models: list[ModelStatus]
