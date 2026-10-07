from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from .detector import ModelRegistry, ModelUnavailableError
from .quality import inspect_image
from .schemas import DetectResponse, ModelName, ModelsResponse

_RETAKE_RECOMMENDATIONS = {
    "image_blurred": "Hold the camera steady and retake the photo in focus.",
    "image_underexposed": "Move to a brighter area and retake the photo without blocking the light.",
    "image_overexposed": "Avoid direct glare and retake the photo in even lighting.",
    "image_too_small": "Retake the photo at a higher resolution, at least 224 × 224 pixels.",
    "image_undecodable": "Upload a valid image file and retake the photo if needed.",
}


def create_app(registry: ModelRegistry | None = None) -> FastAPI:
    app = FastAPI(title="GanodermaScout Vision Service", version="1.0.0")
    app.state.model_registry = registry or ModelRegistry()

    @app.post("/detect", response_model=DetectResponse)
    async def detect(
        file: UploadFile = File(...),
        model_name: ModelName = Query("yolov12"),
    ) -> DetectResponse:
        image_bytes = await file.read()
        quality = inspect_image(image_bytes)
        if not quality.is_acceptable:
            reason = quality.rejection_reason or "image_undecodable"
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "image_quality_rejected",
                    "reason": reason,
                    "retake_recommendation": _RETAKE_RECOMMENDATIONS[reason],
                    "quality": {
                        "blur_score": quality.blur_score,
                        "exposure_score": quality.exposure_score,
                    },
                },
            )

        started = perf_counter()
        try:
            detections = await run_in_threadpool(
                app.state.model_registry.detect, model_name, image_bytes
            )
        except ModelUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "model_unavailable", "model_name": model_name},
            ) from exc
        latency_ms = (perf_counter() - started) * 1000.0
        return DetectResponse(
            model_name=model_name,
            detections=detections,
            inference_latency_ms=latency_ms,
        )

    @app.get("/models", response_model=ModelsResponse)
    async def models() -> ModelsResponse:
        return ModelsResponse(models=app.state.model_registry.available_models())

    return app


app = create_app()
