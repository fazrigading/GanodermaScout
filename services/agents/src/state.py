from __future__ import annotations

from typing import Any, Literal, TypedDict

RequestType = Literal["new_inspection", "history_query", "general_advisory"]
VerificationStatus = Literal["pending", "approved", "rewritten", "refused"]


class AgentState(TypedDict, total=False):
    user_query: str
    request_type: RequestType
    vision_processed: bool
    image_bytes: bytes
    image_detections: list[dict[str, Any]]
    vision_model_name: str
    vision_latency_ms: float
    palm_id: str
    palm_code: str
    block_id: str
    palm_history: list[dict[str, Any]]
    block_history: list[dict[str, Any]]
    retrieved_passages: list[dict[str, Any]]
    drafted_advisory: str
    verification_status: VerificationStatus
    verification_metadata: dict[str, Any]
    metadata: dict[str, Any]
