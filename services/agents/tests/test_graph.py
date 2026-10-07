import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph import build_graph
from src.nodes import AdvisoryDraft, CitedRecommendation, classify_request
from services.core.src.rag.retriever import RetrievedPassage


class VisionStub:
    def __init__(self):
        self.calls = []

    def detect(self, image_bytes: bytes, model_name: str):
        self.calls.append((image_bytes, model_name))
        return {
            "model_name": model_name,
            "inference_latency_ms": 12.5,
            "detections": [
                {
                    "bbox": [12, 24, 160, 190],
                    "class_label": "mature_basidiocarp",
                    "confidence": 0.94,
                }
            ],
        }


class MemoryStub:
    def __init__(self):
        self.arguments = None

    def load_history(self, *, palm_id=None, palm_code=None, block_id=None):
        self.arguments = (palm_id, palm_code, block_id)
        return {
            "palm_history": [{"inspection_id": "inspection-1"}],
            "block_history": [{"inspection_id": "inspection-2"}],
        }


class RetrievalStub:
    def __init__(self):
        self.query = None

    def __call__(self, query, top_k):
        self.query = query
        assert top_k == 5
        return [
            RetrievedPassage(
                chunk_id="BSR-EXT-001#p1",
                document_id="BSR-EXT-001",
                content="Remove visible basidiocarps and clear debris near infected palms.",
                metadata={"title": "Field sanitation"},
                citation_anchor="BSR-EXT-001#p1",
                score=0.03,
            ),
            RetrievedPassage(
                chunk_id="BSR-EXT-001#p2",
                document_id="BSR-EXT-001",
                content="Repeated mature basidiocarp emergence can warrant isolation trenching.",
                metadata={"title": "Severe cases"},
                citation_anchor="BSR-EXT-001#p2",
                score=0.02,
            ),
        ]


class ModelStub:
    def with_structured_output(self, schema):
        assert schema is AdvisoryDraft
        return self

    def invoke(self, messages):
        return AdvisoryDraft(
            recommendations=[
                CitedRecommendation(
                    text="Remove visible fruiting bodies and clear loose debris. [Source: FAKE#p9]",
                    source_ids=["BSR-EXT-001#p1", "FAKE#p9"],
                ),
                CitedRecommendation(
                    text="Repeated mature basidiocarp emergence may warrant isolation trenching.",
                    source_ids=["BSR-EXT-001#p2"],
                ),
            ]
        )


def test_router_classifies_inspection_history_and_advisory_requests():
    assert classify_request({"image_bytes": b"image"}) == "new_inspection"
    assert classify_request({"user_query": "Show the previous inspection history"}) == "history_query"
    assert classify_request({"user_query": "What sanitation steps are recommended?"}) == "general_advisory"


def test_graph_hydrates_memory_retrieves_context_and_cites_synthesis(monkeypatch):
    monkeypatch.setenv("VISION_MODEL_NAME", "yolov12")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "5")
    vision = VisionStub()
    memory = MemoryStub()
    retrieval = RetrievalStub()
    graph = build_graph(
        model=ModelStub(),
        retriever=retrieval,
        memory_store=memory,
        vision_client=vision,
    )

    result = graph.invoke(
        {
            "user_query": "What should I do about this mature conk?",
            "image_bytes": b"palm image",
            "palm_id": "palm-1",
            "block_id": "block-1",
        }
    )

    assert result["request_type"] == "new_inspection"
    assert result["image_detections"][0]["class_label"] == "mature_basidiocarp"
    assert result["vision_model_name"] == "yolov12"
    assert vision.calls == [(b"palm image", "yolov12")]
    assert memory.arguments == ("palm-1", None, "block-1")
    assert result["palm_history"] == [{"inspection_id": "inspection-1"}]
    assert result["block_history"] == [{"inspection_id": "inspection-2"}]
    assert "mature_basidiocarp" in retrieval.query
    assert "What should I do about this mature conk?" in retrieval.query
    assert "[Source: BSR-EXT-001#p1]" in result["drafted_advisory"]
    assert "[Source: BSR-EXT-001#p2]" in result["drafted_advisory"]
    assert "FAKE#p9" not in result["drafted_advisory"]
    assert result["verification_status"] == "pending"
