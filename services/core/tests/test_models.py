import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db import (
    Base,
    Block,
    Detection,
    DetectionCorrection,
    DocumentChunk,
    Inspection,
    Palm,
    Plantation,
    Recommendation,
)


def test_table_names_registered():
    assert set(Base.metadata.tables) == {
        "plantations",
        "blocks",
        "palms",
        "inspections",
        "detections",
        "recommendations",
        "document_chunks",
        "detection_corrections",
    }


def test_hierarchy_relationships():
    plantation = Plantation(name="Estate A")
    block = Block(name="Block B-1", plantation=plantation)
    palm = Palm(palm_code="B-12", block=block)
    inspection = Inspection(image_url="/data/storage/img.png", status="queued", palm=palm)
    assert inspection.palm.block.plantation.name == "Estate A"


def test_detection_and_recommendation_link():
    inspection = Inspection(image_url="/data/storage/img.png", status="queued")
    detection = Detection(
        class_name="mature_basidiocarp",
        confidence=0.91,
        bbox_x_min=10.0,
        bbox_y_min=20.0,
        bbox_x_max=100.0,
        bbox_y_max=120.0,
        inspection=inspection,
    )
    recommendation = Recommendation(
        response_text="Sanitize area.",
        model_name="test-model",
        citations_json={},
        verification_status="approved",
        inspection=inspection,
    )
    assert detection.inspection is inspection
    assert recommendation.inspection is inspection


def test_document_chunk_vector_dim():
    chunk = DocumentChunk(chunk_id="doc-1#p1", document_id="doc-1", content="text", metadata_json={}, embedding=[0.0] * 1536)
    assert len(chunk.embedding) == 1536


def test_correction_link():
    inspection = Inspection(image_url="/data/storage/img.png", status="completed")
    correction = DetectionCorrection(
        corrected_by="agronomist-1",
        original_bbox={"x": 1},
        corrected_bbox={"x": 2},
        inspection=inspection,
    )
    assert correction.inspection is inspection
