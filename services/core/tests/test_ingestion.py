import os
import sys
from pathlib import Path

import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.chunker import chunk_text
from src.rag.ingestion import ingest_corpus

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORPUS = _REPO_ROOT / "data" / "sample_dataset" / "corpus"


def test_sample_corpus_preserves_metadata_sections_and_citation_ids():
    chunks = ingest_corpus(_CORPUS)
    sanitation = [chunk for chunk in chunks if chunk.document_id == "BSR-EXT-001"]

    assert [chunk.chunk_id for chunk in sanitation] == [
        "BSR-EXT-001#p1",
        "BSR-EXT-001#p2",
        "BSR-EXT-001#p3",
    ]
    assert [chunk.section for chunk in sanitation] == [
        "Sanitation around infected palms",
        "Isolation of severe cases",
        "Monitoring after sanitation",
    ]
    first = sanitation[0]
    assert first.metadata["title"] == "Basal Stem Rot Field Sanitation for Oil Palm"
    assert first.metadata["author"] == "Estate Extension Service"
    assert first.metadata["year"] == 2023
    assert first.metadata["source"] == "Sample extension bulletin (synthetic sample for smoke testing)"
    assert first.metadata["license"] == "CC-BY-4.0 (sample text)"
    assert first.citation_anchor == first.chunk_id
    assert "## Sanitation around infected palms" in first.content
    assert first.source_start < first.source_end
    source = (_CORPUS / "BSR-EXT-001-sanitation.md").read_text(encoding="utf-8")
    assert source[first.source_start : first.source_end].startswith(
        "When a mature basidiocarp (developed conk) is observed"
    )


def test_chunker_enforces_token_limit_and_overlaps_adjacent_passages():
    text = " ".join(f"protocol{index}" for index in range(1200))
    chunks = chunk_text(text)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    assert len(chunks) >= 3
    assert all(chunk.token_count <= 500 for chunk in chunks)
    overlap_text = tokenizer.decode(tokenizer.encode(chunks[0].content)[-50:])
    assert chunks[1].content.startswith(overlap_text)


def test_html_documents_extract_metadata_and_headers(tmp_path):
    (tmp_path / "guide.html").write_text(
        """<!doctype html>
        <html><head><title>Guide title</title>
        <meta name="author" content="Estate Agronomist">
        <meta name="date" content="2022-04-01">
        <meta name="publisher" content="Extension Office">
        <meta name="license" content="CC-BY-4.0"></head>
        <body><h1>Guide title</h1><h2>Field practice</h2>
        <p>Inspect the palm base and remove loose debris.</p></body></html>""",
        encoding="utf-8",
    )

    chunks = ingest_corpus(tmp_path)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata["title"] == "Guide title"
    assert chunk.metadata["author"] == "Estate Agronomist"
    assert chunk.metadata["year"] == 2022
    assert chunk.metadata["source"] == "Extension Office"
    assert chunk.metadata["license"] == "CC-BY-4.0"
    assert "## Field practice" in chunk.content
    assert "Inspect the palm base" in chunk.content
