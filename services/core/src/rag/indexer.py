from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ..db.models import DocumentChunk
from .embedding import EMBEDDING_DIMENSIONS, EmbeddingClient
from .ingestion import IngestedChunk, ingest_corpus


def index_chunks(
    session: Session,
    chunks: Sequence[IngestedChunk],
    *,
    embedding_client: EmbeddingClient | None = None,
    batch_size: int = 32,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Chunk IDs must be unique within an indexing batch")
    if not chunks:
        return 0

    client = embedding_client or EmbeddingClient()
    indexed = 0
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        vectors = client.embed_texts([chunk.content for chunk in batch])
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding client returned an unexpected number of vectors")
        if any(len(vector) != EMBEDDING_DIMENSIONS for vector in vectors):
            raise ValueError(f"Each embedding must have {EMBEDDING_DIMENSIONS} dimensions")

        chunk_ids = [chunk.chunk_id for chunk in batch]
        existing = session.scalars(
            select(DocumentChunk).where(DocumentChunk.chunk_id.in_(chunk_ids))
        ).all()
        rows = {row.chunk_id: row for row in existing}
        for chunk, vector in zip(batch, vectors, strict=True):
            metadata = {
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "citation_anchor": chunk.citation_anchor,
                "section": chunk.section,
                "source_start": chunk.source_start,
                "source_end": chunk.source_end,
                "token_count": chunk.token_count,
                "embedding_provider": client.provider,
                "embedding_model": client.model_name,
            }
            row = rows.get(chunk.chunk_id)
            if row is None:
                row = DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    metadata_json=metadata,
                    embedding=vector,
                )
                session.add(row)
                rows[chunk.chunk_id] = row
            else:
                row.document_id = chunk.document_id
                row.content = chunk.content
                row.metadata_json = metadata
                row.embedding = vector
            indexed += 1
        session.flush()
    return indexed


def index_corpus(
    session: Session,
    corpus_dir: str | Path,
    *,
    embedding_client: EmbeddingClient | None = None,
    batch_size: int = 32,
) -> int:
    chunks = ingest_corpus(corpus_dir)
    return index_chunks(
        session,
        chunks,
        embedding_client=embedding_client,
        batch_size=batch_size,
    )


def _sync_database_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index agronomy corpus passages into PostgreSQL")
    parser.add_argument("--corpus", default="data/sample_dataset/corpus")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("--database-url or DATABASE_URL is required")

    engine = create_engine(_sync_database_url(args.database_url))
    try:
        with Session(engine) as session:
            indexed = index_corpus(session, args.corpus)
            session.commit()
    finally:
        engine.dispose()
    print(f"Indexed {indexed} document chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
