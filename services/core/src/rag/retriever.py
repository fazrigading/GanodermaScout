from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Any, Protocol, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import DocumentChunk
from .embedding import EmbeddingClient


@lru_cache(maxsize=1)
def _default_embedding_client() -> EmbeddingClient:
    return EmbeddingClient()


@dataclass(frozen=True)
class RetrievedPassage:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]
    citation_anchor: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PassageReranker(Protocol):
    def rerank(
        self, query: str, passages: Sequence[RetrievedPassage], top_k: int
    ) -> list[RetrievedPassage]: ...


def _passage(row: DocumentChunk, score: float) -> RetrievedPassage:
    metadata = dict(row.metadata_json or {})
    return RetrievedPassage(
        chunk_id=row.chunk_id,
        document_id=row.document_id,
        content=row.content,
        metadata=metadata,
        citation_anchor=str(metadata.get("citation_anchor") or row.chunk_id),
        score=score,
    )


def search_dense(
    session: Session,
    query: str,
    *,
    embedding_client: EmbeddingClient | None = None,
    limit: int = 20,
) -> list[RetrievedPassage]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    client = embedding_client or _default_embedding_client()
    vector = client.embed_query(query)
    distance = DocumentChunk.embedding.cosine_distance(vector)
    statement = (
        select(DocumentChunk, distance.label("distance"))
        .where(DocumentChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(limit)
    )
    return [
        _passage(row, 1.0 - float(distance_value))
        for row, distance_value in session.execute(statement).all()
    ]


def search_sparse(
    session: Session,
    query: str,
    *,
    limit: int = 20,
) -> list[RetrievedPassage]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    query_vector = func.websearch_to_tsquery("english", query)
    score = func.ts_rank_cd(DocumentChunk.search_vector, query_vector)
    statement = (
        select(DocumentChunk, score.label("score"))
        .where(DocumentChunk.search_vector.op("@@")(query_vector))
        .order_by(score.desc())
        .limit(limit)
    )
    return [
        _passage(row, float(score_value))
        for row, score_value in session.execute(statement).all()
    ]


def reciprocal_rank_fusion(
    dense_results: Sequence[RetrievedPassage],
    sparse_results: Sequence[RetrievedPassage],
    *,
    k: int = 60,
) -> list[RetrievedPassage]:
    if k <= 0:
        raise ValueError("k must be positive")
    fused: dict[str, tuple[RetrievedPassage, float]] = {}
    for results in (dense_results, sparse_results):
        seen: set[str] = set()
        for rank, passage in enumerate(results, start=1):
            if passage.chunk_id in seen:
                continue
            seen.add(passage.chunk_id)
            prior, score = fused.get(passage.chunk_id, (passage, 0.0))
            fused[passage.chunk_id] = (prior, score + 1.0 / (k + rank))
    ranked = [replace(passage, score=score) for passage, score in fused.values()]
    return sorted(ranked, key=lambda passage: passage.score, reverse=True)


def hybrid_retrieve(
    session: Session,
    query: str,
    *,
    top_k: int = 5,
    candidate_k: int | None = None,
    embedding_client: EmbeddingClient | None = None,
    reranker: PassageReranker | None = None,
) -> list[RetrievedPassage]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    candidate_limit = candidate_k if candidate_k is not None else max(top_k * 4, top_k)
    if candidate_limit < top_k:
        raise ValueError("candidate_k must be at least top_k")

    dense = search_dense(
        session, query, embedding_client=embedding_client, limit=candidate_limit
    )
    sparse = search_sparse(session, query, limit=candidate_limit)
    fused = reciprocal_rank_fusion(dense, sparse)[:candidate_limit]
    if reranker is not None:
        return reranker.rerank(query, fused, top_k)
    return fused[:top_k]
