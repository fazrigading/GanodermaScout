import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.embedding import EmbeddingClient
from src.rag.reranker import CrossEncoderReranker
from src.rag.retriever import RetrievedPassage, reciprocal_rank_fusion


class LocalModelStub:
    def encode(self, texts, normalize_embeddings):
        assert normalize_embeddings is True
        return [[0.6, 0.8] for _ in texts]


class OpenAIEmbeddingStub:
    def __init__(self):
        self.call = None
        self.embeddings = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.2] * 1536),
                SimpleNamespace(index=0, embedding=[0.1] * 1536),
            ]
        )


class CrossEncoderStub:
    def predict(self, pairs):
        assert pairs == [("query", "A#p1"), ("query", "B#p1")]
        return [0.1, 0.9]


def passage(chunk_id: str, score: float = 0.0) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        document_id=chunk_id.split("#", 1)[0],
        content=chunk_id,
        metadata={"title": chunk_id},
        citation_anchor=chunk_id,
        score=score,
    )


def test_local_bge_embeddings_fit_database_dimension():
    client = EmbeddingClient(provider="huggingface", local_model=LocalModelStub())

    vector = client.embed_query("mature Ganoderma conk")

    assert len(vector) == 1536
    assert vector[:2] == [0.6, 0.8]
    assert vector[2:] == [0.0] * 1534


def test_openai_embeddings_keep_input_order_and_dimension():
    api = OpenAIEmbeddingStub()
    client = EmbeddingClient(provider="openai", openai_client=api)

    vectors = client.embed_texts(["first", "second"])

    assert len(vectors) == 2
    assert vectors[0][0] == 0.1
    assert vectors[1][0] == 0.2
    assert api.call["model"] == "text-embedding-3-small"
    assert api.call["dimensions"] == 1536


def test_rrf_combines_dense_and_sparse_order_with_citations():
    dense = [passage("A#p1"), passage("B#p1"), passage("C#p1")]
    sparse = [passage("B#p1"), passage("C#p1"), passage("D#p1")]

    ranked = reciprocal_rank_fusion(dense, sparse, k=60)

    assert [item.chunk_id for item in ranked] == ["B#p1", "C#p1", "A#p1", "D#p1"]
    assert abs(ranked[0].score - (1 / 61 + 1 / 62)) < 1e-12
    assert ranked[0].score > ranked[1].score > ranked[2].score > ranked[3].score
    assert [item.citation_anchor for item in ranked] == [item.chunk_id for item in ranked]


def test_cross_encoder_reranks_candidates_and_limits_results():
    reranker = CrossEncoderReranker(model=CrossEncoderStub())
    candidates = [passage("A#p1"), passage("B#p1")]

    ranked = reranker.rerank("query", candidates, top_k=1)

    assert [item.chunk_id for item in ranked] == ["B#p1"]
    assert ranked[0].citation_anchor == "B#p1"
