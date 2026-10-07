from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Sequence

from .retriever import RetrievedPassage

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None, *, model: Any | None = None) -> None:
        self.model_name = model_name or os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
        self._model = model

    def rerank(
        self, query: str, passages: Sequence[RetrievedPassage], top_k: int
    ) -> list[RetrievedPassage]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not passages:
            return []
        model = self._model
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self.model_name)
            self._model = model
        scores = model.predict([(query, passage.content) for passage in passages])
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if len(scores) != len(passages):
            raise RuntimeError("Reranker returned an unexpected number of scores")
        ranked = [
            replace(passage, score=float(score))
            for passage, score in zip(passages, scores, strict=True)
        ]
        return sorted(ranked, key=lambda passage: passage.score, reverse=True)[:top_k]
