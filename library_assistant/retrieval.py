"""In-memory vector index for retrieval-augmented generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import Embedder
from .models import Book


@dataclass(frozen=True)
class SearchHit:
    book: Book
    score: float


class VectorIndex:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._books: list[Book] = []
        self._matrix = np.zeros((0, embedder.dim), dtype=np.float32)

    def build(self, books: list[Book]) -> VectorIndex:
        self._books = list(books)
        self._matrix = (
            self._embedder.embed([b.to_document() for b in self._books])
            if self._books
            else np.zeros((0, self._embedder.dim), dtype=np.float32)
        )
        return self

    def __len__(self) -> int:
        return len(self._books)

    def search(self, query: str, k: int = 5, min_score: float = 0.05) -> list[SearchHit]:
        if not self._books or not query.strip():
            return []
        query_vec = self._embedder.embed([query])[0]
        scores = self._matrix @ query_vec
        order = np.argsort(-scores)[:k]
        return [SearchHit(self._books[i], float(scores[i])) for i in order if scores[i] >= min_score]
