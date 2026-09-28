"""Embedding models used for retrieval.

`HashingEmbedder` is fast, deterministic and works offline, so tests and CI never need
a network call. It is lexical: it matches shared words, not meaning.
`SentenceTransformerEmbedder` gives true semantic search; install the optional
`semantic` extra and set LIA_EMBEDDER=sentence-transformers to use it.
"""

from __future__ import annotations

import hashlib
import re
from itertools import pairwise
from typing import Protocol

import numpy as np

_STOPWORDS = frozenset(
    "a an and are as at be by for from how i in into is it me my of on or so that the "
    "this to want what which with you your about any some good best book books".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) array of L2-normalized vectors."""
        ...


class HashingEmbedder:
    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.md5(feature.encode()).digest()
        index = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return index, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in pairwise(tokens)]
            for feature in features:
                index, sign = self._bucket(feature)
                out[row, index] += sign
            # Sublinear term frequency, then unit length so dot product = cosine.
            out[row] = np.sign(out[row]) * np.log1p(np.abs(out[row]))
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def build_embedder(name: str) -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    if name == "sentence-transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedder '{name}'. Use 'hashing' or 'sentence-transformers'.")
