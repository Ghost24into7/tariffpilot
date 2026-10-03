"""Hybrid retrieval: BM25 + vector cosine fused with reciprocal-rank fusion. Pure Python."""
from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

_STOP = {"the", "of", "and", "or", "for", "with", "a", "an", "to", "in", "on", "other", "whether", "not", "such", "as"}


def tokenize(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in _STOP:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic offline embedder (char trigrams + words, hashed). Swap for GeminiEmbedder in production."""
    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in tokenize(t):
                v[zlib.crc32(w.encode()) % self.dim] += 1.0
                p = f"#{w}#"
                for i in range(len(p) - 2):
                    v[zlib.crc32(p[i:i + 3].encode()) % self.dim] += 0.3
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = "gemini-embedding-001", batch: int = 50):
        from google import genai  # optional dependency
        self._c, self.model, self.batch = genai.Client(api_key=api_key), model, batch

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            r = self._c.models.embed_content(model=self.model, contents=texts[i:i + self.batch])
            for e in r.embeddings:
                n = math.sqrt(sum(x * x for x in e.values)) or 1.0
                out.append([x / n for x in e.values])
        return out


@dataclass
class Doc:
    id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    doc: Doc
    score: float
    bm25: float
    cos: float


class HybridIndex:
    def __init__(self, docs: list[Doc], embedder: Embedder | None = None, k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.embedder = embedder or HashEmbedder()
        self._tok = [tokenize(d.text) for d in docs]
        self._tf = []
        self._df: dict[str, int] = {}
        for toks in self._tok:
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            for t in tf:
                self._df[t] = self._df.get(t, 0) + 1
        self._avg = (sum(len(t) for t in self._tok) / len(docs)) if docs else 0.0
        self._vec = self.embedder.embed([d.text for d in docs]) if docs else []

    def _bm25(self, q: list[str], i: int) -> float:
        n, s, dl = len(self.docs), 0.0, len(self._tok[i])
        for t in q:
            f = self._tf[i].get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
            s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self._avg or 1)))
        return s

    def search(self, query: str, k: int = 5, where: Callable[[Doc], bool] | None = None) -> list[Hit]:
        if not self.docs:
            return []
        q = tokenize(query)
        qv = self.embedder.embed([query])[0]
        idx = [i for i, d in enumerate(self.docs) if where is None or where(d)]
        bm = {i: self._bm25(q, i) for i in idx}
        cs = {i: sum(a * b for a, b in zip(qv, self._vec[i])) for i in idx}
        fused: dict[int, float] = {i: 0.0 for i in idx}
        for scores in (bm, cs):
            for rank, i in enumerate(sorted(idx, key=lambda j: (-scores[j], j))):
                if scores[i] > 0:
                    fused[i] += 1.0 / (60 + rank + 1)
        top = sorted(idx, key=lambda j: (-fused[j], j))[:k]
        return [Hit(self.docs[i], round(fused[i], 5), round(bm[i], 4), round(cs[i], 4)) for i in top if fused[i] > 0 and (bm[i] > 0 or cs[i] >= 0.15)]
